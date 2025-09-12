import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import json


def read_tsv(tsv_path):
    """
    qualysisのtsvファイルを読み込み、必要なマーカーのみを抽出して返す
    Args:
        tsv_path (str or Path): 読み込むtsvファイルのパス
    Returns:
        pd.DataFrame: 読み込んだデータフレーム
    """
    df = pd.read_csv(tsv_path, sep='\t', header=10)  #Qualisys
    df = df.apply(pd.to_numeric, errors='coerce')  # 数値に変換
    df.set_index("Frame", inplace=True)  # フレームをインデックスに設定
    df.index = df.index - 1  # フレームを0から始まるように調整(GoProと合わせやすくするため)
    df.dropna(axis=1, how='all', inplace=True)  # 全ての値がNaNの行を削除
    # print(f"df: {df.columns}")
    # df = df.filter(regex='RASI|LASI|RPSI|LPSI|RKNE|LKNE|RANK|LANK|RTOE|LTOE|RHEE|LHEE')
    df = df.filter(regex='RASI|LASI|RPSI|LPSI|RILC|LILC|RKNE|LKNE|RKNE2|LKNE2|RANK|LANK|RANK2|LANK2|RTOE|LTOE|RHEE|LHEE|R5TOE|L5TOE')
    # print(f"filetered_df: {df.columns}")
    return df

def get_marker_data(df: pd.DataFrame, marker_name: str) -> np.ndarray | None:
    """
    データフレームから指定されたマーカーの3次元座標データを抽出します。
    列が存在しない場合は、警告を表示してNoneを返します。

    Args:
        df (pd.DataFrame): データフレーム。
        marker_name (str): マーカー名 (例: 'RASI')。

    Returns:
        np.ndarray | None: マーカーの座標データ配列。存在しない場合はNone。
    """
    required_columns = [f'{marker_name} X', f'{marker_name} Y', f'{marker_name} Z']

    # 必要な列がすべてデータフレームに存在するかチェック
    if all(col in df.columns for col in required_columns):
        return df[required_columns].to_numpy()
    else:
        # 存在しない場合は警告を表示してNoneを返す
        print(f"警告: マーカー '{marker_name}' のデータが見つかりませんでした。このマーカーの処理はスキップされます。")
        return None

def interpolate_short_gaps(df, max_gap_size, method='spline', order=3):
    """
    データフレームの各列で、連続するNaNの数がmax_gap_size以下の区間のみを補間します。
    有効なデータ点が order より少ない列は、エラーを避けるために補間処理をスキップします。
    """
    output_df = df.copy()

    # 各列をループして処理
    for col in df.columns:
        s = df[col]

        # 3次スプライン補間(order=3)には最低でも4点(order+1)のデータが必要。
        # 有効なデータ点の数が足りない場合は、エラーになるためスキップする。
        if s.notna().sum() <= order:
            print(f"警告: 列 '{col}' の有効データ点数が {s.notna().sum()} 点以下のため、スプライン補間をスキップします。")
            continue

        # この列だけを補間したバージョンを作成
        interpolated_series = s.interpolate(method=method, order=order)

        # 連続するNaNのグループに一意のIDを振る
        is_not_nan = s.notna()
        nan_groups = (is_not_nan != is_not_nan.shift()).cumsum()

        # 各NaNが、どのくらいの長さの連続NaNグループに属しているかを計算
        gap_sizes = s[s.isna()].groupby(nan_groups).transform('size')

        # 補間を実行するNaNの位置を特定（連続長がmax_gap_size以下のもの）
        to_interpolate_idx = gap_sizes[gap_sizes <= max_gap_size].index

        # 特定した位置の値を、補間したシリーズの値で置き換える
        output_df.loc[to_interpolate_idx, col] = interpolated_series.loc[to_interpolate_idx]

    return output_df
def _find_rigid_transform(A, B):
    """
    2つの点群AとB間の最適な剛体変換(回転Rと移動t)を計算。
    B = R @ A + t を満たすRとtを算出。
    Kabschアルゴリズムを使用。https://hunterheidenreich.com/posts/kabsch-algorithm/
    """
    assert A.shape == B.shape
    num_rows, num_cols = A.shape
    if num_rows == 0:
        return np.identity(num_cols), np.zeros(num_cols)

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am.T @ Bm

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # 反転を防ぐための処理
    if np.linalg.det(R) < 0:
        Vt[num_cols-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    return R, t



def interpolate_pelvis_rigid_body(df: pd.DataFrame, reference_pos_path: str, target_markers: list) -> pd.DataFrame:
    """
    Tポーズの参照座標を元に、剛体モデルフィッティングで骨盤マーカーの欠損を補間
    """
    try:
        with open(reference_pos_path, 'r') as f:
            ref_pos_dict = json.load(f)
    except FileNotFoundError:
        print(f"エラー: 参照ファイル '{reference_pos_path}' が見つかりません。剛体補間はスキップされます。")
        return df

    interpolated_df = df.copy()

    # データフレームに存在しない参照マーカーをref_pos_dictから除外
    ref_pos_dict = {k: v for k, v in ref_pos_dict.items() if f'{k} X' in df.columns}

    processed_frames = 0

    for frame_idx, row in df.iterrows():
        # ターゲットマーカーが一つでも欠損しているか確認
        is_missing = any(row[[f'{m} X', f'{m} Y', f'{m} Z']].isnull().any() for m in target_markers)
        if not is_missing: #欠損がない場合はスキップ
            continue

        # 現在のフレームで利用可能な参照マーカーの座標を取得
        ref_points = []
        walk_points = []

        for marker_name, ref_coord in ref_pos_dict.items():
            walk_coord = row[[f'{marker_name} X', f'{marker_name} Y', f'{marker_name} Z']].values
            if not np.isnan(walk_coord).any():
                ref_points.append(ref_coord)
                walk_points.append(walk_coord)

        # 3点以上ないと姿勢は決まらない
        if len(ref_points) < 3:
            continue

        ref_points = np.array(ref_points)
        walk_points = np.array(walk_points)

        # 最適な変換を計算 (参照点群 -> 歩行中の点群)
        R, t = _find_rigid_transform(ref_points, walk_points)

        # 欠損しているマーカーを変換して補間
        for marker_name in target_markers:
            # このマーカーが欠損している場合のみ処理
            if row[[f'{marker_name} X', f'{marker_name} Y', f'{marker_name} Z']].isnull().any():
                if marker_name in ref_pos_dict:
                    ref_point_missing = np.array(ref_pos_dict[marker_name])

                    # 参照座標を現在の歩行座標系に変換
                    transformed_point = R @ ref_point_missing + t

                    # データフレームを更新
                    interpolated_df.loc[frame_idx, f'{marker_name} X'] = transformed_point[0]
                    interpolated_df.loc[frame_idx, f'{marker_name} Y'] = transformed_point[1]
                    interpolated_df.loc[frame_idx, f'{marker_name} Z'] = transformed_point[2]

        processed_frames += 1
    return interpolated_df


def butterworth_filter(df, cutoff, order, fs):
    def _butter_lowpass_fillter(column_data, order, cutoff, fs):
        nyquist_freq = fs / 2
        normal_cutoff = cutoff / nyquist_freq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        data_to_filter = column_data
        filtered_data = filtfilt(b, a, data_to_filter)
        column_data = filtered_data
        return column_data

    butter_df = df.copy()
    butter_df = butter_df.apply(_butter_lowpass_fillter, args=(order, cutoff, fs))  #4次のバターワースローパスフィルタ
    return butter_df


def butterworth_filter_no_nan_gaps(df, cutoff, order, fs):
    """
    欠損値のない連続した区間ごとに、バターワースフィルタを適用して平滑化する。
    
    Args:
        df (pd.DataFrame): フィルタリングするデータフレーム
        cutoff (float): カットオフ周波数
        order (int): フィルタの次数
        fs (int): サンプリング周波数
    
    Returns:
        pd.DataFrame: フィルタリングされたデータフレーム
    """
    filtered_df = df.copy()
    
    # 欠損値のない連続したセグメントを見つける
    for col in df.columns:
        s = df[col]
        
        # 連続した非NaNデータのインデックスを見つける
        is_valid = s.notna()
        # 連続した非NaNデータのグループにIDを割り当てる
        group_ids = is_valid.cumsum()
        
        # グループごとに処理
        for group_id, group_df in s[is_valid].groupby(group_ids):
            # グループのデータがフィルタリングに必要な長さ（最低でも order + 1）があるか確認
            # filtfiltはデータを前後に処理するため、最低でも2 * order + 1 点が必要
            if len(group_df) > 2 * order:
                # フィルタリングを実行
                nyquist_freq = fs / 2
                normal_cutoff = cutoff / nyquist_freq
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                
                # filtfiltでフィルタリング
                filtered_series = pd.Series(
                    filtfilt(b, a, group_df.values),
                    index=group_df.index
                )
                
                # 元のデータフレームをフィルタリング結果で更新
                filtered_df.loc[filtered_series.index, col] = filtered_series
    
    return filtered_df



def frame2percent(acc_frame_series):
    ori_idx = acc_frame_series.index.to_numpy()
    normalized_ori_idx = (ori_idx - ori_idx[0]) / (ori_idx[-1] - ori_idx[0]) * 100  # 横軸を0~100に正規化
    acc_data = acc_frame_series.to_numpy()
    # 0~99で1刻みになるようリサンプリング
    new_start_idx = 0
    new_end_idx = 100
    num_points = 100
    new_idx = np.linspace(new_start_idx, new_end_idx, num_points)
    new_acc_data = np.interp(new_idx, normalized_ori_idx, acc_data)
    # print(f"new_acc_data:{new_acc_data}")
    return new_acc_data