from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.ticker as mticker
from scipy.signal import butter, filtfilt

# qualysisファイルのあるディレクトリ
tsv_dir = Path(r"G:\gait_pattern\20251024_fukuyama\qtm_label")
# qualysisisのtsvファイルを取得(QTMで補間処理を実施している前提)
tsv_files = tsv_dir.glob("*.tsv") 
tsv_files = list(tsv_files)

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
    df = df.filter(regex='RASI|LASI|RPSI|LPSI|RKNE|LKNE|RKNE2|LKNE2|RANK|LANK|RANK2|LANK2|RTOE|LTOE|RHEE|LHEE|R5TOE|L5TOE')
    # df = df.filter(regex='RASI|LASI|RPSI|LPSI|RILC|LILC|RKNE|LKNE|RKNE2|LKNE2|RANK|LANK|RANK2|LANK2|RTOE|LTOE|RHEE|LHEE|R5TOE|L5TOE')
    # print(f"filetered_df: {df.columns}")
    return df

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


def main():
    for itsv, tsv_file in enumerate(tsv_files):
        print(f"Processing {itsv+1}/{len(tsv_files)}: {tsv_file.name}")
        full_df = read_tsv(tsv_file)  #tsvファイルの読み込み
        target_df = full_df.copy()
        nan_df = target_df.replace(0, np.nan)  #0をNaNに置き換え
        
        # 補間処理 欠損が20フレーム以下の区間をスプライン補間(QTMでやり残しがある場合の保険)
        interpolated_df = interpolate_short_gaps(nan_df, max_gap_size=20, method='spline', order=3)
        
        
        # 平滑化処理 バターワースフィルタ
        butter_df = butterworth_filter_no_nan_gaps(interpolated_df, cutoff=6, order=4, fs=120) #4次のバターワースローパスフィルタ

        # 角度計算に必要な全てのマーカー名をリストアップ
        required_markers = [
            'RASI', 'LASI', 'RPSI', 'LPSI', 'RKNE', 'LKNE', 'RKNE2', 'LKNE2',
            'RANK', 'LANK', 'RANK2', 'LANK2', 'RTOE', 'LTOE', 'RHEE', 'LHEE'
        ]

        # マーカー名から、対応するすべての列名(X, Y, Z)のリストを作成
        all_marker_columns = []
        for marker in required_markers:
            all_marker_columns.extend([f'{marker} X', f'{marker} Y', f'{marker} Z'])

        # 実際にデータフレームに存在する列のみを対象にする
        existing_columns = [col for col in all_marker_columns if col in butter_df.columns]

        # 必要なマーカーが全て揃っているフレーム（行）のインデックスを抽出
        valid_frames_df = butter_df.dropna(subset=existing_columns)
        valid_frames = valid_frames_df.index

        print(f"全マーカーが存在する有効なフレーム総数: {len(valid_frames)}")
        if len(valid_frames) > 0:
            # 有効フレームのシーケンスの差分を計算し、1より大きい場所（ギャップ）を見つける
            gaps = np.where(np.diff(valid_frames) > 1)[0] + 1
            # ギャップの位置で配列を分割し、連続した範囲のリストを作成する
            ranges = np.split(valid_frames, gaps)

            # print("検出された有効フレームの範囲:")
            # for r in ranges:
            #     if len(r) > 1:
            #         print(f"  - フレーム {r[0]} から {r[-1]} (計 {len(r)} フレーム)")
            #     else: #範囲が1フレームしかない場合
            #         print(f"  - フレーム {r[0]} (単一フレーム)")
        else:
            print("警告: 全てのマーカーが揃っているフレームが1つもありません。角度計算をスキップします。")
            continue

        # 各有効範囲の角度計算結果を保存するリスト
        all_angle_dfs = []

        for _range in ranges:
            print(f"range: {_range[0]} - {_range[-1]} (計 {len(_range)} フレーム)")
            try:
                # 現在の範囲 'r' のデータのみを抽出する
                rasi = butter_df.loc[_range, ['RASI X', 'RASI Y', 'RASI Z']].to_numpy()
                lasi = butter_df.loc[_range, ['LASI X', 'LASI Y', 'LASI Z']].to_numpy()
                rpsi = butter_df.loc[_range, ['RPSI X', 'RPSI Y', 'RPSI Z']].to_numpy()
                lpsi = butter_df.loc[_range, ['LPSI X', 'LPSI Y', 'LPSI Z']].to_numpy()
                rknee = butter_df.loc[_range, ['RKNE X', 'RKNE Y', 'RKNE Z']].to_numpy()
                lknee = butter_df.loc[_range, ['LKNE X', 'LKNE Y', 'LKNE Z']].to_numpy()
                rknee2 = butter_df.loc[_range, ['RKNE2 X', 'RKNE2 Y', 'RKNE2 Z']].to_numpy()
                lknee2 = butter_df.loc[_range, ['LKNE2 X', 'LKNE2 Y', 'LKNE2 Z']].to_numpy()
                rank = butter_df.loc[_range, ['RANK X', 'RANK Y', 'RANK Z']].to_numpy()
                lank = butter_df.loc[_range, ['LANK X', 'LANK Y', 'LANK Z']].to_numpy()
                rank2 = butter_df.loc[_range, ['RANK2 X', 'RANK2 Y', 'RANK2 Z']].to_numpy()
                lank2 = butter_df.loc[_range, ['LANK2 X', 'LANK2 Y', 'LANK2 Z']].to_numpy()
                rtoe = butter_df.loc[_range, ['RTOE X', 'RTOE Y', 'RTOE Z']].to_numpy()
                ltoe = butter_df.loc[_range, ['LTOE X', 'LTOE Y', 'LTOE Z']].to_numpy()
                rhee = butter_df.loc[_range, ['RHEE X', 'RHEE Y', 'RHEE Z']].to_numpy()
                lhee = butter_df.loc[_range, ['LHEE X', 'LHEE Y', 'LHEE Z']].to_numpy()
            except KeyError as e:
                print(f"エラー: 範囲 {_range[0]}-{_range[-1]} で必要なマーカーが不足: {e}")
                continue

            angle_list_range = []

            # ループの範囲を現在の有効範囲 '_range' に限定
            for frame_idx_in_range, original_frame_num in enumerate(_range):
                try:
                    # 座標系定義の計算 (インデックスは範囲内でのインデックス frame_idx_in_range を使用)
                    # print(f"Frame {frame}:")
                    d_asi = np.linalg.norm(rasi[frame_idx_in_range,:] - lasi[frame_idx_in_range,:])
                    d_leg = (np.linalg.norm(rank[frame_idx_in_range,:] - rasi[frame_idx_in_range,:]) + np.linalg.norm(lank[frame_idx_in_range, :] - lasi[frame_idx_in_range,:])) / 2  #大腿長の平均
                    r = 0.012 #使用したマーカー径

                    h = 1.60 #被験者身長

                    k = h/1.7
                    beta = 0.1 * np.pi #[rad]
                    theta = 0.496 #[rad]
                    c = 0.115 * d_leg - 0.00153
                    x_dis = 0.1288 * d_leg - 0.04856

                    # skycom + davis
                    x_rthigh = -(x_dis +r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
                    x_lthigh = -(x_dis +r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
                    y_rthigh = +(c * np.sin(theta) - d_asi/2)
                    y_lthigh = -(c * np.sin(theta)- d_asi/2)
                    z_rthigh = -(x_dis + r) * np.sin(beta) - c * np.cos(theta) * np.cos(beta)
                    z_lthigh = -(x_dis + r) * np.sin(beta) - c * np.cos(theta) * np.cos(beta)
                    rthigh_pelvis = np.array([x_rthigh, y_rthigh, z_rthigh]).T
                    lthigh_pelvis = np.array([x_lthigh, y_lthigh, z_lthigh]).T

                    # 骨盤原点1 ASISの中点
                    hip_0 = (rasi[frame_idx_in_range,:] + lasi[frame_idx_in_range,:]) / 2
                    # 仙骨 PSISの中点
                    sacrum = (rpsi[frame_idx_in_range,:] + lpsi[frame_idx_in_range,:]) / 2

                    #骨盤節座標系1（原点はhip_0）
                    e_y0_pelvis_0 = (lasi[frame_idx_in_range,:] - rasi[frame_idx_in_range,:])/np.linalg.norm(lasi[frame_idx_in_range,:] - rasi[frame_idx_in_range,:])
                    e_x_pelvis_0 = (hip_0 - sacrum)/np.linalg.norm(hip_0 - sacrum)
                    e_z_pelvis_0 = np.cross(e_x_pelvis_0, e_y0_pelvis_0)/np.linalg.norm(np.cross(e_x_pelvis_0, e_y0_pelvis_0))
                    e_y_pelvis_0 = np.cross(e_z_pelvis_0, e_x_pelvis_0)
                    
                    # 骨盤節座標系の定義：Davisモデルに従って骨盤節座標系1をそのまま参照座標系として使用(もともとはL250-255の処理)
                    e_x_pelvis = e_x_pelvis_0
                    e_y_pelvis = e_y_pelvis_0
                    e_z_pelvis = e_z_pelvis_0
                    rot_pelvis = np.array([e_x_pelvis, e_y_pelvis, e_z_pelvis]).T
            
                    transformation_matrix = np.array([[e_x_pelvis_0[0], e_y_pelvis_0[0], e_z_pelvis_0[0], hip_0[0]],
                                                        [e_x_pelvis_0[1], e_y_pelvis_0[1], e_z_pelvis_0[1], hip_0[1]],
                                                        [e_x_pelvis_0[2], e_y_pelvis_0[2], e_z_pelvis_0[2], hip_0[2]],
                                                        [0,       0,       0,       1]])

                    #モーキャプの座標系に変換してもう一度計算
                    rthigh = np.dot(transformation_matrix, np.append(rthigh_pelvis, 1))[:3]
                    lthigh = np.dot(transformation_matrix, np.append(lthigh_pelvis, 1))[:3]
                    hip = (rthigh + lthigh) / 2

                    # 腰椎節原点
                    lumbar = (0.47 * (rasi[frame_idx_in_range,:] + lasi[frame_idx_in_range,:]) / 2 + 0.53 * (rpsi[frame_idx_in_range,:] + lpsi[frame_idx_in_range,:]) / 2) + 0.02 * k * np.array([0, 0, 1])

                    # #SKYCOMマニュアルに従って骨盤節座標系を再定義する場合(現在は未使用→Davisモデルの座標系をそのまま使用する処理L231-235に代替)
                    # e_y0_pelvis = (lthigh - rthigh)/np.linalg.norm(lthigh - rthigh)
                    # e_z_pelvis = (lumbar - hip)/np.linalg.norm(lumbar - hip)
                    # e_x_pelvis = np.cross(e_y0_pelvis, e_z_pelvis)/np.linalg.norm(np.cross(e_y0_pelvis, e_z_pelvis))
                    # e_y_pelvis = np.cross(e_z_pelvis, e_x_pelvis)
                    # rot_pelvis = np.array([e_x_pelvis, e_y_pelvis, e_z_pelvis]).T

                    #必要な原点の設定
                    rshank = (rknee[frame_idx_in_range, :] + rknee2[frame_idx_in_range, :]) / 2
                    lshank = (lknee[frame_idx_in_range, :] + lknee2[frame_idx_in_range, :]) / 2
                    rfoot = (rank[frame_idx_in_range,:] + rank2[frame_idx_in_range,:]) / 2
                    lfoot = (lank[frame_idx_in_range, :] + lank2[frame_idx_in_range,:]) / 2

                    #右大腿節座標系（原点はrthigh）
                    e_y0_rthigh = rknee2[frame_idx_in_range, :] - rknee[frame_idx_in_range, :]
                    e_z_rthigh = (rshank - rthigh)/np.linalg.norm(rshank - rthigh)
                    e_x_rthigh = np.cross(e_y0_rthigh, e_z_rthigh)/np.linalg.norm(np.cross(e_y0_rthigh, e_z_rthigh))
                    e_y_rthigh = np.cross(e_z_rthigh, e_x_rthigh)
                    rot_rthigh = np.array([e_x_rthigh, e_y_rthigh, e_z_rthigh]).T

                    #左大腿節座標系（原点はlthigh）
                    e_y0_lthigh = lknee[frame_idx_in_range, :] - lknee2[frame_idx_in_range, :]
                    e_z_lthigh = (lshank - lthigh)/np.linalg.norm(lshank - lthigh)
                    e_x_lthigh = np.cross(e_y0_lthigh, e_z_lthigh)/np.linalg.norm(np.cross(e_y0_lthigh, e_z_lthigh))
                    e_y_lthigh = np.cross(e_z_lthigh, e_x_lthigh)
                    rot_lthigh = np.array([e_x_lthigh, e_y_lthigh, e_z_lthigh]).T

                    #右下腿節座標系（原点はrshank）
                    e_y0_rshank = rknee2[frame_idx_in_range, :] - rknee[frame_idx_in_range, :]
                    e_z_rshank = (rshank - rfoot)/np.linalg.norm(rshank - rfoot)
                    e_x_rshank = np.cross(e_y0_rshank, e_z_rshank)/np.linalg.norm(np.cross(e_y0_rshank, e_z_rshank))
                    e_y_rshank = np.cross(e_z_rshank, e_x_rshank)
                    rot_rshank = np.array([e_x_rshank, e_y_rshank, e_z_rshank]).T

                    #左下腿節座標系（原点はlshank）
                    e_y0_lshank = lknee[frame_idx_in_range, :] - lknee2[frame_idx_in_range, :]
                    e_z_lshank = (lshank - lfoot)/np.linalg.norm(lshank - lfoot)
                    e_x_lshank = np.cross(e_y0_lshank, e_z_lshank)/np.linalg.norm(np.cross(e_y0_lshank, e_z_lshank))
                    e_y_lshank = np.cross(e_z_lshank, e_x_lshank)
                    rot_lshank = np.array([e_x_lshank, e_y_lshank, e_z_lshank]).T

                    #右足節座標系 AIST参照（原点はrfoot）
                    e_z_rfoot = (rtoe[frame_idx_in_range,:] - rhee[frame_idx_in_range,:]) / np.linalg.norm(rtoe[frame_idx_in_range,:] - rhee[frame_idx_in_range,:])
                    e_y0_rfoot = rank[frame_idx_in_range,:] - rank2[frame_idx_in_range,:]
                    e_x_rfoot = np.cross(e_z_rfoot, e_y0_rfoot)/np.linalg.norm(np.cross(e_z_rfoot, e_y0_rfoot))
                    e_y_rfoot = np.cross(e_z_rfoot, e_x_rfoot)
                    rot_rfoot = np.array([e_x_rfoot, e_y_rfoot, e_z_rfoot]).T

                    #左足節座標系 AIST参照（原点はlfoot）
                    e_z_lfoot = (ltoe[frame_idx_in_range,:] - lhee[frame_idx_in_range,:]) / np.linalg.norm(ltoe[frame_idx_in_range,:] - lhee[frame_idx_in_range,:])
                    e_y0_lfoot = lank2[frame_idx_in_range,:] - lank[frame_idx_in_range,:]
                    e_x_lfoot = np.cross(e_z_lfoot, e_y0_lfoot)/np.linalg.norm(np.cross(e_z_lfoot, e_y0_lfoot))
                    e_y_lfoot = np.cross(e_z_lfoot, e_x_lfoot)
                    rot_lfoot = np.array([e_x_lfoot, e_y_lfoot, e_z_lfoot]).T

                    # 相対回転行列の計算
                    r_hip_realative_rotation = np.dot(np.linalg.inv(rot_rthigh), rot_pelvis)  #骨盤節に合わせるための大腿節の回転行列
                    l_hip_realative_rotation = np.dot(np.linalg.inv(rot_lthigh), rot_pelvis)
                    r_knee_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rthigh)  #大腿節に合わせるための下腿節の回転行列
                    l_knee_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lthigh)
                    r_ankle_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rfoot)  #足節に合わせるための下腿節の回転行列
                    l_ankle_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lfoot)

                    r_hip_angle_rot = R.from_matrix(r_hip_realative_rotation)
                    l_hip_angle_rot = R.from_matrix(l_hip_realative_rotation)
                    r_knee_angle_rot = R.from_matrix(r_knee_realative_rotation)
                    l_knee_angle_rot = R.from_matrix(l_knee_realative_rotation)
                    r_ankle_angle_rot = R.from_matrix(r_ankle_realative_rotation)
                    l_ankle_angle_rot = R.from_matrix(l_ankle_realative_rotation)

                    # 回転行列から回転角を計算 XYZ大文字だと内因性，xyz小文字だと外因性
                    # 屈曲-伸展
                    r_hip_angle = r_hip_angle_rot.as_euler('YZX', degrees=True)[0]
                    l_hip_angle = l_hip_angle_rot.as_euler('YZX', degrees=True)[0]
                    r_knee_angle = r_knee_angle_rot.as_euler('YZX', degrees=True)[0]
                    l_knee_angle = l_knee_angle_rot.as_euler('YZX', degrees=True)[0]
                    r_ankle_angle = r_ankle_angle_rot.as_euler('YZX', degrees=True)[0]
                    l_ankle_angle = l_ankle_angle_rot.as_euler('YZX', degrees=True)[0]

                    angle_list_range.append([r_hip_angle, l_hip_angle, r_knee_angle, l_knee_angle, r_ankle_angle, l_ankle_angle])
                     
                    plot_flag = False
                    if plot_flag:
                        if original_frame_num == 166:
                            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
                            ax.set_xlabel("x")
                            ax.set_ylabel("y")
                            ax.set_zlabel("z")
                            ax.set_xlim(-1000, 1000)
                            ax.set_ylim(-1000, 1000)
                            ax.set_zlim(0, 2000)
                            #frame数を表示
                            ax.text2D(0.5, 0.01, f"frame = {original_frame_num}", transform=ax.transAxes)
                            #方向を設定
                            ax.view_init(elev=0, azim=0)

                            ax.scatter(rasi[frame_idx_in_range,:][0], rasi[frame_idx_in_range,:][1], rasi[frame_idx_in_range,:][2], label='rasi')
                            ax.scatter(lasi[frame_idx_in_range,:][0], lasi[frame_idx_in_range,:][1], lasi[frame_idx_in_range,:][2], label='lasi')
                            ax.scatter(rpsi[frame_idx_in_range,:][0], rpsi[frame_idx_in_range,:][1], rpsi[frame_idx_in_range,:][2], label='rpsi')
                            ax.scatter(lpsi[frame_idx_in_range,:][0], lpsi[frame_idx_in_range,:][1], lpsi[frame_idx_in_range,:][2], label='lpsi')
                            ax.scatter(rfoot[0], rfoot[1], rfoot[2], label='rfoot')
                            ax.scatter(lfoot[0], lfoot[1], lfoot[2], label='lfoot')
                            ax.scatter(rshank[0], rshank[1], rshank[2], label='rshank')
                            ax.scatter(lshank[0], lshank[1], lshank[2], label='lshank')
                            ax.scatter(rtoe[frame_idx_in_range,:][0], rtoe[frame_idx_in_range,:][1], rtoe[frame_idx_in_range,:][2], label='rtoe')
                            ax.scatter(ltoe[frame_idx_in_range,:][0], ltoe[frame_idx_in_range,:][1], ltoe[frame_idx_in_range,:][2], label='ltoe')
                            ax.scatter(rhee[frame_idx_in_range,:][0], rhee[frame_idx_in_range,:][1], rhee[frame_idx_in_range,:][2], label='rhee')
                            ax.scatter(lhee[frame_idx_in_range, :][0], lhee[frame_idx_in_range, :][1], lhee[frame_idx_in_range, :][2], label='lhee')
                            ax.scatter(lumbar[0], lumbar[1], lumbar[2], label='lumbar')
                            ax.scatter(hip[0], hip[1], hip[2], label='hip')
                            ax.scatter(rthigh[0], rthigh[1], rthigh[2], label='rthigh')
                            ax.scatter(lthigh[0], lthigh[1], lthigh[2], label='lthigh')

                            plot_scale = 50
                            e_x_pelvis = e_x_pelvis * plot_scale
                            e_y_pelvis = e_y_pelvis * plot_scale
                            e_z_pelvis = e_z_pelvis * plot_scale
                            e_x_rthigh = e_x_rthigh * plot_scale
                            e_y_rthigh = e_y_rthigh * plot_scale
                            e_z_rthigh = e_z_rthigh * plot_scale
                            e_x_lthigh = e_x_lthigh * plot_scale
                            e_y_lthigh = e_y_lthigh * plot_scale
                            e_z_lthigh = e_z_lthigh * plot_scale
                            e_x_rshank = e_x_rshank * plot_scale
                            e_y_rshank = e_y_rshank * plot_scale
                            e_z_rshank = e_z_rshank * plot_scale
                            e_x_lshank = e_x_lshank * plot_scale
                            e_y_lshank = e_y_lshank * plot_scale
                            e_z_lshank = e_z_lshank * plot_scale
                            e_x_rfoot = e_x_rfoot * plot_scale
                            e_y_rfoot = e_y_rfoot * plot_scale
                            e_z_rfoot = e_z_rfoot * plot_scale
                            e_x_lfoot = e_x_lfoot * plot_scale
                            e_y_lfoot = e_y_lfoot * plot_scale
                            e_z_lfoot = e_z_lfoot * plot_scale

                            e_x_pelvis_0 = e_x_pelvis_0 * plot_scale
                            e_y_pelvis_0 = e_y_pelvis_0 * plot_scale
                            e_z_pelvis_0 = e_z_pelvis_0 * plot_scale
                            
                            ax.plot([hip[0], hip[0] + e_x_pelvis[0]], [hip[1], hip[1] + e_x_pelvis[1]], [hip[2], hip[2] + e_x_pelvis[2]], color='red')
                            ax.plot([hip[0], hip[0] + e_y_pelvis[0]], [hip[1], hip[1] + e_y_pelvis[1]], [hip[2], hip[2] + e_y_pelvis[2]], color='green')
                            ax.plot([hip[0], hip[0] + e_z_pelvis[0]], [hip[1], hip[1] + e_z_pelvis[1]], [hip[2], hip[2] + e_z_pelvis[2]], color='blue')

                            ax.plot([rthigh[0], rthigh[0] + e_x_rthigh[0]], [rthigh[1], rthigh[1] + e_x_rthigh[1]], [rthigh[2], rthigh[2] + e_x_rthigh[2]], color='red')
                            ax.plot([rthigh[0], rthigh[0] + e_y_rthigh[0]], [rthigh[1], rthigh[1] + e_y_rthigh[1]], [rthigh[2], rthigh[2] + e_y_rthigh[2]], color='green')
                            ax.plot([rthigh[0], rthigh[0] + e_z_rthigh[0]], [rthigh[1], rthigh[1] + e_z_rthigh[1]], [rthigh[2], rthigh[2] + e_z_rthigh[2]], color='blue')

                            ax.plot([lthigh[0], lthigh[0] + e_x_lthigh[0]], [lthigh[1], lthigh[1] + e_x_lthigh[1]], [lthigh[2], lthigh[2] + e_x_lthigh[2]], color='red')
                            ax.plot([lthigh[0], lthigh[0] + e_y_lthigh[0]], [lthigh[1], lthigh[1] + e_y_lthigh[1]], [lthigh[2], lthigh[2] + e_y_lthigh[2]], color='green')
                            ax.plot([lthigh[0], lthigh[0] + e_z_lthigh[0]], [lthigh[1], lthigh[1] + e_z_lthigh[1]], [lthigh[2], lthigh[2] + e_z_lthigh[2]], color='blue')

                            ax.plot([rshank[0], rshank[0] + e_x_rshank[0]], [rshank[1], rshank[1] + e_x_rshank[1]], [rshank[2], rshank[2] + e_x_rshank[2]], color='red')
                            ax.plot([rshank[0], rshank[0] + e_y_rshank[0]], [rshank[1], rshank[1] + e_y_rshank[1]], [rshank[2], rshank[2] + e_y_rshank[2]], color='green')
                            ax.plot([rshank[0], rshank[0] + e_z_rshank[0]], [rshank[1], rshank[1] + e_z_rshank[1]], [rshank[2], rshank[2] + e_z_rshank[2]], color='blue')

                            ax.plot([lshank[0], lshank[0] + e_x_lshank[0]], [lshank[1], lshank[1] + e_x_lshank[1]], [lshank[2], lshank[2] + e_x_lshank[2]], color='red')
                            ax.plot([lshank[0], lshank[0] + e_y_lshank[0]], [lshank[1], lshank[1] + e_y_lshank[1]], [lshank[2], lshank[2] + e_y_lshank[2]], color='green')
                            ax.plot([lshank[0], lshank[0] + e_z_lshank[0]], [lshank[1], lshank[1] + e_z_lshank[1]], [lshank[2], lshank[2] + e_z_lshank[2]], color='blue')

                            ax.plot([rfoot[0], rfoot[0] + e_x_rfoot[0]], [rfoot[1], rfoot[1] + e_x_rfoot[1]], [rfoot[2], rfoot[2] + e_x_rfoot[2]], color='red')
                            ax.plot([rfoot[0], rfoot[0] + e_y_rfoot[0]], [rfoot[1], rfoot[1] + e_y_rfoot[1]], [rfoot[2], rfoot[2] + e_y_rfoot[2]], color='green')
                            ax.plot([rfoot[0], rfoot[0] + e_z_rfoot[0]], [rfoot[1], rfoot[1] + e_z_rfoot[1]], [rfoot[2], rfoot[2] + e_z_rfoot[2]], color='blue')

                            ax.plot([lfoot[0], lfoot[0] + e_x_lfoot[0]], [lfoot[1], lfoot[1] + e_x_lfoot[1]], [lfoot[2], lfoot[2] + e_x_lfoot[2]], color='red')
                            ax.plot([lfoot[0], lfoot[0] + e_y_lfoot[0]], [lfoot[1], lfoot[1] + e_y_lfoot[1]], [lfoot[2], lfoot[2] + e_y_lfoot[2]], color='green')
                            ax.plot([lfoot[0], lfoot[0] + e_z_lfoot[0]], [lfoot[1], lfoot[1] + e_z_lfoot[1]], [lfoot[2], lfoot[2] + e_z_lfoot[2]], color='blue')

                            plt.legend()
                            plt.show()

                except Exception as e:
                    print(f"フレーム {original_frame_num} で予期せぬエラー: {e}。このフレームの角度をNaNとします。")
                    angle_list_range.append([np.nan] * 6) # 6つの角度すべてをNaNに

            # この範囲の角度データからデータフレームを作成
            if angle_list_range:
                angle_array_range = np.array(angle_list_range)
                angle_df_range = pd.DataFrame(angle_array_range, columns=["R_Hip", "L_Hip", "R_Knee", "L_Knee", "R_Ankle", "L_Ankle"], index=_range)
                all_angle_dfs.append(angle_df_range)

        # すべての範囲の角度データフレームを結合
        if all_angle_dfs:
            angle_df = pd.concat(all_angle_dfs)
            # 元のdfのインデックスに合わせて振り直し、欠損部分はNaNで埋める
            angle_df = angle_df.reindex(full_df.index)
        else:
            print("警告: 有効な角度データが計算されませんでした。")
            continue


        # 角度データの連続性保つ
        for col in angle_df.columns:
            prev = None
            for i in angle_df.index:
                curr = angle_df.at[i, col]
                if prev is not None and pd.notna(curr) and pd.notna(prev):
                    diff = curr - prev
                    if diff > 180:
                        angle_df.at[i, col] = curr - 360
                    elif diff < -180:
                        angle_df.at[i, col] = curr + 360
                    prev = angle_df.at[i, col]
                elif pd.notna(curr):
                    prev = curr
                    
        # Hip, Knee, Ankle角度のオフセット補正
        if 'R_Hip' in angle_df.columns:
            for frame in angle_df.index:
                if angle_df.at[frame, 'R_Hip'] > 0:
                    angle_df.loc[frame, 'R_Hip'] = angle_df.at[frame, 'R_Hip'] - 180
                else:
                    angle_df.loc[frame, 'R_Hip'] = 180 + angle_df.at[frame, 'R_Hip']
        if 'L_Hip' in angle_df.columns:
            for frame in angle_df.index:
                if angle_df.at[frame, 'L_Hip'] > 0:
                    angle_df.loc[frame, 'L_Hip'] = angle_df.at[frame, 'L_Hip'] - 180
                else:
                    angle_df.loc[frame, 'L_Hip'] = 180 + angle_df.at[frame, 'L_Hip']
        if 'R_Knee' in angle_df.columns:
            for frame in angle_df.index:
                if angle_df.at[frame, 'R_Knee'] > 0:
                    angle_df.loc[frame, 'R_Knee'] = 180 - angle_df.at[frame, 'R_Knee']
                else:
                    angle_df.loc[frame, 'R_Knee'] = - (180 + angle_df.at[frame, 'R_Knee'])
        if 'L_Knee' in angle_df.columns:
            for frame in angle_df.index:
                if angle_df.at[frame, 'L_Knee'] > 0:
                    angle_df.loc[frame, 'L_Knee'] = 180 - angle_df.at[frame, 'L_Knee']
                else:
                    angle_df.loc[frame, 'L_Knee'] = - (180 + angle_df.at[frame, 'L_Knee'])
        if 'R_Ankle' in angle_df.columns:
            for frame in angle_df.index:
                angle_df.loc[frame, 'R_Ankle'] = 90 - angle_df.at[frame, 'R_Ankle']
        if 'L_Ankle' in angle_df.columns:
            for frame in angle_df.index:
                angle_df.loc[frame, 'L_Ankle'] = 90 - angle_df.at[frame, 'L_Ankle']
                
                
        angle_df.to_csv(tsv_file.with_name(f"{tsv_file.stem}_angles.csv"))
        print(f"角度データを保存しました: {tsv_file.with_name(f'{tsv_file.stem}_angles.csv')}")
        
        # 左右股関節の屈曲伸展角度をプロット
        plt.plot(angle_df['L_Hip'], label='Left Hip Flexion/Extension', color='orange')
        plt.plot(angle_df['R_Hip'], label='Right Hip Flexion/Extension', color='blue')
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Hip Flexion/Extension Angles Over Time')
        plt.ylim(-40, 40)
        plt.grid()
        plt.legend()
        plt.savefig(tsv_file.with_name(f"{tsv_file.stem}_Hip_L_Flexion_Extension.png"))
        plt.close()
        
        # 左右膝関節の屈曲伸展角度をプロット
        plt.plot(angle_df['L_Knee'], label='Left Knee Flexion/Extension', color='orange')
        plt.plot(angle_df['R_Knee'], label='Right Knee Flexion/Extension', color='blue')
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Knee Flexion/Extension Angles Over Time')
        plt.ylim(-10, 70)
        plt.grid()
        plt.legend()
        plt.savefig(tsv_file.with_name(f"{tsv_file.stem}_Knee_L_Flexion_Extension.png"))
        plt.close()

        # 左右足関節の底屈背屈角度をプロット
        plt.plot(angle_df['L_Ankle'], label='Left Ankle Plantarflexion/Dorsiflexion', color='orange')
        plt.plot(angle_df['R_Ankle'], label='Right Ankle Plantarflexion/Dorsiflexion', color='blue')
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Ankle Plantarflexion/Dorsiflexion Angles Over Time')
        plt.ylim(-20, 60)
        plt.grid()
        plt.legend()
        plt.savefig(tsv_file.with_name(f"{tsv_file.stem}_Ankle_L_Plantarflexion_Dorsiflexion.png"))
        plt.close()
        
if __name__ == "__main__":
    main()
