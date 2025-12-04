import pandas as pd
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
import json

def read_3d_optitrack(csv_path, start_frame, end_frame, geometry_path=None):
    """
    OptiTrackの3Dデータ(100Hz)を読み込み、データ範囲を決定する.
    剛体定義による補間によりLPSI, LASI, RPSI, RASIは
    RigidBody 01:Marker1~4として読み込まれていることを前提としている.
    """
    df = pd.read_csv(csv_path, skiprows=[0, 1, 2, 4], header=[0, 2])
    df = df.loc[:, df.columns.get_level_values(1).isin(['X', 'Y', 'Z'])]  #XYZとつくラベル付けしたもののみ取得

    start_frame, end_frame = max(0, start_frame), min(len(df)-1, end_frame)
    df = df.loc[start_frame:end_frame].reset_index(drop=True)

    marker_set = ["RASI", "LASI", "RPSI", "LPSI","RKNE","LKNE", "RANK","LANK","RTOE","LTOE","RHEE","LHEE", "RKNE2", "LKNE2", "RANK2", "LANK2",
                  "Marker1", "Marker2", "Marker3", "Marker4"]
    marker_set_df = df[[col for col in df.columns if any(marker in col[0] for marker in marker_set)]].copy()
    
    if marker_set_df.empty:
        print("Error: No marker data found")
        return np.array([]), range(0)

    # RigidBodyの列が存在するかチェック
    rigidbody_exists = any("RigidBody 01" in col[0] for col in marker_set_df.columns)
    
    if rigidbody_exists:
        print("RigidBodyデータが検出されました。欠損値の置き換え処理を実行します。")
        # MarkerSetとRigidBodyのマッピング
        marker_mapping = {
            "MarkerSet 01:LPSI": "RigidBody 01:Marker1",
            "MarkerSet 01:LASI": "RigidBody 01:Marker2",
            "MarkerSet 01:RPSI": "RigidBody 01:Marker3",
            "MarkerSet 01:RASI": "RigidBody 01:Marker4"
        }

        # 各マーカーについて、欠損値をRigidBodyの値で置き換え
        for markerset_name, rigidbody_name in marker_mapping.items():
            # MarkerSetの列を取得
            markerset_cols = [col for col in marker_set_df.columns if markerset_name in col[0]]
            # RigidBodyの列を取得
            rigidbody_cols = [col for col in marker_set_df.columns if rigidbody_name in col[0]]
            
            if markerset_cols and rigidbody_cols:
                # 欠損値がある箇所をRigidBodyの値で置き換え
                for ms_col, rb_col in zip(markerset_cols, rigidbody_cols):
                    mask = marker_set_df[ms_col].isnull()
                    marker_set_df.loc[mask, ms_col] = marker_set_df.loc[mask, rb_col]
                    if mask.any():
                        print(f"  {markerset_name}の{mask.sum()}箇所を{rigidbody_name}で補完")
            else:
                print(f"警告: {markerset_name}または{rigidbody_name}が見つかりません")
    else:
        pass  # RigidBodyデータがない場合は何もしない

    final_marker_set = ["RASI", "LASI", "RPSI", "LPSI","RKNE","LKNE", "RANK","LANK","RTOE","LTOE","RHEE","LHEE", "RKNE2", "LKNE2", "RANK2", "LANK2"]
    final_df = marker_set_df[[col for col in marker_set_df.columns if any(marker in col[0] for marker in final_marker_set)]].copy()

    # すべてのマーカーが揃っているフレームのみを抽出
    valid_frames_mask = ~final_df.isnull().any(axis=1)
    final_df = final_df[valid_frames_mask].copy()
    
    if final_df.empty:
        print("Error: すべてのマーカーが揃っているフレームがありません")
        return np.array([]), range(0)
    
        # 有効なフレームのインデックスを取得（元の相対インデックス）
    valid_frame_indices = final_df.index.tolist()
    
    # 実際に使用する開始・終了フレーム（元のstart_frameからのオフセット）
    actual_start = valid_frame_indices[0]
    actual_end = valid_frame_indices[-1]
    
    # インデックスをリセット（0から始まる連続したインデックスに）
    final_df = final_df.reset_index(drop=True)
    
    # full_rangeは0から始まる連続した範囲
    full_range = range(0, len(final_df))
    
    print(f"有効なフレーム範囲: {actual_start} から {actual_end} ({len(final_df)} フレーム)")
    print(f"元の絶対フレーム範囲: {start_frame + actual_start} から {start_frame + actual_end}")
    
    final_df.to_csv(os.path.join(os.path.dirname(csv_path), f"marker_set_{os.path.basename(csv_path)}"))
    
    keypoints = final_df.values
    keypoints_mocap = keypoints.reshape(-1, len(final_marker_set), 3)
    
    # 返り値のstart_frameとend_frameは絶対フレーム番号
    return keypoints_mocap, full_range, start_frame + actual_start, start_frame + actual_end

    
    # # 有効なフレームのインデックスを取得
    # valid_frame_indices = final_df.index.tolist()
    # full_range = range(valid_frame_indices[0], valid_frame_indices[-1] + 1)
    # start_frame, end_frame = valid_frame_indices[0], valid_frame_indices[-1] + 1 
    
    # final_df.index = full_range
    # print(f"有効なフレーム範囲: {valid_frame_indices[0]} から {valid_frame_indices[-1]} ({len(final_df)} フレーム)")
    
    # final_df.to_csv(os.path.join(os.path.dirname(csv_path), f"marker_set_{os.path.basename(csv_path)}"))
    
    # # rigid bodyを使用していない場合に合わせるために調整（もう少しうまくできそう）
    # full_range = range(0,len(final_df))
    # keypoints = final_df.values
    # keypoints_mocap = keypoints.reshape(-1, len(final_marker_set), 3)

    # return keypoints_mocap, full_range, start_frame, end_frame

def read_3d_optitrack_norigid(csv_path, start_frame, end_frame, geometry_path=None):
    """
    OptiTrackの3Dデータ(100Hz)を読み込み、前処理を行う。
    1. データ範囲を決定する。
    2. データに対して、スプライン補間と幾何学的補間を全て実行する。
    """
    def geometric_interpolation(marker_df, marker_to_fix, geometry, original_missing_mask):
        """
        指定されたマーカーを、幾何学情報を用いて補間する汎用関数。
        引数として「元々欠損していた行のマスク」を受け取るように変更。
        """
        # 補間対象マーカー用の幾何学情報を取得
        if marker_to_fix not in geometry:
            print(f"警告: ジオメトリ情報に '{marker_to_fix}' の定義がありません。")
            return marker_df

        # ★★★ 変更点 ★★★
        # 引数で渡されたマスクを使い、元々欠損があったかどうかを判断する
        if not original_missing_mask.any():
            print(f"{marker_to_fix} に元々の欠損はなかったため、幾何学的な補間はスキップしました。")
            return marker_df

        marker_geometry = geometry[marker_to_fix]
        ref_marker_names = marker_geometry["reference_markers"]

        # 必要な列名を取得
        target_cols = [c for c in marker_df.columns if marker_to_fix in c[0]]
        ref_cols_map = {name: [c for c in marker_df.columns if name in c[0]] for name in ref_marker_names}

        if not target_cols or not all(ref_cols_map.values()):
            print(f"警告: {marker_to_fix} またはその参照マーカーがデータフレームにありません。")
            return marker_df

        print(f"ジオメトリ情報を使用して、元々欠損していた {marker_to_fix} を再計算・補完します。")

        # T-poseでの形状（ソース）を定義
        source_vectors = [np.array(marker_geometry["reference_vectors"][name]) for name in ref_marker_names]
        target_offset_vector = np.array(marker_geometry["target_offset_vector"])

        # ★★★ 変更点 ★★★
        # print(f"marker_df[original_missing_mask].index: {marker_df[original_missing_mask].index}")
        # 引数で渡されたマスクを使って、元々欠損していた行だけをループする
        # for index in marker_df[original_missing_mask].index:
        for index in range(len(marker_df)):  #すべての行を対象とする（もともととれているデータも上書きしてほかのマーカーから補間）
            row = marker_df.loc[index]

            # 参照マーカーのデータが揃っているか確認
            if all(not row[cols].isnull().any() for cols in ref_cols_map.values()):
                ref_positions = [row[ref_cols_map[name]].values for name in ref_marker_names]

                centroid_current = np.mean(ref_positions, axis=0)
                target_vectors = [p - centroid_current for p in ref_positions]

                rot, _ = R.align_vectors(target_vectors, source_vectors)
                estimated_offset = rot.apply(target_offset_vector)
                estimated_target = centroid_current + estimated_offset

                marker_df.loc[index, target_cols] = estimated_target

        return marker_df

    df = pd.read_csv(csv_path, skiprows=[0, 1, 2, 4], header=[0, 2])

    if start_frame >= len(df) or end_frame >= len(df) or start_frame < 0:
        print(f"Error: Requested range ({start_frame}-{end_frame}) is outside available data range (0-{len(df)-1})")
        return np.array([]), range(0)

    start_frame, end_frame = max(0, start_frame), min(len(df)-1, end_frame)
    df = df.loc[start_frame:end_frame].reset_index(drop=True)

    marker_set = ["RASI", "LASI", "RPSI", "LPSI","RKNE","LKNE", "RANK","LANK","RTOE","LTOE","RHEE","LHEE", "RKNE2", "LKNE2", "RANK2", "LANK2"]
    marker_set_df = df[[col for col in df.columns if any(marker in col[0] for marker in marker_set)]].copy()

    if marker_set_df.empty:
        print("Error: No marker data found")
        return np.array([]), range(0)

    # --- ここから補間処理  ---
    markers_to_fix = ["LPSI"]
    original_missing_masks = {}
    # print("スプライン補間を行う前に、元々の欠損箇所を記録します。")
    for marker in markers_to_fix:
        cols = [c for c in marker_set_df.columns if marker in c[0]]
        if cols:
            original_missing_masks[marker] = marker_set_df[cols].isnull().any(axis=1)
    # marker_set_df.to_csv(os.path.join(os.path.dirname(csv_path), f"before_interpolation_{os.path.basename(csv_path)}"))  #確認用

    print("細かい欠損を補完するため、先に三次スプライン補間を実行します。")
    marker_set_df.interpolate(method='cubic', limit_direction='both', inplace=True)
    # marker_set_df.interpolate(method='spline', order=3, limit_direction='both', inplace=True)  #なんかこれだとうまくいかなかった
    # marker_set_df.to_csv(os.path.join(os.path.dirname(csv_path), f"after_interpolation_{os.path.basename(csv_path)}"))  #確認用

    if geometry_path and os.path.exists(geometry_path):
        with open(geometry_path, 'r') as f:
            geometry = json.load(f)
        for marker in markers_to_fix:
            if marker in original_missing_masks:
                marker_set_df = geometric_interpolation(marker_set_df, marker, geometry, original_missing_masks[marker])
    # marker_set_df.to_csv(os.path.join(os.path.dirname(csv_path), f"after_geometric_interpolation_{os.path.basename(csv_path)}"))   #確認用

    if marker_set_df.isnull().values.any():
        print("エラー: 補間後も処理できない欠損値が残っています。")
        return np.array([]), range(0)

    final_df = marker_set_df
    # --- 以降の処理 ---
    full_range = range(0, len(final_df))
    final_df.to_csv(os.path.join(os.path.dirname(csv_path), f"marker_set_{os.path.basename(csv_path)}"))
    keypoints = final_df.values
    keypoints_mocap = keypoints.reshape(-1, len(marker_set), 3)

    return keypoints_mocap, full_range, start_frame, end_frame

def butter_lowpass_filter(data, order, cutoff_freq, frame_list, sampling_freq=100):  #4次のバターワースローパスフィルタ
    # sampling_freq を可変にして、60Hz または 100Hz に対応
    nyquist_freq = sampling_freq / 2
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # print(f"data = {data}")
    # print(f"data.shape = {data.shape}")
    y = filtfilt(b, a, data[frame_list])
    data_fillter = np.copy(data)
    data_fillter[frame_list] = y
    return data_fillter
