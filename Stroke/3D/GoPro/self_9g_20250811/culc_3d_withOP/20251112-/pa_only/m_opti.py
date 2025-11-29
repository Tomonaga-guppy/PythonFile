import pandas as pd
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
import json

def read_3d_optitrack(csv_path, start_frame, end_frame, geometry_path=None):
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

    return keypoints_mocap, full_range

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
