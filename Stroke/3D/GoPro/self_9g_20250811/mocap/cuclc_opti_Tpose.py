"""
Tposeから基本姿勢時の骨盤の向きを計算して保存する
"""

import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt
import json

# T-poseのCSVファイルへのパスを指定
tpose_csv_path = r"G:\gait_pattern\20250811_br\sub0\thera0-14\mocap\0_0_14.csv"
output_dir = os.path.dirname(tpose_csv_path)

def read_3d_optitrack(csv_path):
    df = pd.read_csv(csv_path, skiprows=[0, 1, 2, 4], header=[0, 2])  #Motive
    # print(f"After range selection: {df.shape}")

    marker_set = ["RASI", "LASI", "RPSI", "LPSI","RKNE","LKNE", "RANK","LANK","RTOE","LTOE","RHEE","LHEE", "RKNE2", "LKNE2", "RANK2", "LANK2"]

    marker_set_df = df[[col for col in df.columns if any(marker in col[0] for marker in marker_set)]].copy()

    # print(f"Marker set dataframe shape: {marker_set_df.shape}")

    if marker_set_df.empty:
        print("Error: No marker data found")
        return np.array([]), range(0)

    success_frame_list = []
    for frame in range(0, len(marker_set_df)):
        if not marker_set_df.iloc[frame].isna().any():
            success_frame_list.append(frame)

    if not success_frame_list:
        print("Error: No valid frames found")
        return np.array([]), range(0)

    # full_range をダウンサンプリング後のインデックスベースで設定
    full_range = range(0, len(marker_set_df))
    # print(f"Processing range: {full_range}")

    success_df = marker_set_df.reindex(full_range)
    interpolate_success_df = success_df.interpolate(method='spline', order=3)

    for i, index in enumerate(full_range):
        marker_set_df.loc[index, :] = interpolate_success_df.iloc[i, :]

    marker_set_df.to_csv(os.path.join(os.path.dirname(csv_path), f"marker_set_{os.path.basename(csv_path)}"))

    keypoints = marker_set_df.values
    if keypoints.size == 0:
        print("Error: No keypoint data after processing")
        return np.array([]), range(0)

    keypoints_mocap = keypoints.reshape(-1, len(marker_set), 3)  #xyzで組になるように変形

    return keypoints_mocap, full_range

keypoints_mocap, full_range = read_3d_optitrack(tpose_csv_path)

sampling_freq = 100  #元のデータのサンプリング周波数

"""
T-poseデータから、各骨盤マーカーを残りの3点から計算するための
幾何学情報をすべて計算し、JSONファイルに保存する。
"""
try:
    df = pd.read_csv(tpose_csv_path, skiprows=[0, 1, 2, 4], header=[0, 2])
except Exception as e:
    print(f"エラー: CSVファイルの読み込みに失敗しました。詳細: {e}")

all_markers = ["RASI", "LASI", "RPSI", "LPSI"]

try:
    marker_df = df[[col for col in df.columns if any(m in col[0] for m in all_markers)]].copy()
except KeyError:
    print(f"エラー: CSVファイルに {', '.join(all_markers)} のいずれかのデータが見つかりません。")

valid_rows = marker_df.dropna()
if valid_rows.empty:
    print(f"エラー: T-poseファイルに4つのマーカーが全て揃っているフレームが見つかりませんでした。")

print(f"{len(valid_rows)} フレーム分の有効なデータから平均的な位置関係を計算します。")
mean_coords = valid_rows.mean()

# 各マーカーの平均座標を辞書に格納
marker_positions = {m: mean_coords[[c for c in mean_coords.index if m in c[0]]].values for m in all_markers}

# 全ジオメトリ情報を格納する辞書
full_geometry = {}

# 各マーカーを補完対象（target）としてループ
for target_marker in all_markers:
    # 残りの3つを基準（references）とする
    reference_markers = [m for m in all_markers if m != target_marker]

    ref_positions = [marker_positions[m] for m in reference_markers]

    # 基準点群の中心を計算
    centroid_ref = np.mean(ref_positions, axis=0)

    # 基準点群の中心からの相対ベクトル
    reference_vectors = {name: (marker_positions[name] - centroid_ref).tolist() for name in reference_markers}

    # 補完したい点の、基準点群中心からの相対ベクトル
    target_offset_vector = (marker_positions[target_marker] - centroid_ref).tolist()

    # このマーカー用の情報を辞書に格納
    full_geometry[target_marker] = {
        "reference_markers": reference_markers,
        "reference_vectors": reference_vectors,
        "target_offset_vector": target_offset_vector
    }

geometry_path = os.path.join(output_dir, "geometry.json")
# JSONファイルに保存
with open(geometry_path, 'w') as f:
    json.dump(full_geometry, f, indent=4)

print(f"骨盤ジオメトリ情報を {geometry_path} に保存しました。")
