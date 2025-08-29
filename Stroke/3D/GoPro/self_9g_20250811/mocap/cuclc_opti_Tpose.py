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

def butter_lowpass_fillter(data, order, cutoff_freq, frame_list, sampling_freq=60):  #4次のバターワースローパスフィルタ
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

keypoints_mocap, full_range = read_3d_optitrack(tpose_csv_path)

sampling_freq = 100  #元のデータのサンプリング周波数
rot_pelvis_array = np.zeros((len(full_range), 3, 3))  #各フレームの骨盤の回転行列を保存する配列

# 骨盤マーカーの欠損があった場合の補間用データ
r_asi_array = np.zeros((len(full_range), 3))
l_asi_array = np.zeros((len(full_range), 3))
r_psi_array = np.zeros((len(full_range), 3))
l_psi_array = np.zeros((len(full_range), 3))

for frame_num in full_range:
    # マーカー位置を取得
    rasi = np.array([butter_lowpass_fillter(keypoints_mocap[:, 10, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    lasi = np.array([butter_lowpass_fillter(keypoints_mocap[:, 2, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    rpsi = np.array([butter_lowpass_fillter(keypoints_mocap[:, 14, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    lpsi = np.array([butter_lowpass_fillter(keypoints_mocap[:, 6, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    rank = np.array([butter_lowpass_fillter(keypoints_mocap[:, 8, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    lank = np.array([butter_lowpass_fillter(keypoints_mocap[:, 0, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T

    #メモ
    d_asi = np.linalg.norm(rasi[frame_num,:] - lasi[frame_num,:])
    d_leg = (np.linalg.norm(rank[frame_num,:] - rasi[frame_num,:]) + np.linalg.norm(lank[frame_num, :] - lasi[frame_num,:])) / 2
    r = 0.0127 #[m] Opti確認：https://www.optitrack.jp/products/accessories/marker.html
    h = 1.76 #[m]
    k = h/1.7
    beta = 0.1 * np.pi #[rad]
    theta = 0.496 #[rad]
    c = 0.115 * d_leg - 0.0153  #SKYCOMだと0.00153だけどDavisモデルは0.0153  https://wiki.has-motion.com/doku.php?id=visual3d:documentation:modeling:segments:hip_joint_landmarks
    x_dis = 0.1288 * d_leg - 0.04856

    # skycom + davis
    x_rthigh = -(x_dis +r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
    x_lthigh = -(x_dis +r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
    y_rthigh = +(c * np.sin(theta) - d_asi/2)
    y_lthigh = -(c * np.sin(theta)- d_asi/2)
    z_rthigh = -(x_dis + r) * np.sin(beta) + c * np.cos(theta) * np.cos(beta)
    z_lthigh = -(x_dis + r) * np.sin(beta) + c * np.cos(theta) * np.cos(beta)
    rthigh_pelvis = np.array([x_rthigh, y_rthigh, z_rthigh]).T
    lthigh_pelvis = np.array([x_lthigh, y_lthigh, z_lthigh]).T

    hip_0 = (rasi[frame_num,:] + lasi[frame_num,:]) / 2
    lumbar = (0.47 * (rasi[frame_num,:] + lasi[frame_num,:]) / 2 + 0.53 * (rpsi[frame_num,:] + lpsi[frame_num,:]) / 2) + 0.02 * k * np.array([0, 0, 1])

    #骨盤節座標系（原点はhip）
    e_y0_pelvis = lasi[frame_num,:] - rasi[frame_num,:]
    e_z_pelvis = (lumbar - hip_0)/np.linalg.norm(lumbar - hip_0)
    e_x_pelvis = np.cross(e_y0_pelvis, e_z_pelvis)/np.linalg.norm(np.cross(e_y0_pelvis, e_z_pelvis))
    e_y_pelvis = np.cross(e_z_pelvis, e_x_pelvis)
    rot_pelvis = np.array([e_x_pelvis, e_y_pelvis, e_z_pelvis]).T

    transformation_matrix = np.array([[e_x_pelvis[0], e_y_pelvis[0], e_z_pelvis[0], hip_0[0]],
                                        [e_x_pelvis[1], e_y_pelvis[1], e_z_pelvis[1], hip_0[1]],
                                        [e_x_pelvis[2], e_y_pelvis[2], e_z_pelvis[2], hip_0[2]],
                                        [0,       0,       0,       1]])

    #モーキャプの座標系に変換してもう一度計算
    rthigh = np.dot(transformation_matrix, np.append(rthigh_pelvis, 1))[:3]
    lthigh = np.dot(transformation_matrix, np.append(lthigh_pelvis, 1))[:3]
    hip = (rthigh + lthigh) / 2

    e_y0_pelvis = lthigh - rthigh
    e_z_pelvis = (lumbar - hip)/np.linalg.norm(lumbar - hip)
    e_x_pelvis = np.cross(e_y0_pelvis, e_z_pelvis)/np.linalg.norm(np.cross(e_y0_pelvis, e_z_pelvis))
    e_y_pelvis = np.cross(e_z_pelvis, e_x_pelvis)
    rot_pelvis = np.array([e_x_pelvis, e_y_pelvis, e_z_pelvis]).T


    # 座標系を補正して、Tposeの状態がニュートラルとなる座標系を作成する
    global_up = np.array([0, 0, 1]) # グローバル座標系の上方向ベクトル
    # 上方向ベクトルと左右軸ベクトルの外積で、水平な前後軸を求める
    e_x_corrected = np.cross(e_y_pelvis, global_up)
    e_x_corrected = e_x_corrected / np.linalg.norm(e_x_corrected)
    # 次に、求めた前後軸と、グローバルな上方向ベクトルの外積で、水平な左右軸を再定義
    e_y_corrected = np.cross(global_up, e_x_corrected)
    e_y_corrected = e_y_corrected / np.linalg.norm(e_y_corrected)
    # 補正後のZ軸 (上下軸) は、グローバル座標系の上方向と一致させる
    e_z_corrected = global_up
    # 4. 補正された3つの直交ベクトルから、新しいニュートラルな回転行列を組み立てる
    rot_pelvis_neutral_corrected = np.array([e_x_corrected, e_y_corrected, e_z_corrected]).T

    # 各フレームの骨盤の回転行列を保存
    rot_pelvis_array[frame_num] = rot_pelvis_neutral_corrected


    # 骨盤マーカーの補間用
    r_asi_array[frame_num] = rasi[frame_num,:]
    l_asi_array[frame_num] = lasi[frame_num,:]
    r_psi_array[frame_num] = rpsi[frame_num,:]
    l_psi_array[frame_num] = lpsi[frame_num,:]

rot_pelvis_neutral = np.mean(rot_pelvis_array, axis=0)
# print(f"rot_pelvis_array shape: {rot_pelvis_array.shape}")
# print(rot_pelvis_array[0][:][:])
# print(rot_pelvis_array[-1][:][:])

# 計算した行列を保存
neutral_matrix_path = os.path.join(output_dir, "rot_pelvis_neutral.npy")
np.save(neutral_matrix_path, rot_pelvis_neutral)

print(f"ニュートラルな骨盤の回転行列:\n{rot_pelvis_neutral}")
print(f"ニュートラルな骨盤の回転行列を保存しました: {neutral_matrix_path}")

r_asi = r_asi_array.mean(axis=0)
l_asi = l_asi_array.mean(axis=0)
r_psi = r_psi_array.mean(axis=0)
l_psi = l_psi_array.mean(axis=0)

# 3点 (RASI, LASI, LPSI) を基準点群とする
centroid_ref = (r_asi + l_asi + r_psi) / 3
# 補完したい点 (RPSI) の、基準点群中心からの相対ベクトル
v_rasi_ref = r_asi - centroid_ref
v_lasi_ref = l_asi - centroid_ref
v_rpsi_ref = r_psi - centroid_ref

# 補完したい点 (LPSI) の、基準点群中心からの相対ベクトル
v_lpsi_offset = l_psi - centroid_ref

# JSONデータを作成
geometry = {
    "reference_vectors": {
        "RASI": v_rasi_ref.tolist(),
        "LASI": v_lasi_ref.tolist(),
        "RPSI": v_rpsi_ref.tolist()
    },
    "target_offset_vector": {
        "LPSI": v_lpsi_offset.tolist()
    }
}

geometry_json_path = os.path.join(output_dir, "geometry.json")
# JSONファイルに保存
with open(geometry_json_path, 'w') as f:
    json.dump(geometry, f, indent=4)

print(f"骨盤のジオメトリ情報を {geometry_json_path} に保存しました。")
