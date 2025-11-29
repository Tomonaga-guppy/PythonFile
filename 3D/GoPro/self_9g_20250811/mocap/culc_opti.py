"""
Tposeの際の骨盤の向きにより、各フレームの骨盤の向きを補正してから関節角度を計算する
"""

import pandas as pd
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample
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

def main():
    csv_path_dir = r"G:\gait_pattern\20250811_br\sub0\thera0-16\mocap"
    # csv_path_dir = r"G:\gait_pattern\20250811_br\sub0\thera0-14\mocap"
    # csv_path_dir = r"G:\gait_pattern\20250811_br\sub0\thera0-15\mocap"
    # csv_path_dir = r"G:\gait_pattern\20250811_br\sub1\thera0-2\mocap"
    # csv_path_dir = r"G:\gait_pattern\20250811_br\sub1\thera0-3\mocap"
    # csv_path_dir = r"G:\gait_pattern\20250811_br\sub1\thera1-0\mocap"

    if csv_path_dir == r"G:\gait_pattern\20250811_br\sub1\thera0-2\mocap":
        start_frame = 1000
        end_frame = 1440
    elif csv_path_dir == r"G:\gait_pattern\20250811_br\sub1\thera0-3\mocap":
        start_frame = 943
        end_frame = 1400
    elif csv_path_dir == r"G:\gait_pattern\20250811_br\sub1\thera1-0\mocap":
        start_frame = 1090
        end_frame = 1252
    elif csv_path_dir == r"G:\gait_pattern\20250811_br\sub0\thera0-16\mocap":
        start_frame = 890
        end_frame = 1210
    elif csv_path_dir == r"G:\gait_pattern\20250811_br\sub0\thera0-15\mocap":
        # #0-0-15 で股関節外転 maxは1756くらい（60Hz）
        # start_frame_100hz = 2751
        # end_frame_100hz = 3144
        #0-0-15 で右股関節外旋（右足外転）
        start_frame = 627
        end_frame = 976
    else:
        #適当
        start_frame = 0
        end_frame = 100

    csv_paths = glob.glob(os.path.join(csv_path_dir, "*.csv"))

    # marker_set_で始まるファイルを除外
    csv_paths = [path for path in csv_paths if not os.path.basename(path).startswith("marker_set_")]
    # angle_で始まるファイルも除外（既に処理済みのファイル）
    csv_paths = [path for path in csv_paths if not os.path.basename(path).startswith("angle_")]
    # beforeで始まるファイルも除外（既に処理済みのファイル）
    csv_paths = [path for path in csv_paths if not os.path.basename(path).startswith("before_")]
    #  afterで始まるファイルも除外（既に処理済みのファイル）
    csv_paths = [path for path in csv_paths if not os.path.basename(path).startswith("after_")]

    geometry_json_path = r"G:\gait_pattern\20250811_br\sub0\thera0-16\mocap\geometry.json"

    for i, csv_path in enumerate(csv_paths):
        print(f"Processing: {csv_path}")

        try:
            keypoints_mocap, full_range = read_3d_optitrack(csv_path, start_frame, end_frame,
                                                            geometry_path=geometry_json_path)
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            continue

        if keypoints_mocap.size == 0:
            print(f"Skipping {csv_path}: No valid data")
            continue

        print(f"csv_path = {csv_path}")
        print(f"keypoints_mocap shape: {keypoints_mocap.shape}")

        # サンプリング周波数を設定
        sampling_freq = 100

        angle_list = []
        pel2hee_r_list = []
        pel2hee_l_list = []

        # 前フレームの角度を保存する変数（角度の連続性を保つため）
        prev_angles = None

        # バターワースフィルタのサンプリング周波数を動的に設定
        rasi = np.array([butter_lowpass_filter(keypoints_mocap[:, 10, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lasi = np.array([butter_lowpass_filter(keypoints_mocap[:, 2, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rpsi = np.array([butter_lowpass_filter(keypoints_mocap[:, 14, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lpsi = np.array([butter_lowpass_filter(keypoints_mocap[:, 6, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rank = np.array([butter_lowpass_filter(keypoints_mocap[:, 8, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lank = np.array([butter_lowpass_filter(keypoints_mocap[:, 0, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rank2 = np.array([butter_lowpass_filter(keypoints_mocap[:, 9, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lank2 = np.array([butter_lowpass_filter(keypoints_mocap[:, 1, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rknee = np.array([butter_lowpass_filter(keypoints_mocap[:, 12, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lknee = np.array([butter_lowpass_filter(keypoints_mocap[:, 4, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rknee2 = np.array([butter_lowpass_filter(keypoints_mocap[:, 13, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lknee2 = np.array([butter_lowpass_filter(keypoints_mocap[:, 5, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rtoe = np.array([butter_lowpass_filter(keypoints_mocap[:, 15, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        ltoe = np.array([butter_lowpass_filter(keypoints_mocap[:, 7, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rhee = np.array([butter_lowpass_filter(keypoints_mocap[:, 11, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lhee = np.array([butter_lowpass_filter(keypoints_mocap[:, 3, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T

        # full_range = range(1, len(rasi))  #差分取るために0からではなく1フレーム目からにする
        print(f"full_range(開始点は1フレーム後から): {full_range}")

        hip_list = []
        angle_list = []


        for frame_num in full_range:
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

            """
            変更後
            """
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
            hip_0 = (rasi[frame_num,:] + lasi[frame_num,:]) / 2
            # 仙骨 PSISの中点
            sacrum = (rpsi[frame_num,:] + lpsi[frame_num,:]) / 2

            #骨盤節座標系1（原点はhip_0）
            e_x0_pelvis_0 = (hip_0 - sacrum)/np.linalg.norm(hip_0 - sacrum)
            e_y_pelvis_0 = (lasi[frame_num,:] - rasi[frame_num,:])/np.linalg.norm(lasi[frame_num,:] - rasi[frame_num,:])
            e_z_pelvis_0 = np.cross(e_x0_pelvis_0, e_y_pelvis_0)/np.linalg.norm(np.cross(e_x0_pelvis_0, e_y_pelvis_0))
            e_x_pelvis_0 = np.cross(e_y_pelvis_0, e_z_pelvis_0)
            
            # お試し 産総研##############
            # e_x_pelvis = e_x_pelvis_0
            # e_y_pelvis = e_y_pelvis_0
            # e_z_pelvis = e_z_pelvis_0
            # rot_pelvis = np.array([e_x_pelvis, e_y_pelvis, e_z_pelvis]).T
            ######################

            transformation_matrix = np.array([[e_x_pelvis_0[0], e_y_pelvis_0[0], e_z_pelvis_0[0], hip_0[0]],
                                                [e_x_pelvis_0[1], e_y_pelvis_0[1], e_z_pelvis_0[1], hip_0[1]],
                                                [e_x_pelvis_0[2], e_y_pelvis_0[2], e_z_pelvis_0[2], hip_0[2]],
                                                [0,       0,       0,       1]])

            #グローバル座標に変換して再度計算
            rthigh = np.dot(transformation_matrix, np.append(rthigh_pelvis, 1))[:3]
            lthigh = np.dot(transformation_matrix, np.append(lthigh_pelvis, 1))[:3]
            hip = (rthigh + lthigh) / 2

            # 腰椎節原点
            # lumbar = (0.47 * (rasi[frame_num,:] + lasi[frame_num,:]) / 2 + 0.53 * (rpsi[frame_num,:] + lpsi[frame_num,:]) / 2) 
            lumbar = (0.47 * (rasi[frame_num,:] + lasi[frame_num,:]) / 2 + 0.53 * (rpsi[frame_num,:] + lpsi[frame_num,:]) / 2) + 0.02 * k * np.array([0, 1, 0])
            # lumbar = (0.47 * (rasi[frame_num,:] + lasi[frame_num,:]) / 2 + 0.53 * (rpsi[frame_num,:] + lpsi[frame_num,:]) / 2) + 0.02 * k * np.array([0, 0, 1])
            
            e_y0_pelvis = (lthigh - rthigh)/np.linalg.norm(lthigh - rthigh)
            e_z_pelvis = (lumbar - hip)/np.linalg.norm(lumbar - hip)
            e_x_pelvis = np.cross(e_y0_pelvis, e_z_pelvis)/np.linalg.norm(np.cross(e_y0_pelvis, e_z_pelvis))
            e_y_pelvis = np.cross(e_z_pelvis, e_x_pelvis)
            rot_pelvis = np.array([e_x_pelvis, e_y_pelvis, e_z_pelvis]).T

            hip_list.append(hip)
            hip_array = np.array(hip_list)

            #必要な原点の設定
            rshank = (rknee[frame_num, :] + rknee2[frame_num, :]) / 2
            lshank = (lknee[frame_num, :] + lknee2[frame_num, :]) / 2
            rfoot = (rank[frame_num,:] + rank2[frame_num,:]) / 2
            lfoot = (lank[frame_num, :] + lank2[frame_num,:]) / 2

            #右大腿節座標系（原点はrthigh）
            e_y0_rthigh = rknee2[frame_num, :] - rknee[frame_num, :]
            e_z_rthigh = (rshank - rthigh)/np.linalg.norm(rshank - rthigh)
            e_x_rthigh = np.cross(e_y0_rthigh, e_z_rthigh)/np.linalg.norm(np.cross(e_y0_rthigh, e_z_rthigh))
            e_y_rthigh = np.cross(e_z_rthigh, e_x_rthigh)
            rot_rthigh = np.array([e_x_rthigh, e_y_rthigh, e_z_rthigh]).T

            #左大腿節座標系（原点はlthigh）
            e_y0_lthigh = lknee[frame_num, :] - lknee2[frame_num, :]
            e_z_lthigh = (lshank - lthigh)/np.linalg.norm(lshank - lthigh)
            e_x_lthigh = np.cross(e_y0_lthigh, e_z_lthigh)/np.linalg.norm(np.cross(e_y0_lthigh, e_z_lthigh))
            e_y_lthigh = np.cross(e_z_lthigh, e_x_lthigh)
            rot_lthigh = np.array([e_x_lthigh, e_y_lthigh, e_z_lthigh]).T

            #右下腿節座標系（原点はrshank）
            e_y0_rshank = rknee2[frame_num, :] - rknee[frame_num, :]
            e_z_rshank = (rshank - rfoot)/np.linalg.norm(rshank - rfoot)
            e_x_rshank = np.cross(e_y0_rshank, e_z_rshank)/np.linalg.norm(np.cross(e_y0_rshank, e_z_rshank))
            e_y_rshank = np.cross(e_z_rshank, e_x_rshank)
            rot_rshank = np.array([e_x_rshank, e_y_rshank, e_z_rshank]).T

            #左下腿節座標系（原点はlshank）
            e_y0_lshank = lknee[frame_num, :] - lknee2[frame_num, :]
            e_z_lshank = (lshank - lfoot)/np.linalg.norm(lshank - lfoot)
            e_x_lshank = np.cross(e_y0_lshank, e_z_lshank)/np.linalg.norm(np.cross(e_y0_lshank, e_z_lshank))
            e_y_lshank = np.cross(e_z_lshank, e_x_lshank)
            rot_lshank = np.array([e_x_lshank, e_y_lshank, e_z_lshank]).T

            #右足節座標系 AIST参照（原点はrfoot）
            e_z_rfoot = (rtoe[frame_num,:] - rhee[frame_num,:]) / np.linalg.norm(rtoe[frame_num,:] - rhee[frame_num,:])
            e_y0_rfoot = rank[frame_num,:] - rank2[frame_num,:]
            e_x_rfoot = np.cross(e_z_rfoot, e_y0_rfoot)/np.linalg.norm(np.cross(e_z_rfoot, e_y0_rfoot))
            e_y_rfoot = np.cross(e_z_rfoot, e_x_rfoot)
            rot_rfoot = np.array([e_x_rfoot, e_y_rfoot, e_z_rfoot]).T

            #左足節座標系 AIST参照（原点はlfoot）
            e_z_lfoot = (ltoe[frame_num,:] - lhee[frame_num, :]) / np.linalg.norm(ltoe[frame_num,:] - lhee[frame_num, :])
            e_y0_lfoot = lank2[frame_num,:] - lank[frame_num,:]
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
            r_hip_angle_flex = r_hip_angle_rot.as_euler('YZX', degrees=True)[0]
            l_hip_angle_flex = l_hip_angle_rot.as_euler('YZX', degrees=True)[0]
            r_knee_angle_flex = r_knee_angle_rot.as_euler('YZX', degrees=True)[0]
            l_knee_angle_flex = l_knee_angle_rot.as_euler('YZX', degrees=True)[0]
            r_ankle_angle_pldo = r_ankle_angle_rot.as_euler('YZX', degrees=True)[0]
            l_ankle_angle_pldo = l_ankle_angle_rot.as_euler('YZX', degrees=True)[0]
            
            # 内旋外旋
            r_hip_angle_inex = r_hip_angle_rot.as_euler('YZX', degrees=True)[1]
            l_hip_angle_inex = l_hip_angle_rot.as_euler('YZX', degrees=True)[1]
            r_knee_angle_inex = r_knee_angle_rot.as_euler('YZX', degrees=True)[1]
            l_knee_angle_inex = l_knee_angle_rot.as_euler('YZX', degrees=True)[1]
            r_ankle_angle_inex = r_ankle_angle_rot.as_euler('YZX', degrees=True)[1]
            l_ankle_angle_inex = l_ankle_angle_rot.as_euler('YZX', degrees=True)[1]

            # 内転外転
            r_hip_angle_adab = r_hip_angle_rot.as_euler('YZX', degrees=True)[2]
            l_hip_angle_adab = l_hip_angle_rot.as_euler('YZX', degrees=True)[2]
            r_knee_angle_adab = r_knee_angle_rot.as_euler('YZX', degrees=True)[2]
            l_knee_angle_adab = l_knee_angle_rot.as_euler('YZX', degrees=True)[2]
            r_ankle_angle_adab = r_ankle_angle_rot.as_euler('YZX', degrees=True)[2]
            l_ankle_angle_adab = l_ankle_angle_rot.as_euler('YZX', degrees=True)[2]
            
            
            # ##############################################################################
            # def signed_angle_on_plane(forward_v, asai_v, r_hip_v):
            #     """
            #     forward_v, asai_vで定義される平面上にr_hip_vを射影し、
            #     forward_vとr_hip_vのなす角度（符号付き, degree）を返す
            #     """
            #     # 平面法線
            #     plane_normal = np.cross(forward_v, asai_v)
            #     plane_normal /= np.linalg.norm(plane_normal)

            #     # forward_vを平面上に正規化
            #     f_proj = forward_v - np.dot(forward_v, plane_normal) * plane_normal
            #     f_proj /= np.linalg.norm(f_proj)

            #     # r_hip_vを平面上に射影・正規化
            #     r_proj = r_hip_v - np.dot(r_hip_v, plane_normal) * plane_normal
            #     r_proj /= np.linalg.norm(r_proj)

            #     # atan2で符号付き角度
            #     x = np.dot(f_proj, r_proj)
            #     y = np.dot(np.cross(f_proj, r_proj), plane_normal)
            #     angle_rad = np.arctan2(y, x)
            #     angle_deg = np.degrees(angle_rad)
            #     return angle_deg

            # # 股関節の外旋内旋角度2（やり方別）
            # forward_v = hip_array[frame_num, :] - hip_array[frame_num-1, :]  # 前のフレームから現在のフレームまでの進行方向ベクトル
            # asis_v = lasi[frame_num, :] - rasi[frame_num, :]  # 右と左のASISを結ぶベクトル 右から左
            # r_hip_v = rshank - rthigh  # 右股関節→右膝ベクトル
            # l_hip_v = lshank - lthigh  # 左股関節→左膝ベクトル
            # r_hip_external_rotation_angle_2 = signed_angle_on_plane(forward_v, asis_v, r_hip_v)
            # l_hip_external_rotation_angle_2 = signed_angle_on_plane(forward_v, asis_v, l_hip_v)

            # # 股関節の外転内転角度2（やり方別）
            # down_v = np.cross(asis_v, forward_v)  # 下方向ベクトル（体の軸のイメージ）
            # r_hip_abduction_angle_2 = signed_angle_on_plane(down_v, asis_v, r_hip_v)
            # l_hip_abduction_angle_2 = signed_angle_on_plane(down_v, asis_v, l_hip_v)
            # ##############################################################################

            angle_list.append([r_hip_angle_flex, l_hip_angle_flex, r_knee_angle_flex, l_knee_angle_flex, r_ankle_angle_pldo, l_ankle_angle_pldo,
                                    r_hip_angle_inex, l_hip_angle_inex, r_knee_angle_inex, l_knee_angle_inex, r_ankle_angle_inex, l_ankle_angle_inex,
                                    r_hip_angle_adab, l_hip_angle_adab, r_knee_angle_adab, l_knee_angle_adab, r_ankle_angle_adab, l_ankle_angle_adab])


            plot_flag = True
            if plot_flag:
                # print(frame_num)  #相対フレーム数
                if frame_num == 0:
                # if frame_num == 1756-1650:  #100Hzで2926(足を最大に外転してるくらいの時)
                    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("z")
                    ax.set_xlim(-1.5, 1.5)
                    ax.set_ylim(-1, 2)
                    ax.set_zlim(-2, 1)
                    #frame数を表示
                    ax.text2D(0.5, 0.01, f"frame = {frame_num}", transform=ax.transAxes)
                    #方向を設定
                    ax.view_init(elev=0, azim=0)

                    ax.scatter(rasi[frame_num,:][0], rasi[frame_num,:][1], rasi[frame_num,:][2], label='rasi')
                    ax.scatter(lasi[frame_num,:][0], lasi[frame_num,:][1], lasi[frame_num,:][2], label='lasi')
                    ax.scatter(rpsi[frame_num,:][0], rpsi[frame_num,:][1], rpsi[frame_num,:][2], label='rpsi')
                    ax.scatter(lpsi[frame_num,:][0], lpsi[frame_num,:][1], lpsi[frame_num,:][2], label='lpsi')
                    ax.scatter(rfoot[0], rfoot[1], rfoot[2], label='rfoot')
                    ax.scatter(lfoot[0], lfoot[1], lfoot[2], label='lfoot')
                    ax.scatter(rshank[0], rshank[1], rshank[2], label='rshank')
                    ax.scatter(lshank[0], lshank[1], lshank[2], label='lshank')
                    ax.scatter(rtoe[frame_num,:][0], rtoe[frame_num,:][1], rtoe[frame_num,:][2], label='rtoe')
                    ax.scatter(ltoe[frame_num,:][0], ltoe[frame_num,:][1], ltoe[frame_num,:][2], label='ltoe')
                    ax.scatter(rhee[frame_num,:][0], rhee[frame_num,:][1], rhee[frame_num,:][2], label='rhee')
                    ax.scatter(lhee[frame_num, :][0], lhee[frame_num, :][1], lhee[frame_num, :][2], label='lhee')
                    ax.scatter(lumbar[0], lumbar[1], lumbar[2], label='lumbar')
                    ax.scatter(hip[0], hip[1], hip[2], label='hip')
                    ax.scatter(rthigh[0], rthigh[1], rthigh[2], label='rthigh')
                    ax.scatter(lthigh[0], lthigh[1], lthigh[2], label='lthigh')

                    e_x_pelvis = e_x_pelvis * 0.1
                    e_y_pelvis = e_y_pelvis * 0.1
                    e_z_pelvis = e_z_pelvis * 0.1
                    e_x_rthigh = e_x_rthigh * 0.1
                    e_y_rthigh = e_y_rthigh * 0.1
                    e_z_rthigh = e_z_rthigh * 0.1
                    e_x_lthigh = e_x_lthigh * 0.1
                    e_y_lthigh = e_y_lthigh * 0.1
                    e_z_lthigh = e_z_lthigh * 0.1
                    e_x_rshank = e_x_rshank * 0.1
                    e_y_rshank = e_y_rshank * 0.1
                    e_z_rshank = e_z_rshank * 0.1
                    e_x_lshank = e_x_lshank * 0.1
                    e_y_lshank = e_y_lshank * 0.1
                    e_z_lshank = e_z_lshank * 0.1
                    e_x_rfoot = e_x_rfoot * 0.1
                    e_y_rfoot = e_y_rfoot * 0.1
                    e_z_rfoot = e_z_rfoot * 0.1
                    e_x_lfoot = e_x_lfoot * 0.1
                    e_y_lfoot = e_y_lfoot * 0.1
                    e_z_lfoot = e_z_lfoot * 0.1

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
                    
                    e_x_pelvis_0 = e_x_pelvis_0 * 0.1
                    e_y_pelvis_0 = e_y_pelvis_0 * 0.1
                    e_z_pelvis_0 = e_z_pelvis_0 * 0.1
                    ax.scatter(hip_0[0], hip_0[1], hip_0[2], label='hip_0', color='black')
                    ax.plot([hip_0[0], hip_0[0] + e_x_pelvis_0[0]], [hip_0[1], hip_0[1] + e_x_pelvis_0[1]], [hip_0[2], hip_0[2] + e_x_pelvis_0[2]], color='red')
                    ax.plot([hip_0[0], hip_0[0] + e_y_pelvis_0[0]], [hip_0[1], hip_0[1] + e_y_pelvis_0[1]], [hip_0[2], hip_0[2] + e_y_pelvis_0[2]], color='green')
                    ax.plot([hip_0[0], hip_0[0] + e_z_pelvis_0[0]], [hip_0[1], hip_0[1] + e_z_pelvis_0[1]], [hip_0[2], hip_0[2] + e_z_pelvis_0[2]], color='blue')
                    
                    
                    plt.legend()
                    plt.show()

            # e_z_lshank_list.append(e_z_lshank)
            # e_z_lfoot_list.append(e_z_lfoot)

            #骨盤とかかとのベクトル計算
            pel2hee_r = rhee[frame_num, :] - hip[:]
            pel2hee_r_list.append(pel2hee_r)
            pel2hee_l = lhee[frame_num, :] - hip[:]
            pel2hee_l_list.append(pel2hee_l)


        angle_array = np.array(angle_list)
        angle_df = pd.DataFrame(angle_array, columns=["R_Hip_FlEx", "L_Hip_FlEx", "R_Knee_FlEx", "L_Knee_FlEx", "R_Ankle_PlDo", "L_Ankle_PlDo",
                                                    "R_Hip_InEx", "L_Hip_InEx", "R_Knee_InEx", "L_Knee_InEx", "R_Ankle_InEx", "L_Ankle_InEx",
                                                    "R_Hip_AdAb", "L_Hip_AdAb", "R_Knee_AdAb", "L_Knee_AdAb", "R_Ankle_AdAb", "L_Ankle_AdAb"], index=full_range)

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
        if 'R_Hip_FlEx' in angle_df.columns:
            for frame in angle_df.index:
                if angle_df.at[frame, 'R_Hip_FlEx'] > 0:
                    angle_df.loc[frame, 'R_Hip_FlEx'] = angle_df.at[frame, 'R_Hip_FlEx'] - 180
                else:
                    angle_df.loc[frame, 'R_Hip_FlEx'] = 180 + angle_df.at[frame, 'R_Hip_FlEx']
        if 'L_Hip_FlEx' in angle_df.columns:
            for frame in angle_df.index:
                if angle_df.at[frame, 'L_Hip_FlEx'] > 0:
                    angle_df.loc[frame, 'L_Hip_FlEx'] = angle_df.at[frame, 'L_Hip_FlEx'] - 180
                else:
                    angle_df.loc[frame, 'L_Hip_FlEx'] = 180 + angle_df.at[frame, 'L_Hip_FlEx']
        if 'R_Knee_FlEx' in angle_df.columns:
            for frame in angle_df.index:
                if angle_df.at[frame, 'R_Knee_FlEx'] > 0:
                    angle_df.loc[frame, 'R_Knee_FlEx'] = 180 - angle_df.at[frame, 'R_Knee_FlEx']
                else:
                    angle_df.loc[frame, 'R_Knee_FlEx'] = - (180 + angle_df.at[frame, 'R_Knee_FlEx'])
        if 'L_Knee_FlEx' in angle_df.columns:
            for frame in angle_df.index:
                if angle_df.at[frame, 'L_Knee_FlEx'] > 0:
                    angle_df.loc[frame, 'L_Knee_FlEx'] = 180 - angle_df.at[frame, 'L_Knee_FlEx']
                else:
                    angle_df.loc[frame, 'L_Knee_FlEx'] = - (180 + angle_df.at[frame, 'L_Knee_FlEx'])
        if 'R_Ankle_PlDo' in angle_df.columns:
            for frame in angle_df.index:
                angle_df.loc[frame, 'R_Ankle_PlDo'] = 90 - angle_df.at[frame, 'R_Ankle_PlDo']
        if 'L_Ankle_PlDo' in angle_df.columns:
            for frame in angle_df.index:
                angle_df.loc[frame, 'L_Ankle_PlDo'] = 90 - angle_df.at[frame, 'L_Ankle_PlDo']
                
                
        # DataFrameのインデックスを絶対フレーム番号に設定
        # 100Hzデータの場合、start_frameからの100Hz絶対フレーム番号
        absolute_frame_indices = np.array(full_range) + start_frame

        # print(f"absolute_frame_indices = {absolute_frame_indices}")
        absolute_frame_indices = absolute_frame_indices[1:]
        # print(f"absolute_frame_indices = {absolute_frame_indices}")

        # ファイル名に適切なサンプリング周波数を記載
        angle_df.to_csv(os.path.join(os.path.dirname(csv_path), f"angle_100Hz_{os.path.basename(csv_path)}"))

        pel2hee_r_array = np.array(pel2hee_r_list)
        rhee_pel_z = pel2hee_r_array[:, 2]
        pel2hee_l_array = np.array(pel2hee_l_list)
        lhee_pel_z = pel2hee_l_array[:, 2]

        # 100Hzデータでの相対フレーム番号（0から始まる）
        rel_frames = np.array(full_range)

        # 100Hzデータでの絶対フレーム番号
        abs_frames = rel_frames + start_frame
        
        print(f"frame_100hz_rel sample: {rel_frames[:10]}")
        print(f"frame_100hz_abs sample: {abs_frames[:10]}")

        print(f"len(frame_100hz_rel) = {len(rel_frames)}")
        print(f"len(frame_100hz_abs) = {len(abs_frames)}")
        print(f"len(rhee_pel_z) = {len(rhee_pel_z)}")
        print(f"len(lhee_pel_z) = {len(lhee_pel_z)}")
        
        df_ic_o = pd.DataFrame({
            "frame_100hz_rel": rel_frames,
            "frame_100hz_abs": abs_frames,
            "rhee_pel_z": rhee_pel_z,
            "lhee_pel_z": lhee_pel_z
        })
        
        df_ic_o.index = df_ic_o.index + 1
        df_ic_r = df_ic_o.sort_values(by="rhee_pel_z", ascending=False)
        df_ic_l = df_ic_o.sort_values(by="lhee_pel_z", ascending=False)

        # 初期接地検出（100Hz相対フレーム番号で）
        ic_r_list_100hz_rel = df_ic_r.head(30)["frame_100hz_rel"].values.astype(int)
        ic_r_list_100hz_abs = df_ic_r.head(30)["frame_100hz_abs"].values.astype(int)
        ic_l_list_100hz_rel = df_ic_l.head(30)["frame_100hz_rel"].values.astype(int)
        ic_l_list_100hz_abs = df_ic_l.head(30)["frame_100hz_abs"].values.astype(int)

        print(f"start_frame (100Hz): {start_frame}")
        print(f"end_frame (100Hz): {end_frame}")
        print(f"ic_r_list (100Hz相対フレーム): {ic_r_list_100hz_rel}")
        print(f"ic_l_list (100Hz相対フレーム): {ic_l_list_100hz_rel}")
        print(f"ic_r_list (100Hz絶対フレーム): {ic_r_list_100hz_abs}")
        print(f"ic_l_list (100Hz絶対フレーム): {ic_l_list_100hz_abs}")

        filt_ic_r_list_100hz_rel = []
        skip_values_r = set()
        for value in ic_r_list_100hz_rel:
            if value in skip_values_r:
                continue
            filt_ic_r_list_100hz_rel.append(value)
            # 100Hzでの10フレーム間隔でスキップ
            skip_values_r.update(range(value - 10, value + 11))
        filt_ic_r_list_100hz_rel = sorted(filt_ic_r_list_100hz_rel)
        print(f"フィルタリング後のRリスト (100Hz相対フレーム): {filt_ic_r_list_100hz_rel}")

        filt_ic_l_list_100hz_rel = []
        skip_values_l = set()
        for value in ic_l_list_100hz_rel:
            if value in skip_values_l:
                continue
            filt_ic_l_list_100hz_rel.append(value)
            # 100Hzでの10フレーム間隔でスキップ
            skip_values_l.update(range(value - 10, value + 11))
        filt_ic_l_list_100hz_rel = sorted(filt_ic_l_list_100hz_rel)
        print(f"フィルタリング後のLリスト (100Hz相対フレーム): {filt_ic_l_list_100hz_rel}")

        # 絶対フレーム番号に変換
        filt_ic_r_list_100hz_abs = []
        for relative_ic_r_frame in filt_ic_r_list_100hz_rel:
            # 100Hz絶対フレーム番号
            absolute_100hz = relative_ic_r_frame + start_frame
            filt_ic_r_list_100hz_abs.append(absolute_100hz)
        print(f"フィルタリング後のic_rリスト (100Hz絶対フレーム): {filt_ic_r_list_100hz_abs}")

        filt_ic_l_list_100hz_abs = []
        for relative_ic_l_frame in filt_ic_l_list_100hz_rel:
            # 100Hz絶対フレーム番号
            absolute_100hz = relative_ic_l_frame + start_frame
            filt_ic_l_list_100hz_abs.append(absolute_100hz)
        print(f"フィルタリング後のic_lリスト (100Hz絶対フレーム): {filt_ic_l_list_100hz_abs}")

        # 骨盤とかかとの距離をプロット
        plt.figure()
        plt.plot(abs_frames, df_ic_o["rhee_pel_z"], label="Right Heel - Pelvis Z Position")
        plt.plot(abs_frames, df_ic_o["lhee_pel_z"], label="Left Heel - Pelvis Z Position")
        plt.xlabel("Frame (100Hz absolute)")
        plt.ylabel("Position (mm)")
        plt.title("Heel Z Position")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(csv_path_dir, "heel_z_position_100Hz.png"))
        # plt.show()
        plt.close()

        # 関節角度と初期接地をプロット（100Hz絶対フレーム番号で統一）
                # 左右股関節の屈曲伸展角度をプロット
        plt.plot(abs_frames, angle_df['L_Hip_FlEx'], label='Left Hip Flexion/Extension', color='orange')
        plt.plot(abs_frames, angle_df['R_Hip_FlEx'], label='Right Hip Flexion/Extension', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in filt_ic_l_list_100hz_abs]
        [plt.axvline(x=frame, color='blue', linestyle='--', alpha=0.5) for frame in filt_ic_r_list_100hz_abs]
        plt.xlabel('Frame [-]')
        plt.ylabel('Angle [deg]')
        plt.title('Hip Flexion/Extension Angles Over Time')
        plt.ylim(-40, 40)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(csv_path_dir, f"{os.path.basename(csv_path)}_Hip_1_Flexion_Extension.png"))
        plt.close()
        
        # 左右膝関節の屈曲伸展角度をプロット
        plt.plot(abs_frames, angle_df['L_Knee_FlEx'], label='Left Knee Flexion/Extension', color='orange')
        plt.plot(abs_frames, angle_df['R_Knee_FlEx'], label='Right Knee Flexion/Extension', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in filt_ic_l_list_100hz_abs]
        [plt.axvline(x=frame, color='blue', linestyle='--', alpha=0.5) for frame in filt_ic_r_list_100hz_abs]
        plt.xlabel('Frame [-]')
        plt.ylabel('Angle [deg]')
        plt.title('Knee Flexion/Extension Angles Over Time')
        plt.ylim(-10, 70)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(csv_path_dir, f"{os.path.basename(csv_path)}_Knee_1_Flexion_Extension.png"))
        plt.close()

        # 左右足関節の底屈背屈角度をプロット
        plt.plot(abs_frames, angle_df['L_Ankle_PlDo'], label='Left Ankle Plantarflexion/Dorsiflexion', color='orange')
        plt.plot(abs_frames, angle_df['R_Ankle_PlDo'], label='Right Ankle Plantarflexion/Dorsiflexion', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in filt_ic_l_list_100hz_abs]
        [plt.axvline(x=frame, color='blue', linestyle='--', alpha=0.5) for frame in filt_ic_r_list_100hz_abs]
        plt.xlabel('Frame [-]')
        plt.ylabel('Angle [deg]')
        plt.title('Ankle Plantarflexion/Dorsiflexion Angles Over Time')
        plt.ylim(-20, 60)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(csv_path_dir, f"{os.path.basename(csv_path)}_Ankle_1_Plantarflexion_Dorsiflexion.png"))
        plt.close()
        
        
        # 左右股関節の内転外転角度をプロット
        plt.plot(abs_frames, angle_df['L_Hip_AdAb'], label='Left Hip Adduction/Abduction', color='orange')
        plt.plot(abs_frames, angle_df['R_Hip_AdAb'], label='Right Hip Adduction/Abduction', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in filt_ic_l_list_100hz_abs]
        [plt.axvline(x=frame, color='blue', linestyle='--', alpha=0.5) for frame in filt_ic_r_list_100hz_abs]
        plt.xlabel('Frame [-]')
        plt.ylabel('Angle [deg]')
        plt.title('Hip Adduction/Abduction Angles Over Time')
        # plt.ylim(-20, 60)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(csv_path_dir, f"{os.path.basename(csv_path)}_Hip_2_Adduction_Abduction.png"))
        plt.close()
        
        # 左右膝関節の内転外転角度をプロット
        plt.plot(abs_frames, angle_df['L_Knee_AdAb'], label='Left Knee Adduction/Abduction', color='orange')
        plt.plot(abs_frames, angle_df['R_Knee_AdAb'], label='Right Knee Adduction/Abduction', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in filt_ic_l_list_100hz_abs]
        [plt.axvline(x=frame, color='blue', linestyle='--', alpha=0.5) for frame in filt_ic_r_list_100hz_abs]
        plt.xlabel('Frame [-]')
        plt.ylabel('Angle [deg]')
        plt.title('Knee Adduction/Abduction Angles Over Time')
        # plt.ylim(-20, 60)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(csv_path_dir, f"{os.path.basename(csv_path)}_Knee_2_Adduction_Abduction.png"))
        plt.close()
        
        # 左右足関節の内転外転角度をプロット
        plt.plot(abs_frames, angle_df['L_Ankle_AdAb'], label='Left Ankle Adduction/Abduction', color='orange')
        plt.plot(abs_frames, angle_df['R_Ankle_AdAb'], label='Right Ankle Adduction/Abduction', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in filt_ic_l_list_100hz_abs]
        [plt.axvline(x=frame, color='blue', linestyle='--', alpha=0.5) for frame in filt_ic_r_list_100hz_abs]
        plt.xlabel('Frame [-]')
        plt.ylabel('Angle [deg]')
        plt.title('Ankle Adduction/Abduction Angles Over Time')
        # plt.ylim(-20, 60)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(csv_path_dir, f"{os.path.basename(csv_path)}_Ankle_2_Adduction_Abduction.png"))
        plt.close()
        
        
        # 左右股関節の内旋外旋角度をプロット
        plt.plot(abs_frames, angle_df['L_Hip_InEx'], label='Left Hip Internal/External Rotation', color='orange')
        plt.plot(abs_frames, angle_df['R_Hip_InEx'], label='Right Hip Internal/External Rotation', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in filt_ic_l_list_100hz_abs]
        [plt.axvline(x=frame, color='blue', linestyle='--', alpha=0.5) for frame in filt_ic_r_list_100hz_abs]
        plt.xlabel('Frame [-]')
        plt.ylabel('Angle [deg]')
        plt.title('Hip Internal/External Rotation Angles Over Time')
        # plt.ylim(-20, 60)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(csv_path_dir, f"{os.path.basename(csv_path)}_Hip_3_Internal_External_Rotation.png"))
        plt.close()
        
        # 左右膝関節の内旋外旋角度をプロット
        plt.plot(abs_frames, angle_df['L_Knee_InEx'], label='Left Knee Internal/External Rotation', color='orange')
        plt.plot(abs_frames, angle_df['R_Knee_InEx'], label='Right Knee Internal/External Rotation', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in filt_ic_l_list_100hz_abs]
        [plt.axvline(x=frame, color='blue', linestyle='--', alpha=0.5) for frame in filt_ic_r_list_100hz_abs]
        plt.xlabel('Frame [-]')
        plt.ylabel('Angle [deg]')
        plt.title('Knee Internal/External Rotation Angles Over Time')
        # plt.ylim(-20, 60)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(csv_path_dir, f"{os.path.basename(csv_path)}_Knee_3_Internal_External_Rotation.png"))
        plt.close()
        
        # 左右足関節の内旋外旋角度をプロット
        plt.plot(abs_frames, angle_df['L_Ankle_InEx'], label='Left Ankle Internal/External Rotation', color='orange')
        plt.plot(abs_frames, angle_df['R_Ankle_InEx'], label='Right Ankle Internal/External Rotation', color='blue')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in filt_ic_l_list_100hz_abs]
        [plt.axvline(x=frame, color='blue', linestyle='--', alpha=0.5) for frame in filt_ic_r_list_100hz_abs]
        plt.xlabel('Frame [-]')
        plt.ylabel('Angle [deg]')
        plt.title('Ankle Internal/External Rotation Angles Over Time')
        # plt.ylim(-20, 60)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(csv_path_dir, f"{os.path.basename(csv_path)}_Ankle_3_Internal_External_Rotation.png"))
        plt.close()
        


        # 歩行速度 walk speedの算出
        cycle_frame = [[filt_ic_r_list_100hz_rel[i], filt_ic_r_list_100hz_rel[i+1]] for i in range(len(filt_ic_r_list_100hz_rel)-1)]
        speed = []
        for start, end in cycle_frame:
            duration = (end - start) / 100  # 100Hzなので秒に変換
            distance = np.linalg.norm(hip_array[end,:] - hip_array[start,:])
            speed.append(distance / duration if duration > 0 else 0)
            # print(f"start_end: {start}-{end}, duration: {duration}, distance: {distance}, speed: {speed[-1]}")
        gait_speed_mean = np.mean(speed, axis=0)
        gait_speed_std = np.std(speed, axis=0)
        print(f"歩行速度: {gait_speed_mean} m/s")

        # 歩隔の算出
        print(f"filt_ic_r_list_100hz_relative: {filt_ic_r_list_100hz_rel}")
        print(f"filt_ic_l_list_100hz_relative: {filt_ic_l_list_100hz_rel}")
        step_length_list = []
        for start, end in cycle_frame:
            mid = (start + end) / 2
            l_ic_relative_frame = min(filt_ic_l_list_100hz_rel, key=lambda x: abs(x - mid))
            step_length_1 = abs(rhee[start, 0] - lhee[l_ic_relative_frame, 0])
            step_length_2 = abs(lhee[l_ic_relative_frame, 0] - rhee[end, 0])
            step_length = (step_length_1 + step_length_2) / 2
            step_length_list.append(step_length)
            # print(f"start: {start}, end: {end}, l_ic: {l_ic_relative_frame}, step_length_1: {step_length_1}, step_length_2: {step_length_2}, step_length: {step_length}")
        step_length_mean = np.mean(step_length_list, axis=0)
        step_length_std = np.std(step_length_list, axis=0)
        print(f"歩隔: {step_length_mean} m")

        # 歩行速度及び歩隔をjsonファイルで保存
        gait_data = {
            "start_frame_100Hz": int(start_frame),
            "end_frame_100Hz": int(end_frame),
            "ic_r_list_100hz_abs": [int(x) for x in filt_ic_r_list_100hz_abs],
            "ic_l_list_100hz_abs": [int(x) for x in filt_ic_l_list_100hz_abs],
            "cycle_num": int(len(cycle_frame)),
            "gait speed": float(gait_speed_mean),
            "gait speed std": float(gait_speed_std),
            "step length": float(step_length_mean),
            "step length std": float(step_length_std)
        }
        json_path = os.path.join(os.path.dirname(csv_path), f"gait_data_{os.path.basename(csv_path).split('.')[0]}.json")
        with open(json_path, "w") as json_file:
            json.dump(gait_data, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()