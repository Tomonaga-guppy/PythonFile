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

def read_3d_optitrack(csv_path, down_hz, start_frame_100hz, end_frame_100hz, geometry_path = None):
    df = pd.read_csv(csv_path, skiprows=[0, 1, 2, 4], header=[0, 2])  #Motive

    print(f"元のデータ形状: {df.shape}")
    print(f"解析範囲（100Hz）: {start_frame_100hz} to {end_frame_100hz}")

    # データの範囲をチェック
    if start_frame_100hz >= len(df) or end_frame_100hz >= len(df) or start_frame_100hz < 0:
        print(f"Error: Requested range ({start_frame_100hz}-{end_frame_100hz}) is outside available data range (0-{len(df)-1})")
        return np.array([]), range(0)

    # 範囲を調整
    start_frame = max(0, start_frame_100hz)
    end_frame = min(len(df)-1, end_frame_100hz)
    df = df.loc[start_frame:end_frame]
    print(f"After range selection: {df.shape}")

    if down_hz:
        # 100Hz → 60Hz のダウンサンプリング
        original_length = len(df)
        target_length = int(original_length * 60 / 100)
        print(f"ダウンサンプリング: {original_length} → {target_length} フレーム")
        if target_length <= 0:
            print("Error: Target length is 0 after downsampling")
            return np.array([]), range(0)

        # 各列に対してリサンプリングを適用
        resampled_data = {}
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:  # 数値列のみリサンプリング
                if not df[col].dropna().empty:
                    resampled_data[col] = resample(df[col].dropna(), target_length)
                else:
                    resampled_data[col] = np.full(target_length, np.nan)
            else:
                # 非数値列は単純にインデックスを調整
                if len(df) > 0:
                    step = len(df) / target_length
                    indices = [int(i * step) for i in range(target_length)]
                    resampled_data[col] = df[col].iloc[indices].reset_index(drop=True)
                else:
                    resampled_data[col] = np.full(target_length, np.nan)

        # 辞書から一度にDataFrameを作成
        df_down = pd.DataFrame(resampled_data).reset_index(drop=True)
    else:
        df_down = df

    marker_set = ["RASI", "LASI", "RPSI", "LPSI","RKNE","LKNE", "RANK","LANK","RTOE","LTOE","RHEE","LHEE", "RKNE2", "LKNE2", "RANK2", "LANK2"]

    marker_set_df = df_down[[col for col in df_down.columns if any(marker in col[0] for marker in marker_set)]].copy()

    marker_set_df.to_csv(os.path.join(os.path.dirname(csv_path), f"before_interp_marker_set_{os.path.basename(csv_path)}"))

    if marker_set_df.empty:
        print("Error: No marker data found")
        return np.array([]), range(0)

    # LPSI補間を行う前に、元々LPSIが欠損していた場所を記録しておく
    lpsi_cols = [c for c in marker_set_df.columns if 'LPSI' in c[0]]
    original_lpsi_missing_mask = pd.Series(False, index=marker_set_df.index)
    if lpsi_cols:
        original_lpsi_missing_mask = marker_set_df[lpsi_cols].isnull().any(axis=1)
        print("LPSIの元々の欠損状況:")

    print("LPSI等の細かい欠損を補完するため、先に三次スプライン補間を実行します。")
    marker_set_df.interpolate(method='spline', order=3, limit_direction='both', inplace=True)

    if geometry_path and os.path.exists(geometry_path):
        if original_lpsi_missing_mask.any():
            print(f"ジオメトリ情報 {geometry_path} を使用して、元々欠損していたLPSIを再計算・補完します。")
            with open(geometry_path, 'r') as f:
                geometry = json.load(f)

            ref_vecs = geometry['reference_vectors']
            source_vectors = [np.array(ref_vecs['RASI']), np.array(ref_vecs['LASI']), np.array(ref_vecs['RPSI'])]
            target_offset_vector = np.array(geometry['target_offset_vector']['LPSI'])

            rasi_cols = [c for c in marker_set_df.columns if 'RASI' in c[0]]
            lasi_cols = [c for c in marker_set_df.columns if 'LASI' in c[0]]
            rpsi_cols = [c for c in marker_set_df.columns if 'RPSI' in c[0]]

            # 元々LPSIが欠損していた行だけをループ
            for index in marker_set_df[original_lpsi_missing_mask].index:
                row = marker_set_df.loc[index]
                # 基準点が（スプライン補間後に）有効か確認
                if not row[rasi_cols].isnull().any() and not row[lasi_cols].isnull().any() and not row[rpsi_cols].isnull().any():
                    p_rasi = row[rasi_cols].values
                    p_lasi = row[lasi_cols].values
                    p_rpsi = row[rpsi_cols].values

                    centroid_current = (p_rasi + p_lasi + p_rpsi) / 3
                    target_vectors = [p_rasi - centroid_current, p_lasi - centroid_current, p_rpsi - centroid_current]

                    rot, _ = R.align_vectors(target_vectors, source_vectors)
                    estimated_offset = rot.apply(target_offset_vector)
                    estimated_lpsi = centroid_current + estimated_offset

                    marker_set_df.loc[index, lpsi_cols] = estimated_lpsi
        else:
            print("LPSIに元々の欠損はなかったため、幾何学的な補間はスキップしました。")

    # 最終チェック：それでも欠損が残っている場合は線形補間
    if marker_set_df.isnull().values.any():
        print("警告: 全ての補間処理後もデータに欠損値が残っています。残りの欠損を線形補間します。")
        marker_set_df.interpolate(method='linear', limit_direction='both', inplace=True)

    if marker_set_df.isnull().values.any():
        print("エラー: 補間後も処理できない欠損値が残っています。")
        return np.array([]), range(0)

    # full_range をダウンサンプリング後のインデックスベースで設定
    full_range = range(0, len(marker_set_df))
    print(f"Processing range: {full_range}")

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

def main():
    down_hz = True  # True: 100Hz→60Hz変換, False: 100Hzのまま
    # csv_path_dir = r"G:\gait_pattern\20250811_br\sub0\thera0-16\mocap"
    # csv_path_dir = r"G:\gait_pattern\20250811_br\sub0\thera0-14\mocap"
    # csv_path_dir = r"G:\gait_pattern\20250811_br\sub0\thera0-15\mocap"
    # csv_path_dir = r"G:\gait_pattern\20250811_br\sub1\thera0-2\mocap"
    csv_path_dir = r"G:\gait_pattern\20250811_br\sub1\thera1-1\mocap"
    if csv_path_dir == r"G:\gait_pattern\20250811_br\sub1\thera0-2\mocap":
        start_frame_100hz = 1000
        end_frame_100hz = 1440
    elif csv_path_dir == r"G:\gait_pattern\20250811_br\sub1\thera1-1\mocap":
        start_frame_100hz = 1089
        end_frame_100hz = 1252
    elif csv_path_dir == r"G:\gait_pattern\20250811_br\sub0\thera0-16\mocap":
        start_frame_100hz = 890
        end_frame_100hz = 1210
    elif csv_path_dir == r"G:\gait_pattern\20250811_br\sub0\thera0-15\mocap":
        # #0-0-15 で股関節外転 maxは1756くらい（60Hz）
        # start_frame_100hz = 2751
        # end_frame_100hz = 3144
        #0-0-15 で右股関節外旋（右足外転）
        start_frame_100hz = 627
        end_frame_100hz = 976
    else:
        #適当
        start_frame_100hz = 0
        end_frame_100hz = 100

    csv_paths = glob.glob(os.path.join(csv_path_dir, "*.csv"))

    # marker_set_で始まるファイルを除外
    csv_paths = [path for path in csv_paths if not os.path.basename(path).startswith("marker_set_")]
    # angle_で始まるファイルも除外（既に処理済みのファイル）
    csv_paths = [path for path in csv_paths if not os.path.basename(path).startswith("angle_")]

    # Tpose時の骨盤の姿勢（3x3の回転行列）を読み込み
    neutral_matrix_path = r"G:\gait_pattern\20250811_br\sub0\thera0-14\mocap\rot_pelvis_neutral.npy"
    rot_pelvis_neutral = np.load(neutral_matrix_path)

    geometry_json_path = r"G:\gait_pattern\20250811_br\sub0\thera0-14\mocap\geometry.json"

    for i, csv_path in enumerate(csv_paths):
        print(f"Processing: {csv_path}")

        try:
            keypoints_mocap, full_range = read_3d_optitrack(csv_path, down_hz, start_frame_100hz, end_frame_100hz,
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
        sampling_freq = 60 if down_hz else 100

        angle_list = []
        dist_list = []
        bector_list = []

        # 前フレームの角度を保存する変数（角度の連続性を保つため）
        prev_angles = None

        # バターワースフィルタのサンプリング周波数を動的に設定
        rasi = np.array([butter_lowpass_fillter(keypoints_mocap[:, 10, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lasi = np.array([butter_lowpass_fillter(keypoints_mocap[:, 2, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rpsi = np.array([butter_lowpass_fillter(keypoints_mocap[:, 14, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lpsi = np.array([butter_lowpass_fillter(keypoints_mocap[:, 6, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rank = np.array([butter_lowpass_fillter(keypoints_mocap[:, 8, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lank = np.array([butter_lowpass_fillter(keypoints_mocap[:, 0, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rank2 = np.array([butter_lowpass_fillter(keypoints_mocap[:, 9, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lank2 = np.array([butter_lowpass_fillter(keypoints_mocap[:, 1, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rknee = np.array([butter_lowpass_fillter(keypoints_mocap[:, 12, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lknee = np.array([butter_lowpass_fillter(keypoints_mocap[:, 4, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rknee2 = np.array([butter_lowpass_fillter(keypoints_mocap[:, 13, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lknee2 = np.array([butter_lowpass_fillter(keypoints_mocap[:, 5, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rtoe = np.array([butter_lowpass_fillter(keypoints_mocap[:, 15, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        ltoe = np.array([butter_lowpass_fillter(keypoints_mocap[:, 7, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rhee = np.array([butter_lowpass_fillter(keypoints_mocap[:, 11, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lhee = np.array([butter_lowpass_fillter(keypoints_mocap[:, 3, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T

        print(f"ダウンサンプリング後full_range: {full_range}")
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

            """現在の骨盤の向きがTposeの状態からどれだけずれているかを計算し、その分を補正する（Tposeの状態をゼロとする）"""
            # rot_pelvis = np.dot(np.linalg.inv(rot_pelvis_neutral), rot_pelvis)
            # e_x_pelvis = rot_pelvis[:,0]
            # e_y_pelvis = rot_pelvis[:,1]
            # e_z_pelvis = rot_pelvis[:,2]

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

            # as_eulerが小文字(内因性の回転角度)となるよう設定
            # 屈曲伸展（底屈背屈）角度を計算
            r_hip_realative_rotation = np.dot(np.linalg.inv(rot_pelvis), rot_rthigh)
            l_hip_realative_rotation = np.dot(np.linalg.inv(rot_pelvis), rot_lthigh)
            r_knee_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rthigh)
            l_knee_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lthigh)
            r_ankle_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rfoot)
            l_ankle_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lfoot)
            r_hip_flexion_angle = R.from_matrix(r_hip_realative_rotation).as_euler('YZX', degrees=True)[0]
            l_hip_flexion_angle = R.from_matrix(l_hip_realative_rotation).as_euler('YZX', degrees=True)[0]
            r_knee_flexion_angle =  R.from_matrix(r_knee_realative_rotation).as_euler('YZX', degrees=True)[0]
            l_knee_flexion_angle = R.from_matrix(l_knee_realative_rotation).as_euler('YZX', degrees=True)[0]
            r_ankle_absorption_angle = R.from_matrix(r_ankle_realative_rotation).as_euler('YZX', degrees=True)[0]
            l_ankle_absorption_angle = R.from_matrix(l_ankle_realative_rotation).as_euler('YZX', degrees=True)[0]

            # 股関節の外旋内旋角度
            r_hip_external_rotation_angle = R.from_matrix(r_hip_realative_rotation).as_euler('YZX', degrees=True)[1]
            l_hip_external_rotation_angle = R.from_matrix(l_hip_realative_rotation).as_euler('YZX', degrees=True)[1]
            # # 確認用0-0-15 左足基準で右足の外旋内旋角度を計算(足の場合はXが外転内転, Z内返し外返し)
            # r_ankle_realative_rotation_rl = np.dot(np.linalg.inv(rot_lshank), rot_rfoot)
            # r_hip_external_rotation_angle = R.from_matrix(r_ankle_realative_rotation_rl).as_euler('YZX', degrees=True)[2]


            # 股関節の外転内転角度
            r_hip_abduction_angle = R.from_matrix(r_hip_realative_rotation).as_euler('YZX', degrees=True)[2]
            l_hip_abduction_angle = R.from_matrix(l_hip_realative_rotation).as_euler('YZX', degrees=True)[2]
            # # 確認用0-0-15で
            # rknee_realative_rotation_rl = np.dot(np.linalg.inv(rot_lshank), rot_rshank)
            # r_hip_abduction_angle = R.from_matrix(rknee_realative_rotation_rl).as_euler('YZX', degrees=True)[2]  #左下腿基準で右膝の外転内転角度を計算



            # ただの2つのベクトルのなす角

            # 角度の連続性を保つ処理
            def unwrap_angle(current_angle, prev_angle):
                """角度の連続性を保つため、360度ジャンプを修正"""
                if prev_angle is None:
                    return current_angle

                diff = current_angle - prev_angle
                if diff > 180:
                    return current_angle - 360
                elif diff < -180:
                    return current_angle + 360
                else:
                    return current_angle

            angles = [r_hip_flexion_angle, l_hip_flexion_angle, r_knee_flexion_angle, l_knee_flexion_angle, r_ankle_absorption_angle, l_ankle_absorption_angle,
                      r_hip_abduction_angle, l_hip_abduction_angle, r_hip_external_rotation_angle, l_hip_external_rotation_angle]

            # 角度の連続性を保つ
            if prev_angles is not None:
                angles = [unwrap_angle(angle, prev_angles[i]) for i, angle in enumerate(angles)]

            angles = [angle + 360 if angle < -90 else angle for angle in angles]   #角度が大きく負になる場合は360度足す

            angles[0] = 180 - angles[0]  #hip屈曲伸展
            angles[1] = 180 - angles[1]
            angles[2] = 180 - angles[2]  #knee屈曲伸展
            angles[3] = 180 - angles[3]
            angles[4] = 90 - angles[4]  #ankle底屈背屈
            angles[5] = 90 - angles[5]
            angles[6] = - angles[6]  #hip外転内転
            angles[7] = - angles[7]
            angles[8] = angles[8]  #hip外旋内旋
            angles[9] = angles[9]

            # 前フレームの角度として保存
            prev_angles = angles.copy()

            angle_list.append(angles)

            # def calculate_angle(vector1, vector2):  #(frame, xyz)または(frame, xy)の配列を入力)
            #     angle_list = []
            #     for frame in range(len(vector1)):
            #         dot_product = np.dot(vector1[frame], vector2[frame])
            #         cross_product = np.cross(vector1[frame], vector2[frame])
            #         angle = np.arctan2(np.linalg.norm(cross_product), dot_product)
            #         angle = angle * 180 / np.pi
            #         angle_list.append(angle)

            #     return angle_list

            # trans_mat_pelvis = np.array([[e_x_pelvis[0], e_y_pelvis[0], e_z_pelvis[0], hip[0]],
            #                       [e_x_pelvis[1], e_y_pelvis[1], e_z_pelvis[1], hip[1]],
            #                       [e_x_pelvis[2], e_y_pelvis[2], e_z_pelvis[2], hip[2]],
            #                       [0, 0, 0, 1]])
            # lhee_basse_pelvis = np.dot(trans_mat_pelvis, np.append(lhee[frame_num,:], 1))[:3]
            # print(f"lhee_basse_pelvis = {lhee_basse_pelvis}")

            plot_flag = True
            if plot_flag:
                # print(frame_num)  #相対フレーム数
                if frame_num == 0:
                # if frame_num == 1756-1650:  #100Hzで2926(足を最大に外転してるくらいの時)
                    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("z")
                    ax.set_xlim(-1, 1)
                    ax.set_ylim(-1, 1)
                    ax.set_zlim(-1, 1)
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


                    ax.plot([lhee[frame_num, :][0]- hip[0]], [lhee[frame_num, :][1]- hip[1]], [lhee[frame_num, :][2]- hip[2]], color='red')

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

                    plt.legend()
                    plt.show()

            # e_z_lshank_list.append(e_z_lshank)
            # e_z_lfoot_list.append(e_z_lfoot)

            #骨盤とかかとの距離計算
            dist = np.linalg.norm(rhee[frame_num, :] - hip[:])
            bector = rhee[frame_num, :] - hip[:]
            dist_list.append(dist)
            bector_list.append(bector)
            # bector_list.append(lhee_basse_pelvis)

        # print(f"angle_array = {angle_array}")
        # print(f"angle_array.shape = {angle_array.shape}")

        # angle_listをnumpy配列に変換
        angle_array = np.array(angle_list)

        # DataFrameのインデックスを絶対フレーム番号に設定
        if down_hz:
            # 60Hzデータの場合、start_frameからの60Hz絶対フレーム番号
            absolute_frame_indices = np.array(full_range) + int(start_frame_100hz * 60 / 100)
        else:
            # 100Hzデータの場合、start_frameからの100Hz絶対フレーム番号
            absolute_frame_indices = np.array(full_range) + start_frame_100hz

        df = pd.DataFrame({
            "r_hip_flexion_angle": angle_array[:, 0],
            "r_knee_flexion_angle": angle_array[:, 2],
            "r_ankle_absorption_angle": angle_array[:, 4],
            "l_hip_flexion_angle": angle_array[:, 1],
            "l_knee_flexion_angle": angle_array[:, 3],
            "l_ankle_absorption_angle": angle_array[:, 5],
            "r_hip_abduction_angle": angle_array[:, 6],
            "l_hip_abduction_angle": angle_array[:, 7],
            "r_hip_external_rotation_angle": angle_array[:, 8],
            "l_hip_external_rotation_angle": angle_array[:, 9]
        }, index=absolute_frame_indices)

        # ファイル名に適切なサンプリング周波数を記載
        if down_hz:
            df.to_csv(os.path.join(os.path.dirname(csv_path), f"angle_60Hz_{os.path.basename(csv_path)}"))
        else:
            df.to_csv(os.path.join(os.path.dirname(csv_path), f"angle_100Hz_{os.path.basename(csv_path)}"))

        # ankle_angle = calculate_angle(e_z_lshank_list, e_z_lfoot_list)
        # print(f"ankle_angle = {ankle_angle}")

        # fig, ax = plt.subplots()
        # ax.plot(full_range, dist_list)
        # plt.show()
        # plt.cla

        if down_hz:
            bector_array = np.array(bector_list)
            rhee_pel_z = bector_array[:, 2]

            # 60Hzデータでの相対フレーム番号（0から始まる）
            relative_frames_60hz = np.array(full_range)

            # 60Hzデータでの絶対フレーム番号
            # start_frameは100Hzでの番号なので、60Hzに変換してから相対フレームに加算
            start_frame_60hz = int(start_frame_100hz * 60 / 100)
            absolute_frames_60hz = relative_frames_60hz + start_frame_60hz
            end_frame_60hz = int(end_frame_100hz * 60 / 100)

            # 対応する100Hzでの絶対フレーム番号
            # 60Hz相対フレームを100Hzに変換してstart_frameを加算
            absolute_frames_100hz = (relative_frames_60hz * 100 / 60 + start_frame_100hz).astype(int)

            df_ic = pd.DataFrame({
                "frame_60hz_relative": relative_frames_60hz,
                "frame_60hz_absolute": absolute_frames_60hz,
                "frame_100hz_absolute": absolute_frames_100hz,
                "rhee_pel_z": rhee_pel_z
            })
            df_ic = df_ic.sort_values(by="rhee_pel_z", ascending=False)

            # 初期接地検出（60Hz相対フレーム番号で）
            ic_list_60hz_relative = df_ic.head(30)["frame_60hz_relative"].values.astype(int)
            ic_list_60hz_absolute = df_ic.head(30)["frame_60hz_absolute"].values.astype(int)
            ic_list_100hz_absolute = df_ic.head(30)["frame_100hz_absolute"].values.astype(int)

            print(f"start_frame (100Hz): {start_frame_100hz}")
            print(f"end_frame (100Hz): {end_frame_100hz}")
            print(f"start_frame (60Hz): {start_frame_60hz}")
            print(f"end_frame (60Hz): {end_frame_60hz}")
            print(f"ic_list (60Hz相対フレーム): {ic_list_60hz_relative}")
            print(f"ic_list (60Hz絶対フレーム): {ic_list_60hz_absolute}")
            print(f"ic_list (100Hz絶対フレーム): {ic_list_100hz_absolute}")

            # フィルタリング処理（60Hz相対フレーム番号ベース）
            filtered_list_60hz_relative = []
            skip_values = set()
            for value in ic_list_60hz_relative:
                if value in skip_values:
                    continue
                filtered_list_60hz_relative.append(value)
                # 60Hzでの10フレーム間隔でスキップ
                skip_values.update(range(value - 10, value + 11))
            filtered_list_60hz_relative = sorted(filtered_list_60hz_relative)

            # 絶対フレーム番号に変換
            filtered_list_60hz_absolute = []
            filtered_list_100hz_absolute = []
            for relative_frame in filtered_list_60hz_relative:
                # 60Hz絶対フレーム番号
                absolute_60hz = relative_frame + start_frame_60hz
                filtered_list_60hz_absolute.append(absolute_60hz)

                # 100Hz絶対フレーム番号
                absolute_100hz = int(relative_frame * 100 / 60 + start_frame_100hz)
                filtered_list_100hz_absolute.append(absolute_100hz)

            print(f"フィルタリング後のリスト (60Hz絶対フレーム): {filtered_list_60hz_absolute}")
            print(f"フィルタリング後のリスト (100Hz絶対フレーム): {filtered_list_100hz_absolute}")

            # 絶対フレーム番号として保存
            np.save(os.path.join(os.path.dirname(csv_path), f"ic_frame_60Hz_absolute_{os.path.basename(csv_path).split('.')[0]}"), filtered_list_60hz_absolute)
            np.save(os.path.join(os.path.dirname(csv_path), f"ic_frame_100Hz_absolute_{os.path.basename(csv_path).split('.')[0]}"), filtered_list_100hz_absolute)

            # 関節角度と初期接地をプロット（60Hz絶対フレーム番号で統一）
            plt.figure(figsize=(12, 6))
            # dfのインデックスを使用（60Hz絶対フレーム番号）
            plt.plot(df.index, df["r_hip_flexion_angle"], label="Right Hip Flexion Angle")
            plt.plot(df.index, df["r_knee_flexion_angle"], label="Right Knee Flexion Angle")
            plt.plot(df.index, df["r_ankle_absorption_angle"], label="Right Ankle Absorption Angle")
            plt.plot(df.index, df["l_hip_flexion_angle"], label="Left Hip Flexion Angle")
            plt.plot(df.index, df["l_knee_flexion_angle"], label="Left Knee Flexion Angle")
            plt.plot(df.index, df["l_ankle_absorption_angle"], label="Left Ankle Absorption Angle")
            # 初期接地フレーム（60Hz絶対フレーム）を縦線で表示
            for idx, ic_frame_60hz in enumerate(filtered_list_60hz_absolute):
                plt.axvline(x=ic_frame_60hz, color='red', linestyle='--', alpha=0.5, label='Initial Contact (60Hz)' if idx == 0 else "")
            plt.xlabel("Frame (60Hz absolute)")
            plt.ylabel("Angle (degrees)")
            plt.title("Joint Angles and Initial Contact Frames (60Hz)")
            plt.legend()
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(csv_path_dir, "joint_angles_initial_contact_60Hz.png"))
            plt.close()

            # 股関節の外旋内旋角度プロット
            # plt.plot(df.index, df["l_hip_external_rotation_angle"], label="Left Hip External Rotation Angle")
            plt.plot(df.index, df["r_hip_external_rotation_angle"], label="Right Hip External Rotation Angle")
            for idx, ic_frame_60hz in enumerate(filtered_list_60hz_absolute):
                plt.axvline(x=ic_frame_60hz, color='red', linestyle='--', alpha=0.5, label='Initial Contact (60Hz)' if idx == 0 else "")
            plt.ylabel("Angle (degrees)")
            plt.title("Hip External Rotation Angles")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(csv_path_dir, "hip_external_rotation_angles_60Hz.png"))
            plt.show()
            plt.close()


            # 股関節の外転内転角度プロット
            plt.plot(df.index, df["r_hip_abduction_angle"], label="Right Hip Abduction Angle")
            # plt.plot(df.index, df["l_hip_abduction_angle"], label="Left Hip Abduction Angle")
            for idx, ic_frame_60hz in enumerate(filtered_list_60hz_absolute):
                plt.axvline(x=ic_frame_60hz, color='red', linestyle='--', alpha=0.5, label='Initial Contact (60Hz)' if idx == 0 else "")
            plt.ylabel("Angle (degrees)")
            plt.title("Hip Abduction Angles")
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.savefig(os.path.join(csv_path_dir, "hip_abduction_angles_60Hz.png"))
            plt.close()



            # デバッグ用：角度データとICフレームの対応確認
            print("\n=== デバッグ情報 ===")
            print(f"角度データのインデックス範囲: {df.index.min()} - {df.index.max()}")
            print(f"ICフレーム（60Hz絶対）: {filtered_list_60hz_absolute}")
            print(f"ICフレーム（100Hz絶対）: {filtered_list_100hz_absolute}")

            # ICフレームが角度データの範囲内にあるかチェック
            for ic_frame in filtered_list_60hz_absolute:
                if ic_frame in df.index:
                    print(f"IC Frame {ic_frame}: 角度データ内に存在")
                else:
                    print(f"IC Frame {ic_frame}: 角度データ範囲外")
        else:
            # 100Hzのままの場合
            bector_array = np.array(bector_list)
            lhee_pel_z = bector_array[:, 2]

            # 100Hzデータでの絶対フレーム番号を計算
            absolute_frames_100hz = np.array(full_range) + start_frame_100hz

            df_ic = pd.DataFrame({
                "frame_100hz": absolute_frames_100hz,
                "lhee_pel_z": lhee_pel_z
            })
            df_ic = df_ic.sort_values(by="lhee_pel_z", ascending=False)

            ic_list_100hz = df_ic.head(30)["frame_100hz"].values.astype(int)
            print(f"ic_list (100Hz絶対フレーム) = {ic_list_100hz}")

            # フィルタリング処理
            filtered_list_100hz = []
            skip_values = set()
            for value in ic_list_100hz:
                if value in skip_values:
                    continue
                filtered_list_100hz.append(value)
                # 100Hzでの16フレーム間隔でスキップ（60Hzでの10フレームに相当）
                skip_values.update(range(value - 16, value + 17))
            filtered_list_100hz = sorted(filtered_list_100hz)
            print(f"フィルタリング後のリスト (100Hz絶対フレーム): {filtered_list_100hz}")

            # 絶対フレーム番号として保存
            np.save(os.path.join(os.path.dirname(csv_path), f"ic_frame_100Hz_absolute_{os.path.basename(csv_path).split('.')[0]}"), filtered_list_100hz)

            # 関節角度と初期接地をプロット（100Hzの場合）
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df["r_hip_angle"], label="Right Hip Angle")
            plt.plot(df.index, df["r_knee_angle"], label="Right Knee Angle")
            plt.plot(df.index, df["r_ankle_angle"], label="Right Ankle Angle")
            plt.plot(df.index, df["l_hip_angle"], label="Left Hip Angle")
            plt.plot(df.index, df["l_knee_angle"], label="Left Knee Angle")
            plt.plot(df.index, df["l_ankle_angle"], label="Left Ankle Angle")

            # 初期接地フレーム（100Hz絶対フレーム）を縦線で表示
            for idx, ic_frame_100hz in enumerate(filtered_list_100hz):
                plt.axvline(x=ic_frame_100hz, color='red', linestyle='--', alpha=0.5, label='Initial Contact' if idx == 0 else "")

            plt.xlabel("Frame (100Hz absolute)")
            plt.ylabel("Angle (degrees)")
            plt.title("Joint Angles and Initial Contact Frames")
            plt.legend()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()