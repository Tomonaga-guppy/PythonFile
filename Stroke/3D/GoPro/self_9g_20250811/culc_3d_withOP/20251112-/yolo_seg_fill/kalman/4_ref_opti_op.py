import pandas as pd
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample
import json
import m_opti as opti
import m_openpose as op

def main():
    # csv_path_dir = r"G:\gait_pattern\20250811_br\sub0\thera0-16\mocap"
    # csv_path_dir = r"G:\gait_pattern\20250811_br\sub0\thera0-14\mocap"
    # csv_path_dir = r"G:\gait_pattern\20250811_br\sub0\thera0-15\mocap"
    # csv_path_dir = r"G:\gait_pattern\20250811_br\sub1\thera0-2\mocap"
    # csv_path_dir = Path(r"G:\gait_pattern\20250811_br\sub1\thera0-3\mocap")
    csv_path_dir = Path(r"G:\gait_pattern\20250811_br\sub1\thera1-0\mocap")

    if str(csv_path_dir) == r"G:\gait_pattern\20250811_br\sub1\thera0-2\mocap":
        start_frame = 1000
        end_frame = 1440
    elif str(csv_path_dir) == r"G:\gait_pattern\20250811_br\sub1\thera0-3\mocap":
        start_frame = 943
        end_frame = 1400
    elif str(csv_path_dir) == r"G:\gait_pattern\20250811_br\sub1\thera1-0\mocap":
        start_frame = 1090
        end_frame = 1252
    elif str(csv_path_dir) == r"G:\gait_pattern\20250811_br\sub0\thera0-16\mocap":
        start_frame = 890
        end_frame = 1210
    elif str(csv_path_dir) == r"G:\gait_pattern\20250811_br\sub0\thera0-15\mocap":
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

    csv_paths = list(csv_path_dir.glob("*.csv"))

    # marker_set_で始まるファイルを除外
    csv_paths = [path for path in csv_paths if not path.name.startswith("marker_set_")]
    # angle_で始まるファイルも除外（既に処理済みのファイル）
    csv_paths = [path for path in csv_paths if not path.name.startswith("angle_")]
    # beforeで始まるファイルも除外（既に処理済みのファイル）
    csv_paths = [path for path in csv_paths if not path.name.startswith("before_")]
    #  afterで始まるファイルも除外（既に処理済みのファイル）
    csv_paths = [path for path in csv_paths if not path.name.startswith("after_")]
    # normalized_で始まるファイルも除外（既に処理済みのファイル）
    csv_paths = [path for path in csv_paths if not path.name.startswith("normalized_")]
    # gait_parameters_で始まるファイルも除外（既に処理済みのファイル）
    csv_paths = [path for path in csv_paths if not path.name.startswith("gait_parameters_")]
    # symmetry_indices_で始まるファイルも除外（既に処理済みのファイル）
    csv_paths = [path for path in csv_paths if not path.name.startswith("symmetry_indices_")]


    geometry_json_path = Path(r"G:\gait_pattern\20250811_br\sub0\thera0-16\mocap\geometry.json")

    for i, csv_path in enumerate(csv_paths):
        print(f"Processing: {csv_path}")

        try:
            keypoints_mocap, full_range = opti.read_3d_optitrack(csv_path, start_frame, end_frame,
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

        # バターワースフィルタのサンプリング周波数を動的に設定
        rasi = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 10, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lasi = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 2, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rpsi = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 14, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lpsi = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 6, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rank = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 8, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lank = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 0, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rank2 = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 9, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lank2 = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 1, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rknee = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 12, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lknee = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 4, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rknee2 = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 13, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lknee2 = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 5, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rtoe = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 15, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        ltoe = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 7, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        rhee = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 11, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
        lhee = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 3, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T

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
            e_x_pelvis = e_x_pelvis_0
            e_y_pelvis = e_y_pelvis_0
            e_z_pelvis = e_z_pelvis_0
            rot_pelvis = np.array([e_x_pelvis, e_y_pelvis, e_z_pelvis]).T
            # #####################

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
            
            # e_y0_pelvis = (lthigh - rthigh)/np.linalg.norm(lthigh - rthigh)
            # e_z_pelvis = (lumbar - hip)/np.linalg.norm(lumbar - hip)
            # e_x_pelvis = np.cross(e_y0_pelvis, e_z_pelvis)/np.linalg.norm(np.cross(e_y0_pelvis, e_z_pelvis))
            # e_y_pelvis = np.cross(e_z_pelvis, e_x_pelvis)
            # rot_pelvis = np.array([e_x_pelvis, e_y_pelvis, e_z_pelvis]).T

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

            angle_list.append([r_hip_angle_flex, l_hip_angle_flex, r_knee_angle_flex, l_knee_angle_flex, r_ankle_angle_pldo, l_ankle_angle_pldo,
                                    r_hip_angle_inex, l_hip_angle_inex, r_knee_angle_inex, l_knee_angle_inex, r_ankle_angle_inex, l_ankle_angle_inex,
                                    r_hip_angle_adab, l_hip_angle_adab, r_knee_angle_adab, l_knee_angle_adab, r_ankle_angle_adab, l_ankle_angle_adab])


            plot_flag = False
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
        angle_df.to_csv(csv_path.parent / f"angle_100Hz_{csv_path.name}")

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
        
        df_IcTo = pd.DataFrame({
            "frame_100hz_rel": rel_frames,
            "frame_100hz_abs": abs_frames,
            "rhee_pel_z": rhee_pel_z,
            "lhee_pel_z": lhee_pel_z,
        })
        
        df_IcTo.index = df_IcTo.index + 1
        df_ic_r = df_IcTo.sort_values(by="rhee_pel_z", ascending=False)  #降順: 大きいピークが初期接地
        df_ic_l = df_IcTo.sort_values(by="lhee_pel_z", ascending=False)  #降順
        df_to_r = df_IcTo.sort_values(by="rhee_pel_z")  #昇順  小さいピークがつま先離地
        df_to_l = df_IcTo.sort_values(by="lhee_pel_z")  #昇順

        # 初期接地検出（100Hz相対フレーム番号で）
        ic_r_list_100hz_rel = df_ic_r.head(30)["frame_100hz_rel"].values.astype(int)
        ic_l_list_100hz_rel = df_ic_l.head(30)["frame_100hz_rel"].values.astype(int)
        ic_r_list_100hz_abs = df_ic_r.head(30)["frame_100hz_abs"].values.astype(int)
        ic_l_list_100hz_abs = df_ic_l.head(30)["frame_100hz_abs"].values.astype(int)
        
        to_r_list_100hz_rel = df_to_r.head(30)["frame_100hz_rel"].values.astype(int)
        to_l_list_100hz_rel = df_to_l.head(30)["frame_100hz_rel"].values.astype(int)
        to_r_list_100hz_abs = df_to_r.head(30)["frame_100hz_abs"].values.astype(int)
        to_l_list_100hz_abs = df_to_l.head(30)["frame_100hz_abs"].values.astype(int)

        print(f"start_frame (100Hz): {start_frame}")
        print(f"end_frame (100Hz): {end_frame}")
        print(f"ic_r_list (100Hz相対フレーム): {ic_r_list_100hz_rel}")
        print(f"ic_l_list (100Hz相対フレーム): {ic_l_list_100hz_rel}")
        print(f"ic_r_list (100Hz絶対フレーム): {ic_r_list_100hz_abs}")
        print(f"ic_l_list (100Hz絶対フレーム): {ic_l_list_100hz_abs}")
        
        print(f"to_r_list (100Hz相対フレーム): {to_r_list_100hz_rel}")
        print(f"to_l_list (100Hz相対フレーム): {to_l_list_100hz_rel}")
        print(f"to_r_list (100Hz絶対フレーム): {to_r_list_100hz_abs}")
        print(f"to_l_list (100Hz絶対フレーム): {to_l_list_100hz_abs}")

        filt_ic_r_list_100hz_rel = []
        skip_values_r = set()
        for value in ic_r_list_100hz_rel:
            if value in skip_values_r:
                continue
            filt_ic_r_list_100hz_rel.append(value)
            # 100Hzでの10フレーム間隔でスキップ
            skip_values_r.update(range(value - 10, value + 11))
        filt_ic_r_list_100hz_rel = sorted(filt_ic_r_list_100hz_rel)
        print(f"フィルタリング後のic_rリスト (100Hz相対フレーム): {filt_ic_r_list_100hz_rel}")

        filt_ic_l_list_100hz_rel = []
        skip_values_l = set()
        for value in ic_l_list_100hz_rel:
            if value in skip_values_l:
                continue
            filt_ic_l_list_100hz_rel.append(value)
            # 100Hzでの10フレーム間隔でスキップ
            skip_values_l.update(range(value - 10, value + 11))
        filt_ic_l_list_100hz_rel = sorted(filt_ic_l_list_100hz_rel)
        print(f"フィルタリング後のic_lリスト (100Hz相対フレーム): {filt_ic_l_list_100hz_rel}")

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
        
        
        filt_to_r_list_100hz_rel = []
        skip_values_r = set()
        for value in to_r_list_100hz_rel:
            if value in skip_values_r:
                continue
            filt_to_r_list_100hz_rel.append(value)
            # 100Hzでの10フレーム間隔でスキップ
            skip_values_r.update(range(value - 10, value + 11))
        filt_to_r_list_100hz_rel = sorted(filt_to_r_list_100hz_rel)
        print(f"フィルタリング後のto_rリスト (100Hz相対フレーム): {filt_to_r_list_100hz_rel}")
        filt_to_l_list_100hz_rel = []
        skip_values_l = set()
        for value in to_l_list_100hz_rel:
            if value in skip_values_l:
                continue
            filt_to_l_list_100hz_rel.append(value)
            # 100Hzでの10フレーム間隔でスキップ
            skip_values_l.update(range(value - 10, value + 11))
        filt_to_l_list_100hz_rel = sorted(filt_to_l_list_100hz_rel)
        print(f"フィルタリング後のto_lリスト (100Hz相対フレーム): {filt_to_l_list_100hz_rel}")
        # 絶対フレーム番号に変換
        filt_to_r_list_100hz_abs = []
        for relative_to_r_frame in filt_to_r_list_100hz_rel:
            # 100Hz絶対フレーム番号
            absolute_100hz = relative_to_r_frame + start_frame
            filt_to_r_list_100hz_abs.append(absolute_100hz)
        print(f"フィルタリング後のto_rリスト (100Hz絶対フレーム): {filt_to_r_list_100hz_abs}")
        filt_to_l_list_100hz_abs = []
        for relative_to_l_frame in filt_to_l_list_100hz_rel:
            # 100Hz絶対フレーム番号
            absolute_100hz = relative_to_l_frame + start_frame
            filt_to_l_list_100hz_abs.append(absolute_100hz)
        print(f"フィルタリング後のto_lリスト (100Hz絶対フレーム): {filt_to_l_list_100hz_abs}")
        

        # 右足の歩行周期を作成 [IC, TO, 次のIC]
        gait_cycles_r = []
        for i in range(len(filt_ic_r_list_100hz_rel) - 1):
            ic_current = filt_ic_r_list_100hz_rel[i]
            ic_next = filt_ic_r_list_100hz_rel[i + 1]

            # 現在のICと次のICの間にあるTOを探す
            to_in_cycle = [to for to in filt_to_r_list_100hz_rel if ic_current < to < ic_next]
            
            if len(to_in_cycle) > 0:
                # 最初のTOを使用
                gait_cycles_r.append([ic_current, to_in_cycle[0], ic_next])
        
        print(f"右足の歩行周期 (100Hz): {gait_cycles_r}")
        
        # 左足も同様に作成
        gait_cycles_l = []
        for i in range(len(filt_ic_l_list_100hz_rel) - 1):
            ic_current = filt_ic_l_list_100hz_rel[i]
            ic_next = filt_ic_l_list_100hz_rel[i + 1]

            # 現在のICと次のICの間にあるTOを探す
            to_in_cycle = [to for to in filt_to_l_list_100hz_rel if ic_current < to < ic_next]
            
            if len(to_in_cycle) > 0:
                gait_cycles_l.append([ic_current, to_in_cycle[0], ic_next])
        
        print(f"左足の歩行周期 (100Hz): {gait_cycles_l}")
        
        gait_cycles_r_abs = []
        for ic_rel, to_rel, ic_next_rel in gait_cycles_r:
            ic_abs = ic_rel + start_frame
            to_abs = to_rel + start_frame
            ic_next_abs = ic_next_rel + start_frame
            gait_cycles_r_abs.append([ic_abs, to_abs, ic_next_abs])
        gait_cycles_l_abs = []
        for ic_rel, to_rel, ic_next_rel in gait_cycles_l:
            ic_abs = ic_rel + start_frame
            to_abs = to_rel + start_frame
            ic_next_abs = ic_next_rel + start_frame
            gait_cycles_l_abs.append([ic_abs, to_abs, ic_next_abs])
        print(f"右足の歩行周期 (100Hz絶対フレーム): {gait_cycles_r_abs}")
        print(f"左足の歩行周期 (100Hz絶対フレーム): {gait_cycles_l_abs}")
        
        # 右足の歩行周期ごとに関節角度を100%に正規化
        normalized_gait_cycles_r = []
        for cycle_idx, (ic_start, to, ic_end) in enumerate(gait_cycles_r):
            cycle_length = ic_end - ic_start
            # 0%から100%まで101点（0, 1, 2, ..., 100）にリサンプリング
            normalized_percentage = np.linspace(0, 100, 101)
            
            # 元のフレーム番号（相対）
            original_frames = np.arange(ic_start, ic_end + 1)
            
            # 各関節角度を補間
            # 屈曲伸展・背屈底屈
            rhip_flex_normalized = np.interp(normalized_percentage, 
                                            np.linspace(0, 100, len(original_frames)),
                                            angle_df.loc[original_frames, 'R_Hip_FlEx'].values)
            rknee_flex_normalized = np.interp(normalized_percentage,
                                              np.linspace(0, 100, len(original_frames)),
                                              angle_df.loc[original_frames, 'R_Knee_FlEx'].values)
            rankle_pldo_normalized = np.interp(normalized_percentage,
                                               np.linspace(0, 100, len(original_frames)),
                                               angle_df.loc[original_frames, 'R_Ankle_PlDo'].values)
            # 内旋外旋
            rhip_inex_normalized = np.interp(normalized_percentage,
                                             np.linspace(0, 100, len(original_frames)),
                                                angle_df.loc[original_frames, 'R_Hip_InEx'].values)
            rknee_inex_normalized = np.interp(normalized_percentage,
                                              np.linspace(0, 100, len(original_frames)),
                                              angle_df.loc[original_frames, 'R_Knee_InEx'].values)
            rankle_inex_normalized = np.interp(normalized_percentage,
                                               np.linspace(0, 100, len(original_frames)),
                                               angle_df.loc[original_frames, 'R_Ankle_InEx'].values)
            # 内転外転
            rhip_adab_normalized = np.interp(normalized_percentage,
                                             np.linspace(0, 100, len(original_frames)),
                                                angle_df.loc[original_frames, 'R_Hip_AdAb'].values)
            rknee_adab_normalized = np.interp(normalized_percentage,
                                              np.linspace(0, 100, len(original_frames)),
                                                angle_df.loc[original_frames, 'R_Knee_AdAb'].values)
            rankle_adab_normalized = np.interp(normalized_percentage,
                                               np.linspace(0, 100, len(original_frames)),
                                                  angle_df.loc[original_frames, 'R_Ankle_AdAb'].values)
            
            # 立脚期の割合を計算
            stance_phase_percentage = ((to - ic_start) / cycle_length) * 100
            
            cycle_data = {
                'cycle_index': cycle_idx,
                'ic_start': ic_start,
                'to': to,
                'ic_end': ic_end,
                'cycle_length_frames': cycle_length,
                'stance_phase_percentage': stance_phase_percentage,
                'percentage': normalized_percentage,
                'R_Hip_FlEx': rhip_flex_normalized,
                'R_Knee_FlEx': rknee_flex_normalized,
                'R_Ankle_PlDo': rankle_pldo_normalized,
                'R_Hip_InEx': rhip_inex_normalized,
                'R_Knee_InEx': rknee_inex_normalized,
                'R_Ankle_InEx': rankle_inex_normalized,
                'R_Hip_AdAb': rhip_adab_normalized,
                'R_Knee_AdAb': rknee_adab_normalized,
                'R_Ankle_AdAb': rankle_adab_normalized
            }
            normalized_gait_cycles_r.append(cycle_data)
        
        # 左足の歩行周期ごとに関節角度を100%に正規化
        normalized_gait_cycles_l = []
        for cycle_idx, (ic_start, to, ic_end) in enumerate(gait_cycles_l):
            cycle_length = ic_end - ic_start
            normalized_percentage = np.linspace(0, 100, 101)
            
            original_frames = np.arange(ic_start, ic_end + 1)
            
            lhip_flex_normalized = np.interp(normalized_percentage,
                                            np.linspace(0, 100, len(original_frames)),
                                            angle_df.loc[original_frames, 'L_Hip_FlEx'].values)
            lknee_flex_normalized = np.interp(normalized_percentage,
                                              np.linspace(0, 100, len(original_frames)),
                                              angle_df.loc[original_frames, 'L_Knee_FlEx'].values)
            lankle_pldo_normalized = np.interp(normalized_percentage,
                                               np.linspace(0, 100, len(original_frames)),
                                               angle_df.loc[original_frames, 'L_Ankle_PlDo'].values)
            
            lhip_inex_normalized = np.interp(normalized_percentage,
                                                np.linspace(0, 100, len(original_frames)),
                                                    angle_df.loc[original_frames, 'L_Hip_InEx'].values)
            lknee_inex_normalized = np.interp(normalized_percentage,
                                               np.linspace(0, 100, len(original_frames)),
                                               angle_df.loc[original_frames, 'L_Knee_InEx'].values)
            lankle_inex_normalized = np.interp(normalized_percentage,
                                               np.linspace(0, 100, len(original_frames)),
                                               angle_df.loc[original_frames, 'L_Ankle_InEx'].values)
            
            lhip_adab_normalized = np.interp(normalized_percentage,
                                              np.linspace(0, 100, len(original_frames)),
                                              angle_df.loc[original_frames, 'L_Hip_AdAb'].values)
            lknee_adab_normalized = np.interp(normalized_percentage,
                                               np.linspace(0, 100, len(original_frames)),
                                               angle_df.loc[original_frames, 'L_Knee_AdAb'].values)
            lankle_adab_normalized = np.interp(normalized_percentage,
                                                np.linspace(0, 100, len(original_frames)),
                                                angle_df.loc[original_frames, 'L_Ankle_AdAb'].values)
            

            stance_phase_percentage = ((to - ic_start) / cycle_length) * 100
            
            cycle_data = {
                'cycle_index': cycle_idx,
                'ic_start': ic_start,
                'to': to,
                'ic_end': ic_end,
                'cycle_length_frames': cycle_length,
                'stance_phase_percentage': stance_phase_percentage,
                'percentage': normalized_percentage,
                'L_Hip_FlEx': lhip_flex_normalized,
                'L_Knee_FlEx': lknee_flex_normalized,
                'L_Ankle_PlDo': lankle_pldo_normalized,
                'L_Hip_InEx': lhip_inex_normalized,
                'L_Knee_InEx': lknee_inex_normalized,
                'L_Ankle_InEx': lankle_inex_normalized,
                'L_Hip_AdAb': lhip_adab_normalized,
                'L_Knee_AdAb': lknee_adab_normalized,
                'L_Ankle_AdAb': lankle_adab_normalized
            }
            normalized_gait_cycles_l.append(cycle_data)
        
        # 正規化されたデータを保存
        for cycle_data in normalized_gait_cycles_r:
            cycle_df = pd.DataFrame({
                'Percentage': cycle_data['percentage'],
                'R_Hip_FlEx': cycle_data['R_Hip_FlEx'],
                'R_Knee_FlEx': cycle_data['R_Knee_FlEx'],
                'R_Ankle_PlDo': cycle_data['R_Ankle_PlDo'],
                'R_Hip_InEx': cycle_data['R_Hip_InEx'],
                'R_Knee_InEx': cycle_data['R_Knee_InEx'],
                'R_Ankle_InEx': cycle_data['R_Ankle_InEx'],
                'R_Hip_AdAb': cycle_data['R_Hip_AdAb'],
                'R_Knee_AdAb': cycle_data['R_Knee_AdAb'],
                'R_Ankle_AdAb': cycle_data['R_Ankle_AdAb']
            })
            cycle_df.to_csv(csv_path_dir / f"normalized_cycle_R_{cycle_data['cycle_index']}_{csv_path.stem}.csv", index=False)
        
        for cycle_data in normalized_gait_cycles_l:
            cycle_df = pd.DataFrame({
                'Percentage': cycle_data['percentage'],
                'L_Hip_FlEx': cycle_data['L_Hip_FlEx'],
                'L_Knee_FlEx': cycle_data['L_Knee_FlEx'],
                'L_Ankle_PlDo': cycle_data['L_Ankle_PlDo'],
                'L_Hip_InEx': cycle_data['L_Hip_InEx'],
                'L_Knee_InEx': cycle_data['L_Knee_InEx'],
                'L_Ankle_InEx': cycle_data['L_Ankle_InEx'],
                'L_Hip_AdAb': cycle_data['L_Hip_AdAb'],
                'L_Knee_AdAb': cycle_data['L_Knee_AdAb'],
                'L_Ankle_AdAb': cycle_data['L_Ankle_AdAb']
            })
            cycle_df.to_csv(csv_path_dir / f"normalized_cycle_L_{cycle_data['cycle_index']}_{csv_path.stem}.csv", index=False)
        
        # 各サイクルの平均と標準偏差を計算
        if len(normalized_gait_cycles_r) > 0:
            all_rhip = np.array([cycle['R_Hip_FlEx'] for cycle in normalized_gait_cycles_r])
            all_rknee = np.array([cycle['R_Knee_FlEx'] for cycle in normalized_gait_cycles_r])
            all_rankle = np.array([cycle['R_Ankle_PlDo'] for cycle in normalized_gait_cycles_r])
            all_rhip_inex = np.array([cycle['R_Hip_InEx'] for cycle in normalized_gait_cycles_r])
            all_rknee_inex = np.array([cycle['R_Knee_InEx'] for cycle in normalized_gait_cycles_r])
            all_rankle_inex = np.array([cycle['R_Ankle_InEx'] for cycle in normalized_gait_cycles_r])
            all_rhip_adab = np.array([cycle['R_Hip_AdAb'] for cycle in normalized_gait_cycles_r])
            all_rknee_adab = np.array([cycle['R_Knee_AdAb'] for cycle in normalized_gait_cycles_r])
            all_rankle_adab = np.array([cycle['R_Ankle_AdAb'] for cycle in normalized_gait_cycles_r])

            mean_cycle_r = pd.DataFrame({
                'Percentage': normalized_percentage,
                'R_Hip_FlEx_mean': np.mean(all_rhip, axis=0),
                'R_Hip_FlEx_std': np.std(all_rhip, axis=0),
                'R_Knee_FlEx_mean': np.mean(all_rknee, axis=0),
                'R_Knee_FlEx_std': np.std(all_rknee, axis=0),
                'R_Ankle_PlDo_mean': np.mean(all_rankle, axis=0),
                'R_Ankle_PlDo_std': np.std(all_rankle, axis=0),
                'R_Hip_InEx_mean': np.mean(all_rhip_inex, axis=0),
                'R_Hip_InEx_std': np.std(all_rhip_inex, axis=0),
                'R_Knee_InEx_mean': np.mean(all_rknee_inex, axis=0),
                'R_Knee_InEx_std': np.std(all_rknee_inex, axis=0),
                'R_Ankle_InEx_mean': np.mean(all_rankle_inex, axis=0),
                'R_Ankle_InEx_std': np.std(all_rankle_inex, axis=0),
                'R_Hip_AdAb_mean': np.mean(all_rhip_adab, axis=0),
                'R_Hip_AdAb_std': np.std(all_rhip_adab, axis=0),
                'R_Knee_AdAb_mean': np.mean(all_rknee_adab, axis=0),
                'R_Knee_AdAb_std': np.std(all_rknee_adab, axis=0),
                'R_Ankle_AdAb_mean': np.mean(all_rankle_adab, axis=0),
                'R_Ankle_AdAb_std': np.std(all_rankle_adab, axis=0)
            })
            mean_cycle_r.to_csv(csv_path_dir / f"normalized_cycle_R_mean_{csv_path.stem}.csv", index=False)
        
        if len(normalized_gait_cycles_l) > 0:
            all_lhip = np.array([cycle['L_Hip_FlEx'] for cycle in normalized_gait_cycles_l])
            all_lknee = np.array([cycle['L_Knee_FlEx'] for cycle in normalized_gait_cycles_l])
            all_lankle = np.array([cycle['L_Ankle_PlDo'] for cycle in normalized_gait_cycles_l])
            all_lhip_inex = np.array([cycle['L_Hip_InEx'] for cycle in normalized_gait_cycles_l])
            all_lknee_inex = np.array([cycle['L_Knee_InEx'] for cycle in normalized_gait_cycles_l])
            all_lankle_inex = np.array([cycle['L_Ankle_InEx'] for cycle in normalized_gait_cycles_l])
            all_lhip_adab = np.array([cycle['L_Hip_AdAb'] for cycle in normalized_gait_cycles_l])
            all_lknee_adab = np.array([cycle['L_Knee_AdAb'] for cycle in normalized_gait_cycles_l])
            all_lankle_adab = np.array([cycle['L_Ankle_AdAb'] for cycle in normalized_gait_cycles_l])

            mean_cycle_l = pd.DataFrame({
                'Percentage': normalized_percentage,
                'L_Hip_FlEx_mean': np.mean(all_lhip, axis=0),
                'L_Hip_FlEx_std': np.std(all_lhip, axis=0),
                'L_Knee_FlEx_mean': np.mean(all_lknee, axis=0),
                'L_Knee_FlEx_std': np.std(all_lknee, axis=0),
                'L_Ankle_PlDo_mean': np.mean(all_lankle, axis=0),
                'L_Ankle_PlDo_std': np.std(all_lankle, axis=0),
                'L_Hip_InEx_mean': np.mean(all_lhip_inex, axis=0),
                'L_Hip_InEx_std': np.std(all_lhip_inex, axis=0),
                'L_Knee_InEx_mean': np.mean(all_lknee_inex, axis=0),
                'L_Knee_InEx_std': np.std(all_lknee_inex, axis=0),
                'L_Ankle_InEx_mean': np.mean(all_lankle_inex, axis=0),
                'L_Ankle_InEx_std': np.std(all_lankle_inex, axis=0),
                'L_Hip_AdAb_mean': np.mean(all_lhip_adab, axis=0),
                'L_Hip_AdAb_std': np.std(all_lhip_adab, axis=0),
                'L_Knee_AdAb_mean': np.mean(all_lknee_adab, axis=0),
                'L_Knee_AdAb_std': np.std(all_lknee_adab, axis=0),
                'L_Ankle_AdAb_mean': np.mean(all_lankle_adab, axis=0),
                'L_Ankle_AdAb_std': np.std(all_lankle_adab, axis=0)
            })
            mean_cycle_l.to_csv(csv_path_dir / f"normalized_cycle_L_mean_{csv_path.stem}.csv", index=False)
        
        # 平均サイクルをプロット（右足）
        if len(normalized_gait_cycles_r) > 0:
            # 屈曲伸展・背屈底屈のプロット
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            
            # 股関節
            axes[0].plot(normalized_percentage, mean_cycle_r['R_Hip_FlEx_mean'], 'b-', label='Mean')
            axes[0].fill_between(normalized_percentage,
                                mean_cycle_r['R_Hip_FlEx_mean'] - mean_cycle_r['R_Hip_FlEx_std'],
                                mean_cycle_r['R_Hip_FlEx_mean'] + mean_cycle_r['R_Hip_FlEx_std'],
                                alpha=0.3, color='b')
            axes[0].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_r]),
                          color='r', linestyle='--', label='Toe Off')
            axes[0].set_ylabel('Hip Angle [deg]')
            axes[0].set_title('Right Hip Flexion/Extension')
            axes[0].legend()
            axes[0].grid(True)
            
            # 膝関節
            axes[1].plot(normalized_percentage, mean_cycle_r['R_Knee_FlEx_mean'], 'b-', label='Mean')
            axes[1].fill_between(normalized_percentage,
                                mean_cycle_r['R_Knee_FlEx_mean'] - mean_cycle_r['R_Knee_FlEx_std'],
                                mean_cycle_r['R_Knee_FlEx_mean'] + mean_cycle_r['R_Knee_FlEx_std'],
                                alpha=0.3, color='b')
            axes[1].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_r]),
                          color='r', linestyle='--', label='Toe Off')
            axes[1].set_ylabel('Knee Angle [deg]')
            axes[1].set_title('Right Knee Flexion/Extension')
            axes[1].legend()
            axes[1].grid(True)
            
            # 足関節
            axes[2].plot(normalized_percentage, mean_cycle_r['R_Ankle_PlDo_mean'], 'b-', label='Mean')
            axes[2].fill_between(normalized_percentage,
                                mean_cycle_r['R_Ankle_PlDo_mean'] - mean_cycle_r['R_Ankle_PlDo_std'],
                                mean_cycle_r['R_Ankle_PlDo_mean'] + mean_cycle_r['R_Ankle_PlDo_std'],
                                alpha=0.3, color='b')
            axes[2].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_r]),
                          color='r', linestyle='--', label='Toe Off')
            axes[2].set_xlabel('Gait Cycle [%]')
            axes[2].set_ylabel('Ankle Angle [deg]')
            axes[2].set_title('Right Ankle Plantarflexion/Dorsiflexion')
            axes[2].legend()
            axes[2].grid(True)
            
            plt.tight_layout()
            plt.savefig(csv_path_dir / f"gait_cycle_FlEx_R_{csv_path.stem}.png")
            plt.close()
            
            # 内旋外旋のプロット
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            # 股関節
            axes[0].plot(normalized_percentage, mean_cycle_r['R_Hip_InEx_mean'], 'b-', label='Mean')
            axes[0].fill_between(normalized_percentage,
                                mean_cycle_r['R_Hip_InEx_mean'] - mean_cycle_r['R_Hip_InEx_std'],
                                mean_cycle_r['R_Hip_InEx_mean'] + mean_cycle_r['R_Hip_InEx_std'],
                                alpha=0.3, color='b')
            axes[0].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_r]),
                          color='r', linestyle='--', label='Toe Off')
            axes[0].set_ylabel('Hip Angle [deg]')
            axes[0].set_title('Right Hip Internal/External Rotation')
            axes[0].legend()
            axes[0].grid(True)
            # 膝関節
            axes[1].plot(normalized_percentage, mean_cycle_r['R_Knee_InEx_mean'], 'b-', label='Mean')
            axes[1].fill_between(normalized_percentage,
                                mean_cycle_r['R_Knee_InEx_mean'] - mean_cycle_r['R_Knee_InEx_std'],
                                mean_cycle_r['R_Knee_InEx_mean'] + mean_cycle_r['R_Knee_InEx_std'],
                                alpha=0.3, color='b')
            axes[1].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_r]),
                          color='r', linestyle='--', label='Toe Off')
            axes[1].set_ylabel('Knee Angle [deg]')
            axes[1].set_title('Right Knee Internal/External Rotation')
            axes[1].legend()
            axes[1].grid(True)
            # 足関節
            axes[2].plot(normalized_percentage, mean_cycle_r['R_Ankle_InEx_mean'], 'b-', label='Mean')
            axes[2].fill_between(normalized_percentage,
                                mean_cycle_r['R_Ankle_InEx_mean'] - mean_cycle_r['R_Ankle_InEx_std'],
                                mean_cycle_r['R_Ankle_InEx_mean'] + mean_cycle_r['R_Ankle_InEx_std'],
                                alpha=0.3, color='b')
            axes[2].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_r]),
                          color='r', linestyle='--', label='Toe Off')
            axes[2].set_ylabel('Ankle Angle [deg]')
            axes[2].set_title('Right Ankle Internal/External Rotation')
            axes[2].legend()
            axes[2].grid(True)
            plt.tight_layout()
            plt.savefig(csv_path_dir / f"gait_cycle_InEx_R_{csv_path.stem}.png")
            plt.close()

            # 内転外転のプロット
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            # 股関節
            axes[0].plot(normalized_percentage, mean_cycle_r['R_Hip_AdAb_mean'], 'b-', label='Mean')
            axes[0].fill_between(normalized_percentage,
                                mean_cycle_r['R_Hip_AdAb_mean'] - mean_cycle_r['R_Hip_AdAb_std'],
                                mean_cycle_r['R_Hip_AdAb_mean'] + mean_cycle_r['R_Hip_AdAb_std'],
                                alpha=0.3, color='b')
            axes[0].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_r]),
                          color='r', linestyle='--', label='Toe Off')
            axes[0].set_ylabel('Hip Angle [deg]')
            axes[0].set_title('Right Hip Adduction/Abduction')
            axes[0].legend()
            axes[0].grid(True)

            # 膝関節
            axes[1].plot(normalized_percentage, mean_cycle_r['R_Knee_AdAb_mean'], 'b-', label='Mean')
            axes[1].fill_between(normalized_percentage,
                                mean_cycle_r['R_Knee_AdAb_mean'] - mean_cycle_r['R_Knee_AdAb_std'],
                                mean_cycle_r['R_Knee_AdAb_mean'] + mean_cycle_r['R_Knee_AdAb_std'],
                                alpha=0.3, color='b')
            axes[1].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_r]),
                          color='r', linestyle='--', label='Toe Off')
            axes[1].set_ylabel('Knee Angle [deg]')
            axes[1].set_title('Right Knee Adduction/Abduction')
            axes[1].legend()
            axes[1].grid(True)

            # 足関節
            axes[2].plot(normalized_percentage, mean_cycle_r['R_Ankle_AdAb_mean'], 'b-', label='Mean')
            axes[2].fill_between(normalized_percentage,
                                mean_cycle_r['R_Ankle_AdAb_mean'] - mean_cycle_r['R_Ankle_AdAb_std'],
                                mean_cycle_r['R_Ankle_AdAb_mean'] + mean_cycle_r['R_Ankle_AdAb_std'],
                                alpha=0.3, color='b')
            axes[2].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_r]),
                          color='r', linestyle='--', label='Toe Off')
            axes[2].set_xlabel('Gait Cycle [%]')
            axes[2].set_ylabel('Ankle Angle [deg]')
            axes[2].set_title('Right Ankle Adduction/Abduction')
            axes[2].legend()
            axes[2].grid(True)

            plt.tight_layout()
            plt.savefig(csv_path_dir / f"gait_cycle_AdAb_R_{csv_path.stem}.png")
            plt.close()
            
        # 左足の平均サイクルをプロット
        if len(normalized_gait_cycles_l) > 0:
            # 屈曲伸展・背屈底屈のプロット
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            
            # 股関節
            axes[0].plot(normalized_percentage, mean_cycle_l['L_Hip_FlEx_mean'], 'b-', label='Mean')
            axes[0].fill_between(normalized_percentage,
                                mean_cycle_l['L_Hip_FlEx_mean'] - mean_cycle_l['L_Hip_FlEx_std'],
                                mean_cycle_l['L_Hip_FlEx_mean'] + mean_cycle_l['L_Hip_FlEx_std'],
                                alpha=0.3, color='b')
            axes[0].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_l]),
                          color='r', linestyle='--', label='Toe Off')
            axes[0].set_ylabel('Hip Angle [deg]')
            axes[0].set_title('Left Hip Flexion/Extension')
            axes[0].legend()
            axes[0].grid(True)
            
            # 膝関節
            axes[1].plot(normalized_percentage, mean_cycle_l['L_Knee_FlEx_mean'], 'b-', label='Mean')
            axes[1].fill_between(normalized_percentage,
                                mean_cycle_l['L_Knee_FlEx_mean'] - mean_cycle_l['L_Knee_FlEx_std'],
                                mean_cycle_l['L_Knee_FlEx_mean'] + mean_cycle_l['L_Knee_FlEx_std'],
                                alpha=0.3, color='b')
            axes[1].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_l]),
                          color='r', linestyle='--', label='Toe Off')
            axes[1].set_ylabel('Knee Angle [deg]')
            axes[1].set_title('Left Knee Flexion/Extension')
            axes[1].legend()
            axes[1].grid(True)
            
            # 足関節
            axes[2].plot(normalized_percentage, mean_cycle_l['L_Ankle_PlDo_mean'], 'b-', label='Mean')
            axes[2].fill_between(normalized_percentage,
                                mean_cycle_l['L_Ankle_PlDo_mean'] - mean_cycle_l['L_Ankle_PlDo_std'],
                                mean_cycle_l['L_Ankle_PlDo_mean'] + mean_cycle_l['L_Ankle_PlDo_std'],
                                alpha=0.3, color='b')
            axes[2].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_l]),
                          color='r', linestyle='--', label='Toe Off')
            axes[2].set_xlabel('Gait Cycle [%]')
            axes[2].set_ylabel('Ankle Angle [deg]')
            axes[2].set_title('Left Ankle Plantarflexion/Dorsiflexion')
            axes[2].legend()
            axes[2].grid(True)
            plt.tight_layout()
            plt.savefig(csv_path_dir / f"gait_cycle_FlEx_L_{csv_path.stem}.png")
            plt.close()
            # 内旋外旋のプロット
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            # 股関節
            axes[0].plot(normalized_percentage, mean_cycle_l['L_Hip_InEx_mean'], 'b-', label='Mean')
            axes[0].fill_between(normalized_percentage,
                                mean_cycle_l['L_Hip_InEx_mean'] - mean_cycle_l['L_Hip_InEx_std'],
                                mean_cycle_l['L_Hip_InEx_mean'] + mean_cycle_l['L_Hip_InEx_std'],
                                alpha=0.3, color='b')
            axes[0].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_l]),
                          color='r', linestyle='--', label='Toe Off')
            axes[0].set_ylabel('Hip Angle [deg]')
            axes[0].set_title('Left Hip Internal/External Rotation')
            axes[0].legend()
            axes[0].grid(True)

            # 膝関節
            axes[1].plot(normalized_percentage, mean_cycle_l['L_Knee_InEx_mean'], 'b-', label='Mean')
            axes[1].fill_between(normalized_percentage,
                                mean_cycle_l['L_Knee_InEx_mean'] - mean_cycle_l['L_Knee_InEx_std'],
                                mean_cycle_l['L_Knee_InEx_mean'] + mean_cycle_l['L_Knee_InEx_std'],
                                alpha=0.3, color='b')
            axes[1].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_l]),
                          color='r', linestyle='--', label='Toe Off')
            axes[1].set_ylabel('Knee Angle [deg]')
            axes[1].set_title('Left Knee Internal/External Rotation')
            axes[1].legend()
            axes[1].grid(True)

            # 足関節
            axes[2].plot(normalized_percentage, mean_cycle_l['L_Ankle_InEx_mean'], 'b-', label='Mean')
            axes[2].fill_between(normalized_percentage,
                                mean_cycle_l['L_Ankle_InEx_mean'] - mean_cycle_l['L_Ankle_InEx_std'],
                                mean_cycle_l['L_Ankle_InEx_mean'] + mean_cycle_l['L_Ankle_InEx_std'],
                                alpha=0.3, color='b')
            axes[2].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_l]),
                          color='r', linestyle='--', label='Toe Off')
            axes[2].set_xlabel('Gait Cycle [%]')
            axes[2].set_ylabel('Ankle Angle [deg]')
            axes[2].set_title('Left Ankle Internal/External Rotation')
            axes[2].legend()
            axes[2].grid(True)
            plt.tight_layout()
            plt.savefig(csv_path_dir / f"gait_cycle_InEx_L_{csv_path.stem}.png")
            plt.close()
            # 内転外転のプロット
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            # 股関節
            axes[0].plot(normalized_percentage, mean_cycle_l['L_Hip_AdAb_mean'], 'b-', label='Mean')
            axes[0].fill_between(normalized_percentage,
                                mean_cycle_l['L_Hip_AdAb_mean'] - mean_cycle_l['L_Hip_AdAb_std'],
                                mean_cycle_l['L_Hip_AdAb_mean'] + mean_cycle_l['L_Hip_AdAb_std'],
                                alpha=0.3, color='b')
            axes[0].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_l]),
                          color='r', linestyle='--', label='Toe Off')
            axes[0].set_ylabel('Hip Angle [deg]')
            axes[0].set_title('Left Hip Adduction/Abduction')
            axes[0].legend()
            axes[0].grid(True)

            # 膝関節
            axes[1].plot(normalized_percentage, mean_cycle_l['L_Knee_AdAb_mean'], 'b-', label='Mean')
            axes[1].fill_between(normalized_percentage,
                                mean_cycle_l['L_Knee_AdAb_mean'] - mean_cycle_l['L_Knee_AdAb_std'],
                                mean_cycle_l['L_Knee_AdAb_mean'] + mean_cycle_l['L_Knee_AdAb_std'],
                                alpha=0.3, color='b')
            axes[1].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_l]),
                          color='r', linestyle='--', label='Toe Off')
            axes[1].set_ylabel('Knee Angle [deg]')
            axes[1].set_title('Left Knee Adduction/Abduction')
            axes[1].legend()
            axes[1].grid(True)

            # 足関節
            axes[2].plot(normalized_percentage, mean_cycle_l['L_Ankle_AdAb_mean'], 'b-', label='Mean')
            axes[2].fill_between(normalized_percentage,
                                mean_cycle_l['L_Ankle_AdAb_mean'] - mean_cycle_l['L_Ankle_AdAb_std'],
                                mean_cycle_l['L_Ankle_AdAb_mean'] + mean_cycle_l['L_Ankle_AdAb_std'],
                                alpha=0.3, color='b')
            axes[2].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_l]),
                          color='r', linestyle='--', label='Toe Off')
            axes[2].set_xlabel('Gait Cycle [%]')
            axes[2].set_ylabel('Ankle Angle [deg]')
            axes[2].set_title('Left Ankle Adduction/Abduction')
            axes[2].legend()
            axes[2].grid(True)
            plt.tight_layout()
            plt.savefig(csv_path_dir / f"gait_cycle_AdAb_L_{csv_path.stem}.png")
            plt.close()
            
            
        ###########################################
        # 歩行パラメータの計算（Mocap）
        ###########################################
        def calculate_gait_parameters(gait_cycles, hip_array, rhee, lhee, sampling_freq=100):
            """
            歩行パラメータを計算
            
            Parameters:
            -----------
            gait_cycles : list
                [[ic_start, to, ic_end], ...] の形式
            hip_array : np.ndarray
                骨盤中心の3D座標 (n_frames, 3)
            rhee : np.ndarray
                右踵の3D座標 (n_frames, 3)
            lhee : np.ndarray
                左踵の3D座標 (n_frames, 3)
            sampling_freq : int
                サンプリング周波数 [Hz]
                
            Returns:
            --------
            gait_params : list of dict
                各歩行周期のパラメータ
            """
            gait_params = []
            
            for cycle_idx, (ic_start, to, ic_end) in enumerate(gait_cycles):
                # 歩行周期時間 [s]
                cycle_duration = (ic_end - ic_start) / sampling_freq
                
                # 立脚期時間 [s]
                stance_duration = (to - ic_start) / sampling_freq
                
                # 遊脚期時間 [s]
                swing_duration = (ic_end - to) / sampling_freq
                
                # 歩行速度 [m/s]
                # 骨盤の移動距離を時間で割る
                hip_displacement = np.linalg.norm(hip_array[ic_end] - hip_array[ic_start])
                gait_speed = hip_displacement / cycle_duration
                
                # ストライド長 [m]
                stride_length = hip_displacement
                
                # 歩隔 [m]
                # 立脚期中の左右踵の平均距離
                step_width_samples = []
                for frame in range(ic_start, to + 1):
                    # 左右踵のY座標の差（左右方向）
                    lateral_distance = abs(rhee[frame, 1] - lhee[frame, 1])
                    step_width_samples.append(lateral_distance)
                step_width = np.mean(step_width_samples)
                
                # ケイデンス [steps/min]
                cadence = 60.0 / cycle_duration
                
                params = {
                    'cycle_index': cycle_idx,
                    'cycle_duration': cycle_duration,
                    'stance_duration': stance_duration,
                    'swing_duration': swing_duration,
                    'stance_phase_percent': (stance_duration / cycle_duration) * 100,
                    'swing_phase_percent': (swing_duration / cycle_duration) * 100,
                    'gait_speed': gait_speed,
                    'stride_length': stride_length,
                    'step_width': step_width,
                    'cadence': cadence
                }
                gait_params.append(params)
            
            return gait_params
        
        # 右足と左足の歩行パラメータを計算
        gait_params_r = calculate_gait_parameters(gait_cycles_r, hip_array, rhee, lhee, sampling_freq=100)
        gait_params_l = calculate_gait_parameters(gait_cycles_l, hip_array, rhee, lhee, sampling_freq=100)
        
        # シンメトリインデックスを計算
        def calculate_symmetry_index(params_r, params_l):
            """
            シンメトリインデックスを計算
            SI = 100 * (|R - L| / (0.5 * (R + L)))
            
            Parameters:
            -----------
            params_r : list of dict
                右足の歩行パラメータ
            params_l : list of dict
                左足の歩行パラメータ
                
            Returns:
            --------
            symmetry_indices : dict
                各パラメータのシンメトリインデックス
            """
            # 各パラメータの平均を計算
            keys = ['cycle_duration', 'stance_duration', 'swing_duration', 'gait_speed', 
                    'stride_length', 'step_width', 'cadence', 'stance_phase_percent', 'swing_phase_percent']
            
            symmetry_indices = {}
            
            for key in keys:
                r_values = [p[key] for p in params_r]
                l_values = [p[key] for p in params_l]
                
                r_mean = np.mean(r_values)
                l_mean = np.mean(l_values)
                
                # シンメトリインデックス
                if (r_mean + l_mean) != 0:
                    si = 100 * abs(r_mean - l_mean) / (0.5 * (r_mean + l_mean))
                else:
                    si = 0
                
                symmetry_indices[key] = {
                    'right_mean': r_mean,
                    'left_mean': l_mean,
                    'symmetry_index': si
                }
            
            return symmetry_indices
        
        # シンメトリインデックスを計算
        symmetry_indices = calculate_symmetry_index(gait_params_r, gait_params_l)
        
        # 歩行パラメータをCSVに保存
        gait_params_r_df = pd.DataFrame(gait_params_r)
        gait_params_l_df = pd.DataFrame(gait_params_l)
        
        gait_params_r_df.to_csv(csv_path_dir / f"gait_parameters_R_{csv_path.stem}.csv", index=False)
        gait_params_l_df.to_csv(csv_path_dir / f"gait_parameters_L_{csv_path.stem}.csv", index=False)
        
        # シンメトリインデックスをCSVに保存
        si_data = []
        for key, values in symmetry_indices.items():
            si_data.append({
                'Parameter': key,
                'Right_Mean': values['right_mean'],
                'Left_Mean': values['left_mean'],
                'Symmetry_Index': values['symmetry_index']
            })
        si_df = pd.DataFrame(si_data)
        si_df.to_csv(csv_path_dir / f"symmetry_indices_{csv_path.stem}.csv", index=False)
        
        print(f"\n歩行パラメータ（Mocap）:")
        print(f"右足: 平均歩行速度 = {np.mean([p['gait_speed'] for p in gait_params_r]):.3f} m/s")
        print(f"左足: 平均歩行速度 = {np.mean([p['gait_speed'] for p in gait_params_l]):.3f} m/s")
        print(f"右足: 平均歩隔 = {np.mean([p['step_width'] for p in gait_params_r]):.3f} m")
        print(f"左足: 平均歩隔 = {np.mean([p['step_width'] for p in gait_params_l]):.3f} m")
        print(f"\nシンメトリインデックス:")
        print(si_df)
        
        
            
        # #####################################
        # #####################################
        # OpenPose3D結果との比較
        # #####################################
        # #####################################
        # 結果保存ディレクトリ作成
        op_result_dir = csv_path_dir.parent / "OpenPose3D_results"
        op_result_dir.mkdir(parents=True, exist_ok=True)

        openpose_npz_path = csv_path_dir.parent / "3d_kp_data_openpose_kalman.npz"
        if not openpose_npz_path.exists():
            print(f"OpenPose3Dデータが見つかりません: {openpose_npz_path}")
            return
    
        print(f"OpenPose3Dデータとの比較を実行: {openpose_npz_path}")
        openpose_data = np.load(openpose_npz_path)
        print(f"openpose_data keys: {openpose_data.files}")
        op_frame = openpose_data['frame']
        op_raw_data = openpose_data['raw']  # shape: (num_frames, num_joints, 3)
        op_filt_data = openpose_data['filt']  # shape: (num_frames, num_joints, 3)
        op_conf = openpose_data['conf']  # shape: (num_frames, num_joints)
        print(f"op_frame: {op_frame[:5]} ...{op_frame[-5:]}")
        print(f"op_raw_data shape: {op_raw_data.shape}")
        
        ###########################################
        # タイミング合わせ  股関節中心が任意の位置を通るタイミングでフレーム調整
        ###########################################
        base_point = int(0)
        # opti用
        hip_z_opti = (((rasi + lasi) / 2 + (rpsi + lpsi) / 2) / 2)[:, 2]
        # hip_z_optiが0より大きくなった最初のフレームを取得
        base_passing_frame = np.argmax(hip_z_opti > base_point) + start_frame
        print(f"100Hz Opti計測開始から原点通過までのフレーム数: {base_passing_frame}")
        
        
        # OpenPose用
        hip_z_op = op_filt_data[:, 8, 2]
        # hip_zが0より大きくなった最初のフレームを取得
        base_passing_idx_op = np.argmax(hip_z_op > base_point)
        base_passing_frame_op = op_frame[base_passing_idx_op]
        print(f"60Hz OpenPoseLED発光検出から原点通過までのフレーム数: {base_passing_frame_op}")

        # フレーム数を揃える
        if csv_path.stem == "1_0_3":
            frame_offset_cut = 394 + 100 #LED発光フレームと動画トリミング開始フレームの差 本当は自動で計算可能
            print(f"フレームオフセット調整: {frame_offset_cut}")

        frame_offset_60Hz = base_passing_frame_op + frame_offset_cut - base_passing_frame * 0.6  # 100Hzは60Hzに変換
        print(f"LED発光からMC開始までのフレームオフセット 60Hz: {frame_offset_60Hz}")
        # 100Hz基準のオフセット(MC開始からGoProトリミング開始までの100Hzフレーム数)
        mc_frame_offset = (frame_offset_cut - frame_offset_60Hz ) / 0.6 
        print(f"MC開始からGoProトリミング開始までのフレームオフセット 100Hz: {mc_frame_offset}")
        
        gait_cycles_r_op = []
        gait_cycles_l_op = []
        
        for ic_rel, to_rel, ic_next_rel in gait_cycles_r_abs:
            ic_abs = (ic_rel - mc_frame_offset) * 0.6
            to_abs = (to_rel - mc_frame_offset) * 0.6
            ic_next_abs = (ic_next_rel - mc_frame_offset) * 0.6
            gait_cycles_r_op.append([ic_abs, to_abs, ic_next_abs])
        for ic_rel, to_rel, ic_next_rel in gait_cycles_l_abs:
            ic_abs = (ic_rel - mc_frame_offset) * 0.6
            to_abs = (to_rel - mc_frame_offset) * 0.6
            ic_next_abs = (ic_next_rel - mc_frame_offset) * 0.6
            gait_cycles_l_op.append([ic_abs, to_abs, ic_next_abs])
            
        print(f"右足歩行サイクル(OpenPose): {gait_cycles_r_op}")
        print(f"左足歩行サイクル(OpenPose): {gait_cycles_l_op}")

        ###########################################
        # 関節角度比較
        ###########################################
        #OpenPoseの各関節点抽出 3次元
        neck_op, midhip_op  = op_filt_data[:, 1, :], op_filt_data[:, 8, :]
        rhip_op, rknee_op, rankle_op, rhee_op  = op_filt_data[:, 9, :], op_filt_data[:, 10, :], op_filt_data[:, 11, :], op_filt_data[:, 24, :]
        rtoe_op = (op_filt_data[:, 22, :] + op_filt_data[:, 23, :]) / 2
        lhip_op, lknee_op, lankle_op, lhee_op  = op_filt_data[:, 12, :], op_filt_data[:, 13, :], op_filt_data[:, 14, :], op_filt_data[:, 21, :]
        ltoe_op = (op_filt_data[:, 19, :] + op_filt_data[:, 20, :]) / 2
        
        # 使用するベクトルを定義
        pel_bec_op = neck_op - midhip_op  #上向き
        thigh_r_op = rknee_op - rhip_op  #下向き
        shank_r_op = rankle_op - rknee_op  #下向き 
        foot_r_op = rtoe_op - rhee_op  #前向き
        thigh_l_op = lknee_op - lhip_op  
        shank_l_op = lankle_op - lknee_op
        foot_l_op = ltoe_op - lhee_op

        # 関節角度計算
        # 股関節、膝関節、足関節の屈曲伸展角度
        n_axis = rhip_op - lhip_op  # 右股関節から左股関節へのベクトル
        rhip_flex_op = op.culc_angle_all_frames(pel_bec_op, thigh_r_op, n_axis, degrees=True, angle_type='hip')
        lhip_flex_op = op.culc_angle_all_frames(pel_bec_op, thigh_l_op, n_axis, degrees=True, angle_type='hip')
        rknee_flex_op = op.culc_angle_all_frames(thigh_r_op, shank_r_op, n_axis, degrees=True, angle_type='knee')
        lknee_flex_op = op.culc_angle_all_frames(thigh_l_op, shank_l_op, n_axis, degrees=True, angle_type='knee')
        rankle_pldo_op = op.culc_angle_all_frames(shank_r_op, foot_r_op, n_axis, degrees=True, angle_type='ankle')
        lankle_pldo_op = op.culc_angle_all_frames(shank_l_op, foot_l_op, n_axis, degrees=True, angle_type='ankle')
        # 股関節の内転外転
        n_axis_adab = np.cross(pel_bec_op, n_axis)  # pel_vecとn_axisの外積を計算して直交ベクトルを取得 進行方向向き
        rhip_adab_op = op.culc_angle_all_frames(pel_bec_op, thigh_r_op, n_axis_adab, degrees=True, angle_type='hip_adab')
        lhip_adab_op = op.culc_angle_all_frames(pel_bec_op, thigh_l_op, n_axis_adab, degrees=True, angle_type='hip_adab')
        # 股関節の内旋外旋
        n_axis_inex = pel_bec_op  # pel_vecを内旋外旋の回転軸とする
        rhip_inex_op = op.culc_angle_all_frames(thigh_r_op, n_axis_inex, n_axis_inex, degrees=True, angle_type='hip_inex')
        lhip_inex_op = op.culc_angle_all_frames(thigh_l_op, n_axis_inex, n_axis_inex, degrees=True, angle_type='hip_inex')


        # 右足の歩行周期ごとに関節角度を100%に正規化
        normalized_gait_cycles_r_op = []
        for cycle_idx, (ic_start, to, ic_end) in enumerate(gait_cycles_r_op):
            cycle_length = ic_end - ic_start
            # 0%から100%まで101点（0, 1, 2, ..., 100）にリサンプリング
            normalized_percentage = np.linspace(0, 100, 101)
            
            # 元のフレーム番号（相対）を整数インデックスに変換
            ic_start_idx = int(np.round(ic_start))
            ic_end_idx = int(np.round(ic_end))
            original_frames = np.arange(ic_start_idx, ic_end_idx + 1)
            
            # 各関節角度を補間
            # 屈曲伸展・背屈底屈
            rhip_flex_normalized_op = np.interp(normalized_percentage, 
                                            np.linspace(0, 100, len(original_frames)),
                                            rhip_flex_op[original_frames])
            rknee_flex_normalized_op = np.interp(normalized_percentage,
                                              np.linspace(0, 100, len(original_frames)),
                                              rknee_flex_op[original_frames])
            rankle_pldo_normalized_op = np.interp(normalized_percentage,
                                               np.linspace(0, 100, len(original_frames)),
                                               rankle_pldo_op[original_frames])
            # 内旋外旋
            rhip_inex_normalized_op = np.interp(normalized_percentage,
                                             np.linspace(0, 100, len(original_frames)),
                                                rhip_inex_op[original_frames])
            # 内転外転
            rhip_adab_normalized_op = np.interp(normalized_percentage,
                                             np.linspace(0, 100, len(original_frames)),
                                                rhip_adab_op[original_frames])

            # 立脚期の割合を計算
            stance_phase_percentage = ((to - ic_start) / cycle_length) * 100
            
            cycle_data = {
                'cycle_index': cycle_idx,
                'ic_start': ic_start,
                'to': to,
                'ic_end': ic_end,
                'cycle_length_frames': cycle_length,
                'stance_phase_percentage': stance_phase_percentage,
                'percentage': normalized_percentage,
                'R_Hip_FlEx': rhip_flex_normalized_op,
                'R_Knee_FlEx': rknee_flex_normalized_op,
                'R_Ankle_PlDo': rankle_pldo_normalized_op,
                'R_Hip_InEx': rhip_inex_normalized_op,
                'R_Hip_AdAb': rhip_adab_normalized_op
            }
            normalized_gait_cycles_r_op.append(cycle_data)
        
        # 左足の歩行周期ごとに関節角度を100%に正規化
        normalized_gait_cycles_l_op = []
        
        for cycle_idx, (ic_start, to, ic_end) in enumerate(gait_cycles_l_op):
            cycle_length = ic_end - ic_start
            # 0%から100%まで101点（0, 1, 2, ..., 100）にリサンプリング
            normalized_percentage = np.linspace(0, 100, 101)
            
            # 元のフレーム番号（相対）
            ic_start_idx = int(np.round(ic_start))
            ic_end_idx = int(np.round(ic_end))
            original_frames = np.arange(ic_start_idx, ic_end_idx + 1)

            # 各関節角度を補間
            # 屈曲伸展・背屈底屈
            lhip_flex_normalized_op = np.interp(normalized_percentage, 
                                            np.linspace(0, 100, len(original_frames)),
                                            lhip_flex_op[original_frames])
            lknee_flex_normalized_op = np.interp(normalized_percentage,
                                              np.linspace(0, 100, len(original_frames)),
                                              lknee_flex_op[original_frames])
            lankle_pldo_normalized_op = np.interp(normalized_percentage,
                                               np.linspace(0, 100, len(original_frames)),
                                               lankle_pldo_op[original_frames])
            # 内旋外旋
            lhip_inex_normalized_op = np.interp(normalized_percentage,
                                             np.linspace(0, 100, len(original_frames)),
                                                lhip_inex_op[original_frames])
            # 内転外転
            lhip_adab_normalized_op = np.interp(normalized_percentage,
                                             np.linspace(0, 100, len(original_frames)),
                                                lhip_adab_op[original_frames])

            # 立脚期の割合を計算
            stance_phase_percentage = ((to - ic_start) / cycle_length) * 100
            
            cycle_data = {
                'cycle_index': cycle_idx,
                'ic_start': ic_start,
                'to': to,
                'ic_end': ic_end,
                'cycle_length_frames': cycle_length,
                'stance_phase_percentage': stance_phase_percentage,
                'percentage': normalized_percentage,
                'L_Hip_FlEx': lhip_flex_normalized_op,
                'L_Knee_FlEx': lknee_flex_normalized_op,
                'L_Ankle_PlDo': lankle_pldo_normalized_op,
                'L_Hip_InEx': lhip_inex_normalized_op,
                'L_Hip_AdAb': lhip_adab_normalized_op
            }
            normalized_gait_cycles_l_op.append(cycle_data)
                
        ###########################################
        # 歩行パラメータの計算（OpenPose）
        ###########################################
        def calculate_gait_parameters_op(gait_cycles, midhip_op, rhee_op, lhee_op, sampling_freq=60):
            """
            OpenPoseデータから歩行パラメータを計算
            """
            gait_params = []
            
            for cycle_idx, (ic_start, to, ic_end) in enumerate(gait_cycles):
                # 整数インデックスに変換
                ic_start_idx = int(np.round(ic_start))
                to_idx = int(np.round(to))
                ic_end_idx = int(np.round(ic_end))
                
                # 歩行周期時間 [s]
                cycle_duration = (ic_end_idx - ic_start_idx) / sampling_freq
                
                # 立脚期時間 [s]
                stance_duration = (to_idx - ic_start_idx) / sampling_freq
                
                # 遊脚期時間 [s]
                swing_duration = (ic_end_idx - to_idx) / sampling_freq
                
                # 歩行速度 [m/s]
                hip_displacement = np.linalg.norm(midhip_op[ic_end_idx] - midhip_op[ic_start_idx]) / 1000 # mmからmに変換
                gait_speed = hip_displacement / cycle_duration
                
                # ストライド長 [m]
                stride_length = hip_displacement
                
                # 歩隔 [m]
                step_width_samples = []
                for frame in range(ic_start_idx, to_idx + 1):
                    lateral_distance = abs(rhee_op[frame, 1] - lhee_op[frame, 1]) / 1000  # mmからmに変換
                    step_width_samples.append(lateral_distance)
                step_width = np.mean(step_width_samples)
                
                # ケイデンス [steps/min]
                cadence = 60.0 / cycle_duration
                
                params = {
                    'cycle_index': cycle_idx,
                    'cycle_duration': cycle_duration,
                    'stance_duration': stance_duration,
                    'swing_duration': swing_duration,
                    'stance_phase_percent': (stance_duration / cycle_duration) * 100,
                    'swing_phase_percent': (swing_duration / cycle_duration) * 100,
                    'gait_speed': gait_speed,
                    'stride_length': stride_length,
                    'step_width': step_width,
                    'cadence': cadence
                }
                gait_params.append(params)
            
            return gait_params
        
        # OpenPoseの歩行パラメータを計算
        gait_params_r_op = calculate_gait_parameters_op(gait_cycles_r_op, midhip_op, rhee_op, lhee_op, sampling_freq=60)
        gait_params_l_op = calculate_gait_parameters_op(gait_cycles_l_op, midhip_op, rhee_op, lhee_op, sampling_freq=60)
        
        # OpenPoseのシンメトリインデックスを計算
        symmetry_indices_op = calculate_symmetry_index(gait_params_r_op, gait_params_l_op)
        
        # OpenPoseの歩行パラメータをCSVに保存
        gait_params_r_op_df = pd.DataFrame(gait_params_r_op)
        gait_params_l_op_df = pd.DataFrame(gait_params_l_op)
        
        gait_params_r_op_df.to_csv(op_result_dir / f"gait_parameters_R_{csv_path.stem}_OpenPose.csv", index=False)
        gait_params_l_op_df.to_csv(op_result_dir / f"gait_parameters_L_{csv_path.stem}_OpenPose.csv", index=False)
        
        # OpenPoseのシンメトリインデックスをCSVに保存
        si_data_op = []
        for key, values in symmetry_indices_op.items():
            si_data_op.append({
                'Parameter': key,
                'Right_Mean': values['right_mean'],
                'Left_Mean': values['left_mean'],
                'Symmetry_Index': values['symmetry_index']
            })
        si_df_op = pd.DataFrame(si_data_op)
        si_df_op.to_csv(op_result_dir / f"symmetry_indices_{csv_path.stem}_OpenPose.csv", index=False)
        
        print(f"\n歩行パラメータ（OpenPose）:")
        print(f"右足: 平均歩行速度 = {np.mean([p['gait_speed'] for p in gait_params_r_op]):.3f} m/s")
        print(f"左足: 平均歩行速度 = {np.mean([p['gait_speed'] for p in gait_params_l_op]):.3f} m/s")
        print(f"右足: 平均歩隔 = {np.mean([p['step_width'] for p in gait_params_r_op]):.3f} m")
        print(f"左足: 平均歩隔 = {np.mean([p['step_width'] for p in gait_params_l_op]):.3f} m")
        print(f"\nシンメトリインデックス（OpenPose）:")
        print(si_df_op)
        
        ###########################################
        # MocapとOpenPoseの歩行パラメータ比較（平均と標準偏差）
        ###########################################
        # 比較する主要パラメータ
        if len(gait_params_r) > 0 and len(gait_params_r_op) > 0:
            params_to_compare = [
                ('gait_speed', 'Gait Speed [m/s]'),
                ('step_width', 'Step Width [m]'),
                ('stance_phase_percent', 'Stance Phase [%]')
            ]
            
            # 平均と標準偏差を計算
            comparison_stats = []
            for param_key, param_label in params_to_compare:
                # Mocap右足
                r_values_mocap = [p[param_key] for p in gait_params_r]
                r_mean_mocap = np.mean(r_values_mocap)
                r_std_mocap = np.std(r_values_mocap)
                
                # Mocap左足
                l_values_mocap = [p[param_key] for p in gait_params_l]
                l_mean_mocap = np.mean(l_values_mocap)
                l_std_mocap = np.std(l_values_mocap)
                
                # OpenPose右足
                r_values_op = [p[param_key] for p in gait_params_r_op]
                r_mean_op = np.mean(r_values_op)
                r_std_op = np.std(r_values_op)
                
                # OpenPose左足
                l_values_op = [p[param_key] for p in gait_params_l_op]
                l_mean_op = np.mean(l_values_op)
                l_std_op = np.std(l_values_op)
                
                # RMSE計算（平均値間の差）
                rmse_r = abs(r_mean_mocap - r_mean_op)
                rmse_l = abs(l_mean_mocap - l_mean_op)
                
                comparison_stats.append({
                    'Parameter': param_label,
                    'Mocap_R_Mean': r_mean_mocap,
                    'Mocap_R_Std': r_std_mocap,
                    'Mocap_L_Mean': l_mean_mocap,
                    'Mocap_L_Std': l_std_mocap,
                    'OpenPose_R_Mean': r_mean_op,
                    'OpenPose_R_Std': r_std_op,
                    'OpenPose_L_Mean': l_mean_op,
                    'OpenPose_L_Std': l_std_op,
                    'Diff_R': rmse_r,
                    'Diff_L': rmse_l
                })
            
            # データフレームに変換
            comparison_stats_df = pd.DataFrame(comparison_stats)
            comparison_stats_df.to_csv(op_result_dir / f"gait_parameters_mean_std_comparison_{csv_path.stem}.csv", index=False)
            
            print(f"\n歩行パラメータ比較（平均±標準偏差）:")
            print(comparison_stats_df)
            
            # 棒グラフで比較（平均±標準偏差）
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for idx, (param_key, param_label) in enumerate(params_to_compare):
                ax = axes[idx]
                
                # データ取得
                stats = comparison_stats[idx]
                
                # X軸の位置
                x = np.arange(2)  # 右足、左足
                width = 0.35
                
                # Mocapの平均と標準偏差
                mocap_means = [stats['Mocap_R_Mean'], stats['Mocap_L_Mean']]
                mocap_stds = [stats['Mocap_R_Std'], stats['Mocap_L_Std']]
                
                # OpenPoseの平均と標準偏差
                op_means = [stats['OpenPose_R_Mean'], stats['OpenPose_L_Mean']]
                op_stds = [stats['OpenPose_R_Std'], stats['OpenPose_L_Std']]
                
                # 棒グラフとエラーバー
                bars1 = ax.bar(x - width/2, mocap_means, width, yerr=mocap_stds, 
                              label='Mocap', color='blue', alpha=0.7, capsize=5)
                bars2 = ax.bar(x + width/2, op_means, width, yerr=op_stds, 
                              label='OpenPose', color='red', alpha=0.7, capsize=5)
                
                # グラフの装飾
                ax.set_xlabel('Foot Side', fontsize=12)
                ax.set_ylabel(param_label, fontsize=12)
                ax.set_title(f'{param_label}\n(Diff R: {stats["Diff_R"]:.3f}, Diff L: {stats["Diff_L"]:.3f})', 
                           fontsize=13)
                ax.set_xticks(x)
                ax.set_xticklabels(['Right', 'Left'])
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3, axis='y')
                
                # 値をバーの上に表示
                for bar in bars1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)
                for bar in bars2:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"comparison_gait_parameters_mean_std_{csv_path.stem}.png", dpi=300)
            plt.close()
            
            print(f"\n平均と標準偏差の比較グラフを保存しました: comparison_gait_parameters_mean_std_{csv_path.stem}.png")
            
            ###########################################
            # シンメトリインデックスの比較グラフ（立脚期のみ）
            ###########################################
            # 立脚期のシンメトリインデックスのデータを取得
            si_stance_mocap = symmetry_indices['stance_phase_percent']['symmetry_index']
            si_stance_op = symmetry_indices_op['stance_phase_percent']['symmetry_index']
            
            si_comparison = {
                'Parameter': 'Stance Phase [%]',
                'SI_Mocap': si_stance_mocap,
                'SI_OpenPose': si_stance_op,
                'Difference': abs(si_stance_mocap - si_stance_op)
            }
            
            # データフレームに変換
            si_comparison_df = pd.DataFrame([si_comparison])
            si_comparison_df.to_csv(op_result_dir / f"symmetry_index_comparison_{csv_path.stem}.csv", index=False)
            
            print(f"\nシンメトリインデックス比較（立脚期）:")
            print(si_comparison_df)
            
            # シンメトリインデックスの棒グラフ（立脚期のみ）
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # X軸の位置
            x = np.array([0])
            width = 0.35
            
            # データ取得
            si_mocap_values = [si_comparison['SI_Mocap']]
            si_op_values = [si_comparison['SI_OpenPose']]
            
            # 棒グラフ
            bars1 = ax.bar(x - width/2, si_mocap_values, width, 
                          label='Mocap', color='blue', alpha=0.7)
            bars2 = ax.bar(x + width/2, si_op_values, width, 
                          label='OpenPose', color='red', alpha=0.7)
            
            # グラフの装飾
            ax.set_ylabel('Symmetry Index [%]', fontsize=14)
            ax.set_title('Stance Phase Symmetry Index Comparison\n(Lower values indicate better symmetry)', fontsize=15)
            ax.set_xticks(x)
            ax.set_xticklabels(['Stance Phase'], fontsize=13)
            ax.legend(fontsize=13, loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xlim(-0.5, 0.5)
            
            # 値をバーの上に表示
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # 参考線（完全な対称性）
            ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect Symmetry')
            
            # 差分を表示
            ax.text(0, max(si_mocap_values[0], si_op_values[0]) * 1.15, 
                   f'Difference: {si_comparison["Difference"]:.1f}%',
                   ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"symmetry_index_comparison_stance_{csv_path.stem}.png", dpi=300)
            plt.close()
            
            print(f"\n立脚期シンメトリインデックス比較グラフを保存しました: symmetry_index_comparison_stance_{csv_path.stem}.png")
            
            
            
        
        # 右足平均サイクル計算
        if len(normalized_gait_cycles_r_op) > 0:
            all_rhip = np.array([cycle['R_Hip_FlEx'] for cycle in normalized_gait_cycles_r_op])
            all_rknee = np.array([cycle['R_Knee_FlEx'] for cycle in normalized_gait_cycles_r_op])
            all_rankle = np.array([cycle['R_Ankle_PlDo'] for cycle in normalized_gait_cycles_r_op])
            all_rhip_inex = np.array([cycle['R_Hip_InEx'] for cycle in normalized_gait_cycles_r_op])
            all_rhip_adab = np.array([cycle['R_Hip_AdAb'] for cycle in normalized_gait_cycles_r_op])

            mean_cycle_r_op = pd.DataFrame({
                'Percentage': normalized_percentage,
                'R_Hip_FlEx_mean': np.mean(all_rhip, axis=0),
                'R_Hip_FlEx_std': np.std(all_rhip, axis=0),
                'R_Knee_FlEx_mean': np.mean(all_rknee, axis=0),
                'R_Knee_FlEx_std': np.std(all_rknee, axis=0),
                'R_Ankle_PlDo_mean': np.mean(all_rankle, axis=0),
                'R_Ankle_PlDo_std': np.std(all_rankle, axis=0),
                'R_Hip_InEx_mean': np.mean(all_rhip_inex, axis=0),
                'R_Hip_InEx_std': np.std(all_rhip_inex, axis=0),
                'R_Hip_AdAb_mean': np.mean(all_rhip_adab, axis=0),
                'R_Hip_AdAb_std': np.std(all_rhip_adab, axis=0)
            })
            mean_cycle_r_op.to_csv(op_result_dir / f"normalized_cycle_R_mean_{csv_path.stem}_OpenPose.csv", index=False)
        # 左足平均サイクル計算
        if len(normalized_gait_cycles_l_op) > 0:
            all_lhip = np.array([cycle['L_Hip_FlEx'] for cycle in normalized_gait_cycles_l_op])
            all_lknee = np.array([cycle['L_Knee_FlEx'] for cycle in normalized_gait_cycles_l_op])
            all_lankle = np.array([cycle['L_Ankle_PlDo'] for cycle in normalized_gait_cycles_l_op])
            all_lhip_inex = np.array([cycle['L_Hip_InEx'] for cycle in normalized_gait_cycles_l_op])
            all_lhip_adab = np.array([cycle['L_Hip_AdAb'] for cycle in normalized_gait_cycles_l_op])

            mean_cycle_l_op = pd.DataFrame({
                'Percentage': normalized_percentage,
                'L_Hip_FlEx_mean': np.mean(all_lhip, axis=0),
                'L_Hip_FlEx_std': np.std(all_lhip, axis=0),
                'L_Knee_FlEx_mean': np.mean(all_lknee, axis=0),
                'L_Knee_FlEx_std': np.std(all_lknee, axis=0),
                'L_Ankle_PlDo_mean': np.mean(all_lankle, axis=0),
                'L_Ankle_PlDo_std': np.std(all_lankle, axis=0),
                'L_Hip_InEx_mean': np.mean(all_lhip_inex, axis=0),
                'L_Hip_InEx_std': np.std(all_lhip_inex, axis=0),
                'L_Hip_AdAb_mean': np.mean(all_lhip_adab, axis=0),
                'L_Hip_AdAb_std': np.std(all_lhip_adab, axis=0)
            })
            mean_cycle_l_op.to_csv(op_result_dir / f"normalized_cycle_L_mean_{csv_path.stem}_OpenPose.csv", index=False)
            
        # 関節角度をMCとOpenPoseで比較プロット
        # RMSEを計算する関数
        def calculate_rmse(data1, data2):
            """RMSEを計算"""
            return np.sqrt(np.mean((data1 - data2)**2))
        
        
        
        
        # 右足の比較プロット
        print(f"len normalized_gait_cycles_r: {len(normalized_gait_cycles_r)}, len normalized_gait_cycles_r_op: {len(normalized_gait_cycles_r_op)}")
        if len(normalized_gait_cycles_r) > 0 and len(normalized_gait_cycles_r_op) > 0:
            print("右足関節角度の比較プロットを作成中...")
            # 屈曲伸展・背屈底屈の比較
            fig, axes = plt.subplots(3, 1, figsize=(12, 14))
            
            # 股関節屈曲伸展
            axes[0].plot(normalized_percentage, mean_cycle_r['R_Hip_FlEx_mean'], 'b-', label='Mocap', linewidth=2)
            axes[0].fill_between(normalized_percentage,
                                mean_cycle_r['R_Hip_FlEx_mean'] - mean_cycle_r['R_Hip_FlEx_std'],
                                mean_cycle_r['R_Hip_FlEx_mean'] + mean_cycle_r['R_Hip_FlEx_std'],
                                alpha=0.2, color='b')
            axes[0].plot(normalized_percentage, mean_cycle_r_op['R_Hip_FlEx_mean'], 'r--', label='OpenPose', linewidth=2)
            axes[0].fill_between(normalized_percentage,
                                mean_cycle_r_op['R_Hip_FlEx_mean'] - mean_cycle_r_op['R_Hip_FlEx_std'],
                                mean_cycle_r_op['R_Hip_FlEx_mean'] + mean_cycle_r_op['R_Hip_FlEx_std'],
                                alpha=0.2, color='r')
            
            rmse_rhip = calculate_rmse(mean_cycle_r['R_Hip_FlEx_mean'], mean_cycle_r_op['R_Hip_FlEx_mean'])
            axes[0].set_ylabel('Hip Angle [deg]', fontsize=12)
            axes[0].set_title(f'Right Hip Flexion/Extension (RMSE: {rmse_rhip:.2f}°)', fontsize=14)
            axes[0].legend(fontsize=11)
            axes[0].grid(True, alpha=0.3)
            
            # 膝関節屈曲伸展
            axes[1].plot(normalized_percentage, mean_cycle_r['R_Knee_FlEx_mean'], 'b-', label='Mocap', linewidth=2)
            axes[1].fill_between(normalized_percentage,
                                mean_cycle_r['R_Knee_FlEx_mean'] - mean_cycle_r['R_Knee_FlEx_std'],
                                mean_cycle_r['R_Knee_FlEx_mean'] + mean_cycle_r['R_Knee_FlEx_std'],
                                alpha=0.2, color='b')
            axes[1].plot(normalized_percentage, mean_cycle_r_op['R_Knee_FlEx_mean'], 'r--', label='OpenPose', linewidth=2)
            axes[1].fill_between(normalized_percentage,
                                mean_cycle_r_op['R_Knee_FlEx_mean'] - mean_cycle_r_op['R_Knee_FlEx_std'],
                                mean_cycle_r_op['R_Knee_FlEx_mean'] + mean_cycle_r_op['R_Knee_FlEx_std'],
                                alpha=0.2, color='r')
            
            rmse_rknee = calculate_rmse(mean_cycle_r['R_Knee_FlEx_mean'], mean_cycle_r_op['R_Knee_FlEx_mean'])
            axes[1].set_ylabel('Knee Angle [deg]', fontsize=12)
            axes[1].set_title(f'Right Knee Flexion/Extension (RMSE: {rmse_rknee:.2f}°)', fontsize=14)
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
            
            # 足関節背屈底屈
            axes[2].plot(normalized_percentage, mean_cycle_r['R_Ankle_PlDo_mean'], 'b-', label='Mocap', linewidth=2)
            axes[2].fill_between(normalized_percentage,
                                mean_cycle_r['R_Ankle_PlDo_mean'] - mean_cycle_r['R_Ankle_PlDo_std'],
                                mean_cycle_r['R_Ankle_PlDo_mean'] + mean_cycle_r['R_Ankle_PlDo_std'],
                                alpha=0.2, color='b')
            axes[2].plot(normalized_percentage, mean_cycle_r_op['R_Ankle_PlDo_mean'], 'r--', label='OpenPose', linewidth=2)
            axes[2].fill_between(normalized_percentage,
                                mean_cycle_r_op['R_Ankle_PlDo_mean'] - mean_cycle_r_op['R_Ankle_PlDo_std'],
                                mean_cycle_r_op['R_Ankle_PlDo_mean'] + mean_cycle_r_op['R_Ankle_PlDo_std'],
                                alpha=0.2, color='r')
            
            rmse_rankle = calculate_rmse(mean_cycle_r['R_Ankle_PlDo_mean'], mean_cycle_r_op['R_Ankle_PlDo_mean'])
            axes[2].set_xlabel('Gait Cycle [%]', fontsize=12)
            axes[2].set_ylabel('Ankle Angle [deg]', fontsize=12)
            axes[2].set_title(f'Right Ankle Plantarflexion/Dorsiflexion (RMSE: {rmse_rankle:.2f}°)', fontsize=14)
            axes[2].legend(fontsize=11)
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"comparison_FlEx_R_{csv_path.stem}.png", dpi=300)
            plt.close()
            
            # 股関節内旋外旋の比較
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            ax.plot(normalized_percentage, mean_cycle_r['R_Hip_InEx_mean'], 'b-', label='Mocap', linewidth=2)
            ax.fill_between(normalized_percentage,
                           mean_cycle_r['R_Hip_InEx_mean'] - mean_cycle_r['R_Hip_InEx_std'],
                           mean_cycle_r['R_Hip_InEx_mean'] + mean_cycle_r['R_Hip_InEx_std'],
                           alpha=0.2, color='b')
            ax.plot(normalized_percentage, mean_cycle_r_op['R_Hip_InEx_mean'], 'r--', label='OpenPose', linewidth=2)
            ax.fill_between(normalized_percentage,
                           mean_cycle_r_op['R_Hip_InEx_mean'] - mean_cycle_r_op['R_Hip_InEx_std'],
                           mean_cycle_r_op['R_Hip_InEx_mean'] + mean_cycle_r_op['R_Hip_InEx_std'],
                           alpha=0.2, color='r')
            
            rmse_rhip_inex = calculate_rmse(mean_cycle_r['R_Hip_InEx_mean'], mean_cycle_r_op['R_Hip_InEx_mean'])
            ax.set_xlabel('Gait Cycle [%]', fontsize=12)
            ax.set_ylabel('Hip Angle [deg]', fontsize=12)
            ax.set_title(f'Right Hip Internal/External Rotation (RMSE: {rmse_rhip_inex:.2f}°)', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"comparison_InEx_R_{csv_path.stem}.png", dpi=300)
            plt.close()
            
            # 股関節内転外転の比較
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            ax.plot(normalized_percentage, mean_cycle_r['R_Hip_AdAb_mean'], 'b-', label='Mocap', linewidth=2)
            ax.fill_between(normalized_percentage,
                           mean_cycle_r['R_Hip_AdAb_mean'] - mean_cycle_r['R_Hip_AdAb_std'],
                           mean_cycle_r['R_Hip_AdAb_mean'] + mean_cycle_r['R_Hip_AdAb_std'],
                           alpha=0.2, color='b')
            ax.plot(normalized_percentage, mean_cycle_r_op['R_Hip_AdAb_mean'], 'r--', label='OpenPose', linewidth=2)
            ax.fill_between(normalized_percentage,
                           mean_cycle_r_op['R_Hip_AdAb_mean'] - mean_cycle_r_op['R_Hip_AdAb_std'],
                           mean_cycle_r_op['R_Hip_AdAb_mean'] + mean_cycle_r_op['R_Hip_AdAb_std'],
                           alpha=0.2, color='r')
            
            rmse_rhip_adab = calculate_rmse(mean_cycle_r['R_Hip_AdAb_mean'], mean_cycle_r_op['R_Hip_AdAb_mean'])
            ax.set_xlabel('Gait Cycle [%]', fontsize=12)
            ax.set_ylabel('Hip Angle [deg]', fontsize=12)
            ax.set_title(f'Right Hip Adduction/Abduction (RMSE: {rmse_rhip_adab:.2f}°)', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"comparison_AdAb_R_{csv_path.stem}.png", dpi=300)
            plt.close()
            
            # RMSEサマリーを保存
            rmse_summary_r = pd.DataFrame({
                'Joint_Movement': ['Hip_FlEx', 'Knee_FlEx', 'Ankle_PlDo', 'Hip_InEx', 'Hip_AdAb'],
                'RMSE [deg]': [rmse_rhip, rmse_rknee, rmse_rankle, rmse_rhip_inex, rmse_rhip_adab]
            })
            rmse_summary_r.to_csv(op_result_dir / f"RMSE_summary_R_{csv_path.stem}.csv", index=False)
            print(f"\n右足RMSE:")
            print(rmse_summary_r)
        
        # 左足の比較プロット
        if len(normalized_gait_cycles_l) > 0 and len(normalized_gait_cycles_l_op) > 0:
            print("左足関節角度の比較プロットを作成中...")
            # 屈曲伸展・背屈底屈の比較
            fig, axes = plt.subplots(3, 1, figsize=(12, 14))
            
            # 股関節屈曲伸展
            axes[0].plot(normalized_percentage, mean_cycle_l['L_Hip_FlEx_mean'], 'b-', label='Mocap', linewidth=2)
            axes[0].fill_between(normalized_percentage,
                                mean_cycle_l['L_Hip_FlEx_mean'] - mean_cycle_l['L_Hip_FlEx_std'],
                                mean_cycle_l['L_Hip_FlEx_mean'] + mean_cycle_l['L_Hip_FlEx_std'],
                                alpha=0.2, color='b')
            axes[0].plot(normalized_percentage, mean_cycle_l_op['L_Hip_FlEx_mean'], 'r--', label='OpenPose', linewidth=2)
            axes[0].fill_between(normalized_percentage,
                                mean_cycle_l_op['L_Hip_FlEx_mean'] - mean_cycle_l_op['L_Hip_FlEx_std'],
                                mean_cycle_l_op['L_Hip_FlEx_mean'] + mean_cycle_l_op['L_Hip_FlEx_std'],
                                alpha=0.2, color='r')
            
            rmse_lhip = calculate_rmse(mean_cycle_l['L_Hip_FlEx_mean'], mean_cycle_l_op['L_Hip_FlEx_mean'])
            axes[0].set_ylabel('Hip Angle [deg]', fontsize=12)
            axes[0].set_title(f'Left Hip Flexion/Extension (RMSE: {rmse_lhip:.2f}°)', fontsize=14)
            axes[0].legend(fontsize=11)
            axes[0].grid(True, alpha=0.3)
            
            # 膝関節屈曲伸展
            axes[1].plot(normalized_percentage, mean_cycle_l['L_Knee_FlEx_mean'], 'b-', label='Mocap', linewidth=2)
            axes[1].fill_between(normalized_percentage,
                                mean_cycle_l['L_Knee_FlEx_mean'] - mean_cycle_l['L_Knee_FlEx_std'],
                                mean_cycle_l['L_Knee_FlEx_mean'] + mean_cycle_l['L_Knee_FlEx_std'],
                                alpha=0.2, color='b')
            axes[1].plot(normalized_percentage, mean_cycle_l_op['L_Knee_FlEx_mean'], 'r--', label='OpenPose', linewidth=2)
            axes[1].fill_between(normalized_percentage,
                                mean_cycle_l_op['L_Knee_FlEx_mean'] - mean_cycle_l_op['L_Knee_FlEx_std'],
                                mean_cycle_l_op['L_Knee_FlEx_mean'] + mean_cycle_l_op['L_Knee_FlEx_std'],
                                alpha=0.2, color='r')
            
            rmse_lknee = calculate_rmse(mean_cycle_l['L_Knee_FlEx_mean'], mean_cycle_l_op['L_Knee_FlEx_mean'])
            axes[1].set_ylabel('Knee Angle [deg]', fontsize=12)
            axes[1].set_title(f'Left Knee Flexion/Extension (RMSE: {rmse_lknee:.2f}°)', fontsize=14)
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
            
            # 足関節背屈底屈
            axes[2].plot(normalized_percentage, mean_cycle_l['L_Ankle_PlDo_mean'], 'b-', label='Mocap', linewidth=2)
            axes[2].fill_between(normalized_percentage,
                                mean_cycle_l['L_Ankle_PlDo_mean'] - mean_cycle_l['L_Ankle_PlDo_std'],
                                mean_cycle_l['L_Ankle_PlDo_mean'] + mean_cycle_l['L_Ankle_PlDo_std'],
                                alpha=0.2, color='b')
            axes[2].plot(normalized_percentage, mean_cycle_l_op['L_Ankle_PlDo_mean'], 'r--', label='OpenPose', linewidth=2)
            axes[2].fill_between(normalized_percentage,
                                mean_cycle_l_op['L_Ankle_PlDo_mean'] - mean_cycle_l_op['L_Ankle_PlDo_std'],
                                mean_cycle_l_op['L_Ankle_PlDo_mean'] + mean_cycle_l_op['L_Ankle_PlDo_std'],
                                alpha=0.2, color='r')
            
            rmse_lankle = calculate_rmse(mean_cycle_l['L_Ankle_PlDo_mean'], mean_cycle_l_op['L_Ankle_PlDo_mean'])
            axes[2].set_xlabel('Gait Cycle [%]', fontsize=12)
            axes[2].set_ylabel('Ankle Angle [deg]', fontsize=12)
            axes[2].set_title(f'Left Ankle Plantarflexion/Dorsiflexion (RMSE: {rmse_lankle:.2f}°)', fontsize=14)
            axes[2].legend(fontsize=11)
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"comparison_FlEx_L_{csv_path.stem}.png", dpi=300)
            plt.close()
            
            # 股関節内旋外旋の比較
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            ax.plot(normalized_percentage, mean_cycle_l['L_Hip_InEx_mean'], 'b-', label='Mocap', linewidth=2)
            ax.fill_between(normalized_percentage,
                           mean_cycle_l['L_Hip_InEx_mean'] - mean_cycle_l['L_Hip_InEx_std'],
                           mean_cycle_l['L_Hip_InEx_mean'] + mean_cycle_l['L_Hip_InEx_std'],
                           alpha=0.2, color='b')
            ax.plot(normalized_percentage, mean_cycle_l_op['L_Hip_InEx_mean'], 'r--', label='OpenPose', linewidth=2)
            ax.fill_between(normalized_percentage,
                           mean_cycle_l_op['L_Hip_InEx_mean'] - mean_cycle_l_op['L_Hip_InEx_std'],
                           mean_cycle_l_op['L_Hip_InEx_mean'] + mean_cycle_l_op['L_Hip_InEx_std'],
                           alpha=0.2, color='r')
            
            rmse_lhip_inex = calculate_rmse(mean_cycle_l['L_Hip_InEx_mean'], mean_cycle_l_op['L_Hip_InEx_mean'])
            ax.set_xlabel('Gait Cycle [%]', fontsize=12)
            ax.set_ylabel('Hip Angle [deg]', fontsize=12)
            ax.set_title(f'Left Hip Internal/External Rotation (RMSE: {rmse_lhip_inex:.2f}°)', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"comparison_InEx_L_{csv_path.stem}.png", dpi=300)
            plt.close()
            
            # 股関節内転外転の比較
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            ax.plot(normalized_percentage, mean_cycle_l['L_Hip_AdAb_mean'], 'b-', label='Mocap', linewidth=2)
            ax.fill_between(normalized_percentage,
                           mean_cycle_l['L_Hip_AdAb_mean'] - mean_cycle_l['L_Hip_AdAb_std'],
                           mean_cycle_l['L_Hip_AdAb_mean'] + mean_cycle_l['L_Hip_AdAb_std'],
                           alpha=0.2, color='b')
            ax.plot(normalized_percentage, mean_cycle_l_op['L_Hip_AdAb_mean'], 'r--', label='OpenPose', linewidth=2)
            ax.fill_between(normalized_percentage,
                           mean_cycle_l_op['L_Hip_AdAb_mean'] - mean_cycle_l_op['L_Hip_AdAb_std'],
                           mean_cycle_l_op['L_Hip_AdAb_mean'] + mean_cycle_l_op['L_Hip_AdAb_std'],
                           alpha=0.2, color='r')
            
            rmse_lhip_adab = calculate_rmse(mean_cycle_l['L_Hip_AdAb_mean'], mean_cycle_l_op['L_Hip_AdAb_mean'])
            ax.set_xlabel('Gait Cycle [%]', fontsize=12)
            ax.set_ylabel('Hip Angle [deg]', fontsize=12)
            ax.set_title(f'Left Hip Adduction/Abduction (RMSE: {rmse_lhip_adab:.2f}°)', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"comparison_AdAb_L_{csv_path.stem}.png", dpi=300)
            plt.close()
            
            # RMSEサマリーを保存
            rmse_summary_l = pd.DataFrame({
                'Joint_Movement': ['Hip_FlEx', 'Knee_FlEx', 'Ankle_PlDo', 'Hip_InEx', 'Hip_AdAb'],
                'RMSE [deg]': [rmse_lhip, rmse_lknee, rmse_lankle, rmse_lhip_inex, rmse_lhip_adab]
            })
            rmse_summary_l.to_csv(op_result_dir / f"RMSE_summary_L_{csv_path.stem}.csv", index=False)
            print(f"\n左足RMSE:")
            print(rmse_summary_l)

if __name__ == "__main__":
    main()