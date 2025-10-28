import pandas as pd
import os
import glob
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
    csv_path_dir = r"G:\gait_pattern\20250811_br\sub1\thera0-3\mocap"
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
            
        # #####################################
        # #####################################
        # OpenPose3D結果との比較
        # #####################################
        # #####################################
        # 結果保存ディレクトリ作成
        op_result_dir = os.path.join(os.path.dirname(os.path.dirname(csv_path)), "OpenPose3D_results")
        if not os.path.exists(op_result_dir):
            os.makedirs(op_result_dir)

        openpose_npz_path = os.path.join(os.path.dirname(csv_path_dir), "3d_kp_data_openpose.npz")
        if not os.path.exists(openpose_npz_path):
            print(f"OpenPose3Dデータが見つかりません: {openpose_npz_path}")
            return
    
        print(f"OpenPose3Dデータとの比較を実行: {openpose_npz_path}")
        openpose_data = np.load(openpose_npz_path)
        print(f"openpose_data keys: {openpose_data.files}")
        op_frame = openpose_data['frame']
        op_raw_data = openpose_data['raw']  # shape: (num_frames, num_joints, 3)
        op_filt_data = openpose_data['filt']  # shape: (num_frames, num_joints, 3)
        op_conf = openpose_data['conf']  # shape: (num_frames, num_joints)
        print(f"op_frame: {op_frame}")
        print(f"op_raw_data shape: {op_raw_data.shape}")
        
        ###########################################
        # タイミング合わせ  股関節中心が任意の位置を通るタイミングでフレーム調整
        ###########################################
        base_point = int(0)
        # opti用
        hip_z_opti = (((rasi + lasi) / 2 + (rpsi + lpsi) / 2) / 2)[:, 2]
        # hip_z_optiが0より大きくなった最初のフレームを取得
        start_frame_opti = np.argmax(hip_z_opti > base_point) + start_frame
        print(f"100Hz Opti計測開始から原点通過までのフレーム数: {start_frame_opti}")
        
        # OpenPose用
        hip_z_op = op_filt_data[:, 8, 2]
        # hip_zが0より大きくなった最初のフレームを取得
        start_idx_op = np.argmax(hip_z_op > base_point)
        start_frame_op = op_frame[start_idx_op]
        print(f"60Hz OpenPoseLED発光検出から原点通過までのフレーム数: {start_frame_op}")
        
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
        
        start_frame_op_tekitou = 170
        end_frame_op_tekitou = 459
        
        # pel_bec_op = pel_bec_op[start_frame_op_tekitou:end_frame_op_tekitou, :]
        # thigh_r_op = thigh_r_op[start_frame_op_tekitou:end_frame_op_tekitou, :]
        # shank_r_op = shank_r_op[start_frame_op_tekitou:end_frame_op_tekitou, :]
        # foot_r_op = foot_r_op[start_frame_op_tekitou:end_frame_op_tekitou, :]
        # thigh_l_op = thigh_l_op[start_frame_op_tekitou:end_frame_op_tekitou, :]
        # shank_l_op = shank_l_op[start_frame_op_tekitou:end_frame_op_tekitou, :]
        # foot_l_op = foot_l_op[start_frame_op_tekitou:end_frame_op_tekitou, :]
        
        # 初期接地算出
        pel2hee_z_r_op = (rhee_op - midhip_op)[:,2]
        pel2hee_z_l_op = (lhee_op - midhip_op)[:,2]

        df_ic_op_o = pd.DataFrame({"frame_60hz": op_frame,
                                    "rhee_pel_z": pel2hee_z_r_op,
                                    "lhee_pel_z": pel2hee_z_l_op})
        df_ic_op_r = df_ic_op_o.sort_values(by="rhee_pel_z", ascending=False)
        df_ic_op_l = df_ic_op_o.sort_values(by="lhee_pel_z", ascending=False)
        # 初期接地検出
        ic_r_list_op = df_ic_op_r.head(30)["frame_60hz"].values.astype(int)
        ic_l_list_op = df_ic_op_l.head(30)["frame_60hz"].values.astype(int)
        print(f"ic_r_list_op: {ic_r_list_op}")
        print(f"ic_l_list_op: {ic_l_list_op}")
        filt_ic_r_list_op = []
        skip_values_op_r = set()
        for value in ic_r_list_op:
            if value in skip_values_op_r:
                continue
            filt_ic_r_list_op.append(value)
            # 60Hzでの10フレーム間隔でスキップ
            skip_values_op_r.update(range(value - 10, value + 11))
        filt_ic_r_list_op = sorted(filt_ic_r_list_op)
        print(f"フィルタリング後のRリスト (60Hz相対フレーム): {filt_ic_r_list_op}")
        filt_ic_l_list_op = []
        skip_values_op_l = set()
        for value in ic_l_list_op:
            if value in skip_values_op_l:
                continue
            filt_ic_l_list_op.append(value)
            # 60Hzでの10フレーム間隔でスキップ
            skip_values_op_l.update(range(value - 10, value + 11))
        filt_ic_l_list_op = sorted(filt_ic_l_list_op)
        print(f"フィルタリング後のLリスト (60Hz相対フレーム): {filt_ic_l_list_op}")


        # 関節角度計算
        # 股関節、膝関節、足関節の屈曲伸展角度
        n_axis = rhip_op - lhip_op  # 右股関節から左股関節へのベクトル

        rhip_flex_op = op.culc_angle_all_frames(pel_bec_op, thigh_r_op, n_axis, degrees=True, angle_type='hip')
        lhip_flex_op = op.culc_angle_all_frames(pel_bec_op, thigh_l_op, n_axis, degrees=True, angle_type='hip')
        rknee_flex_op = op.culc_angle_all_frames(thigh_r_op, shank_r_op, n_axis, degrees=True, angle_type='knee')
        lknee_flex_op = op.culc_angle_all_frames(thigh_l_op, shank_l_op, n_axis, degrees=True, angle_type='knee')
        rankle_pldo_op = op.culc_angle_all_frames(shank_r_op, foot_r_op, n_axis, degrees=True, angle_type='ankle')
        lankle_pldo_op = op.culc_angle_all_frames(shank_l_op, foot_l_op, n_axis, degrees=True, angle_type='ankle')

        print(f"rhip_flex_op[265]: {rhip_flex_op[265]}")
        print(f"lhip_flex_op[265]: {lhip_flex_op[265]}")
        
        # rhip_flex_op = [180 - rhip_flex_angle if rhip_flex_angle > 0 else - (180 - rhip_flex_angle) for rhip_flex_angle in rhip_flex_op]
        # lhip_flex_op = [180 - lhip_flex_angle if lhip_flex_angle > 0 else - (180 - lhip_flex_angle) for lhip_flex_angle in lhip_flex_op]

        plt.plot(range(start_frame_op_tekitou, end_frame_op_tekitou), rhip_flex_op[start_frame_op_tekitou:end_frame_op_tekitou], label='OpenPose Right Hip Flexion/Extension', color='blue')
        plt.plot(range(start_frame_op_tekitou, end_frame_op_tekitou), lhip_flex_op[start_frame_op_tekitou:end_frame_op_tekitou], label='OpenPose Left Hip Flexion/Extension', color='orange')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in filt_ic_l_list_op]
        [plt.axvline(x=frame, color='blue', linestyle='--', alpha=0.5) for frame in filt_ic_r_list_op]
        plt.xlabel('Frame [-]')
        plt.ylabel('Angle [deg]')
        plt.title('OpenPose Hip Flexion/Extension Angles Over Time')
        # plt.ylim(-40, 40)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(op_result_dir, f"{os.path.basename(csv_path)}_Hip_Flexion_Extension_{openpose_npz_path.stem}.png"))
        # plt.show()
        plt.close()
        
        plt.plot(range(start_frame_op_tekitou, end_frame_op_tekitou), rknee_flex_op[start_frame_op_tekitou:end_frame_op_tekitou], label='OpenPose Right Knee Flexion/Extension', color='blue')
        plt.plot(range(start_frame_op_tekitou, end_frame_op_tekitou), lknee_flex_op[start_frame_op_tekitou:end_frame_op_tekitou], label='OpenPose Left Knee Flexion/Extension', color='orange')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in filt_ic_l_list_op]
        [plt.axvline(x=frame, color='blue', linestyle='--', alpha=0.5) for frame in filt_ic_r_list_op]
        plt.xlabel('Frame [-]')
        plt.ylabel('Angle [deg]')
        plt.title('OpenPose Knee Flexion/Extension Angles Over Time')
        # plt.ylim(-10, 70)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(op_result_dir, f"{os.path.basename(csv_path)}_Knee_Flexion_Extension_{openpose_npz_path.stem}.png"))
        # plt.show()
        plt.close()
        
        plt.plot(range(start_frame_op_tekitou, end_frame_op_tekitou), rankle_pldo_op[start_frame_op_tekitou:end_frame_op_tekitou], label='OpenPose Right Ankle Plantarflexion/Dorsiflexion', color='blue')
        plt.plot(range(start_frame_op_tekitou, end_frame_op_tekitou), lankle_pldo_op[start_frame_op_tekitou:end_frame_op_tekitou], label='OpenPose Left Ankle Plantarflexion/Dorsiflexion', color='orange')
        [plt.axvline(x=frame, color='orange', linestyle='--', alpha=0.5) for frame in filt_ic_l_list_op]   
        [plt.axvline(x=frame, color='blue', linestyle='--', alpha=0.5) for frame in filt_ic_r_list_op]
        plt.xlabel('Frame [-]')
        plt.ylabel('Angle [deg]')
        plt.title('OpenPose Ankle Plantarflexion/Dorsiflexion Angles Over Time')
        # plt.ylim(-20, 60)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(op_result_dir, f"{os.path.basename(csv_path)}_Ankle_Plantarflexion_Dorsiflexion_{openpose_npz_path.stem}.png"))
        # plt.show()
        plt.close()

if __name__ == "__main__":
    main()