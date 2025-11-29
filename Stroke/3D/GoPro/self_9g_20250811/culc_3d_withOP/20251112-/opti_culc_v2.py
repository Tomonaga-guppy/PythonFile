import pandas as pd
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample
import json
import m_opti as opti

def main():
    # csv_path_dir = Path(r"G:\gait_pattern\BR9G_shuron\sub1\thera0-3\mocap")
    csv_path_dir = Path(r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\mocap")
    start_frame = 0
    end_frame = 10000

    if str(csv_path_dir) == r"G:\gait_pattern\BR9G_shuron\sub1\thera0-2\mocap":
        start_frame = 1000
        end_frame = 1440
    elif str(csv_path_dir) == r"G:\gait_pattern\BR9G_shuron\sub1\thera0-3\mocap":
        start_frame = 943
        end_frame = 1400
    elif str(csv_path_dir) == r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\mocap":
        pass
        start_frame = 1000
        # end_frame = 1252
    elif str(csv_path_dir) == r"G:\gait_pattern\BR9G_shuron\sub0\thera0-16\mocap":
        start_frame = 890
        end_frame = 1210
    elif str(csv_path_dir) == r"G:\gait_pattern\BR9G_shuron\sub0\thera0-15\mocap":
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

    csv_paths = list(csv_path_dir.glob("[0-9]*_[0-9]*_[0-9].csv"))

    geometry_json_path = Path(r"G:\gait_pattern\20250811_br\sub0\thera0-14\mocap\geometry.json")

    for i, csv_path in enumerate(csv_paths):
        print(f"Processing: {csv_path}")

        try:
            keypoints_mocap, full_range, start_frame, end_frame = opti.read_3d_optitrack(csv_path, start_frame, end_frame,
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
        sac2hee_r_list = []
        sac2hee_l_list = []
        heel_z_list = []

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
            
            # Davidモデルを参考に骨盤座標系1を骨盤の座標系として定義
            e_x_pelvis = e_x_pelvis_0
            e_y_pelvis = e_y_pelvis_0
            e_z_pelvis = e_z_pelvis_0
            rot_pelvis = np.array([e_x_pelvis, e_y_pelvis, e_z_pelvis]).T

            transformation_matrix = np.array([[e_x_pelvis_0[0], e_y_pelvis_0[0], e_z_pelvis_0[0], hip_0[0]],
                                                [e_x_pelvis_0[1], e_y_pelvis_0[1], e_z_pelvis_0[1], hip_0[1]],
                                                [e_x_pelvis_0[2], e_y_pelvis_0[2], e_z_pelvis_0[2], hip_0[2]],
                                                [0,       0,       0,       1]])

            #グローバル座標に変換して再度計算
            rthigh = np.dot(transformation_matrix, np.append(rthigh_pelvis, 1))[:3]
            lthigh = np.dot(transformation_matrix, np.append(lthigh_pelvis, 1))[:3]
            hip = (rthigh + lthigh) / 2

            # 腰椎節原点
            lumbar = (0.47 * (rasi[frame_num,:] + lasi[frame_num,:]) / 2 + 0.53 * (rpsi[frame_num,:] + lpsi[frame_num,:]) / 2) + 0.02 * k * np.array([0, 1, 0])

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
            e_z_rshank = (rfoot - rshank)/np.linalg.norm(rfoot - rshank)
            e_x_rshank = np.cross(e_y0_rshank, e_z_rshank)/np.linalg.norm(np.cross(e_y0_rshank, e_z_rshank))
            e_y_rshank = np.cross(e_z_rshank, e_x_rshank)
            rot_rshank = np.array([e_x_rshank, e_y_rshank, e_z_rshank]).T

            #左下腿節座標系（原点はlshank）
            e_y0_lshank = lknee[frame_num, :] - lknee2[frame_num, :]
            e_z_lshank = (lfoot - lshank)/np.linalg.norm(lfoot - lshank)
            e_x_lshank = np.cross(e_y0_lshank, e_z_lshank)/np.linalg.norm(np.cross(e_y0_lshank, e_z_lshank))
            e_y_lshank = np.cross(e_z_lshank, e_x_lshank)
            rot_lshank = np.array([e_x_lshank, e_y_lshank, e_z_lshank]).T

            #右足節座標系 AIST参照（原点はrfoot）
            e_x_rfoot = (rtoe[frame_num,:] - rhee[frame_num,:]) / np.linalg.norm(rtoe[frame_num,:] - rhee[frame_num,:])
            e_y0_rfoot = rank2[frame_num,:] - rank[frame_num,:]
            e_z_rfoot = np.cross(e_x_rfoot, e_y0_rfoot)/np.linalg.norm(np.cross(e_x_rfoot, e_y0_rfoot))
            e_y_rfoot = np.cross(e_z_rfoot, e_x_rfoot)
            rot_rfoot = np.array([e_x_rfoot, e_y_rfoot, e_z_rfoot]).T
            

            #左足節座標系 AIST参照（原点はlfoot）
            e_x_lfoot = (ltoe[frame_num,:] - lhee[frame_num, :]) / np.linalg.norm(ltoe[frame_num,:] - lhee[frame_num, :])
            e_y0_lfoot = lank[frame_num,:] - lank2[frame_num,:]
            e_z_lfoot = np.cross(e_x_lfoot, e_y0_lfoot)/np.linalg.norm(np.cross(e_x_lfoot, e_y0_lfoot))
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
                if frame_num == 98:
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

                    ax.scatter(rasi[frame_num,:][0], rasi[frame_num,:][1], rasi[frame_num,:][2], color='black', s=5)
                    ax.scatter(lasi[frame_num,:][0], lasi[frame_num,:][1], lasi[frame_num,:][2], color='black', s=5)
                    ax.scatter(rpsi[frame_num,:][0], rpsi[frame_num,:][1], rpsi[frame_num,:][2], color='black', s=5)
                    ax.scatter(lpsi[frame_num,:][0], lpsi[frame_num,:][1], lpsi[frame_num,:][2], color='black', s=5)
                    ax.scatter(rank[frame_num,:][0], rank[frame_num,:][1], rank[frame_num,:][2], color='black', s=5)
                    ax.scatter(lank[frame_num,:][0], lank[frame_num,:][1], lank[frame_num,:][2], color='black', s=5)
                    ax.scatter(rank2[frame_num,:][0], rank2[frame_num,:][1], rank2[frame_num,:][2], color='black', s=5)
                    ax.scatter(lank2[frame_num,:][0], lank2[frame_num,:][1], lank2[frame_num,:][2], color='black', s=5)
                    ax.scatter(rknee[frame_num,:][0], rknee[frame_num,:][1], rknee[frame_num,:][2], color='black', s=5)
                    ax.scatter(lknee[frame_num,:][0], lknee[frame_num,:][1], lknee[frame_num,:][2], color='black', s=5)
                    ax.scatter(rknee2[frame_num,:][0], rknee2[frame_num,:][1], rknee2[frame_num,:][2], color='black', s=5)
                    ax.scatter(lknee2[frame_num,:][0], lknee2[frame_num,:][1], lknee2[frame_num,:][2], color='black', s=5)
                    ax.scatter(rtoe[frame_num,:][0], rtoe[frame_num,:][1], rtoe[frame_num,:][2], color='black', s=5)
                    ax.scatter(ltoe[frame_num,:][0], ltoe[frame_num,:][1], ltoe[frame_num,:][2], color='black', s=5)
                    ax.scatter(rhee[frame_num,:][0], rhee[frame_num,:][1], rhee[frame_num,:][2], color='black', s=5)
                    ax.scatter(lhee[frame_num, :][0], lhee[frame_num, :][1], lhee[frame_num, :][2], color='black', s=5)
                    
                    ax.scatter(rfoot[0], rfoot[1], rfoot[2], label='rfoot')
                    ax.scatter(lfoot[0], lfoot[1], lfoot[2], label='lfoot')
                    ax.scatter(rshank[0], rshank[1], rshank[2], label='rshank')
                    ax.scatter(lshank[0], lshank[1], lshank[2], label='lshank')
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

            #仙骨とかかとのベクトル計算
            sacuram = (rpsi[frame_num, :] + lpsi[frame_num, :]) / 2
            sac2hee_r = rhee[frame_num, :] - sacuram
            sac2hee_r_list.append(sac2hee_r)
            sac2hee_l = lhee[frame_num, :] - sacuram
            sac2hee_l_list.append(sac2hee_l)
            
            # 踵のZ座標記録
            heel_z_list.append((rhee[frame_num, 2], lhee[frame_num, 2]))

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
                angle_df.loc[frame, 'R_Knee_FlEx'] = - angle_df.at[frame, 'R_Knee_FlEx']
        if 'L_Knee_FlEx' in angle_df.columns:
            for frame in angle_df.index:
                angle_df.loc[frame, 'L_Knee_FlEx'] = - angle_df.at[frame, 'L_Knee_FlEx']
        if 'R_Ankle_PlDo' in angle_df.columns:
            for frame in angle_df.index:
                angle_df.loc[frame, 'R_Ankle_PlDo'] = 180 - angle_df.at[frame, 'R_Ankle_PlDo']
        if 'L_Ankle_PlDo' in angle_df.columns:
            for frame in angle_df.index:
                angle_df.loc[frame, 'L_Ankle_PlDo'] = 180 - angle_df.at[frame, 'L_Ankle_PlDo']
        
                
        # DataFrameのインデックスを絶対フレーム番号に設定
        # 100Hzデータの場合、start_frameからの100Hz絶対フレーム番号
        absolute_frame_indices = np.array(full_range) + start_frame

        # print(f"absolute_frame_indices = {absolute_frame_indices}")
        absolute_frame_indices = absolute_frame_indices[1:]
        # print(f"absolute_frame_indices = {absolute_frame_indices}")

        # ファイル名に適切なサンプリング周波数を記載
        angle_df.to_csv(csv_path.parent / f"angle_100Hz_{csv_path.name}")

        sac2hee_r_array = np.array(sac2hee_r_list)
        rsac2hee_z = sac2hee_r_array[:, 2]
        sac2hee_l_array = np.array(sac2hee_l_list)
        lsac2hee_z = sac2hee_l_array[:, 2]

        # 100Hzデータでの相対フレーム番号（0から始まる）
        rel_frames = np.array(full_range)

        # 100Hzデータでの絶対フレーム番号
        abs_frames = rel_frames + start_frame
        
        print(f"frame_100hz_rel sample: {rel_frames[:10]}")
        print(f"frame_100hz_abs sample: {abs_frames[:10]}")

        print(f"len(frame_100hz_rel) = {len(rel_frames)}")
        print(f"len(frame_100hz_abs) = {len(abs_frames)}")
        print(f"len(rsac2hee_z) = {len(rsac2hee_z)}")
        print(f"len(lsac2hee_z) = {len(lsac2hee_z)}")
        
        df_IcTo = pd.DataFrame({
            "frame_100hz_rel": rel_frames,
            "frame_100hz_abs": abs_frames,
            "rsac2hee_z": rsac2hee_z,
            "lsac2hee_z": lsac2hee_z,
        })
        
        plt.figure(figsize=(10, 7))
        plt.plot(df_IcTo["frame_100hz_abs"], df_IcTo["rsac2hee_z"], label='rsac2hee')
        plt.plot(df_IcTo["frame_100hz_abs"], df_IcTo["lsac2hee_z"], label='lsac2hee')
        plt.title("Heel to sacrum distance")
        plt.xlabel("Frame number [-]")
        plt.ylabel("Distance [mm]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(csv_path.parent / "heel to sacrum distance.png")
        plt.close()
        
        df_IcTo.index = df_IcTo.index + 1
        df_ic_r = df_IcTo.sort_values(by="rsac2hee_z", ascending=False)  #降順: 大きいピークが初期接地
        df_ic_l = df_IcTo.sort_values(by="lsac2hee_z", ascending=False)  #降順
        df_to_r = df_IcTo.sort_values(by="rsac2hee_z")  #昇順  小さいピークがつま先離地
        df_to_l = df_IcTo.sort_values(by="lsac2hee_z")  #昇順

        # 初期接地検出（100Hz相対フレーム番号で）
        cycle_guide = len(full_range) / 100 #100Hzで1サイクルの目安フレーム数を100にしておく
        check_num = cycle_guide * 20 #一つのピークで20個くらい候補が出る想定(適当)
        ic_r_list_100hz_rel = df_ic_r.head(int(check_num))["frame_100hz_rel"].values.astype(int)
        ic_l_list_100hz_rel = df_ic_l.head(int(check_num))["frame_100hz_rel"].values.astype(int)
        ic_r_list_100hz_abs = df_ic_r.head(int(check_num))["frame_100hz_abs"].values.astype(int)
        ic_l_list_100hz_abs = df_ic_l.head(int(check_num))["frame_100hz_abs"].values.astype(int)
        
        to_r_list_100hz_rel = df_to_r.head(int(check_num))["frame_100hz_rel"].values.astype(int)
        to_l_list_100hz_rel = df_to_l.head(int(check_num))["frame_100hz_rel"].values.astype(int)
        to_r_list_100hz_abs = df_to_r.head(int(check_num))["frame_100hz_abs"].values.astype(int)
        to_l_list_100hz_abs = df_to_l.head(int(check_num))["frame_100hz_abs"].values.astype(int)

        print(f"start_frame (100Hz): {start_frame}")
        print(f"end_frame (100Hz): {end_frame}")
        # print(f"ic_r_list (100Hz相対フレーム): {ic_r_list_100hz_rel}")
        # print(f"ic_l_list (100Hz相対フレーム): {ic_l_list_100hz_rel}")
        # print(f"ic_r_list (100Hz絶対フレーム): {ic_r_list_100hz_abs}")
        # print(f"ic_l_list (100Hz絶対フレーム): {ic_l_list_100hz_abs}")
        
        # print(f"to_r_list (100Hz相対フレーム): {to_r_list_100hz_rel}")
        # print(f"to_l_list (100Hz相対フレーム): {to_l_list_100hz_rel}")
        # print(f"to_r_list (100Hz絶対フレーム): {to_r_list_100hz_abs}")
        # print(f"to_l_list (100Hz絶対フレーム): {to_l_list_100hz_abs}")

        filt_ic_r_list_100hz_rel = []
        skip_values_r = set()
        for value in ic_r_list_100hz_rel:
            if value in skip_values_r:
                continue
            filt_ic_r_list_100hz_rel.append(value)
            # 100Hzでの20フレーム間隔でスキップ
            skip_values_r.update(range(value - 20, value + 21))
        filt_ic_r_list_100hz_rel = sorted(filt_ic_r_list_100hz_rel)
        print(f"フィルタリング後のic_rリスト (100Hz相対フレーム): {filt_ic_r_list_100hz_rel}")

        filt_ic_l_list_100hz_rel = []
        skip_values_l = set()
        for value in ic_l_list_100hz_rel:
            if value in skip_values_l:
                continue
            filt_ic_l_list_100hz_rel.append(value)
            # 100Hzでの20フレーム間隔でスキップ
            skip_values_l.update(range(value - 20, value + 21))
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
            # 100Hzでの20フレーム間隔でスキップ
            skip_values_r.update(range(value - 20, value + 21))
        filt_to_r_list_100hz_rel = sorted(filt_to_r_list_100hz_rel)
        print(f"フィルタリング後のto_rリスト (100Hz相対フレーム): {filt_to_r_list_100hz_rel}")
        filt_to_l_list_100hz_rel = []
        skip_values_l = set()
        for value in to_l_list_100hz_rel:
            if value in skip_values_l:
                continue
            filt_to_l_list_100hz_rel.append(value)
            # 100Hzでの20フレーム間隔でスキップ
            skip_values_l.update(range(value - 20, value + 21))
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
            
            # 現在のICと次のICの間にある左のICを探す
            ic_l_in_cycle = [ic for ic in filt_ic_l_list_100hz_rel if ic_current < ic < ic_next]

            # 現在のICと次のICの間にあるTOを探す
            to_in_cycle = [to for to in filt_to_r_list_100hz_rel if ic_current < to < ic_next]
            
            if len(to_in_cycle) > 0 and len(ic_l_in_cycle) > 0:
                # 最初のTOを使用
                gait_cycles_r.append([ic_current, ic_l_in_cycle[0], to_in_cycle[0], ic_next])
        
        print(f"右足の歩行周期 IC→対IC→TO→次IC (100Hz): {gait_cycles_r}")
        
        # 左足も同様に作成
        gait_cycles_l = []
        for i in range(len(filt_ic_l_list_100hz_rel) - 1):
            ic_current = filt_ic_l_list_100hz_rel[i]
            ic_next = filt_ic_l_list_100hz_rel[i + 1]
            
            # 現在のICと次のICの間にある右のICを探す
            ic_r_in_cycle = [ic for ic in filt_ic_r_list_100hz_rel if ic_current < ic < ic_next]

            # 現在のICと次のICの間にあるTOを探す
            to_in_cycle = [to for to in filt_to_l_list_100hz_rel if ic_current < to < ic_next]
            
            if len(to_in_cycle) > 0 and len(ic_r_in_cycle) > 0:
                gait_cycles_l.append([ic_current, ic_r_in_cycle[0], to_in_cycle[0], ic_next])
        
        print(f"左足の歩行周期  IC→対IC→TO→次IC (100Hz): {gait_cycles_l}")
        
        gait_cycles_r_abs = []
        for ic_rel, ic_l_rel, to_rel, ic_next_rel in gait_cycles_r:
            ic_abs = ic_rel + start_frame
            ic_l_abs = ic_l_rel + start_frame
            to_abs = to_rel + start_frame
            ic_next_abs = ic_next_rel + start_frame
            gait_cycles_r_abs.append([ic_abs, ic_l_abs, to_abs, ic_next_abs])
        gait_cycles_l_abs = []
        for ic_rel, ic_r_rel, to_rel, ic_next_rel in gait_cycles_l:
            ic_abs = ic_rel + start_frame
            ic_r_abs = ic_r_rel + start_frame
            to_abs = to_rel + start_frame
            ic_next_abs = ic_next_rel + start_frame
            gait_cycles_l_abs.append([ic_abs, ic_r_abs, to_abs, ic_next_abs])
        print(f"右足の歩行周期 (100Hz絶対フレーム): {gait_cycles_r_abs}")
        print(f"左足の歩行周期 (100Hz絶対フレーム): {gait_cycles_l_abs}")
        
        # 最終的に使用した絶対フレーム範囲
        all_cycles = gait_cycles_r + gait_cycles_l
        final_start_frame_100hz = min([cycle[0] for cycle in all_cycles])
        final_start_frame_100hz_abs = final_start_frame_100hz + start_frame
        final_end_frame_100hz = max([cycle[-1] for cycle in all_cycles])
        final_end_frame_100hz_abs = final_end_frame_100hz + start_frame
        print(f"最終的に使用したフレーム範囲 (100Hz絶対フレーム): {final_start_frame_100hz_abs} 〜 {final_end_frame_100hz_abs}")
        
        # 開始フレームが右足のどのサイクルに含まれるか確認（すべてのサイクルをチェック）
        start_is_right = False
        for cycle in gait_cycles_r:
            if cycle[0] == final_start_frame_100hz:  # サイクルの開始フレーム（IC）
                start_is_right = True
                break
        if start_is_right:
            start_heel_pos = heel_z_list[final_start_frame_100hz][0]
            print(f"{final_start_frame_100hz_abs}フレーム  右足初期接地から開始，右足踵位置: {start_heel_pos:.3f} m")
        else:
            start_heel_pos = heel_z_list[final_start_frame_100hz][1]
            print(f"{final_start_frame_100hz_abs}フレーム  左足初期接地から開始，左足踵位置: {start_heel_pos:.3f} m")

        # 終了フレームが右足のどのサイクルに含まれるか確認（すべてのサイクルをチェック）
        end_is_right = False
        for cycle in gait_cycles_r:
            if cycle[-1] == final_end_frame_100hz:  # サイクルの終了フレーム（次のIC）
                end_is_right = True
                break

        if end_is_right:
            end_heel_pos = heel_z_list[final_end_frame_100hz][0]
            print(f"{final_end_frame_100hz_abs}フレーム  右足初期接地で終了，右足踵位置: {end_heel_pos:.3f} m")
        else:
            end_heel_pos = heel_z_list[final_end_frame_100hz][1]
            print(f"{final_end_frame_100hz_abs}フレーム  左足初期接地で終了，左足踵位置: {end_heel_pos:.3f} m")
        
        # 右足の歩行周期ごとに関節角度を100%に正規化
        normalized_gait_cycles_r = []
        for cycle_idx, (ic_start, ic_l, to, ic_end) in enumerate(gait_cycles_r):
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
                'ic_l': ic_l,
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
        for cycle_idx, (ic_start, ic_r, to, ic_end) in enumerate(gait_cycles_l):
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
                'ic_l': ic_l,
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
            fig, axes = plt.subplots(3, 1, figsize=(10, 15))
            
            # 股関節
            axes[0].plot(normalized_percentage, mean_cycle_r['R_Hip_FlEx_mean'], 'b-', label='Mean')
            axes[0].fill_between(normalized_percentage,
                                mean_cycle_r['R_Hip_FlEx_mean'] - mean_cycle_r['R_Hip_FlEx_std'],
                                mean_cycle_r['R_Hip_FlEx_mean'] + mean_cycle_r['R_Hip_FlEx_std'],
                                alpha=0.3, color='b')
            axes[0].axvline(x=np.mean([c['stance_phase_percentage'] for c in normalized_gait_cycles_r]),
                          color='r', linestyle='--', label='Toe Off')
            axes[0].set_ylabel('Hip Angle [deg]')
            axes[0].set_ylim(-40,50)
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
            axes[1].set_ylim(-10,75)
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
            axes[2].set_ylim(-30,50)
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
            axes[0].set_ylim(-30,30)
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
            axes[1].set_ylim(-30,30)
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
            axes[2].set_ylim(-30,30)
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
            axes[0].set_ylim(-30,30)
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
            axes[1].set_ylim(-30,30)
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
            axes[2].set_ylim(-30,30)
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
            axes[0].set_ylim(-40,50)
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
            axes[1].set_ylim(-10,75)
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
            axes[2].set_ylim(-30,50)
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
            axes[0].set_ylim(-30,30)
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
            axes[1].set_ylim(-30,30)
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
            axes[2].set_ylim(-30,30)
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
            axes[0].set_ylim(-30,30)
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
            axes[1].set_ylim(-30,30)
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
            axes[2].set_ylim(-30,30)
            axes[2].set_title('Left Ankle Adduction/Abduction')
            axes[2].legend()
            axes[2].grid(True)
            plt.tight_layout()
            plt.savefig(csv_path_dir / f"gait_cycle_AdAb_L_{csv_path.stem}.png")
            plt.close()
            
            
        ###########################################
        # 歩行パラメータの計算（Mocap）
        ###########################################
        def calculate_gait_parameters(gait_cycles, hip_array, rhee, lhee, side, sampling_freq=100):
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
            
            for cycle_idx, (ic_start, ic_opp, to, ic_end) in enumerate(gait_cycles):
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
                
                # ストライド長，ステップ長，歩隔の算出方法参考：https://www.sciencedirect.com/science/article/pii/S0966636205000081?via%3Dihub
                # ストライド長 [m] 
                # 初期接地から次の初期接地までの踵の移動距離
                if side == 'R':
                    stride_length = np.linalg.norm([rhee[ic_end] - rhee[ic_start]])
                else:
                    stride_length = np.linalg.norm([lhee[ic_end] - lhee[ic_start]])
                
                # 歩隔 [m]
                # 初期接地時のX座標の差を計算
                if side == 'R':
                    v_stride = rhee[ic_end] - rhee[ic_start]
                    v_step = lhee[ic_opp] - rhee[ic_start]
                    step_width = abs(np.cross(v_stride, v_step)) / np.linalg.norm(v_stride)
                else:
                    v_stride = lhee[ic_end] - lhee[ic_start]
                    v_step = rhee[ic_opp] - lhee[ic_start]
                    step_width = abs(np.cross(v_stride, v_step)) / np.linalg.norm(v_stride)
                    
                # ステップ長 *対側のパラメータなので注意* [m]
                if side == 'R':
                    step_length_l = np.linalg.norm([lhee[ic_opp] - rhee[ic_start]])
                else:
                    step_length_r = np.linalg.norm([rhee[ic_opp] - lhee[ic_start]])
                
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
                    'step_length_opposite': step_length_l if side == 'R' else step_length_r,
                }
                gait_params.append(params)
            
            return gait_params
        
        # 右足と左足の歩行パラメータを計算
        gait_params_r = calculate_gait_parameters(gait_cycles_r, hip_array, rhee, lhee, side='R', sampling_freq=100)
        gait_params_l = calculate_gait_parameters(gait_cycles_l, hip_array, rhee, lhee, side='L', sampling_freq=100)
        
        # step_length_oppositeはそれぞれ対側のパラメータなので入れ替える
        for params_r, params_l in zip(gait_params_r, gait_params_l):
            params_r['step_length'] = params_l['step_length_opposite']
            params_l['step_length'] = params_r['step_length_opposite']
            del params_r['step_length_opposite']
            del params_l['step_length_opposite']
        
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
                    'stride_length', 'step_width', 'step_length', 'stance_phase_percent', 'swing_phase_percent']
            
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

if __name__ == "__main__":
    main()