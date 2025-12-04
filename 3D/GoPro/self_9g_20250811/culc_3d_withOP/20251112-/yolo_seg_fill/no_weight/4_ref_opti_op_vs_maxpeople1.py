import pandas as pd
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample
import json
import m_opti as opti
import m_openpose as op

# グラフのフォントサイズを全体的に大きく設定
plt.rcParams.update({
    'font.size': 18,              # 基本フォントサイズ
    'axes.titlesize': 20,         # タイトル
    'axes.labelsize': 18,         # 軸ラベル
    'xtick.labelsize': 16,        # x軸目盛り
    'ytick.labelsize': 16,        # y軸目盛り
    'legend.fontsize': 14,        # 凡例
    'figure.titlesize': 22,       # 図タイトル
    'lines.linewidth': 2.5,       # 線の太さ
    'axes.linewidth': 1.5,        # 軸の太さ
    'xtick.major.width': 1.5,     # x軸目盛りの太さ
    'ytick.major.width': 1.5,     # y軸目盛りの太さ
    'xtick.major.size': 6,        # x軸目盛りの長さ
    'ytick.major.size': 6,        # y軸目盛りの長さ
    'figure.autolayout': True,    # 自動レイアウト調整
})

def main():
    # csv_path_dir = Path(r"G:\gait_pattern\BR9G_shuron\sub0\thera0-16\mocap")
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

    csv_paths = list(csv_path_dir.glob("[0-9]*_[0-9]*_*[0-9].csv"))

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
        
        if 'R_Hip_InEx' in angle_df.columns:
            for frame in angle_df.index:
                angle_df.loc[frame, 'R_Hip_InEx'] = angle_df.at[frame, 'R_Hip_InEx'] #外旋ex+, 内旋in-
        if 'L_Hip_InEx' in angle_df.columns:
            for frame in angle_df.index:
                angle_df.loc[frame, 'L_Hip_InEx'] = - angle_df.at[frame, 'L_Hip_InEx'] # 外旋ex+, 内旋in-
        
        
        if 'R_Hip_AdAb' in angle_df.columns:
            for frame in angle_df.index:
                angle_df.loc[frame, 'R_Hip_AdAb'] = angle_df.at[frame, 'R_Hip_AdAb']  # 外転ab+, 内転ad-
        if 'L_Hip_AdAb' in angle_df.columns:
            for frame in angle_df.index:
                angle_df.loc[frame, 'L_Hip_AdAb'] = - angle_df.at[frame, 'L_Hip_AdAb'] # 外転ab+, 内転ad-
        
        
        
                
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
            print(f"正規化した右足サイクルデータ {cycle_data['cycle_index']+1}/{len(normalized_gait_cycles_r)}を保存: normalized_cycle_R_{cycle_data['cycle_index']}_{csv_path.stem}.csv")
        
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
            print(f"正規化した左足サイクルデータ {cycle_data['cycle_index']+1}/{len(normalized_gait_cycles_l)}を保存: normalized_cycle_L_{cycle_data['cycle_index']}_{csv_path.stem}.csv")
        
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
        
        # step_length_oppositeを入れ替える
        # 右足の各サイクルに対して処理
        for i, params_r in enumerate(gait_params_r):
            if i < len(gait_params_l):
                # 対応する左足サイクルがある場合
                params_r['step_length'] = gait_params_l[i]['step_length_opposite']
            else:
                # 対応する左足サイクルがない場合はNoneに設定
                params_r['step_length'] = None

        # 左足の各サイクルに対して処理
        for i, params_l in enumerate(gait_params_l):
            if i < len(gait_params_r):
                # 対応する右足サイクルがある場合
                params_l['step_length'] = gait_params_r[i]['step_length_opposite']
            else:
                # 対応する右足サイクルがない場合はNoneに設定
                params_l['step_length'] = None

        # step_length_oppositeを削除
        for params in gait_params_r:
            del params['step_length_opposite']
        for params in gait_params_l:
            del params['step_length_opposite']
            
        print(f"gait_paramsのキー例: {gait_params_r[0].keys()}")
        
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
                # None値を除外してリストを作成
                r_values = [p[key] for p in params_r if p.get(key) is not None]
                l_values = [p[key] for p in params_l if p.get(key) is not None]
                
                # 有効な値がない場合はスキップ
                if len(r_values) == 0 or len(l_values) == 0:
                    print(f"Warning: No valid values for {key}. Skipping.")
                    continue
                
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

        openpose_npz_path = csv_path_dir.parent / "3d_kp_data_openpose_yoloseg.npz"
        if not openpose_npz_path.exists():
            print(f"OpenPose3Dデータが見つかりません: {openpose_npz_path}")
            return
    
        print(f"OpenPose3Dデータとの比較を実行: {openpose_npz_path}")
        openpose_data = np.load(openpose_npz_path)
        print(f"openpose_data keys: {openpose_data.files}")
        op_frame = openpose_data['frame']
        op_raw_data = openpose_data['raw']  # shape: (num_frames, num_joints, 3)
        op_filt_data = openpose_data['butter_filt']  # shape: (num_frames, num_joints, 3)
        op_conf = openpose_data['conf']  # shape: (num_frames, num_joints)
        print(f"op_frame: {op_frame[:5]} ...{op_frame[-5:]}")
        print(f"op_raw_data shape: {op_raw_data.shape}")
        
        ###########################################
        # タイミング合わせ  股関節中心が任意の位置を通るタイミングでフレーム調整
        ###########################################
        # base_point = int(0)
        # # opti用
        # hip_z_opti = (((rasi + lasi) / 2 + (rpsi + lpsi) / 2) / 2)[:, 2]
        # # hip_z_optiが0より大きくなった最初のフレームを取得
        # base_passing_frame = np.argmax(hip_z_opti > base_point) + start_frame
        # print(f"100Hz Opti計測開始から原点通過までのフレーム数: {base_passing_frame}")
        
        # # OpenPose用
        # hip_z_op = op_filt_data[:, 8, 2]
        # # hip_zが0より大きくなった最初のフレームを取得
        # base_passing_idx_op = np.argmax(hip_z_op > base_point)
        # base_passing_frame_op = op_frame[base_passing_idx_op]
        # print(f"60Hz OpenPoseトリミング開始から原点通過までのフレーム数: {base_passing_frame_op}")
        
        # mc_frame_offset = base_passing_frame - base_passing_frame_op / 0.6  # 100Hzに変換
        # print(f"MC開始からGoProトリミング開始までのフレームオフセット 100Hz: {mc_frame_offset}")
        
        mc_frame_offset = 602  # 手動設定(なにもなしで介助条件で一人検出した場合の値を使用)
        
        
        
        gait_cycles_r_op = []
        gait_cycles_l_op = []
        
        for ic_rel, ic_opp_rel, to_rel, ic_next_rel in gait_cycles_r_abs:
            ic_abs = (ic_rel - mc_frame_offset) * 0.6
            ic_opp_abs = (ic_opp_rel - mc_frame_offset) * 0.6
            to_abs = (to_rel - mc_frame_offset) * 0.6
            ic_next_abs = (ic_next_rel - mc_frame_offset) * 0.6
            gait_cycles_r_op.append([ic_abs, ic_opp_abs, to_abs, ic_next_abs])
        for ic_rel, ic_opp_rel, to_rel, ic_next_rel in gait_cycles_l_abs:
            ic_abs = (ic_rel - mc_frame_offset) * 0.6
            ic_opp_abs = (ic_opp_rel - mc_frame_offset) * 0.6
            to_abs = (to_rel - mc_frame_offset) * 0.6
            ic_next_abs = (ic_next_rel - mc_frame_offset) * 0.6
            gait_cycles_l_op.append([ic_abs, ic_opp_abs, to_abs, ic_next_abs])
            
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
        pel_up_bec_op = neck_op - midhip_op  #上向き
        thigh_r_op = rknee_op - rhip_op  #下向き
        shank_r_op = rankle_op - rknee_op  #下向き 
        foot_r_op = rtoe_op - rhee_op  #前向き
        thigh_l_op = lknee_op - lhip_op  
        shank_l_op = lankle_op - lknee_op
        foot_l_op = ltoe_op - lhee_op

        # 関節角度計算
        # 股関節、膝関節、足関節の屈曲伸展角度
        n_axis = rhip_op - lhip_op  # 左股関節から右股関節へのベクトル
        rhip_flex_op = op.culc_angle_all_frames(pel_up_bec_op, thigh_r_op, n_axis, degrees=True, angle_type='hip')
        lhip_flex_op = op.culc_angle_all_frames(pel_up_bec_op, thigh_l_op, n_axis, degrees=True, angle_type='hip')
        rknee_flex_op = op.culc_angle_all_frames(thigh_r_op, shank_r_op, n_axis, degrees=True, angle_type='knee')
        lknee_flex_op = op.culc_angle_all_frames(thigh_l_op, shank_l_op, n_axis, degrees=True, angle_type='knee')
        rankle_pldo_op = op.culc_angle_all_frames(shank_r_op, foot_r_op, n_axis, degrees=True, angle_type='ankle')
        lankle_pldo_op = op.culc_angle_all_frames(shank_l_op, foot_l_op, n_axis, degrees=True, angle_type='ankle')
        # 股関節の内転外転
        n_axis_adab = np.cross(pel_up_bec_op, n_axis)  # pel_vecとn_axisの外積を計算して直交ベクトルを取得 進行方向向き
        # n_axis_adab = np.zeros_like(pel_up_bec_op)
        # n_axis_adab[:, 2] = 1.0  # 全フレームでZ軸方向 [0, 0, 1]
        rhip_adab_op = op.culc_angle_all_frames(pel_up_bec_op, thigh_r_op, n_axis_adab, degrees=True, angle_type='hip_adab')
        lhip_adab_op = op.culc_angle_all_frames(pel_up_bec_op, thigh_l_op, n_axis_adab, degrees=True, angle_type='hip_adab')
        # 股関節の内旋外旋
        # pel_vecを内旋外旋の回転軸とする
        rhip_inex_op = op.culc_angle_all_frames(thigh_r_op, n_axis_adab, pel_up_bec_op, degrees=True, angle_type='hip_inex')
        lhip_inex_op = op.culc_angle_all_frames(thigh_l_op, n_axis_adab, pel_up_bec_op, degrees=True, angle_type='hip_inex')

        # 右足の歩行周期ごとに関節角度を100%に正規化
        normalized_gait_cycles_r_op = []
        for cycle_idx, (ic_start, ic_opp, to, ic_end) in enumerate(gait_cycles_r_op):
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
        
        for cycle_idx, (ic_start, ic_opp, to, ic_end) in enumerate(gait_cycles_l_op):
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
        def calculate_gait_parameters_op(gait_cycles, midhip_op, rhee_op, lhee_op, side, sampling_freq=60):
            """
            OpenPoseデータから歩行パラメータを計算
            """
            gait_params = []
            
            for cycle_idx, (ic_start, ic_opp, to, ic_end) in enumerate(gait_cycles):
                # 整数インデックスに変換
                ic_start_idx = int(np.round(ic_start))
                ic_opp_idx = int(np.round(ic_opp))
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
                # 初期接地から次の初期接地までの踵の移動距離
                if side == 'R':
                    stride_length = np.linalg.norm([rhee_op[ic_end_idx] - rhee_op[ic_start_idx]]) / 1000 # mmからmに変換
                else:
                    stride_length = np.linalg.norm([lhee_op[ic_end_idx] - lhee_op[ic_start_idx]]) / 1000 # mmからmに変換    
                    
                # 歩隔 [m]
                # 初期接地時のX座標の差を計算
                if side == 'R':
                    v_stride = rhee_op[ic_end_idx] - rhee_op[ic_start_idx]
                    v_step = lhee_op[ic_opp_idx] - rhee_op[ic_start_idx]
                    step_width = abs(np.cross(v_stride, v_step)) / np.linalg.norm(v_stride) / 1000 # mmからmに変換
                else:
                    v_stride = lhee_op[ic_end_idx] - lhee_op[ic_start_idx]
                    v_step = rhee_op[ic_opp_idx] - lhee_op[ic_start_idx]
                    step_width = abs(np.cross(v_stride, v_step)) / np.linalg.norm(v_stride) / 1000 # mmからmに変換
                    
                # ステップ長 *対側のパラメータなので注意* [m]
                if side == 'R':
                    step_length_l = np.linalg.norm([lhee_op[ic_opp_idx] - rhee_op[ic_start_idx]]) / 1000 # mmからmに変換
                else:
                    step_length_r = np.linalg.norm([rhee_op[ic_opp_idx] - lhee_op[ic_start_idx]]) / 1000 # mmからmに変換
                
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
        
        # OpenPoseの歩行パラメータを計算
        gait_params_r_op = calculate_gait_parameters_op(gait_cycles_r_op, midhip_op, rhee_op, lhee_op, side = "R", sampling_freq=60)
        gait_params_l_op = calculate_gait_parameters_op(gait_cycles_l_op, midhip_op, rhee_op, lhee_op, side = "L", sampling_freq=60)
        
                # step_length_oppositeはそれぞれ対側のパラメータなので入れ替える
        for params_r, params_l in zip(gait_params_r_op, gait_params_l_op):
            params_r['step_length'] = params_l['step_length_opposite']
            params_l['step_length'] = params_r['step_length_opposite']
            del params_r['step_length_opposite']
            del params_l['step_length_opposite']
            
        print(f"OpenPose 右足歩行パラメータ: {gait_params_r_op}")
        print(f"OpenPose 左足歩行パラメータ: {gait_params_l_op}")
        
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
                ax.set_xlabel('Foot Side')
                ax.set_ylabel(param_label)
                ax.set_title(f'{param_label}\n(Diff R: {stats["Diff_R"]:.3f}, Diff L: {stats["Diff_L"]:.3f})', 
                           fontsize=13)
                ax.set_xticks(x)
                ax.set_xticklabels(['Right', 'Left'])
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                # 値をバーの上に表示
                for bar in bars1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom')
                for bar in bars2:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"comparison_gait_parameters_mean_std_{csv_path.stem}.png", dpi=300)
            plt.close()
            
            print(f"\n平均と標準偏差の比較グラフを保存しました: comparison_gait_parameters_mean_std_{csv_path.stem}.png")




            # maxpeople1のデータを読み込む
            max_people1_dir = csv_path_dir.parent / "OpenPose3D_results_maxpeople1"
            normalized_gait_cycles_r_op_max1_csv_path = max_people1_dir / f"normalized_cycle_R_mean_{csv_path.stem}_OpenPose.csv"
            normalized_gait_cycles_l_op_max1_csv_path = max_people1_dir / f"normalized_cycle_L_mean_{csv_path.stem}_OpenPose.csv"
            
            # maxpeople1のデータが存在するか確認
            has_max1_data = False
            if max_people1_dir.exists():
                if normalized_gait_cycles_r_op_max1_csv_path.exists() and normalized_gait_cycles_l_op_max1_csv_path.exists():
                    has_max1_data = True
                    print(f"\nmaxpeople1のデータを読み込み中...")
                    
                    # 関節角度の平均データを読み込み
                    mean_cycle_r_op_max1 = pd.read_csv(normalized_gait_cycles_r_op_max1_csv_path)
                    mean_cycle_l_op_max1 = pd.read_csv(normalized_gait_cycles_l_op_max1_csv_path)
                    
                    # MAEデータを読み込み
                    mae_both_max1_path = max_people1_dir / f"MAE_gait_parameters_both_{csv_path.stem}.csv"
                    mae_summary_r_max1_path = max_people1_dir / f"MAE_summary_R_{csv_path.stem}.csv"
                    mae_summary_l_max1_path = max_people1_dir / f"MAE_summary_L_{csv_path.stem}.csv"
                    
                    if mae_both_max1_path.exists():
                        mae_gait_params_both_max1 = pd.read_csv(mae_both_max1_path)
                        print(f"歩行パラメータMAE (maxpeople1):")
                        print(mae_gait_params_both_max1)
                    
                    if mae_summary_r_max1_path.exists():
                        mae_summary_r_max1 = pd.read_csv(mae_summary_r_max1_path)
                        print(f"\n右足関節角度MAE (maxpeople1):")
                        print(mae_summary_r_max1)
                    
                    if mae_summary_l_max1_path.exists():
                        mae_summary_l_max1 = pd.read_csv(mae_summary_l_max1_path)
                        print(f"\n左足関節角度MAE (maxpeople1):")
                        print(mae_summary_l_max1)
                else:
                    print(f"\nmaxpeople1のデータが見つかりません")
            else:
                print(f"\nmaxpeople1ディレクトリが見つかりません: {max_people1_dir}")




            ###########################################
            # 歩行パラメータのMAE計算（Mocapを正解として）
            ###########################################
            def calculate_gait_param_mae(params_mocap, params_op, param_key):
                """
                各歩行周期ごとにMocapを正解としてOpenPoseのMAEを計算
                
                Parameters:
                -----------
                params_mocap : list of dict
                    Mocapの歩行パラメータ
                params_op : list of dict
                    OpenPoseの歩行パラメータ
                param_key : str
                    パラメータのキー
                    
                Returns:
                --------
                mae : float
                    MAE値
                errors : list
                    各サイクルの絶対誤差リスト
                """
                n_cycles = min(len(params_mocap), len(params_op))
                errors = []
                
                for i in range(n_cycles):
                    mocap_val = params_mocap[i].get(param_key)
                    op_val = params_op[i].get(param_key)
                    
                    if mocap_val is not None and op_val is not None:
                        errors.append(abs(mocap_val - op_val))
                
                mae = np.mean(errors) if len(errors) > 0 else np.nan
                return mae, errors
            
            # 計算対象のパラメータ
            gait_param_keys = [
                ('gait_speed', 'Gait Speed', 'm/s'),
                ('stride_length', 'Stride Length', 'm'),
                ('step_length', 'Step Length', 'm'),
                ('step_width', 'Step Width', 'm'),
                ('cycle_duration', 'Cycle Duration', 's'),
                ('stance_duration', 'Stance Duration', 's'),
                ('swing_duration', 'Swing Duration', 's'),
                ('stance_phase_percent', 'Stance Phase', '%'),
                ('swing_phase_percent', 'Swing Phase', '%'),
            ]
            
            # 右足のMAE計算
            mae_results_r = []
            for param_key, param_label, unit in gait_param_keys:
                mae_val, errors = calculate_gait_param_mae(gait_params_r, gait_params_r_op, param_key)
                mae_results_r.append({
                    'Parameter': param_label,
                    'Unit': unit,
                    'MAE': mae_val,
                    'N_cycles': len(errors)
                })
            
            # 左足のMAE計算
            mae_results_l = []
            for param_key, param_label, unit in gait_param_keys:
                mae_val, errors = calculate_gait_param_mae(gait_params_l, gait_params_l_op, param_key)
                mae_results_l.append({
                    'Parameter': param_label,
                    'Unit': unit,
                    'MAE': mae_val,
                    'N_cycles': len(errors)
                })
            
            # データフレームに変換して保存
            mae_gait_params_r_df = pd.DataFrame(mae_results_r)
            mae_gait_params_l_df = pd.DataFrame(mae_results_l)
            
            mae_gait_params_r_df.to_csv(op_result_dir / f"MAE_gait_parameters_R_{csv_path.stem}.csv", index=False)
            mae_gait_params_l_df.to_csv(op_result_dir / f"MAE_gait_parameters_L_{csv_path.stem}.csv", index=False)
            
            print(f"\n歩行パラメータMAE（Mocapを正解として）:")
            print(f"\n右足:")
            print(mae_gait_params_r_df.to_string(index=False))
            print(f"\n左足:")
            print(mae_gait_params_l_df.to_string(index=False))
            
            # 両足の平均MAEも計算
            mae_both_sides = []
            for param_key, param_label, unit in gait_param_keys:
                mae_r, _ = calculate_gait_param_mae(gait_params_r, gait_params_r_op, param_key)
                mae_l, _ = calculate_gait_param_mae(gait_params_l, gait_params_l_op, param_key)
                mae_avg = np.nanmean([mae_r, mae_l])
                mae_both_sides.append({
                    'Parameter': param_label,
                    'Unit': unit,
                    'MAE_Right': mae_r,
                    'MAE_Left': mae_l,
                    'MAE_Average': mae_avg
                })
            
            mae_both_df = pd.DataFrame(mae_both_sides)
            mae_both_df.to_csv(op_result_dir / f"MAE_gait_parameters_both_{csv_path.stem}.csv", index=False)
            
            print(f"\n両足の歩行パラメータMAE:")
            print(mae_both_df.to_string(index=False))
            
            # 棒グラフで可視化
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 距離系パラメータ（m）
            distance_params = ['Gait Speed', 'Stride Length', 'Step Length', 'Step Width']
            distance_mae_r = [r['MAE'] for r in mae_results_r if r['Parameter'] in distance_params]
            distance_mae_l = [r['MAE'] for r in mae_results_l if r['Parameter'] in distance_params]
            
            x = np.arange(len(distance_params))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, distance_mae_r, width, label='Right', color='blue', alpha=0.7)
            axes[0, 0].bar(x + width/2, distance_mae_l, width, label='Left', color='red', alpha=0.7)
            axes[0, 0].set_ylabel('MAE [m or m/s]')
            axes[0, 0].set_title('Distance/Speed Parameters MAE')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(distance_params, rotation=15, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            # 時間系パラメータ（s）
            time_params = ['Cycle Duration', 'Stance Duration', 'Swing Duration']
            time_mae_r = [r['MAE'] for r in mae_results_r if r['Parameter'] in time_params]
            time_mae_l = [r['MAE'] for r in mae_results_l if r['Parameter'] in time_params]
            
            x = np.arange(len(time_params))
            
            axes[0, 1].bar(x - width/2, time_mae_r, width, label='Right', color='blue', alpha=0.7)
            axes[0, 1].bar(x + width/2, time_mae_l, width, label='Left', color='red', alpha=0.7)
            axes[0, 1].set_ylabel('MAE [s]')
            axes[0, 1].set_title('Time Parameters MAE')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(time_params, rotation=15, ha='right')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # 割合系パラメータ（%）
            percent_params = ['Stance Phase', 'Swing Phase']
            percent_mae_r = [r['MAE'] for r in mae_results_r if r['Parameter'] in percent_params]
            percent_mae_l = [r['MAE'] for r in mae_results_l if r['Parameter'] in percent_params]
            
            x = np.arange(len(percent_params))
            
            axes[1, 0].bar(x - width/2, percent_mae_r, width, label='Right', color='blue', alpha=0.7)
            axes[1, 0].bar(x + width/2, percent_mae_l, width, label='Left', color='red', alpha=0.7)
            axes[1, 0].set_ylabel('MAE [%]')
            axes[1, 0].set_title('Phase Parameters MAE')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(percent_params)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # 全パラメータの両足平均MAE（正規化して比較）
            # 各パラメータの平均値で正規化
            normalized_mae = []
            param_names = []
            for result in mae_both_sides:
                param_key_match = [k for k, l, u in gait_param_keys if l == result['Parameter']][0]
                # Mocapの平均値を取得
                r_vals = [p[param_key_match] for p in gait_params_r if p.get(param_key_match) is not None]
                l_vals = [p[param_key_match] for p in gait_params_l if p.get(param_key_match) is not None]
                mean_val = np.mean(r_vals + l_vals) if len(r_vals + l_vals) > 0 else 1
                
                # 正規化MAE（%）
                if mean_val != 0:
                    norm_mae = (result['MAE_Average'] / abs(mean_val)) * 100
                else:
                    norm_mae = 0
                normalized_mae.append(norm_mae)
                param_names.append(result['Parameter'])
            
            axes[1, 1].barh(param_names, normalized_mae, color='green', alpha=0.7)
            axes[1, 1].set_xlabel('Normalized MAE [%]')
            axes[1, 1].set_title('Normalized MAE (MAE / Mean × 100)')
            axes[1, 1].grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"MAE_gait_parameters_comparison_{csv_path.stem}.png", dpi=300)
            plt.close()
            
            print(f"\n歩行パラメータMAEグラフを保存しました: MAE_gait_parameters_comparison_{csv_path.stem}.png")
            
            
            
            
            



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
            ax.set_ylabel('Symmetry Index [%]')
            ax.set_title('Stance Phase Symmetry Index Comparison\n(Lower values indicate better symmetry)')
            ax.set_xticks(x)
            ax.set_xticklabels(['Stance Phase'])
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xlim(-0.5, 0.5)
            
            # 値をバーの上に表示
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontweight='bold')
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontweight='bold')
            
            # 参考線（完全な対称性）
            ax.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect Symmetry')
            
            # 差分を表示
            ax.text(0, max(si_mocap_values[0], si_op_values[0]) * 1.15, 
                   f'Difference: {si_comparison["Difference"]:.1f}%',
                   ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
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
        
        # MAEを計算する関数
        def calculate_mae(data1, data2):
            """MAEを計算"""
            return np.mean(np.abs(data1 - data2))
        
        # 各歩行周期ごとのMAEを計算し、全周期の平均を取る関数
        def calculate_mae_all_cycles(cycles_mocap, cycles_op, joint_key):
            """
            全歩行周期でMAEを計算
            
            Parameters:
            -----------
            cycles_mocap : list of dict
                Mocapの正規化された歩行周期データ
            cycles_op : list of dict
                OpenPoseの正規化された歩行周期データ
            joint_key : str
                関節角度のキー（例: 'R_Hip_FlEx'）
                
            Returns:
            --------
            mae : float
                全周期でのMAE
            """
            all_errors = []
            n_cycles = min(len(cycles_mocap), len(cycles_op))
            
            for i in range(n_cycles):
                mocap_data = cycles_mocap[i][joint_key]
                op_data = cycles_op[i][joint_key]
                # 各フレームでの絶対誤差を計算
                errors = np.abs(mocap_data - op_data)
                all_errors.extend(errors)
            
            return np.mean(all_errors) if len(all_errors) > 0 else np.nan
        
        # 右足MAEの計算
        mae_rhip = calculate_mae_all_cycles(normalized_gait_cycles_r, normalized_gait_cycles_r_op, 'R_Hip_FlEx')
        mae_rknee = calculate_mae_all_cycles(normalized_gait_cycles_r, normalized_gait_cycles_r_op, 'R_Knee_FlEx')
        mae_rankle = calculate_mae_all_cycles(normalized_gait_cycles_r, normalized_gait_cycles_r_op, 'R_Ankle_PlDo')
        mae_rhip_inex = calculate_mae_all_cycles(normalized_gait_cycles_r, normalized_gait_cycles_r_op, 'R_Hip_InEx')
        mae_rhip_adab = calculate_mae_all_cycles(normalized_gait_cycles_r, normalized_gait_cycles_r_op, 'R_Hip_AdAb')
        print(f"\n右足関節角度の全歩行周期MAE:")
        print(f"股関節屈曲伸展 MAE: {mae_rhip:.2f}°")
        print(f"膝関節屈曲伸展 MAE: {mae_rknee:.2f}°")
        print(f"足関節背屈底屈 MAE: {mae_rankle:.2f}°")
        print(f"股関節内旋外旋 MAE: {mae_rhip_inex:.2f}°")
        print(f"股関節内転外転 MAE: {mae_rhip_adab:.2f}°")
        
        # 左足MAEの計算
        mae_lhip = calculate_mae_all_cycles(normalized_gait_cycles_l, normalized_gait_cycles_l_op, 'L_Hip_FlEx')
        mae_lknee = calculate_mae_all_cycles(normalized_gait_cycles_l, normalized_gait_cycles_l_op, 'L_Knee_FlEx')
        mae_lankle = calculate_mae_all_cycles(normalized_gait_cycles_l, normalized_gait_cycles_l_op, 'L_Ankle_PlDo')
        mae_lhip_inex = calculate_mae_all_cycles(normalized_gait_cycles_l, normalized_gait_cycles_l_op, 'L_Hip_InEx')
        mae_lhip_adab = calculate_mae_all_cycles(normalized_gait_cycles_l, normalized_gait_cycles_l_op, 'L_Hip_AdAb')
        print(f"\n左足関節角度の全歩行周期MAE:")    
        print(f"股関節屈曲伸展 MAE: {mae_lhip:.2f}°")
        print(f"膝関節屈曲伸展 MAE: {mae_lknee:.2f}°")
        print(f"足関節背屈底屈 MAE: {mae_lankle:.2f}°")
        print(f"股関節内旋外旋 MAE: {mae_lhip_inex:.2f}°")
        print(f"股関節内転外転 MAE: {mae_lhip_adab:.2f}°")
        
        
        
        ###########################################
        # 各歩行周期ごとの絶対誤差プロット
        ###########################################
        n_cycles_r = min(len(normalized_gait_cycles_r), len(normalized_gait_cycles_r_op))
        n_cycles_l = min(len(normalized_gait_cycles_l), len(normalized_gait_cycles_l_op))
        
        # 右足の絶対誤差プロット
        if n_cycles_r > 0:
            print(f"右足の各歩行周期ごとの絶対誤差プロットを作成中...")
            
            # 屈曲伸展・背屈底屈の絶対誤差
            fig, axes = plt.subplots(3, 1, figsize=(12, 14))
            colors = plt.cm.tab10(np.linspace(0, 1, n_cycles_r))
            
            for cycle_idx in range(n_cycles_r):
                # 股関節屈曲伸展の絶対誤差
                error_hip = np.abs(normalized_gait_cycles_r[cycle_idx]['R_Hip_FlEx'] - 
                                   normalized_gait_cycles_r_op[cycle_idx]['R_Hip_FlEx'])
                axes[0].plot(normalized_percentage, error_hip, color=colors[cycle_idx], 
                            label=f'Cycle {cycle_idx+1}', linewidth=1.5)
                
                # 膝関節屈曲伸展の絶対誤差
                error_knee = np.abs(normalized_gait_cycles_r[cycle_idx]['R_Knee_FlEx'] - 
                                    normalized_gait_cycles_r_op[cycle_idx]['R_Knee_FlEx'])
                axes[1].plot(normalized_percentage, error_knee, color=colors[cycle_idx], 
                            label=f'Cycle {cycle_idx+1}', linewidth=1.5)
                
                # 足関節背屈底屈の絶対誤差
                error_ankle = np.abs(normalized_gait_cycles_r[cycle_idx]['R_Ankle_PlDo'] - 
                                     normalized_gait_cycles_r_op[cycle_idx]['R_Ankle_PlDo'])
                axes[2].plot(normalized_percentage, error_ankle, color=colors[cycle_idx], 
                            label=f'Cycle {cycle_idx+1}', linewidth=1.5)
            
            # グラフの装飾
            axes[0].set_ylabel('Absolute Error [deg]')
            axes[0].set_title(f'Right Hip Flexion/Extension - Absolute Error per Cycle (MAE: {mae_rhip:.2f}°)')
            axes[0].legend(loc='upper right')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, None)
            
            axes[1].set_ylabel('Absolute Error [deg]')
            axes[1].set_title(f'Right Knee Flexion/Extension - Absolute Error per Cycle (MAE: {mae_rknee:.2f}°)')
            axes[1].legend(loc='upper right')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, None)
            
            axes[2].set_xlabel('Gait Cycle [%]')
            axes[2].set_ylabel('Absolute Error [deg]')
            axes[2].set_title(f'Right Ankle Plantarflexion/Dorsiflexion - Absolute Error per Cycle (MAE: {mae_rankle:.2f}°)')
            axes[2].legend(loc='upper right')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim(0, None)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"absolute_error_FlEx_R_{csv_path.stem}.png", dpi=300)
            plt.close()
            
            # 股関節内旋外旋・内転外転の絶対誤差
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            mae_rhip_inex = calculate_mae_all_cycles(normalized_gait_cycles_r, normalized_gait_cycles_r_op, 'R_Hip_InEx')
            mae_rhip_adab = calculate_mae_all_cycles(normalized_gait_cycles_r, normalized_gait_cycles_r_op, 'R_Hip_AdAb')
            
            for cycle_idx in range(n_cycles_r):
                # 股関節内旋外旋の絶対誤差
                error_inex = np.abs(normalized_gait_cycles_r[cycle_idx]['R_Hip_InEx'] - 
                                    normalized_gait_cycles_r_op[cycle_idx]['R_Hip_InEx'])
                axes[0].plot(normalized_percentage, error_inex, color=colors[cycle_idx], 
                            label=f'Cycle {cycle_idx+1}', linewidth=1.5)
                
                # 股関節内転外転の絶対誤差
                error_adab = np.abs(normalized_gait_cycles_r[cycle_idx]['R_Hip_AdAb'] - 
                                    normalized_gait_cycles_r_op[cycle_idx]['R_Hip_AdAb'])
                axes[1].plot(normalized_percentage, error_adab, color=colors[cycle_idx], 
                            label=f'Cycle {cycle_idx+1}', linewidth=1.5)
            
            axes[0].set_ylabel('Absolute Error [deg]')
            axes[0].set_title(f'Right Hip Internal/External Rotation - Absolute Error per Cycle (MAE: {mae_rhip_inex:.2f}°)')
            axes[0].legend(loc='upper right')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, None)
            
            axes[1].set_xlabel('Gait Cycle [%]')
            axes[1].set_ylabel('Absolute Error [deg]')
            axes[1].set_title(f'Right Hip Adduction/Abduction - Absolute Error per Cycle (MAE: {mae_rhip_adab:.2f}°)')
            axes[1].legend(loc='upper right')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, None)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"absolute_error_InEx_AdAb_R_{csv_path.stem}.png", dpi=300)
            plt.close()
            
            print(f"右足絶対誤差プロットを保存しました")
        
        # 左足の絶対誤差プロット
        if n_cycles_l > 0:
            print(f"左足の各歩行周期ごとの絶対誤差プロットを作成中...")
            
            # 屈曲伸展・背屈底屈の絶対誤差
            fig, axes = plt.subplots(3, 1, figsize=(12, 14))
            colors = plt.cm.tab10(np.linspace(0, 1, n_cycles_l))
            
            for cycle_idx in range(n_cycles_l):
                # 股関節屈曲伸展の絶対誤差
                error_hip = np.abs(normalized_gait_cycles_l[cycle_idx]['L_Hip_FlEx'] - 
                                   normalized_gait_cycles_l_op[cycle_idx]['L_Hip_FlEx'])
                axes[0].plot(normalized_percentage, error_hip, color=colors[cycle_idx], 
                            label=f'Cycle {cycle_idx+1}', linewidth=1.5)
                
                # 膝関節屈曲伸展の絶対誤差
                error_knee = np.abs(normalized_gait_cycles_l[cycle_idx]['L_Knee_FlEx'] - 
                                    normalized_gait_cycles_l_op[cycle_idx]['L_Knee_FlEx'])
                axes[1].plot(normalized_percentage, error_knee, color=colors[cycle_idx], 
                            label=f'Cycle {cycle_idx+1}', linewidth=1.5)
                
                # 足関節背屈底屈の絶対誤差
                error_ankle = np.abs(normalized_gait_cycles_l[cycle_idx]['L_Ankle_PlDo'] - 
                                     normalized_gait_cycles_l_op[cycle_idx]['L_Ankle_PlDo'])
                axes[2].plot(normalized_percentage, error_ankle, color=colors[cycle_idx], 
                            label=f'Cycle {cycle_idx+1}', linewidth=1.5)
            
            # グラフの装飾
            axes[0].set_ylabel('Absolute Error [deg]')
            axes[0].set_title(f'Left Hip Flexion/Extension - Absolute Error per Cycle (MAE: {mae_lhip:.2f}°)')
            axes[0].legend(loc='upper right')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, None)
            
            axes[1].set_ylabel('Absolute Error [deg]')
            axes[1].set_title(f'Left Knee Flexion/Extension - Absolute Error per Cycle (MAE: {mae_lknee:.2f}°)')
            axes[1].legend(loc='upper right')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, None)
            
            axes[2].set_xlabel('Gait Cycle [%]')
            axes[2].set_ylabel('Absolute Error [deg]')
            axes[2].set_title(f'Left Ankle Plantarflexion/Dorsiflexion - Absolute Error per Cycle (MAE: {mae_lankle:.2f}°)')
            axes[2].legend(loc='upper right')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim(0, None)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"absolute_error_FlEx_L_{csv_path.stem}.png", dpi=300)
            plt.close()
            
            # 股関節内旋外旋・内転外転の絶対誤差
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            mae_lhip_inex = calculate_mae_all_cycles(normalized_gait_cycles_l, normalized_gait_cycles_l_op, 'L_Hip_InEx')
            mae_lhip_adab = calculate_mae_all_cycles(normalized_gait_cycles_l, normalized_gait_cycles_l_op, 'L_Hip_AdAb')
            
            for cycle_idx in range(n_cycles_l):
                # 股関節内旋外旋の絶対誤差
                error_inex = np.abs(normalized_gait_cycles_l[cycle_idx]['L_Hip_InEx'] - 
                                    normalized_gait_cycles_l_op[cycle_idx]['L_Hip_InEx'])
                axes[0].plot(normalized_percentage, error_inex, color=colors[cycle_idx], 
                            label=f'Cycle {cycle_idx+1}', linewidth=1.5)
                
                # 股関節内転外転の絶対誤差
                error_adab = np.abs(normalized_gait_cycles_l[cycle_idx]['L_Hip_AdAb'] - 
                                    normalized_gait_cycles_l_op[cycle_idx]['L_Hip_AdAb'])
                axes[1].plot(normalized_percentage, error_adab, color=colors[cycle_idx], 
                            label=f'Cycle {cycle_idx+1}', linewidth=1.5)
            
            axes[0].set_ylabel('Absolute Error [deg]')
            axes[0].set_title(f'Left Hip Internal/External Rotation - Absolute Error per Cycle (MAE: {mae_lhip_inex:.2f}°)')
            axes[0].legend(loc='upper right')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, None)
            
            axes[1].set_xlabel('Gait Cycle [%]')
            axes[1].set_ylabel('Absolute Error [deg]')
            axes[1].set_title(f'Left Hip Adduction/Abduction - Absolute Error per Cycle (MAE: {mae_lhip_adab:.2f}°)')
            axes[1].legend(loc='upper right')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, None)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"absolute_error_InEx_AdAb_L_{csv_path.stem}.png", dpi=300)
            plt.close()
            
            print(f"左足絶対誤差プロットを保存しました")
        
        # 右足の比較プロット
        print(f"len normalized_gait_cycles_r: {len(normalized_gait_cycles_r)}, len normalized_gait_cycles_r_op: {len(normalized_gait_cycles_r_op)}")





        
        
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
            axes[0].plot(normalized_percentage, mean_cycle_r_op['R_Hip_FlEx_mean'], 'r--', label='Proposed', linewidth=2)
            axes[0].fill_between(normalized_percentage,
                                mean_cycle_r_op['R_Hip_FlEx_mean'] - mean_cycle_r_op['R_Hip_FlEx_std'],
                                mean_cycle_r_op['R_Hip_FlEx_mean'] + mean_cycle_r_op['R_Hip_FlEx_std'],
                                alpha=0.2, color='r')
            
            # maxpeople1のデータがある場合は追加
            if has_max1_data:
                axes[0].plot(normalized_percentage, mean_cycle_r_op_max1['R_Hip_FlEx_mean'], color='orange', linestyle='-.', label='Baseline', linewidth=2)
                axes[0].fill_between(normalized_percentage,
                                    mean_cycle_r_op_max1['R_Hip_FlEx_mean'] - mean_cycle_r_op_max1['R_Hip_FlEx_std'],
                                    mean_cycle_r_op_max1['R_Hip_FlEx_mean'] + mean_cycle_r_op_max1['R_Hip_FlEx_std'],
                                    alpha=0.2, color='orange')
            
            mae_rhip = calculate_mae_all_cycles(normalized_gait_cycles_r, normalized_gait_cycles_r_op, 'R_Hip_FlEx')
            title_str = f'Right Hip Flexion/Extension (Proposed: {mae_rhip:.2f}°'
            if has_max1_data and 'mae_summary_r_max1' in dir():
                mae_rhip_max1 = mae_summary_r_max1[mae_summary_r_max1['Joint_Movement'] == 'Hip_FlEx']['MAE [deg]'].values[0]
                title_str += f', Baseline: {mae_rhip_max1:.2f}°'
            title_str += ')'
            axes[0].set_ylabel('Hip Angle [deg]')
            axes[0].set_title(title_str)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 膝関節屈曲伸展
            axes[1].plot(normalized_percentage, mean_cycle_r['R_Knee_FlEx_mean'], 'b-', label='Mocap', linewidth=2)
            axes[1].fill_between(normalized_percentage,
                                mean_cycle_r['R_Knee_FlEx_mean'] - mean_cycle_r['R_Knee_FlEx_std'],
                                mean_cycle_r['R_Knee_FlEx_mean'] + mean_cycle_r['R_Knee_FlEx_std'],
                                alpha=0.2, color='b')
            axes[1].plot(normalized_percentage, mean_cycle_r_op['R_Knee_FlEx_mean'], 'r--', label='Proposed', linewidth=2)
            axes[1].fill_between(normalized_percentage,
                                mean_cycle_r_op['R_Knee_FlEx_mean'] - mean_cycle_r_op['R_Knee_FlEx_std'],
                                mean_cycle_r_op['R_Knee_FlEx_mean'] + mean_cycle_r_op['R_Knee_FlEx_std'],
                                alpha=0.2, color='r')
            
            if has_max1_data:
                axes[1].plot(normalized_percentage, mean_cycle_r_op_max1['R_Knee_FlEx_mean'], color='orange', linestyle='-.', label='Baseline', linewidth=2)
                axes[1].fill_between(normalized_percentage,
                                    mean_cycle_r_op_max1['R_Knee_FlEx_mean'] - mean_cycle_r_op_max1['R_Knee_FlEx_std'],
                                    mean_cycle_r_op_max1['R_Knee_FlEx_mean'] + mean_cycle_r_op_max1['R_Knee_FlEx_std'],
                                    alpha=0.2, color='orange')
            
            mae_rknee = calculate_mae_all_cycles(normalized_gait_cycles_r, normalized_gait_cycles_r_op, 'R_Knee_FlEx')
            title_str = f'Right Knee Flexion/Extension (Proposed: {mae_rknee:.2f}°'
            if has_max1_data and 'mae_summary_r_max1' in dir():
                mae_rknee_max1 = mae_summary_r_max1[mae_summary_r_max1['Joint_Movement'] == 'Knee_FlEx']['MAE [deg]'].values[0]
                title_str += f', Baseline: {mae_rknee_max1:.2f}°'
            title_str += ')'
            axes[1].set_ylabel('Knee Angle [deg]')
            axes[1].set_title(title_str)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # 足関節背屈底屈
            axes[2].plot(normalized_percentage, mean_cycle_r['R_Ankle_PlDo_mean'], 'b-', label='Mocap', linewidth=2)
            axes[2].fill_between(normalized_percentage,
                                mean_cycle_r['R_Ankle_PlDo_mean'] - mean_cycle_r['R_Ankle_PlDo_std'],
                                mean_cycle_r['R_Ankle_PlDo_mean'] + mean_cycle_r['R_Ankle_PlDo_std'],
                                alpha=0.2, color='b')
            axes[2].plot(normalized_percentage, mean_cycle_r_op['R_Ankle_PlDo_mean'], 'r--', label='Proposed', linewidth=2)
            axes[2].fill_between(normalized_percentage,
                                mean_cycle_r_op['R_Ankle_PlDo_mean'] - mean_cycle_r_op['R_Ankle_PlDo_std'],
                                mean_cycle_r_op['R_Ankle_PlDo_mean'] + mean_cycle_r_op['R_Ankle_PlDo_std'],
                                alpha=0.2, color='r')
            
            if has_max1_data:
                axes[2].plot(normalized_percentage, mean_cycle_r_op_max1['R_Ankle_PlDo_mean'], color='orange', linestyle='-.', label='Baseline', linewidth=2)
                axes[2].fill_between(normalized_percentage,
                                    mean_cycle_r_op_max1['R_Ankle_PlDo_mean'] - mean_cycle_r_op_max1['R_Ankle_PlDo_std'],
                                    mean_cycle_r_op_max1['R_Ankle_PlDo_mean'] + mean_cycle_r_op_max1['R_Ankle_PlDo_std'],
                                    alpha=0.2, color='orange')
            
            mae_rankle = calculate_mae_all_cycles(normalized_gait_cycles_r, normalized_gait_cycles_r_op, 'R_Ankle_PlDo')
            title_str = f'Right Ankle Plantarflexion/Dorsiflexion (Proposed: {mae_rankle:.2f}°'
            if has_max1_data and 'mae_summary_r_max1' in dir():
                mae_rankle_max1 = mae_summary_r_max1[mae_summary_r_max1['Joint_Movement'] == 'Ankle_PlDo']['MAE [deg]'].values[0]
                title_str += f', Baseline: {mae_rankle_max1:.2f}°'
            title_str += ')'
            axes[2].set_xlabel('Gait Cycle [%]')
            axes[2].set_ylabel('Ankle Angle [deg]')
            axes[2].set_title(title_str)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"comparison_FlEx_R_{csv_path.stem}_vs_max1.png", dpi=300)
            plt.close()
            
            # 股関節内旋外旋の比較
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            ax.plot(normalized_percentage, mean_cycle_r['R_Hip_InEx_mean'], 'b-', label='Mocap', linewidth=2)
            ax.fill_between(normalized_percentage,
                           mean_cycle_r['R_Hip_InEx_mean'] - mean_cycle_r['R_Hip_InEx_std'],
                           mean_cycle_r['R_Hip_InEx_mean'] + mean_cycle_r['R_Hip_InEx_std'],
                           alpha=0.2, color='b')
            ax.plot(normalized_percentage, mean_cycle_r_op['R_Hip_InEx_mean'], 'r--', label='Proposed', linewidth=2)
            ax.fill_between(normalized_percentage,
                           mean_cycle_r_op['R_Hip_InEx_mean'] - mean_cycle_r_op['R_Hip_InEx_std'],
                           mean_cycle_r_op['R_Hip_InEx_mean'] + mean_cycle_r_op['R_Hip_InEx_std'],
                           alpha=0.2, color='r')
            
            if has_max1_data:
                ax.plot(normalized_percentage, mean_cycle_r_op_max1['R_Hip_InEx_mean'], color='orange', linestyle='-.', label='Baseline', linewidth=2)
                ax.fill_between(normalized_percentage,
                               mean_cycle_r_op_max1['R_Hip_InEx_mean'] - mean_cycle_r_op_max1['R_Hip_InEx_std'],
                               mean_cycle_r_op_max1['R_Hip_InEx_mean'] + mean_cycle_r_op_max1['R_Hip_InEx_std'],
                               alpha=0.2, color='orange')
            
            mae_rhip_inex = calculate_mae_all_cycles(normalized_gait_cycles_r, normalized_gait_cycles_r_op, 'R_Hip_InEx')
            title_str = f'Right Hip Internal/External Rotation (Proposed: {mae_rhip_inex:.2f}°'
            if has_max1_data and 'mae_summary_r_max1' in dir():
                mae_rhip_inex_max1 = mae_summary_r_max1[mae_summary_r_max1['Joint_Movement'] == 'Hip_InEx']['MAE [deg]'].values[0]
                title_str += f', Baseline: {mae_rhip_inex_max1:.2f}°'
            title_str += ')'
            ax.set_xlabel('Gait Cycle [%]')
            ax.set_ylabel('Hip Angle [deg]')
            ax.set_title(title_str)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"comparison_InEx_R_{csv_path.stem}_vs_max1.png", dpi=300)
            plt.close()
            
            # 股関節内転外転の比較
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            ax.plot(normalized_percentage, mean_cycle_r['R_Hip_AdAb_mean'], 'b-', label='Mocap', linewidth=2)
            ax.fill_between(normalized_percentage,
                           mean_cycle_r['R_Hip_AdAb_mean'] - mean_cycle_r['R_Hip_AdAb_std'],
                           mean_cycle_r['R_Hip_AdAb_mean'] + mean_cycle_r['R_Hip_AdAb_std'],
                           alpha=0.2, color='b')
            ax.plot(normalized_percentage, mean_cycle_r_op['R_Hip_AdAb_mean'], 'r--', label='Proposed', linewidth=2)
            ax.fill_between(normalized_percentage,
                           mean_cycle_r_op['R_Hip_AdAb_mean'] - mean_cycle_r_op['R_Hip_AdAb_std'],
                           mean_cycle_r_op['R_Hip_AdAb_mean'] + mean_cycle_r_op['R_Hip_AdAb_std'],
                           alpha=0.2, color='r')
            
            if has_max1_data:
                ax.plot(normalized_percentage, mean_cycle_r_op_max1['R_Hip_AdAb_mean'], color='orange', linestyle='-.', label='Baseline', linewidth=2)
                ax.fill_between(normalized_percentage,
                               mean_cycle_r_op_max1['R_Hip_AdAb_mean'] - mean_cycle_r_op_max1['R_Hip_AdAb_std'],
                               mean_cycle_r_op_max1['R_Hip_AdAb_mean'] + mean_cycle_r_op_max1['R_Hip_AdAb_std'],
                               alpha=0.2, color='orange')
            
            mae_rhip_adab = calculate_mae_all_cycles(normalized_gait_cycles_r, normalized_gait_cycles_r_op, 'R_Hip_AdAb')
            title_str = f'Right Hip Adduction/Abduction (Proposed: {mae_rhip_adab:.2f}°'
            if has_max1_data and 'mae_summary_r_max1' in dir():
                mae_rhip_adab_max1 = mae_summary_r_max1[mae_summary_r_max1['Joint_Movement'] == 'Hip_AdAb']['MAE [deg]'].values[0]
                title_str += f', Baseline: {mae_rhip_adab_max1:.2f}°'
            title_str += ')'
            ax.set_xlabel('Gait Cycle [%]')
            ax.set_ylabel('Hip Angle [deg]')
            ax.set_title(title_str)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"comparison_AdAb_R_{csv_path.stem}_vs_max1.png", dpi=300)
            plt.close()
            
            # MAEサマリーを保存
            mae_summary_r = pd.DataFrame({
                'Joint_Movement': ['Hip_FlEx', 'Knee_FlEx', 'Ankle_PlDo', 'Hip_InEx', 'Hip_AdAb'],
                'MAE [deg]': [mae_rhip, mae_rknee, mae_rankle, mae_rhip_inex, mae_rhip_adab]
            })
            mae_summary_r.to_csv(op_result_dir / f"MAE_summary_R_{csv_path.stem}.csv", index=False)
            print(f"\n右足MAE:")
            print(mae_summary_r)
        
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
            axes[0].plot(normalized_percentage, mean_cycle_l_op['L_Hip_FlEx_mean'], 'r--', label='Proposed', linewidth=2)
            axes[0].fill_between(normalized_percentage,
                                mean_cycle_l_op['L_Hip_FlEx_mean'] - mean_cycle_l_op['L_Hip_FlEx_std'],
                                mean_cycle_l_op['L_Hip_FlEx_mean'] + mean_cycle_l_op['L_Hip_FlEx_std'],
                                alpha=0.2, color='r')
            
            if has_max1_data:
                axes[0].plot(normalized_percentage, mean_cycle_l_op_max1['L_Hip_FlEx_mean'], color='orange', linestyle='-.', label='Baseline', linewidth=2)
                axes[0].fill_between(normalized_percentage,
                                    mean_cycle_l_op_max1['L_Hip_FlEx_mean'] - mean_cycle_l_op_max1['L_Hip_FlEx_std'],
                                    mean_cycle_l_op_max1['L_Hip_FlEx_mean'] + mean_cycle_l_op_max1['L_Hip_FlEx_std'],
                                    alpha=0.2, color='orange')
            
            mae_lhip = calculate_mae_all_cycles(normalized_gait_cycles_l, normalized_gait_cycles_l_op, 'L_Hip_FlEx')
            title_str = f'Left Hip Flexion/Extension (Proposed: {mae_lhip:.2f}°'
            if has_max1_data and 'mae_summary_l_max1' in dir():
                mae_lhip_max1 = mae_summary_l_max1[mae_summary_l_max1['Joint_Movement'] == 'Hip_FlEx']['MAE [deg]'].values[0]
                title_str += f', Baseline: {mae_lhip_max1:.2f}°'
            title_str += ')'
            axes[0].set_ylabel('Hip Angle [deg]')
            axes[0].set_title(title_str)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 膝関節屈曲伸展
            axes[1].plot(normalized_percentage, mean_cycle_l['L_Knee_FlEx_mean'], 'b-', label='Mocap', linewidth=2)
            axes[1].fill_between(normalized_percentage,
                                mean_cycle_l['L_Knee_FlEx_mean'] - mean_cycle_l['L_Knee_FlEx_std'],
                                mean_cycle_l['L_Knee_FlEx_mean'] + mean_cycle_l['L_Knee_FlEx_std'],
                                alpha=0.2, color='b')
            axes[1].plot(normalized_percentage, mean_cycle_l_op['L_Knee_FlEx_mean'], 'r--', label='Proposed', linewidth=2)
            axes[1].fill_between(normalized_percentage,
                                mean_cycle_l_op['L_Knee_FlEx_mean'] - mean_cycle_l_op['L_Knee_FlEx_std'],
                                mean_cycle_l_op['L_Knee_FlEx_mean'] + mean_cycle_l_op['L_Knee_FlEx_std'],
                                alpha=0.2, color='r')
            
            if has_max1_data:
                axes[1].plot(normalized_percentage, mean_cycle_l_op_max1['L_Knee_FlEx_mean'], color='orange', linestyle='-.', label='Baseline', linewidth=2)
                axes[1].fill_between(normalized_percentage,
                                    mean_cycle_l_op_max1['L_Knee_FlEx_mean'] - mean_cycle_l_op_max1['L_Knee_FlEx_std'],
                                    mean_cycle_l_op_max1['L_Knee_FlEx_mean'] + mean_cycle_l_op_max1['L_Knee_FlEx_std'],
                                    alpha=0.2, color='orange')
            
            mae_lknee = calculate_mae_all_cycles(normalized_gait_cycles_l, normalized_gait_cycles_l_op, 'L_Knee_FlEx')
            title_str = f'Left Knee Flexion/Extension (Proposed: {mae_lknee:.2f}°'
            if has_max1_data and 'mae_summary_l_max1' in dir():
                mae_lknee_max1 = mae_summary_l_max1[mae_summary_l_max1['Joint_Movement'] == 'Knee_FlEx']['MAE [deg]'].values[0]
                title_str += f', Baseline: {mae_lknee_max1:.2f}°'
            title_str += ')'
            axes[1].set_ylabel('Knee Angle [deg]')
            axes[1].set_title(title_str)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # 足関節背屈底屈
            axes[2].plot(normalized_percentage, mean_cycle_l['L_Ankle_PlDo_mean'], 'b-', label='Mocap', linewidth=2)
            axes[2].fill_between(normalized_percentage,
                                mean_cycle_l['L_Ankle_PlDo_mean'] - mean_cycle_l['L_Ankle_PlDo_std'],
                                mean_cycle_l['L_Ankle_PlDo_mean'] + mean_cycle_l['L_Ankle_PlDo_std'],
                                alpha=0.2, color='b')
            axes[2].plot(normalized_percentage, mean_cycle_l_op['L_Ankle_PlDo_mean'], 'r--', label='Proposed', linewidth=2)
            axes[2].fill_between(normalized_percentage,
                                mean_cycle_l_op['L_Ankle_PlDo_mean'] - mean_cycle_l_op['L_Ankle_PlDo_std'],
                                mean_cycle_l_op['L_Ankle_PlDo_mean'] + mean_cycle_l_op['L_Ankle_PlDo_std'],
                                alpha=0.2, color='r')
            
            if has_max1_data:
                axes[2].plot(normalized_percentage, mean_cycle_l_op_max1['L_Ankle_PlDo_mean'], color='orange', linestyle='-.', label='Baseline', linewidth=2)
                axes[2].fill_between(normalized_percentage,
                                    mean_cycle_l_op_max1['L_Ankle_PlDo_mean'] - mean_cycle_l_op_max1['L_Ankle_PlDo_std'],
                                    mean_cycle_l_op_max1['L_Ankle_PlDo_mean'] + mean_cycle_l_op_max1['L_Ankle_PlDo_std'],
                                    alpha=0.2, color='orange')
            
            mae_lankle = calculate_mae_all_cycles(normalized_gait_cycles_l, normalized_gait_cycles_l_op, 'L_Ankle_PlDo')
            title_str = f'Left Ankle Plantarflexion/Dorsiflexion (Proposed: {mae_lankle:.2f}°'
            if has_max1_data and 'mae_summary_l_max1' in dir():
                mae_lankle_max1 = mae_summary_l_max1[mae_summary_l_max1['Joint_Movement'] == 'Ankle_PlDo']['MAE [deg]'].values[0]
                title_str += f', Baseline: {mae_lankle_max1:.2f}°'
            title_str += ')'
            axes[2].set_xlabel('Gait Cycle [%]')
            axes[2].set_ylabel('Ankle Angle [deg]')
            axes[2].set_title(title_str)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"comparison_FlEx_L_{csv_path.stem}_vs_max1.png", dpi=300)
            plt.close()
            
            # 股関節内旋外旋の比較
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            ax.plot(normalized_percentage, mean_cycle_l['L_Hip_InEx_mean'], 'b-', label='Mocap', linewidth=2)
            ax.fill_between(normalized_percentage,
                           mean_cycle_l['L_Hip_InEx_mean'] - mean_cycle_l['L_Hip_InEx_std'],
                           mean_cycle_l['L_Hip_InEx_mean'] + mean_cycle_l['L_Hip_InEx_std'],
                           alpha=0.2, color='b')
            ax.plot(normalized_percentage, mean_cycle_l_op['L_Hip_InEx_mean'], 'r--', label='Proposed', linewidth=2)
            ax.fill_between(normalized_percentage,
                           mean_cycle_l_op['L_Hip_InEx_mean'] - mean_cycle_l_op['L_Hip_InEx_std'],
                           mean_cycle_l_op['L_Hip_InEx_mean'] + mean_cycle_l_op['L_Hip_InEx_std'],
                           alpha=0.2, color='r')
            
            if has_max1_data:
                ax.plot(normalized_percentage, mean_cycle_l_op_max1['L_Hip_InEx_mean'], color='orange', linestyle='-.', label='Baseline', linewidth=2)
                ax.fill_between(normalized_percentage,
                               mean_cycle_l_op_max1['L_Hip_InEx_mean'] - mean_cycle_l_op_max1['L_Hip_InEx_std'],
                               mean_cycle_l_op_max1['L_Hip_InEx_mean'] + mean_cycle_l_op_max1['L_Hip_InEx_std'],
                               alpha=0.2, color='orange')
            
            mae_lhip_inex = calculate_mae_all_cycles(normalized_gait_cycles_l, normalized_gait_cycles_l_op, 'L_Hip_InEx')
            title_str = f'Left Hip Internal/External Rotation (Proposed: {mae_lhip_inex:.2f}°'
            if has_max1_data and 'mae_summary_l_max1' in dir():
                mae_lhip_inex_max1 = mae_summary_l_max1[mae_summary_l_max1['Joint_Movement'] == 'Hip_InEx']['MAE [deg]'].values[0]
                title_str += f', Baseline: {mae_lhip_inex_max1:.2f}°'
            title_str += ')'
            ax.set_xlabel('Gait Cycle [%]')
            ax.set_ylabel('Hip Angle [deg]')
            ax.set_title(title_str)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"comparison_InEx_L_{csv_path.stem}_vs_max1.png", dpi=300)
            plt.close()
            
            # 股関節内転外転の比較
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            ax.plot(normalized_percentage, mean_cycle_l['L_Hip_AdAb_mean'], 'b-', label='Mocap', linewidth=2)
            ax.fill_between(normalized_percentage,
                           mean_cycle_l['L_Hip_AdAb_mean'] - mean_cycle_l['L_Hip_AdAb_std'],
                           mean_cycle_l['L_Hip_AdAb_mean'] + mean_cycle_l['L_Hip_AdAb_std'],
                           alpha=0.2, color='b')
            ax.plot(normalized_percentage, mean_cycle_l_op['L_Hip_AdAb_mean'], 'r--', label='Proposed', linewidth=2)
            ax.fill_between(normalized_percentage,
                           mean_cycle_l_op['L_Hip_AdAb_mean'] - mean_cycle_l_op['L_Hip_AdAb_std'],
                           mean_cycle_l_op['L_Hip_AdAb_mean'] + mean_cycle_l_op['L_Hip_AdAb_std'],
                           alpha=0.2, color='r')
            
            if has_max1_data:
                ax.plot(normalized_percentage, mean_cycle_l_op_max1['L_Hip_AdAb_mean'], color='orange', linestyle='-.', label='Baseline', linewidth=2)
                ax.fill_between(normalized_percentage,
                               mean_cycle_l_op_max1['L_Hip_AdAb_mean'] - mean_cycle_l_op_max1['L_Hip_AdAb_std'],
                               mean_cycle_l_op_max1['L_Hip_AdAb_mean'] + mean_cycle_l_op_max1['L_Hip_AdAb_std'],
                               alpha=0.2, color='orange')
            
            mae_lhip_adab = calculate_mae_all_cycles(normalized_gait_cycles_l, normalized_gait_cycles_l_op, 'L_Hip_AdAb')
            title_str = f'Left Hip Adduction/Abduction (Proposed: {mae_lhip_adab:.2f}°'
            if has_max1_data and 'mae_summary_l_max1' in dir():
                mae_lhip_adab_max1 = mae_summary_l_max1[mae_summary_l_max1['Joint_Movement'] == 'Hip_AdAb']['MAE [deg]'].values[0]
                title_str += f', Baseline: {mae_lhip_adab_max1:.2f}°'
            title_str += ')'
            ax.set_xlabel('Gait Cycle [%]')
            ax.set_ylabel('Hip Angle [deg]')
            ax.set_title(title_str)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(op_result_dir / f"comparison_AdAb_L_{csv_path.stem}_vs_max1.png", dpi=300)
            plt.close()
            
            # MAEサマリーを保存
            mae_summary_l = pd.DataFrame({
                'Joint_Movement': ['Hip_FlEx', 'Knee_FlEx', 'Ankle_PlDo', 'Hip_InEx', 'Hip_AdAb'],
                'MAE [deg]': [mae_lhip, mae_lknee, mae_lankle, mae_lhip_inex, mae_lhip_adab]
            })
            mae_summary_l.to_csv(op_result_dir / f"MAE_summary_L_{csv_path.stem}.csv", index=False)
            print(f"\n左足MAE:")
            print(mae_summary_l)
            
            ###########################################
            # MAE比較バープロット（YoloSeg vs maxpeople1）
            ###########################################
            if has_max1_data and 'mae_summary_r_max1' in dir() and 'mae_summary_l_max1' in dir():
                print("\nMAE比較バープロットを作成中...")
                
                # 右足
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                
                joint_names = ['Hip FlEx', 'Knee FlEx', 'Ankle PlDo', 'Hip InEx', 'Hip AdAb']
                x = np.arange(len(joint_names))
                width = 0.35
                
                mae_yoloseg_r = [mae_rhip, mae_rknee, mae_rankle, mae_rhip_inex, mae_rhip_adab]
                mae_max1_r = mae_summary_r_max1['MAE [deg]'].values
                
                bars1 = ax.bar(x - width/2, mae_yoloseg_r, width, label='proposed', color='red', alpha=0.7)
                bars2 = ax.bar(x + width/2, mae_max1_r, width, label='baseline', color='orange', alpha=0.7)
                
                ax.set_ylabel('MAE [deg]')
                ax.set_title('Right Leg Joint Angle MAE Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(joint_names, rotation=15, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                # 値をバーの上に表示
                for bar in bars1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}°', ha='center', va='bottom')
                for bar in bars2:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}°', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(op_result_dir / f"MAE_comparison_R_{csv_path.stem}_yoloseg_vs_max1.png", dpi=300)
                plt.close()
                
                # 左足
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                
                mae_yoloseg_l = [mae_lhip, mae_lknee, mae_lankle, mae_lhip_inex, mae_lhip_adab]
                mae_max1_l = mae_summary_l_max1['MAE [deg]'].values
                
                bars1 = ax.bar(x - width/2, mae_yoloseg_l, width, label='proposed', color='red', alpha=0.7)
                bars2 = ax.bar(x + width/2, mae_max1_l, width, label='baseline', color='orange', alpha=0.7)
                
                ax.set_ylabel('MAE [deg]')
                ax.set_title('Left Leg Joint Angle MAE Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(joint_names, rotation=15, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                for bar in bars1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}°', ha='center', va='bottom')
                for bar in bars2:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}°', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(op_result_dir / f"MAE_comparison_L_{csv_path.stem}_yoloseg_vs_max1.png", dpi=300)
                plt.close()
                
                print(f"MAE比較バープロットを保存しました")
                
                ###########################################
                # 歩行パラメータMAEの比較グラフ（YoloSeg vs maxpeople1）
                ###########################################
                if 'mae_gait_params_both_max1' in dir():
                    print("\n歩行パラメータMAE比較グラフを作成中...")
                    
                    # 距離・速度系パラメータ
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    
                    # 距離系パラメータ
                    distance_params = ['Gait Speed', 'Stride Length', 'Step Length', 'Step Width']
                    x = np.arange(len(distance_params))
                    width = 0.35
                    
                    distance_mae_yoloseg = [mae_both_df[mae_both_df['Parameter'] == p]['MAE_Average'].values[0] for p in distance_params]
                    distance_mae_max1 = [mae_gait_params_both_max1[mae_gait_params_both_max1['Parameter'] == p]['MAE_Average'].values[0] for p in distance_params]
                    
                    bars1 = axes[0, 0].bar(x - width/2, distance_mae_yoloseg, width, label='proposed', color='red', alpha=0.7)
                    bars2 = axes[0, 0].bar(x + width/2, distance_mae_max1, width, label='baseline', color='orange', alpha=0.7)
                    
                    axes[0, 0].set_ylabel('MAE [m or m/s]')
                    axes[0, 0].set_title('Distance/Speed Parameters MAE Comparison')
                    axes[0, 0].set_xticks(x)
                    axes[0, 0].set_xticklabels(distance_params, rotation=15, ha='right')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3, axis='y')
                    
                    for bar in bars1:
                        height = bar.get_height()
                        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom')
                    for bar in bars2:
                        height = bar.get_height()
                        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom')
                    
                    # 時間系パラメータ
                    time_params = ['Cycle Duration', 'Stance Duration', 'Swing Duration']
                    x = np.arange(len(time_params))
                    
                    time_mae_yoloseg = [mae_both_df[mae_both_df['Parameter'] == p]['MAE_Average'].values[0] for p in time_params]
                    time_mae_max1 = [mae_gait_params_both_max1[mae_gait_params_both_max1['Parameter'] == p]['MAE_Average'].values[0] for p in time_params]
                    
                    bars1 = axes[0, 1].bar(x - width/2, time_mae_yoloseg, width, label='proposed', color='red', alpha=0.7)
                    bars2 = axes[0, 1].bar(x + width/2, time_mae_max1, width, label='baseline', color='orange', alpha=0.7)
                    
                    axes[0, 1].set_ylabel('MAE [s]')
                    axes[0, 1].set_title('Time Parameters MAE Comparison')
                    axes[0, 1].set_xticks(x)
                    axes[0, 1].set_xticklabels(time_params, rotation=15, ha='right')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3, axis='y')
                    
                    for bar in bars1:
                        height = bar.get_height()
                        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom')
                    for bar in bars2:
                        height = bar.get_height()
                        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom')
                    
                    # 割合系パラメータ
                    percent_params = ['Stance Phase', 'Swing Phase']
                    x = np.arange(len(percent_params))
                    
                    percent_mae_yoloseg = [mae_both_df[mae_both_df['Parameter'] == p]['MAE_Average'].values[0] for p in percent_params]
                    percent_mae_max1 = [mae_gait_params_both_max1[mae_gait_params_both_max1['Parameter'] == p]['MAE_Average'].values[0] for p in percent_params]
                    
                    bars1 = axes[1, 0].bar(x - width/2, percent_mae_yoloseg, width, label='proposed', color='red', alpha=0.7)
                    bars2 = axes[1, 0].bar(x + width/2, percent_mae_max1, width, label='baseline', color='orange', alpha=0.7)
                    
                    axes[1, 0].set_ylabel('MAE [%]')
                    axes[1, 0].set_title('Phase Parameters MAE Comparison')
                    axes[1, 0].set_xticks(x)
                    axes[1, 0].set_xticklabels(percent_params)
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3, axis='y')
                    
                    for bar in bars1:
                        height = bar.get_height()
                        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}', ha='center', va='bottom')
                    for bar in bars2:
                        height = bar.get_height()
                        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}', ha='center', va='bottom')
                    
                    # 全パラメータの比較（横棒グラフ）
                    all_params = distance_params + time_params + percent_params
                    all_mae_yoloseg = distance_mae_yoloseg + time_mae_yoloseg + percent_mae_yoloseg
                    all_mae_max1 = distance_mae_max1 + time_mae_max1 + percent_mae_max1
                    
                    # 正規化MAEで比較（YoloSeg vs maxpeople1）
                    y = np.arange(len(all_params))
                    height = 0.35
                    
                    # どちらが良いかを色で示す（小さい方が良い）
                    colors_yoloseg = ['green' if yolo < max1 else 'red' for yolo, max1 in zip(all_mae_yoloseg, all_mae_max1)]
                    colors_max1 = ['green' if max1 < yolo else 'orange' for yolo, max1 in zip(all_mae_yoloseg, all_mae_max1)]
                    
                    axes[1, 1].barh(y + height/2, all_mae_yoloseg, height, label='proposed', color='red', alpha=0.7)
                    axes[1, 1].barh(y - height/2, all_mae_max1, height, label='baseline', color='orange', alpha=0.7)
                    
                    axes[1, 1].set_xlabel('MAE (各単位)')
                    axes[1, 1].set_title('All Gait Parameters MAE Comparison')
                    axes[1, 1].set_yticks(y)
                    axes[1, 1].set_yticklabels(all_params)
                    axes[1, 1].legend(loc='lower right')
                    axes[1, 1].grid(True, alpha=0.3, axis='x')
                    
                    plt.tight_layout()
                    plt.savefig(op_result_dir / f"MAE_gait_parameters_comparison_{csv_path.stem}_yoloseg_vs_max1.png", dpi=300)
                    plt.close()
                    
                    print(f"歩行パラメータMAE比較グラフを保存しました")
                    
                    # 比較サマリーをCSVに保存
                    comparison_summary = []
                    for p in all_params:
                        yolo_mae = mae_both_df[mae_both_df['Parameter'] == p]['MAE_Average'].values[0]
                        max1_mae = mae_gait_params_both_max1[mae_gait_params_both_max1['Parameter'] == p]['MAE_Average'].values[0]
                        better = 'YoloSeg' if yolo_mae < max1_mae else 'maxpeople1'
                        comparison_summary.append({
                            'Parameter': p,
                            'MAE_YoloSeg': yolo_mae,
                            'MAE_maxpeople1': max1_mae,
                            'Better': better,
                            'Difference': abs(yolo_mae - max1_mae)
                        })
                    
                    comparison_summary_df = pd.DataFrame(comparison_summary)
                    comparison_summary_df.to_csv(op_result_dir / f"MAE_gait_parameters_comparison_summary_{csv_path.stem}.csv", index=False)
                    print(f"\n歩行パラメータMAE比較サマリー:")
                    print(comparison_summary_df.to_string(index=False))


                    comparison_summary_df = pd.DataFrame(comparison_summary)
                    comparison_summary_df.to_csv(op_result_dir / f"MAE_gait_parameters_comparison_summary_{csv_path.stem}.csv", index=False)
                    print(f"\n歩行パラメータMAE比較サマリー:")
                    print(comparison_summary_df.to_string(index=False))
                    
                    ###########################################
                    # gait_speed, step_length, step_widthのMAE比較（Proposed vs Baseline）
                    ###########################################
                    print("\n歩行速度・ステップ長・歩隔のMAE比較グラフを作成中...")
                    
                    # 比較対象パラメータ
                    target_params = ['Gait Speed', 'Step Length', 'Step Width']
                    target_units = ['m/s', 'm', 'm']
                    
                    # MAE値を取得
                    mae_proposed = []
                    mae_baseline = []
                    for p in target_params:
                        proposed_val = mae_both_df[mae_both_df['Parameter'] == p]['MAE_Average'].values[0]
                        baseline_val = mae_gait_params_both_max1[mae_gait_params_both_max1['Parameter'] == p]['MAE_Average'].values[0]
                        mae_proposed.append(proposed_val)
                        mae_baseline.append(baseline_val)
                    
                    # 棒グラフ作成
                    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                    
                    x = np.arange(len(target_params))
                    width = 0.35
                    
                    bars1 = ax.bar(x - width/2, mae_proposed, width, label='Proposed', color='red', alpha=0.7)
                    bars2 = ax.bar(x + width/2, mae_baseline, width, label='Baseline', color='orange', alpha=0.7)
                    
                    ax.set_ylabel('MAE')
                    ax.set_title('Gait Parameters MAE Comparison: Proposed vs Baseline')
                    ax.set_xticks(x)
                    ax.set_xticklabels([f'{p}\n[{u}]' for p, u in zip(target_params, target_units)])
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # 値をバーの上に表示
                    for bar in bars1:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom')
                    for bar in bars2:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    plt.savefig(op_result_dir / f"MAE_gait_speed_step_comparison_{csv_path.stem}.png", dpi=300)
                    plt.close()
                    
                    print(f"歩行速度・ステップ長・歩隔のMAE比較グラフを保存しました: MAE_gait_speed_step_comparison_{csv_path.stem}.png")
                    
                    # 比較結果を表示
                    print(f"\n歩行速度・ステップ長・歩隔のMAE比較:")
                    for p, u, prop, base in zip(target_params, target_units, mae_proposed, mae_baseline):
                        better = 'Proposed' if prop < base else 'Baseline'
                        diff = abs(prop - base)
                        print(f"  {p}: Proposed={prop:.4f} {u}, Baseline={base:.4f} {u}, Better={better}, Diff={diff:.4f}")


                    # 比較結果を表示
                    print(f"\n歩行速度・ステップ長・歩隔のMAE比較:")
                    for p, u, prop, base in zip(target_params, target_units, mae_proposed, mae_baseline):
                        better = 'Proposed' if prop < base else 'Baseline'
                        diff = abs(prop - base)
                        print(f"  {p}: Proposed={prop:.4f} {u}, Baseline={base:.4f} {u}, Better={better}, Diff={diff:.4f}")
                    
                    ###########################################
                    # gait_speed, step_length, step_widthの正規化MAE比較（Proposed vs Baseline）
                    ###########################################
                    print("\n歩行速度・ステップ長・歩隔の正規化MAE比較グラフを作成中...")
                    
                    # 比較対象パラメータ
                    target_params_norm = ['Gait Speed', 'Step Length', 'Step Width']
                    target_keys = ['gait_speed', 'step_length', 'step_width']
                    
                    # 正規化MAE値を計算（Mocapの平均値で正規化）
                    mae_proposed_norm = []
                    mae_baseline_norm = []
                    
                    for p, key in zip(target_params_norm, target_keys):
                        # Mocapの平均値を取得
                        r_vals = [params[key] for params in gait_params_r if params.get(key) is not None]
                        l_vals = [params[key] for params in gait_params_l if params.get(key) is not None]
                        mocap_mean = np.mean(r_vals + l_vals) if len(r_vals + l_vals) > 0 else 1
                        
                        # ProposedのMAE
                        proposed_mae = mae_both_df[mae_both_df['Parameter'] == p]['MAE_Average'].values[0]
                        # BaselineのMAE
                        baseline_mae = mae_gait_params_both_max1[mae_gait_params_both_max1['Parameter'] == p]['MAE_Average'].values[0]
                        
                        # 正規化（%）
                        if mocap_mean != 0:
                            proposed_norm = (proposed_mae / abs(mocap_mean)) * 100
                            baseline_norm = (baseline_mae / abs(mocap_mean)) * 100
                        else:
                            proposed_norm = 0
                            baseline_norm = 0
                        
                        mae_proposed_norm.append(proposed_norm)
                        mae_baseline_norm.append(baseline_norm)
                    
                    # 棒グラフ作成
                    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                    
                    x = np.arange(len(target_params_norm))
                    width = 0.35
                    
                    bars1 = ax.bar(x - width/2, mae_proposed_norm, width, label='Proposed', color='red', alpha=0.7)
                    bars2 = ax.bar(x + width/2, mae_baseline_norm, width, label='Baseline', color='orange', alpha=0.7)
                    
                    ax.set_ylabel('Normalized MAE [%]')
                    ax.set_title('Gait Parameters Normalized MAE Comparison\n(MAE / Mocap Mean × 100)')
                    ax.set_xticks(x)
                    ax.set_xticklabels(target_params_norm)
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # 値をバーの上に表示
                    for bar in bars1:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom')
                    for bar in bars2:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    plt.savefig(op_result_dir / f"MAE_gait_speed_step_normalized_comparison_{csv_path.stem}.png", dpi=300)
                    plt.close()
                    
                    print(f"正規化MAE比較グラフを保存しました: MAE_gait_speed_step_normalized_comparison_{csv_path.stem}.png")
                    
                    # 比較結果を表示
                    print(f"\n歩行速度・ステップ長・歩隔の正規化MAE比較:")
                    for p, key, prop_norm, base_norm in zip(target_params_norm, target_keys, mae_proposed_norm, mae_baseline_norm):
                        # Mocapの平均値も表示
                        r_vals = [params[key] for params in gait_params_r if params.get(key) is not None]
                        l_vals = [params[key] for params in gait_params_l if params.get(key) is not None]
                        mocap_mean = np.mean(r_vals + l_vals)
                        
                        better = 'Proposed' if prop_norm < base_norm else 'Baseline'
                        diff = abs(prop_norm - base_norm)
                        print(f"  {p}: Mocap Mean={mocap_mean:.4f}, Proposed={prop_norm:.1f}%, Baseline={base_norm:.1f}%, Better={better}, Diff={diff:.1f}%")



if __name__ == "__main__":
    main()