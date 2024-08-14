import pandas as pd
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def read_3d_optitrack(csv_path):
    df = pd.read_csv(csv_path, skiprows=[0, 1, 2, 4], header=[0, 2])

    df_down = df[::4].reset_index(drop=True)

    marker_set = ["RASI", "LASI", "RPSI", "LPSI","RKNE","LKNE", "RANK","LANK","RTOE","LTOE","RHEE","LHEE", "RKNE2", "LKNE2", "RANK2", "LANK2"]

    # marker_set = ["RASI[frame_num,:]", "LASI[frame_num,:])","RPSI[frame_num,:]","LPSI[frame_num,:]","RKNE","LKNE", "RTHI", "LTHI", "RANK[frame_num,:]","LANK", "RTIB", "LTIB","RTOE[frame_num,:]","LTOE[frame_num,:]","RHEE[frame_num,:]","LHEE[frame_num, :]",
    #             "RSHO", "LSHO","C7", "T10", "CLAV", "STRN", "RBAK", "RKNE2", "LKNE2", "RANK[frame_num,:]2", "LANK2[frame_num,:]"]

    marker_set_df = df_down[[col for col in df_down.columns if any(marker in col[0] for marker in marker_set)]].copy()
    success_frame_list = []

    for frame in range(0, len(marker_set_df)):
        if not marker_set_df.iloc[frame].isna().any():
            success_frame_list.append(frame)

    full_range = range(min(success_frame_list), max(success_frame_list) + 1)
    success_df = marker_set_df.reindex(full_range)
    interpolate_success_df = success_df.interpolate(method='spline', order=3)

    for i, index in enumerate(full_range):
        marker_set_df.loc[index, :] = interpolate_success_df.iloc[i, :]
    marker_set_df.to_csv(os.path.join(os.path.dirname(csv_path), f"marker_set_{os.path.basename(csv_path)}"))

    # print(f"colums = {marker_set_df.columns}")

    '''
    表示の順番
    columns = MultiIndex([(   'MarkerSet 01:C7', 'X'), 0
                ( 'MarkerSet 01:CLAV', 'X'), 1
                ( 'MarkerSet 01:LANK', 'X'), 2
                ('MarkerSet 01:LANK2[frame_num,:]', 'X'), 3
                ( 'MarkerSet 01:LASI[frame_num,:])', 'X'), 4
                ( 'MarkerSet 01:LHEE[frame_num, :]', 'X'), 5
                ( 'MarkerSet 01:LKNE', 'X'), 6
                ('MarkerSet 01:LKNE2', 'X'), 7
                ( 'MarkerSet 01:LPSI[frame_num,:]', 'X'), 8
                ( 'MarkerSet 01:LSHO', 'X'), 9
                ( 'MarkerSet 01:LTHI', 'X'), 10
                ( 'MarkerSet 01:LTIB', 'X'), 11
                ( 'MarkerSet 01:LTOE[frame_num,:]', 'X'), 12
                ( 'MarkerSet 01:RANK[frame_num,:]', 'X'), 13
                ('MarkerSet 01:RANK[frame_num,:]2', 'X'), 14
                ( 'MarkerSet 01:RASI[frame_num,:]', 'X'), 15
                ( 'MarkerSet 01:RBAK', 'X'), 16
                ( 'MarkerSet 01:RHEE[frame_num,:]', 'X'), 17
                ( 'MarkerSet 01:RKNE', 'X'), 18
                ('MarkerSet 01:RKNE2', 'X'), 19
                ( 'MarkerSet 01:RPSI[frame_num,:]', 'X'), 20
                ( 'MarkerSet 01:RSHO', 'X'), 21
                ( 'MarkerSet 01:RTHI', 'X'), 22
                ( 'MarkerSet 01:RTIB', 'X'), 23
                ( 'MarkerSet 01:RTOE[frame_num,:]', 'X'), 24
                ( 'MarkerSet 01:STRN', 'X'), 25
                (  'MarkerSet 01:T10', 'X'), 26
            )

    columns = MultiIndex([( 'MarkerSet 01:LANK', 'X'), 0
                ('MarkerSet 01:LANK2[frame_num,:]', 'X'), 1
                ( 'MarkerSet 01:LASI[frame_num,:])', 'X'), 2
                ( 'MarkerSet 01:LHEE[frame_num, :]', 'X'), 3
                ( 'MarkerSet 01:LKNE', 'X'), 4
                ('MarkerSet 01:LKNE2', 'X'), 5
                ( 'MarkerSet 01:LPSI[frame_num,:]', 'X'), 6
                ( 'MarkerSet 01:LTOE[frame_num,:]', 'X'), 7
                ( 'MarkerSet 01:RANK[frame_num,:]', 'X'), 8
                ('MarkerSet 01:RANK[frame_num,:]2', 'X'), 9
                ( 'MarkerSet 01:RASI[frame_num,:]', 'X'), 10
                ( 'MarkerSet 01:RHEE[frame_num,:]', 'X'), 11
                ( 'MarkerSet 01:RKNE', 'X'), 12
                ('MarkerSet 01:RKNE2', 'X'), 13
                ( 'MarkerSet 01:RPSI[frame_num,:]', 'X'), 14
                ( 'MarkerSet 01:RTOE[frame_num,:]', 'X'), 15
            )
    '''

    keypoints = marker_set_df.values
    keypoints_mocap = keypoints.reshape(-1, len(marker_set), 3)  #xyzで組になるように変形

    return keypoints_mocap, full_range

def butter_lowpass_fillter(data, order, cutoff_freq, frame_list):  #4次のバターワースローパスフィルタ
    sampling_freq = 30
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
    csv_path_dir = r"F:\Tomson\gait_pattern\20240808\Motive"
    csv_paths = glob.glob(os.path.join(csv_path_dir, "1*.csv"))
    # csv_paths = glob.glob(os.path.join(csv_path_dir, "[1-9]*.csv"))

    for i, csv_path in enumerate(csv_paths):
        keypoints_mocap, full_range = read_3d_optitrack(csv_path)
        print(f"csv_path = {csv_path}")

        angle_list = []
        dist_list = []
        bector_list = []

        e_z_lshank_list = []
        e_z_lfoot_list = []

        rasi = np.array([butter_lowpass_fillter(keypoints_mocap[:, 10, x], order=4, cutoff_freq=6, frame_list=full_range) for x in range(3)]).T
        lasi = np.array([butter_lowpass_fillter(keypoints_mocap[:, 2, x], order=4, cutoff_freq=6, frame_list=full_range) for x in range(3)]).T
        rpsi = np.array([butter_lowpass_fillter(keypoints_mocap[:, 14, x], order=4, cutoff_freq=6, frame_list=full_range) for x in range(3)]).T
        lpsi = np.array([butter_lowpass_fillter(keypoints_mocap[:, 6, x], order=4, cutoff_freq=6, frame_list=full_range) for x in range(3)]).T
        rank = np.array([butter_lowpass_fillter(keypoints_mocap[:, 8, x], order=4, cutoff_freq=6, frame_list=full_range) for x in range(3)]).T
        lank = np.array([butter_lowpass_fillter(keypoints_mocap[:, 0, x], order=4, cutoff_freq=6, frame_list=full_range) for x in range(3)]).T
        rank2 = np.array([butter_lowpass_fillter(keypoints_mocap[:, 9, x], order=4, cutoff_freq=6, frame_list=full_range) for x in range(3)]).T
        lank2 = np.array([butter_lowpass_fillter(keypoints_mocap[:, 1, x], order=4, cutoff_freq=6, frame_list=full_range) for x in range(3)]).T
        rknee = np.array([butter_lowpass_fillter(keypoints_mocap[:, 12, x], order=4, cutoff_freq=6, frame_list=full_range) for x in range(3)]).T
        lknee = np.array([butter_lowpass_fillter(keypoints_mocap[:, 4, x], order=4, cutoff_freq=6, frame_list=full_range) for x in range(3)]).T
        rknee2 = np.array([butter_lowpass_fillter(keypoints_mocap[:, 13, x], order=4, cutoff_freq=6, frame_list=full_range) for x in range(3)]).T
        lknee2 = np.array([butter_lowpass_fillter(keypoints_mocap[:, 5, x], order=4, cutoff_freq=6, frame_list=full_range) for x in range(3)]).T
        rtoe = np.array([butter_lowpass_fillter(keypoints_mocap[:, 15, x], order=4, cutoff_freq=6, frame_list=full_range) for x in range(3)]).T
        ltoe = np.array([butter_lowpass_fillter(keypoints_mocap[:, 7, x], order=4, cutoff_freq=6, frame_list=full_range) for x in range(3)]).T
        rhee = np.array([butter_lowpass_fillter(keypoints_mocap[:, 11, x], order=4, cutoff_freq=6, frame_list=full_range) for x in range(3)]).T
        lhee = np.array([butter_lowpass_fillter(keypoints_mocap[:, 3, x], order=4, cutoff_freq=6, frame_list=full_range) for x in range(3)]).T

        for frame_num in full_range:
            #メモ
            d_asi = np.linalg.norm(rasi[frame_num,:] - lasi[frame_num,:])
            d_leg = (np.linalg.norm(rank[frame_num,:] - rasi[frame_num,:]) + np.linalg.norm(lank[frame_num, :] - lasi[frame_num,:]) / 2)
            # r = 0.0159 #[m] サイズ違うかも確認必要 https://www.optitrack.jp/products/accessories/marker.html
            r = 0.0127 #[m] 三宅君
            h = 1.7 #[m] 山本先生1.703 三宅君1.7
            k = h/1.7
            beta = 0.1 * np.pi #[rad]
            theta = 0.496 #[rad]
            c = 0.115 * d_leg - 0.00153
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

            # #右足節座標系（原点はrfoot）
            # e_z_rfoot = e_z_pelvis
            # e_x0_rfoot = rtoe[frame_num,:] - rhee[frame_num,:]
            # e_y_rfoot = np.cross(e_z_rfoot, e_x0_rfoot)/np.linalg.norm(np.cross(e_z_rfoot, e_x0_rfoot))
            # e_x_rfoot = np.cross(e_y_rfoot, e_z_rfoot)
            # rot_rfoot = np.array([e_x_rfoot, e_y_rfoot, e_z_rfoot]).T

            # #左足節座標系（原点はlfoot）
            # e_z_lfoot = e_z_pelvis
            # e_x0_lfoot = ltoe[frame_num,:] - lhee[frame_num, :]
            # e_y_lfoot = np.cross(e_z_lfoot, e_x0_lfoot)/np.linalg.norm(np.cross(e_z_lfoot, e_x0_lfoot))
            # e_x_lfoot = np.cross(e_y_lfoot, e_z_lfoot)
            # rot_lfoot = np.array([e_x_lfoot, e_y_lfoot, e_z_lfoot]).T

            #右足節座標系 AIST参照（原点はrfoot）
            e_z_rfoot = (rtoe[frame_num,:] - rhee[frame_num,:]) / np.linalg.norm(rtoe[frame_num,:] - rhee[frame_num,:])
            e_y0_rfoot = rank[frame_num,:] - rank2[frame_num,:]
            e_x_rfoot = np.cross(e_z_rfoot, e_y0_rfoot)/np.linalg.norm(np.cross(e_z_rfoot, e_y0_rfoot))
            e_y_rfoot = np.cross(e_z_rfoot, e_x_rfoot)
            rot_rfoot = np.array([e_x_rfoot, e_y_rfoot, e_z_rfoot]).T

            # e_x_rfoot = (rtoe[frame_num,:] - rhee[frame_num,:]) / np.linalg.norm(rtoe[frame_num,:] - rhee[frame_num,:])
            # e_y0_rfoot = rank[frame_num,:]2 - rank[frame_num,:]
            # e_z_rfoot = np.cross(e_x_rfoot, e_y0_rfoot)/np.linalg.norm(np.cross(e_x_rfoot, e_y0_rfoot))
            # e_y_rfoot = np.cross(e_z_rfoot, e_x_rfoot)
            # rot_rfoot = np.array([e_x_rfoot, e_y_rfoot, e_z_rfoot]).T

            #左足節座標系 AIST参照（原点はlfoot）
            e_z_lfoot = (ltoe[frame_num,:] - lhee[frame_num, :]) / np.linalg.norm(ltoe[frame_num,:] - lhee[frame_num, :])
            e_y0_lfoot = lank2[frame_num,:] - lank[frame_num,:]
            e_x_lfoot = np.cross(e_z_lfoot, e_y0_lfoot)/np.linalg.norm(np.cross(e_z_lfoot, e_y0_lfoot))
            e_y_lfoot = np.cross(e_z_lfoot, e_x_lfoot)
            rot_lfoot = np.array([e_x_lfoot, e_y_lfoot, e_z_lfoot]).T

            # e_x_lfoot = (ltoe[frame_num,:] - lhee[frame_num, :]) / np.linalg.norm(ltoe[frame_num,:] - lhee[frame_num, :])
            # e_y0_lfoot = lank[frame_num, :] - lank2[frame_num,:]
            # e_z_lfoot = np.cross(e_x_lfoot, e_y0_lfoot)/np.linalg.norm(np.cross(e_x_lfoot, e_y0_lfoot))
            # e_y_lfoot = np.cross(e_z_lfoot, e_x_lfoot)
            # rot_lfoot = np.array([e_x_lfoot, e_y_lfoot, e_z_lfoot]).T


            r_hip_realative_rotation = np.dot(np.linalg.inv(rot_pelvis), rot_rthigh)
            r_hip_angle = R.from_matrix(r_hip_realative_rotation).as_euler('yzx', degrees=True)[0]
            l_hip_realative_rotation = np.dot(np.linalg.inv(rot_pelvis), rot_lthigh)
            l_hip_angle = R.from_matrix(l_hip_realative_rotation).as_euler('yzx', degrees=True)[0]
            r_knee_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rthigh)
            r_knee_angle = R.from_matrix(r_knee_realative_rotation).as_euler('yzx', degrees=True)[0]
            l_knee_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lthigh)
            l_knee_angle = R.from_matrix(l_knee_realative_rotation).as_euler('yzx', degrees=True)[0]
            r_ankle_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rfoot)
            r_ankle_angle = R.from_matrix(r_ankle_realative_rotation).as_euler('yzx', degrees=True)[0]
            l_ankle_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lfoot)
            l_ankle_angle = R.from_matrix(l_ankle_realative_rotation).as_euler('yzx', degrees=True)[0]

            # r_hip_realative_rotation = np.dot(np.linalg.inv(rot_pelvis), rot_rthigh)
            # r_hip_angle = R.from_matrix(r_hip_realative_rotation).as_euler('yzx', degrees=True)[0]
            # l_hip_realative_rotation = np.dot(np.linalg.inv(rot_pelvis), rot_lthigh)
            # l_hip_angle = R.from_matrix(l_hip_realative_rotation).as_euler('yzx', degrees=True)[0]
            # r_knee_realative_rotation = np.dot(np.linalg.inv(rot_rthigh), rot_rshank)
            # r_knee_angle = R.from_matrix(r_knee_realative_rotation).as_euler('yzx', degrees=True)[0]
            # l_knee_realative_rotation = np.dot(np.linalg.inv(rot_lthigh), rot_lshank)
            # l_knee_angle = R.from_matrix(l_knee_realative_rotation).as_euler('yzx', degrees=True)[0]
            # r_ankle_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rfoot)
            # r_ankle_angle = R.from_matrix(r_ankle_realative_rotation).as_euler('yzx', degrees=True)[0]
            # l_ankle_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lfoot)
            # l_ankle_angle = R.from_matrix(l_ankle_realative_rotation).as_euler('yzx', degrees=True)[0]

            angles = [r_hip_angle, l_hip_angle, r_knee_angle, l_knee_angle, r_ankle_angle, l_ankle_angle]
            angles = [angle + 360 if angle < 0 else angle for angle in angles]  #オイラー角(-180~180)が負の値の場合は正に変換
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

            plot_flag = False
            if plot_flag:
                fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
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
            dist = np.linalg.norm(lhee[frame_num, :] - hip[:])
            bector = lhee[frame_num, :] - hip[:]
            dist_list.append(dist)
            bector_list.append(bector)
            # bector_list.append(lhee_basse_pelvis)

        angle_array = np.array(angle_list)
        # print(f"angle_array = {angle_array}")
        # print(f"angle_array.shape = {angle_array.shape}")
        df = pd.DataFrame({"r_hip_angle": angle_array[:, 0], "r_knee_angle": angle_array[:, 2], "r_ankle_angle": angle_array[:, 4], "l_hip_angle": angle_array[:, 1], "l_knee_angle": angle_array[:, 3], "l_ankle_angle": angle_array[:, 5]})
        df.index = df.index + full_range.start
        df.to_csv(os.path.join(os.path.dirname(csv_path), f"angle_{os.path.basename(csv_path)}"))

        # ankle_angle = calculate_angle(e_z_lshank_list, e_z_lfoot_list)
        # print(f"ankle_angle = {ankle_angle}")

        # fig, ax = plt.subplots()
        # ax.plot(full_range, dist_list)
        # plt.show()
        # plt.cla

        bector_array = np.array(bector_list)
        lhee_pel_z = bector_array[:, 2]
        df = pd.DataFrame({"frame":full_range, "lhee_pel_z":lhee_pel_z})
        df = df.sort_values(by="lhee_pel_z", ascending=False)
        print(df)
        ic_list = df.head(30)["frame"].values
        print(f"ic_list = {ic_list}")

        filtered_list = []
        skip_values = set()
        for value in ic_list:
            # すでにスキップリストにある場合はスキップ
            if value in skip_values:
                continue
            # 現在の値を結果リストに追加
            filtered_list.append(value)
            # 10個以内の数値をスキップリストに追加
            skip_values.update(range(value - 10, value + 11))
        print(f"フィルタリング後のリスト:{filtered_list}")


        # print(f"bector_list = {bector_list}")
        fig, ax = plt.subplots()
        # ax.plot(full_range, bector_array[:, 0], label="x")
        # ax.plot(full_range, bector_array[:, 1], label="y")
        ax.plot(full_range, bector_array[:, 2], label="z")
        plt.legend()
        # plt.show()








if __name__ == "__main__":
    main()

