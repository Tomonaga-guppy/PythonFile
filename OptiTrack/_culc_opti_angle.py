import pandas as pd
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def read_3d_optitrack(csv_path):
    df = pd.read_csv(csv_path, skiprows=[0, 1, 2, 4], header=[0, 2])

    df_down = df[::4].reset_index(drop=True)

    marker_set = ["RASI", "LASI","RPSI","LPSI","RKNE","LKNE", "RANK","LANK","RTOE","LTOE","RHEE","LHEE", "RKNE2", "LKNE2", "RANK2", "LANK2"]  # 16個

    # marker_set = ["RASI", "LASI","RPSI","LPSI","RKNE","LKNE", "RTHI", "LTHI", "RANK","LANK", "RTIB", "LTIB","RTOE","LTOE","RHEE","LHEE",
    #             "RSHO", "LSHO","C7", "T10", "CLAV", "STRN", "RBAK", "RKNE2", "LKNE2", "RANK2", "LANK2"]  # 27個

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
                ('MarkerSet 01:LANK2', 'X'), 3
                ( 'MarkerSet 01:LASI', 'X'), 4
                ( 'MarkerSet 01:LHEE', 'X'), 5
                ( 'MarkerSet 01:LKNE', 'X'), 6
                ('MarkerSet 01:LKNE2', 'X'), 7
                ( 'MarkerSet 01:LPSI', 'X'), 8
                ( 'MarkerSet 01:LSHO', 'X'), 9
                ( 'MarkerSet 01:LTHI', 'X'), 10
                ( 'MarkerSet 01:LTIB', 'X'), 11
                ( 'MarkerSet 01:LTOE', 'X'), 12
                ( 'MarkerSet 01:RANK', 'X'), 13
                ('MarkerSet 01:RANK2', 'X'), 14
                ( 'MarkerSet 01:RASI', 'X'), 15
                ( 'MarkerSet 01:RBAK', 'X'), 16
                ( 'MarkerSet 01:RHEE', 'X'), 17
                ( 'MarkerSet 01:RKNE', 'X'), 18
                ('MarkerSet 01:RKNE2', 'X'), 19
                ( 'MarkerSet 01:RPSI', 'X'), 20
                ( 'MarkerSet 01:RSHO', 'X'), 21
                ( 'MarkerSet 01:RTHI', 'X'), 22
                ( 'MarkerSet 01:RTIB', 'X'), 23
                ( 'MarkerSet 01:RTOE', 'X'), 24
                ( 'MarkerSet 01:STRN', 'X'), 25
                (  'MarkerSet 01:T10', 'X'), 26
            )

    columns = MultiIndex([( 'MarkerSet 01:LANK', 'X'), 0
                ('MarkerSet 01:LANK2', 'X'), 1
                ( 'MarkerSet 01:LASI', 'X'), 2
                ( 'MarkerSet 01:LHEE', 'X'), 3
                ( 'MarkerSet 01:LKNE', 'X'), 4
                ('MarkerSet 01:LKNE2', 'X'), 5
                ( 'MarkerSet 01:LPSI', 'X'), 6
                ( 'MarkerSet 01:LTOE', 'X'), 7
                ( 'MarkerSet 01:RANK', 'X'), 8
                ('MarkerSet 01:RANK2', 'X'), 9
                ( 'MarkerSet 01:RASI', 'X'), 10
                ( 'MarkerSet 01:RHEE', 'X'), 11
                ( 'MarkerSet 01:RKNE', 'X'), 12
                ('MarkerSet 01:RKNE2', 'X'), 13
                ( 'MarkerSet 01:RPSI', 'X'), 14
                ( 'MarkerSet 01:RTOE', 'X'), 15
            )
    '''

    keypoints = marker_set_df.values
    keypoints_mocap = keypoints.reshape(-1, len(marker_set), 3)  #xyzで組になるように変形

    return keypoints_mocap, full_range

def main():
    csv_path_dir = r"F:\Tomson\gait_pattern\20240712\Motive"
    csv_paths = glob.glob(os.path.join(csv_path_dir, "[0-9]*.csv"))

    for i, csv_path in enumerate(csv_paths):
        keypoints_mocap, full_range = read_3d_optitrack(csv_path)
        print(f"csv_path = {csv_path}")

        angle_list = []

        for frame_num in full_range:
            #使う関節の選択
            rasi = keypoints_mocap[frame_num, 10, :]
            lasi = keypoints_mocap[frame_num, 2, :]
            rpsi = keypoints_mocap[frame_num, 14, :]
            lpsi = keypoints_mocap[frame_num, 6, :]
            rank = keypoints_mocap[frame_num, 8, :]
            lank = keypoints_mocap[frame_num, 0, :]
            rank2 = keypoints_mocap[frame_num, 9, :]
            lank2 = keypoints_mocap[frame_num, 1, :]
            rknee = keypoints_mocap[frame_num, 12, :]
            lknee = keypoints_mocap[frame_num, 4, :]
            rknee2 = keypoints_mocap[frame_num, 13, :]
            lknee2 = keypoints_mocap[frame_num, 5, :]
            rtoe = keypoints_mocap[frame_num, 15, :]
            ltoe = keypoints_mocap[frame_num, 7, :]
            rhee = keypoints_mocap[frame_num, 11, :]
            lhee = keypoints_mocap[frame_num, 3, :]


            #メモ
            d_asi = np.linalg.norm(rasi - lasi)
            d_leg = (np.linalg.norm(rank - rasi) + np.linalg.norm(lank - lasi)) / 2
            r = 0.0159 #[m] サイズ違うかも確認必要 https://www.optitrack.jp/products/accessories/marker.html
            h = 1.7 #[m] とりあえず
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

            hip_0 = (rasi + lasi) / 2
            lumbar = (0.47 * (rasi + lasi) / 2 + 0.53 * (rpsi + lpsi) / 2) + 0.02 * k * np.array([0, 0, 1])

            #骨盤節座標系（原点はhip）
            e_y0_pelvis = lasi - rasi
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
            rshank = (rknee + rknee2) / 2
            lshank = (lknee + lknee2) / 2
            rfoot = (rank + rank2) / 2
            lfoot = (lank + lank2) / 2

            #右大腿節座標系（原点はrthigh）
            e_y0_rthigh = rknee2 - rknee
            e_z_rthigh = (rshank - rthigh)/np.linalg.norm(rshank - rthigh)
            e_x_rthigh = np.cross(e_y0_rthigh, e_z_rthigh)/np.linalg.norm(np.cross(e_y0_rthigh, e_z_rthigh))
            e_y_rthigh = np.cross(e_z_rthigh, e_x_rthigh)
            rot_rthigh = np.array([e_x_rthigh, e_y_rthigh, e_z_rthigh]).T

            #左大腿節座標系（原点はlthigh）
            e_y0_lthigh = lknee - lknee2
            e_z_lthigh = (lshank - lthigh)/np.linalg.norm(lshank - lthigh)
            e_x_lthigh = np.cross(e_y0_lthigh, e_z_lthigh)/np.linalg.norm(np.cross(e_y0_lthigh, e_z_lthigh))
            e_y_lthigh = np.cross(e_z_lthigh, e_x_lthigh)
            rot_lthigh = np.array([e_x_lthigh, e_y_lthigh, e_z_lthigh]).T

            #右下腿節座標系（原点はrshank）
            e_y0_rshank = rknee2 - rknee
            e_z_rshank = (rfoot - rshank)/np.linalg.norm(rfoot - rshank)
            e_x_rshank = np.cross(e_y0_rshank, e_z_rshank)/np.linalg.norm(np.cross(e_y0_rshank, e_z_rshank))
            e_y_rshank = np.cross(e_z_rshank, e_x_rshank)
            rot_rshank = np.array([e_x_rshank, e_y_rshank, e_z_rshank]).T

            #左下腿節座標系（原点はlshank）
            e_y0_lshank = lknee - lknee2
            e_z_lshank = (lfoot - lshank)/np.linalg.norm(lfoot - lshank)
            e_x_lshank = np.cross(e_y0_lshank, e_z_lshank)/np.linalg.norm(np.cross(e_y0_lshank, e_z_lshank))
            e_y_lshank = np.cross(e_z_lshank, e_x_lshank)
            rot_lshank = np.array([e_x_lshank, e_y_lshank, e_z_lshank]).T

            # #右足節座標系（原点はrfoot）
            # e_z_rfoot = e_z_pelvis
            # e_x0_rfoot = rtoe - rhee
            # e_y_rfoot = np.cross(e_z_rfoot, e_x0_rfoot)/np.linalg.norm(np.cross(e_z_rfoot, e_x0_rfoot))
            # e_x_rfoot = np.cross(e_y_rfoot, e_z_rfoot)
            # rot_rfoot = np.array([e_x_rfoot, e_y_rfoot, e_z_rfoot]).T

            # #左足節座標系（原点はlfoot）
            # e_z_lfoot = e_z_pelvis
            # e_x0_lfoot = ltoe - lhee
            # e_y_lfoot = np.cross(e_z_lfoot, e_x0_lfoot)/np.linalg.norm(np.cross(e_z_lfoot, e_x0_lfoot))
            # e_x_lfoot = np.cross(e_y_lfoot, e_z_lfoot)
            # rot_lfoot = np.array([e_x_lfoot, e_y_lfoot, e_z_lfoot]).T

            #右足節座標系 AIST参照（原点はrfoot）
            e_z_rfoot = (rtoe - rhee) / np.linalg.norm(rtoe - rhee)
            e_y0_rfoot = rank - rank2
            e_x_rfoot = np.cross(e_z_rfoot, e_y0_rfoot)/np.linalg.norm(np.cross(e_z_rfoot, e_y0_rfoot))
            e_y_rfoot = np.cross(e_z_rfoot, e_x_rfoot)
            rot_rfoot = np.array([e_x_rfoot, e_y_rfoot, e_z_rfoot]).T

            # e_x_rfoot = (rtoe - rhee) / np.linalg.norm(rtoe - rhee)
            # e_y0_rfoot = rank2 - rank
            # e_z_rfoot = np.cross(e_x_rfoot, e_y0_rfoot)/np.linalg.norm(np.cross(e_x_rfoot, e_y0_rfoot))
            # e_y_rfoot = np.cross(e_z_rfoot, e_x_rfoot)
            # rot_rfoot = np.array([e_x_rfoot, e_y_rfoot, e_z_rfoot]).T

            #左足節座標系 AIST参照（原点はlfoot）
            e_z_lfoot = (ltoe - lhee) / np.linalg.norm(ltoe - lhee)
            e_y0_lfoot = lank2 - lank
            e_x_lfoot = np.cross(e_z_lfoot, e_y0_lfoot)/np.linalg.norm(np.cross(e_z_lfoot, e_y0_lfoot))
            e_y_lfoot = np.cross(e_z_lfoot, e_x_lfoot)
            rot_lfoot = np.array([e_x_lfoot, e_y_lfoot, e_z_lfoot]).T

            # e_x_lfoot = (ltoe - lhee) / np.linalg.norm(ltoe - lhee)
            # e_y0_lfoot = lank - lank2
            # e_z_lfoot = np.cross(e_x_lfoot, e_y0_lfoot)/np.linalg.norm(np.cross(e_x_lfoot, e_y0_lfoot))
            # e_y_lfoot = np.cross(e_z_lfoot, e_x_lfoot)
            # rot_lfoot = np.array([e_x_lfoot, e_y_lfoot, e_z_lfoot]).T


            r_hip_realative_rotation = np.dot(np.linalg.inv(rot_pelvis), rot_rthigh)
            r_hip_angle = R.from_matrix(r_hip_realative_rotation).as_euler('yzx', degrees=True)[0]
            l_hip_realative_rotation = np.dot(np.linalg.inv(rot_pelvis), rot_lthigh)
            l_hip_angle = R.from_matrix(l_hip_realative_rotation).as_euler('yzx', degrees=True)[0]
            r_knee_realative_rotation = np.dot(np.linalg.inv(rot_rthigh), rot_rshank)
            r_knee_angle = R.from_matrix(r_knee_realative_rotation).as_euler('yzx', degrees=True)[0]
            l_knee_realative_rotation = np.dot(np.linalg.inv(rot_lthigh), rot_lshank)
            l_knee_angle = R.from_matrix(l_knee_realative_rotation).as_euler('yzx', degrees=True)[0]
            r_ankle_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rfoot)
            r_ankle_angle = R.from_matrix(r_ankle_realative_rotation).as_euler('yzx', degrees=True)[0]
            l_ankle_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lfoot)
            l_ankle_angle = R.from_matrix(l_ankle_realative_rotation).as_euler('yzx', degrees=True)[0]


            angle_list.append([r_hip_angle, l_hip_angle, r_knee_angle, l_knee_angle, r_ankle_angle, l_ankle_angle])






            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)

            ax.scatter(rasi[0], rasi[1], rasi[2], label='rasi')
            ax.scatter(lasi[0], lasi[1], lasi[2], label='lasi')
            ax.scatter(rpsi[0], rpsi[1], rpsi[2], label='rpsi')
            ax.scatter(lpsi[0], lpsi[1], lpsi[2], label='lpsi')
            ax.scatter(rfoot[0], rfoot[1], rfoot[2], label='rfoot')
            ax.scatter(lfoot[0], lfoot[1], lfoot[2], label='lfoot')
            ax.scatter(rshank[0], rshank[1], rshank[2], label='rshank')
            ax.scatter(lshank[0], lshank[1], lshank[2], label='lshank')
            ax.scatter(rtoe[0], rtoe[1], rtoe[2], label='rtoe')
            ax.scatter(ltoe[0], ltoe[1], ltoe[2], label='ltoe')
            ax.scatter(rhee[0], rhee[1], rhee[2], label='rhee')
            ax.scatter(lhee[0], lhee[1], lhee[2], label='lhee')
            ax.scatter(lumbar[0], lumbar[1], lumbar[2], label='lumbar')

            # ax.plot([rasi[0], lasi[0]], [rasi[1], lasi[1]], [rasi[2], lasi[2]], color='blue')
            # ax.plot([rpsi[0], lpsi[0]], [rpsi[1], lpsi[1]], [rpsi[2], lpsi[2]], color='blue')
            # ax.plot([rasi[0], rpsi[0]], [rasi[1], rpsi[1]], [rasi[2], rpsi[2]], color='blue')
            # ax.plot([lasi[0], lpsi[0]], [lasi[1], lpsi[1]], [lasi[2], lpsi[2]], color='blue')
            # ax.plot([rasi[0], rshank[0]], [rasi[1], rshank[1]], [rasi[2], rshank[2]], color='blue')
            # ax.plot([lasi[0], lshank[0]], [lasi[1], lshank[1]], [lasi[2], lshank[2]], color='blue')
            # ax.plot([rshank[0], rfoot[0]], [rshank[1], rfoot[1]], [rshank[2], rfoot[2]], color='blue')
            # ax.plot([lshank[0], lfoot[0]], [lshank[1], lfoot[1]], [lshank[2], lfoot[2]], color='blue')
            # ax.plot([rhee[0], rtoe[0]], [rhee[1], rtoe[1]], [rhee[2], rtoe[2]], color='blue')
            # ax.plot([lhee[0], ltoe[0]], [lhee[1], ltoe[1]], [lhee[2], ltoe[2]], color='blue')

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

            ax.scatter(hip[0], hip[1], hip[2], label='hip')
            ax.scatter(rthigh[0], rthigh[1], rthigh[2], label='rthigh')
            ax.scatter(lthigh[0], lthigh[1], lthigh[2], label='lthigh')
            ax.scatter(rshank[0], rshank[1], rshank[2], label='rshank')
            ax.scatter(lshank[0], lshank[1], lshank[2], label='lshank')
            ax.scatter(rfoot[0], rfoot[1], rfoot[2], label='rfoot')
            ax.scatter(lfoot[0], lfoot[1], lfoot[2], label='lfoot')

            plt.legend()
            plt.show()




        angle_array = np.array(angle_list)
        df = pd.DataFrame({"r_hip_angle": angle_array[:, 0], "r_knee_angle": angle_array[:, 2], "r_ankle_angle": angle_array[:, 4], "l_hip_angle": angle_array[:, 1], "l_knee_angle": angle_array[:, 3], "l_ankle_angle": angle_array[:, 5]})
        df.index = df.index + full_range.start
        df.to_csv(os.path.join(os.path.dirname(csv_path), f"angle_{os.path.basename(csv_path)}"))

if __name__ == "__main__":
    main()

