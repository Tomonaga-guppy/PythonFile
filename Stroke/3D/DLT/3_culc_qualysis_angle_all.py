import pandas as pd
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import json

def read_3DMC(csv_path, down_hz):
    col_names = range(1,100)  #データの形が汚い場合に対応するためあらかじめ列数(100:適当)を設定
    df = pd.read_csv(csv_path, names=col_names, sep='\t', skiprows=[0,1,2,3,4,5,6,7,8,10])  #Qualisis
    df.columns = df.iloc[0]  # 最初の行をヘッダーに
    df = df.drop(0).reset_index(drop=True)  # ヘッダーにした行をデータから削除し、インデックスをリセット
    # print(f"df = {df}")

    # # dfの両端2.5%を削除 補間時の乱れを防ぐため
    df_index = df.index
    df = df.iloc[int(len(df)*0.025):int(len(df)*0.975)]

    if down_hz:
        df_down = df[::4].reset_index()
        df_down.index = df_down.index + df_down['index'].iloc[0]//4
        df_down = df_down.drop("index", axis=1)
        df95_index = df_down.index
        sampling_freq = 30
    else:
        df_down = df
        df95_index = df_index[df.index]
        df.index = df95_index
        sampling_freq = 120

    marker_set = ["RASI", "LASI", "RPSI", "LPSI","RKNE","LKNE", "RANK","LANK","RTOE","LTOE","RHEE","LHEE", "RKNE2", "LKNE2", "RANK2", "LANK2"]
    marker_dict = dict()
    xyz_list = ['X', 'Y', 'Z']

    for marker in marker_set:
        for i, xyz in enumerate(xyz_list):
            key_index = df_down.columns.get_loc(f'{marker}')
            marker_rows = (key_index-1)*3 + i
            marker_dict[f"{marker}_{xyz}"] = marker_rows

    marker_set_df = pd.DataFrame(columns=marker_dict.keys())
    for column in marker_set_df.columns:
        marker_set_df[column] = df_down.iloc[:, marker_dict[column]].values
    marker_set_df.index = df95_index

    marker_set_df = marker_set_df.apply(pd.to_numeric, errors='coerce')  #文字列として読み込まれたデータを数値に変換
    interpolated_df = marker_set_df.interpolate(method='spline', order=3)  #3次スプライン補間
    marker_set_df = interpolated_df.apply(butter_lowpass_fillter, args=(4, 6, sampling_freq, df95_index))

    marker_set_df.to_csv(os.path.join(os.path.dirname(csv_path), f"marker_set_{os.path.basename(csv_path).split('.')[0]}.csv"))

    return marker_set_df, df95_index

def butter_lowpass_fillter(column_data, order, cutoff_freq, sampling_freq, frame_list):  #4次のバターワースローパスフィルタ
    nyquist_freq = sampling_freq / 2
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    data_to_filter = column_data[frame_list]
    filtered_data = filtfilt(b, a, data_to_filter)
    column_data[frame_list] = filtered_data
    return column_data

def main():
    down_hz = False
    csv_path_dir = r"F:\Tomson\gait_pattern\20240912\qualisys"
    csv_paths = glob.glob(os.path.join(csv_path_dir, "sub3*normal*f0001*.tsv"))

    for i, csv_path in enumerate(csv_paths):
        marker_set_df, df_index = read_3DMC(csv_path, down_hz)
        # keypoints = marker_set_df.values
        # keypoints_mocap = keypoints.reshape(-1, len(marker_set), 3)  #xyzで組になるように変形
        # print(keypoints_mocap.shape)
        # print(f"marker_set_df = {marker_set_df}")
        print(f"csv_path = {csv_path}")

        angle_list = []
        bector_list = []
        dist_list = []

        rasi = marker_set_df[['RASI_X', 'RASI_Y', 'RASI_Z']].to_numpy()
        lasi = marker_set_df[['LASI_X', 'LASI_Y', 'LASI_Z']].to_numpy()
        rpsi = marker_set_df[['RPSI_X', 'RPSI_Y', 'RPSI_Z']].to_numpy()
        lpsi = marker_set_df[['LPSI_X', 'LPSI_Y', 'LPSI_Z']].to_numpy()
        rknee = marker_set_df[['RKNE_X', 'RKNE_Y', 'RKNE_Z']].to_numpy()
        lknee = marker_set_df[['LKNE_X', 'LKNE_Y', 'LKNE_Z']].to_numpy()
        rank = marker_set_df[['RANK_X', 'RANK_Y', 'RANK_Z']].to_numpy()
        lank = marker_set_df[['LANK_X', 'LANK_Y', 'LANK_Z']].to_numpy()
        rtoe = marker_set_df[['RTOE_X', 'RTOE_Y', 'RTOE_Z']].to_numpy()
        ltoe = marker_set_df[['LTOE_X', 'LTOE_Y', 'LTOE_Z']].to_numpy()
        rhee = marker_set_df[['RHEE_X', 'RHEE_Y', 'RHEE_Z']].to_numpy()
        lhee = marker_set_df[['LHEE_X', 'LHEE_Y', 'LHEE_Z']].to_numpy()
        rknee2 = marker_set_df[['RKNE2_X', 'RKNE2_Y', 'RKNE2_Z']].to_numpy()
        lknee2 = marker_set_df[['LKNE2_X', 'LKNE2_Y', 'LKNE2_Z']].to_numpy()
        rank2 = marker_set_df[['RANK2_X', 'RANK2_Y', 'RANK2_Z']].to_numpy()
        lank2 = marker_set_df[['LANK2_X', 'LANK2_Y', 'LANK2_Z']].to_numpy()

        # for frame_num in full_range:
        for frame_num in df_index:
            frame_num = frame_num - df_index[0]
            #メモ
            d_asi = np.linalg.norm(rasi[frame_num,:] - lasi[frame_num,:])
            d_leg = (np.linalg.norm(rank[frame_num,:] - rasi[frame_num,:]) + np.linalg.norm(lank[frame_num, :] - lasi[frame_num,:]) / 2)
            r = 0.012 #9/12
            h = 1.74 #sub3
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

            r_hip_realative_rotation = np.dot(np.linalg.inv(rot_pelvis), rot_rthigh)
            l_hip_realative_rotation = np.dot(np.linalg.inv(rot_pelvis), rot_lthigh)
            r_knee_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rthigh)
            l_knee_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lthigh)
            r_ankle_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rfoot)
            l_ankle_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lfoot)

            r_hip_angle_flex_ext_flec_ext = R.from_matrix(r_hip_realative_rotation).as_euler('yzx', degrees=True)[0]
            l_hip_angle_flex_ext = R.from_matrix(l_hip_realative_rotation).as_euler('yzx', degrees=True)[0]
            r_knee_angle_flex_ext =  R.from_matrix(r_knee_realative_rotation).as_euler('yzx', degrees=True)[0]
            l_knee_angle_flex_ext = R.from_matrix(l_knee_realative_rotation).as_euler('yzx', degrees=True)[0]
            r_ankle_angle_flex_ext = R.from_matrix(r_ankle_realative_rotation).as_euler('yzx', degrees=True)[0]
            l_ankle_angle_flex_ext = R.from_matrix(l_ankle_realative_rotation).as_euler('yzx', degrees=True)[0]

            r_hip_angle_abd_add = R.from_matrix(r_hip_realative_rotation).as_euler('yzx', degrees=True)[1]
            l_hip_angle_abd_add = R.from_matrix(l_hip_realative_rotation).as_euler('yzx', degrees=True)[1]
            r_knee_angle_abd_add =  R.from_matrix(r_knee_realative_rotation).as_euler('yzx', degrees=True)[1]
            l_knee_angle_abd_add = R.from_matrix(l_knee_realative_rotation).as_euler('yzx', degrees=True)[1]
            r_ankle_angle_abd_add = R.from_matrix(r_ankle_realative_rotation).as_euler('yzx', degrees=True)[1]
            l_ankle_angle_abd_add = R.from_matrix(l_ankle_realative_rotation).as_euler('yzx', degrees=True)[1]

            r_hip_angle_int_ext = R.from_matrix(r_hip_realative_rotation).as_euler('yzx', degrees=True)[2]
            l_hip_angle_int_ext = R.from_matrix(l_hip_realative_rotation).as_euler('yzx', degrees=True)[2]
            r_knee_angle_int_ext =  R.from_matrix(r_knee_realative_rotation).as_euler('yzx', degrees=True)[2]
            l_knee_angle_int_ext = R.from_matrix(l_knee_realative_rotation).as_euler('yzx', degrees=True)[2]
            r_ankle_angle_int_ext = R.from_matrix(r_ankle_realative_rotation).as_euler('yzx', degrees=True)[2]
            l_ankle_angle_int_ext = R.from_matrix(l_ankle_realative_rotation).as_euler('yzx', degrees=True)[2]

            r_hip_angle_flex_ext_flec_ext = 360 + r_hip_angle_flex_ext_flec_ext if r_hip_angle_flex_ext_flec_ext < 0 else r_hip_angle_flex_ext_flec_ext
            l_hip_angle_flex_ext = 360 + l_hip_angle_flex_ext if l_hip_angle_flex_ext < 0 else l_hip_angle_flex_ext
            r_knee_angle_flex_ext = 360 + r_knee_angle_flex_ext if r_knee_angle_flex_ext < 0 else r_knee_angle_flex_ext
            l_knee_angle_flex_ext = 360 + l_knee_angle_flex_ext if l_knee_angle_flex_ext < 0 else l_knee_angle_flex_ext
            r_ankle_angle_flex_ext = 360 + r_ankle_angle_flex_ext if r_ankle_angle_flex_ext < 0 else r_ankle_angle_flex_ext
            l_ankle_angle_flex_ext = 360 + l_ankle_angle_flex_ext if l_ankle_angle_flex_ext < 0 else l_ankle_angle_flex_ext

            r_hip_angle_flex_ext_flec_ext = 180 - r_hip_angle_flex_ext_flec_ext
            l_hip_angle_flex_ext = 180 - l_hip_angle_flex_ext
            r_knee_angle_flex_ext = 180 - r_knee_angle_flex_ext
            l_knee_angle_flex_ext = 180 - l_knee_angle_flex_ext
            r_ankle_angle_flex_ext = 90 - r_ankle_angle_flex_ext
            l_ankle_angle_flex_ext = 90 - l_ankle_angle_flex_ext

            angles = [r_hip_angle_flex_ext_flec_ext, l_hip_angle_flex_ext, r_knee_angle_flex_ext, l_knee_angle_flex_ext, r_ankle_angle_flex_ext, l_ankle_angle_flex_ext,
                      r_hip_angle_abd_add, l_hip_angle_abd_add, r_knee_angle_abd_add, l_knee_angle_abd_add, r_ankle_angle_abd_add, l_ankle_angle_abd_add,
                      r_hip_angle_int_ext, l_hip_angle_int_ext, r_knee_angle_int_ext, l_knee_angle_int_ext, r_ankle_angle_int_ext, l_ankle_angle_int_ext]
            angle_list.append(angles)

            plot_flag = False
            if plot_flag:
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

            #骨盤とかかとの距離計算
            heel = lhee[frame_num, :]
            bector = heel - hip[:]
            bector_list.append(bector)

        angle_array = np.array(angle_list)
        # print(f"angle_array = {angle_array}")
        # print(f"angle_array.shape = {angle_array.shape}")
        df = pd.DataFrame({"r_hip_angle_flex_ext_flec_ext": angle_array[:, 0], "r_knee_angle_flex_ext": angle_array[:, 2], "r_ankle_angle_flex_ext": angle_array[:, 4],
                           "l_hip_angle_flex_ext": angle_array[:, 1], "l_knee_angle_flex_ext": angle_array[:, 3], "l_ankle_angle_flex_ext": angle_array[:, 5],
                           "r_hip_angle_abd_add": angle_array[:, 6], "r_knee_angle_abd_add": angle_array[:, 8], "r_ankle_angle_abd_add": angle_array[:, 10],
                           "l_hip_angle_abd_add": angle_array[:, 7], "l_knee_angle_abd_add": angle_array[:, 9], "l_ankle_angle_abd_add": angle_array[:, 11],
                           "r_hip_angle_int_ext": angle_array[:, 12], "r_knee_angle_int_ext": angle_array[:, 14], "r_ankle_angle_int_ext": angle_array[:, 16],
                           "l_hip_angle_int_ext": angle_array[:, 13], "l_knee_angle_int_ext": angle_array[:, 15], "l_ankle_angle_int_ext": angle_array[:, 17]})
        # df.index = df.index + full_range.start
        df.index = df_index
        if down_hz:
            df.to_csv(os.path.join(os.path.dirname(csv_path), f"angle_30Hz_{os.path.basename(csv_path).split('.')[0]}.csv"))
            print(f"angle_df saved in angle_30Hz_{os.path.basename(csv_path).split('.')[0]}.csv")
        else:
            df.to_csv(os.path.join(os.path.dirname(csv_path), f"angle_120Hz_{os.path.basename(csv_path).split('.')[0]}.csv"))
            print(f"aangle_df saved in angle_120Hz_{os.path.basename(csv_path).split('.')[0]}.csv")


        bector_array = np.array(bector_list)
        lhee_pel_z = bector_array[:, 0]
        # lhee_pel_z = bector_array[:, 2]  #motiveの場合
        df = pd.DataFrame({"lhee_pel_z":lhee_pel_z})
        df.index = df_index
        df = df.sort_values(by="lhee_pel_z", ascending=True)
        # df = df.sort_values(by="lhee_pel_z", ascending=False)  #motiveの場合
        # print(f"df2 = {df}")
        ic_list = df.index[:120].values
        # print(f"ic_list = {ic_list}")

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
        filtered_list = sorted(filtered_list)
        print(f"フィルタリング後のリスト:{filtered_list}")

        if down_hz:
            np.save(os.path.join(os.path.dirname(csv_path), f"ic_frame_30Hz_{os.path.basename(csv_path).split('.')[0]}.npy"), filtered_list)
        else:
            np.save(os.path.join(os.path.dirname(csv_path), f"ic_frame_120Hz_{os.path.basename(csv_path).split('.')[0]}.npy"), filtered_list)


if __name__ == "__main__":
    main()

