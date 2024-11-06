import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

down_hz = True
csv_path_dir = Path(r"G:\gait_pattern\20240912\qualisys")
csv_paths = list(csv_path_dir.glob("sub3*normal*.tsv"))

def read_3DMC(csv_path, down_hz):
    col_names = range(1,100)  #データの形が汚い場合に対応するためあらかじめ列数(100:適当)を設定
    df = pd.read_csv(csv_path, names=col_names, sep='\t', skiprows=[0,1,2,3,4,5,6,7,8,10])  #Qualisis
    df.columns = df.iloc[0]  # 最初の行をヘッダーに
    df = df.drop(0).reset_index(drop=True)  # ヘッダーにした行をデータから削除し、インデックスをリセット

    if down_hz:
        df_down = df[::4].reset_index(drop=True)  #30Hzにダウンサンプリング
        sampling_freq = 30
    else:
        df_down = df
        sampling_freq = 120

    print(f"df: {df}")
    print(f"df_down: {df_down}")

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

    marker_set_df = marker_set_df.apply(pd.to_numeric, errors='coerce')  #文字列として読み込まれたデータを数値に変換
    marker_set_df.replace(0, np.nan, inplace=True)  #0をnanに変換

    df_copy = marker_set_df.copy()
    valid_index_mask = df_copy.notna().all(axis=1)
    valid_index = df_copy[valid_index_mask].index
    valid_index = pd.Index(range(valid_index.min(), valid_index.max() + 1))  #欠損値がない行のインデックスを範囲で取得、この範囲の値を解析に使用する
    marker_set_df = marker_set_df.loc[valid_index, :]  #欠損値のない行のみを抽出
    interpolated_df = marker_set_df.interpolate(method='spline', order=3)  #3次スプライン補間
    marker_set_fin_df = interpolated_df.apply(butter_lowpass_fillter, args=(4, 6, sampling_freq))  #4次のバターワースローパスフィルタ

    output_csv_path = csv_path.with_name(f"marker_set_{csv_path.stem}.csv")
    marker_set_fin_df.to_csv(output_csv_path)

    return marker_set_fin_df, valid_index

def butter_lowpass_fillter(column_data, order, cutoff_freq, sampling_freq):  #4次のバターワースローパスフィルタ
    nyquist_freq = sampling_freq / 2
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    data_to_filter = column_data
    filtered_data = filtfilt(b, a, data_to_filter)
    column_data = filtered_data
    return column_data

def main():

    for i, csv_path in enumerate(csv_paths):
        marker_set_df, valid_index = read_3DMC(csv_path, down_hz)

        print(f"csv_path = {csv_path}")
        print(f"valid_index = {valid_index}")

        angle_list = []
        bector_list_r = []
        bector_list_l = []

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


        for frame_num in valid_index:
            frame_num = frame_num - valid_index.min()
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

            r_hip_angle_rot = R.from_matrix(r_hip_realative_rotation)
            l_hip_angle_rot = R.from_matrix(l_hip_realative_rotation)
            r_knee_angle_rot = R.from_matrix(r_knee_realative_rotation)
            l_knee_angle_rot = R.from_matrix(l_knee_realative_rotation)
            r_ankle_angle_rot = R.from_matrix(r_ankle_realative_rotation)
            l_ankle_angle_rot = R.from_matrix(l_ankle_realative_rotation)

            # r_hip_angle = r_hip_angle_rot.as_euler('yzx', degrees=True)[0]
            # l_hip_angle = l_hip_angle_rot.as_euler('yzx', degrees=True)[0]
            # r_knee_angle = r_knee_angle_rot.as_euler('yzx', degrees=True)[0]
            # l_knee_angle = l_knee_angle_rot.as_euler('yzx', degrees=True)[0]
            # r_ankle_angle = r_ankle_angle_rot.as_euler('yzx', degrees=True)[0]
            # l_ankle_angle = l_ankle_angle_rot.as_euler('yzx', degrees=True)[0]

            # r_hip_angle = 360 + r_hip_angle if r_hip_angle < 0 else r_hip_angle
            # l_hip_angle = 360 + l_hip_angle if l_hip_angle < 0 else l_hip_angle
            # r_knee_angle = 360 + r_knee_angle if r_knee_angle < 0 else r_knee_angle
            # l_knee_angle = 360 + l_knee_angle if l_knee_angle < 0 else l_knee_angle
            # r_ankle_angle = 360 + r_ankle_angle if r_ankle_angle < 0 else r_ankle_angle
            # l_ankle_angle = 360 + l_ankle_angle if l_ankle_angle < 0 else l_ankle_angle

            # r_hip_angle = 180 - r_hip_angle
            # l_hip_angle = 180 - l_hip_angle
            # r_knee_angle = 180 - r_knee_angle
            # l_knee_angle = 180 - l_knee_angle
            # r_ankle_angle = 90 - r_ankle_angle
            # l_ankle_angle = 90 - l_ankle_angle

            # 屈曲（flexion）・伸展（extension）の角度を計算
            r_hip_angle_flex_ext = r_hip_angle_rot.as_euler('yzx', degrees=True)[0]
            l_hip_angle_flex_ext = l_hip_angle_rot.as_euler('yzx', degrees=True)[0]
            r_knee_angle_flex_ext = r_knee_angle_rot.as_euler('yzx', degrees=True)[0]
            l_knee_angle_flex_ext = l_knee_angle_rot.as_euler('yzx', degrees=True)[0]
            r_ankle_angle_flex_ext = r_ankle_angle_rot.as_euler('yzx', degrees=True)[0]
            l_ankle_angle_flex_ext = l_ankle_angle_rot.as_euler('yzx', degrees=True)[0]

            r_hip_angle_flex_ext = 360 + r_hip_angle_flex_ext if r_hip_angle_flex_ext < 0 else r_hip_angle_flex_ext
            l_hip_angle_flex_ext = 360 + l_hip_angle_flex_ext if l_hip_angle_flex_ext < 0 else l_hip_angle_flex_ext
            r_knee_angle_flex_ext = 360 + r_knee_angle_flex_ext if r_knee_angle_flex_ext < 0 else r_knee_angle_flex_ext
            l_knee_angle_flex_ext = 360 + l_knee_angle_flex_ext if l_knee_angle_flex_ext < 0 else l_knee_angle_flex_ext
            r_ankle_angle_flex_ext = 360 + r_ankle_angle_flex_ext if r_ankle_angle_flex_ext < 0 else r_ankle_angle_flex_ext
            l_ankle_angle_flex_ext = 360 + l_ankle_angle_flex_ext if l_ankle_angle_flex_ext < 0 else l_ankle_angle_flex_ext

            r_hip_angle_flex_ext = 180 - r_hip_angle_flex_ext
            l_hip_angle_flex_ext = 180 - l_hip_angle_flex_ext
            r_knee_angle_flex_ext = 180 - r_knee_angle_flex_ext
            l_knee_angle_flex_ext = 180 - l_knee_angle_flex_ext
            r_ankle_angle_flex_ext = 90 - r_ankle_angle_flex_ext
            l_ankle_angle_flex_ext = 90 - l_ankle_angle_flex_ext

            # # 外転・内転/ 外旋・内旋の角度を計算
            # taikan = e_z_pelvis.copy()
            # r_thigh = e_z_rthigh.copy()
            # l_thigh = e_z_lthigh.copy()
            # r_shank = e_z_rshank.copy()
            # l_shank = e_z_lshank.copy()
            # r_foot = e_z_rfoot.copy()
            # l_foot = e_z_lfoot.copy()

            # def calculate_angle(vector1, vector2):  #(frame, xyz)または(frame, xy)の配列を入力)
            #     if len(vector1) == 3:  #3Dベクトル
            #         dot_product = np.dot(vector1, vector2)
            #         cross_product = np.cross(vector1, vector2)
            #         angle = np.rad2deg(np.arctan2(cross_product, dot_product))  # atan2(y, x : y/x)を使って角度を計算
            #     return angle

            # r_hip_angle = calculate_angle(r_thigh, taikan)
            # l_hip_angle = calculate_angle(l_thigh, taikan)
            # r_knee_angle = calculate_angle(r_thigh, r_shank)
            # l_knee_angle = calculate_angle(l_thigh, l_shank)
            # r_ankle_angle = calculate_angle(r_foot, r_shank)
            # l_ankle_angle = calculate_angle(l_foot, l_shank)

            # r_hip_angle_abd_add = r_hip_angle[0]
            # l_hip_angle_abd_add = l_hip_angle[0]
            # r_knee_angle_abd_add = r_knee_angle[0]
            # l_knee_angle_abd_add = l_knee_angle[0]
            # r_ankle_angle_abd_add = r_ankle_angle[0]
            # l_ankle_angle_abd_add = l_ankle_angle[0]

            # r_hip_angle_ext_int = r_hip_angle[1]
            # l_hip_angle_ext_int = l_hip_angle[1]
            # r_knee_angle_ext_int = r_knee_angle[1]
            # l_knee_angle_ext_int = l_knee_angle[1]
            # r_ankle_angle_ext_int = r_ankle_angle[1]
            # l_ankle_angle_ext_int = l_ankle_angle[1]


            # 外転（abduction）・内転（adduction）の角度を計算
            r_hip_angle_abd_add = - r_hip_angle_rot.as_euler('xyz', degrees=True)[0]
            l_hip_angle_abd_add = l_hip_angle_rot.as_euler('xyz', degrees=True)[0]
            r_knee_angle_abd_add = - r_knee_angle_rot.as_euler('xyz', degrees=True)[0]
            l_knee_angle_abd_add = l_knee_angle_rot.as_euler('xyz', degrees=True)[0]
            r_ankle_angle_abd_add = - r_ankle_angle_rot.as_euler('xyz', degrees=True)[0]
            l_ankle_angle_abd_add = l_ankle_angle_rot.as_euler('xyz', degrees=True)[0]

            r_hip_angle_abd_add = 180- r_hip_angle_abd_add
            l_hip_angle_abd_add = 180- l_hip_angle_abd_add

            r_hip_angle_abd_add = r_hip_angle_abd_add - 360 if r_hip_angle_abd_add > 180 else r_hip_angle_abd_add
            l_hip_angle_abd_add = l_hip_angle_abd_add - 360 if l_hip_angle_abd_add > 180 else l_hip_angle_abd_add


            # r_hip_angle_abd_add = - r_hip_angle_rot.as_euler('yzx', degrees=True)[2]
            # l_hip_angle_abd_add = l_hip_angle_rot.as_euler('yzx', degrees=True)[2]
            # r_knee_angle_abd_add = - r_knee_angle_rot.as_euler('yzx', degrees=True)[2]
            # l_knee_angle_abd_add = l_knee_angle_rot.as_euler('yzx', degrees=True)[2]
            # r_ankle_angle_abd_add = - r_ankle_angle_rot.as_euler('yzx', degrees=True)[2]
            # l_ankle_angle_abd_add = l_ankle_angle_rot.as_euler('yzx', degrees=True)[2]


            # 外旋（external rotation）・内旋（internal rotation）の角度を計算
            r_hip_angle_ext_int = r_hip_angle_rot.as_euler('yzx', degrees=True)[1]
            l_hip_angle_ext_int = l_hip_angle_rot.as_euler('yzx', degrees=True)[1]
            r_knee_angle_ext_int = r_knee_angle_rot.as_euler('yzx', degrees=True)[1]
            l_knee_angle_ext_int = l_knee_angle_rot.as_euler('yzx', degrees=True)[1]
            r_ankle_angle_ext_int = r_ankle_angle_rot.as_euler('yzx', degrees=True)[1]
            l_ankle_angle_ext_int = l_ankle_angle_rot.as_euler('yzx', degrees=True)[1]

            # angles = [r_hip_angle, l_hip_angle, r_knee_angle, l_knee_angle, r_ankle_angle, l_ankle_angle]
            angles = [r_hip_angle_flex_ext, l_hip_angle_flex_ext, r_knee_angle_flex_ext, l_knee_angle_flex_ext, r_ankle_angle_flex_ext, l_ankle_angle_flex_ext,
                      r_hip_angle_abd_add, l_hip_angle_abd_add, r_knee_angle_abd_add, l_knee_angle_abd_add, r_ankle_angle_abd_add, l_ankle_angle_abd_add,
                      r_hip_angle_ext_int, l_hip_angle_ext_int, r_knee_angle_ext_int, l_knee_angle_ext_int, r_ankle_angle_ext_int, l_ankle_angle_ext_int]

            angle_list.append(angles)

            plot_flag = False
            if plot_flag:  #各キーポイントの位置をプロット
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
            heel_r = rhee[frame_num, :]
            bector_r = heel_r - hip
            bector_list_r.append(bector_r)

            heel_l = lhee[frame_num, :]
            bector_l = heel_l - hip[:]
            bector_list_l.append(bector_l)

        angle_array = np.array(angle_list)
        # angle_df = pd.DataFrame({"r_hip_angle": angle_array[:, 0], "r_knee_angle": angle_array[:, 2], "r_ankle_angle": angle_array[:, 4], "l_hip_angle": angle_array[:, 1], "l_knee_angle": angle_array[:, 3], "l_ankle_angle": angle_array[:, 5]})

        angle_df = pd.DataFrame({"r_hip_angle_flex_ext": angle_array[:, 0], "l_hip_angle_flex_ext": angle_array[:, 1], "r_knee_angle_flex_ext": angle_array[:, 2],
                                 "l_knee_angle_flex_ext": angle_array[:, 3], "r_ankle_angle_flex_ext": angle_array[:, 4], "l_ankle_angle_flex_ext": angle_array[:, 5],
                                 "r_hip_angle_abd_add": angle_array[:, 6], "l_hip_angle_abd_add": angle_array[:, 7], "r_knee_angle_abd_add": angle_array[:, 8],
                                 "l_knee_angle_abd_add": angle_array[:, 9], "r_ankle_angle_abd_add": angle_array[:, 10], "l_ankle_angle_abd_add": angle_array[:, 11],
                                 "r_hip_angle_ext_int": angle_array[:, 12], "l_hip_angle_ext_int": angle_array[:, 13], "r_knee_angle_ext_int": angle_array[:, 14],
                                 "l_knee_angle_ext_int": angle_array[:, 15], "r_ankle_angle_ext_int": angle_array[:, 16], "l_ankle_angle_ext_int": angle_array[:, 17]})

        angle_df.index = valid_index
        if down_hz:
            angle_df.to_csv(csv_path.with_name(f"angle_30Hz_{csv_path.stem}.csv"))
        else:
            angle_df.to_csv(csv_path.with_name(f"angle_120Hz_{csv_path.stem}.csv"))

        bector_r_array = np.array(bector_list_r)
        rhee_pel_z = bector_r_array[:, 0]
        # rhee_pel_z = bector_array[:, 2]  #motiveの場合
        df_r = pd.DataFrame({"rhee_pel_z":rhee_pel_z})
        df_r.index = valid_index
        df_r = df_r.sort_values(by="rhee_pel_z", ascending=True)
        # df = df.sort_values(by="rhee_pel_z", ascending=False)  #motiveの場合
        ic_list_r = df_r.index[:120].values

        bector_l_array = np.array(bector_list_l)
        lhee_pel_z = bector_l_array[:, 0]
        # lhee_pel_z = bector_array[:, 2]  #motiveの場合
        df_l = pd.DataFrame({"lhee_pel_z":lhee_pel_z})
        df_l.index = valid_index
        df_l = df_l.sort_values(by="lhee_pel_z", ascending=True)
        # df = df.sort_values(by="lhee_pel_z", ascending=False)  #motiveの場合
        ic_list_l = df_l.index[:120].values


        # plt.figure()
        # plt.plot(df.index, df["dist"])
        # plt.show()

        filtered_list_r = []
        skip_values_r = set()
        for value in ic_list_r:
            # すでにスキップリストにある場合はスキップ
            if value in skip_values_r:
                continue
            # 現在の値を結果リストに追加
            filtered_list_r.append(value)
            # 10個以内の数値をスキップリストに追加
            skip_values_r.update(range(value - 10, value + 11))
        filtered_list_r = sorted(filtered_list_r)
        print(f"フィルタリング後のリスト(右):{filtered_list_r}\n")

        filtered_list_l = []
        skip_values_l = set()
        for value in ic_list_l:
            # すでにスキップリストにある場合はスキップ
            if value in skip_values_l:
                continue
            # 現在の値を結果リストに追加
            filtered_list_l.append(value)
            # 10個以内の数値をスキップリストに追加
            skip_values_l.update(range(value - 10, value + 11))
        filtered_list_l = sorted(filtered_list_l)
        print(f"フィルタリング後のリスト(左):{filtered_list_l}\n")

        if down_hz:
            np.save(csv_path.with_name(f"ic_frame_30Hz_{csv_path.stem}_r.npy"), filtered_list_r)
            np.save(csv_path.with_name(f"ic_frame_30Hz_{csv_path.stem}_l.npy"), filtered_list_l)
        else:
            np.save(csv_path.with_name(f"ic_frame_120Hz_{csv_path.stem}_r.npy"), filtered_list_r)
            np.save(csv_path.with_name(f"ic_frame_120Hz_{csv_path.stem}_l.npy"), filtered_list_l)

        # 参考可動域 https://www.sakaimed.co.jp/knowledge/jointrange-of-motion/measure-joint-range-of-motion/measure-joint-range-of-motion09/
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6))
        ax1.plot(angle_df.index, angle_df["r_hip_angle_flex_ext"], label="r_hip_angle_flex_ext")
        ax1.plot(angle_df.index, angle_df["l_hip_angle_flex_ext"], label="l_hip_angle_flex_ext")
        ax1.set_ylim(-40, 70)
        ax1.set_title("hip flexion/extension")
        ax1.legend()
        [ax1.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_r]
        # [ax1.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_l]
        ax2.plot(angle_df.index, angle_df["r_knee_angle_flex_ext"], label="r_knee_angle_flex_ext")
        ax2.plot(angle_df.index, angle_df["l_knee_angle_flex_ext"], label="l_knee_angle_flex_ext")
        ax2.set_ylim(-40, 70)
        ax2.set_title("knee flexion/extension")
        ax2.legend()
        [ax2.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_r]
        # [ax2.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_l]
        ax3.plot(angle_df.index, angle_df["r_ankle_angle_flex_ext"], label="r_ankle_angle_flex_ext")
        ax3.plot(angle_df.index, angle_df["l_ankle_angle_flex_ext"], label="l_ankle_angle_flex_ext")
        ax3.set_ylim(-40, 70)
        ax3.set_title("ankle flexion/extension")
        ax3.legend()
        [ax3.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_r]
        # [ax3.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_l]

        plt.show()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6))
        ax1.plot(angle_df.index, angle_df["r_hip_angle_abd_add"], label="r_hip_angle_abd_add")
        ax1.plot(angle_df.index, angle_df["l_hip_angle_abd_add"], label="l_hip_angle_abd_add")
        ax1.set_title("hip abduction/adduction")
        ax1.legend()
        [ax1.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_r]
        # [ax1.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_l]
        ax2.plot(angle_df.index, angle_df["r_knee_angle_abd_add"], label="r_knee_angle_abd_add")
        ax2.plot(angle_df.index, angle_df["l_knee_angle_abd_add"], label="l_knee_angle_abd_add")
        ax2.set_title("knee abduction/adduction")
        ax2.legend()
        [ax2.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_r]
        # [ax2.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_l]
        ax3.plot(angle_df.index, angle_df["r_ankle_angle_abd_add"], label="r_ankle_angle_abd_add")
        ax3.plot(angle_df.index, angle_df["l_ankle_angle_abd_add"], label="l_ankle_angle_abd_add")
        ax3.set_title("ankle abduction/adduction")
        ax3.legend()
        [ax3.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_r]
        # [ax3.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_l]
        plt.show()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6))
        ax1.plot(angle_df.index, angle_df["r_hip_angle_ext_int"], label="r_hip_angle_ext_int")
        ax1.plot(angle_df.index, angle_df["l_hip_angle_ext_int"], label="l_hip_angle_ext_int")
        ax1.set_title("hip external/internal rotation")
        ax1.legend()
        [ax1.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_r]
        # [ax1.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_l]
        ax2.plot(angle_df.index, angle_df["r_knee_angle_ext_int"], label="r_knee_angle_ext_int")
        ax2.plot(angle_df.index, angle_df["l_knee_angle_ext_int"], label="l_knee_angle_ext_int")
        ax2.set_title("knee external/internal rotation")
        ax2.legend()
        [ax2.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_r]
        # [ax2.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_l]
        ax3.plot(angle_df.index, angle_df["r_ankle_angle_ext_int"], label="r_ankle_angle_ext_int")
        ax3.plot(angle_df.index, angle_df["l_ankle_angle_ext_int"], label="l_ankle_angle_ext_int")
        ax3.set_title("ankle external/internal rotation")
        ax3.legend()
        [ax3.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_r]
        # [ax3.axvline(x=x_value, color='gray', linestyle='--') for x_value in filtered_list_l]
        plt.show()

if __name__ == "__main__":
    main()
