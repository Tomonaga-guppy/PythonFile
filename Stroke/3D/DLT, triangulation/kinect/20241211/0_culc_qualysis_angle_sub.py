import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import sys

down_hz = False
csv_path_dir = Path(r"G:\gait_pattern\20241016\qualisys")
csv_paths = list(csv_path_dir.glob("sub4*.tsv"))

#補間、フィルタ処理した後に欠損値補間（元々）

def culc_interpolate_frame(df):
    """
    膝、足首で連続して5フレーム以上欠損しているフレームを算出
    """
    df.to_csv("df.csv")
    mideal_dict = dict()
    #5フレーム以上連続して欠損値がある場合はSKYCOMに従って補間
    for column in df.columns:
        #5つ以上連続した欠損値を持つフレーム番号を取得
        is_missing = df[column].isnull()  # 欠損値のインデックスを取得
        group_id = (is_missing != is_missing.shift()).cumsum()  # 連続する欠損値に同じ番号を付けるためのグループIDを作成
        consecutive_missing_counts = is_missing.groupby(group_id).cumsum()  # グループごとに、連続する欠損値の数をカウント
        long_missing_group_numbers = consecutive_missing_counts[consecutive_missing_counts >= 5].groupby(is_missing).first().index   #連続欠損数が5フレーム以上のグループ番号を取得
        long_missing_indices = df.groupby(is_missing).filter(lambda x: x.name in long_missing_group_numbers).index.tolist()  # 連続欠損数が5フレーム以上のグループに属するインデックスを取得

        dict_keys = ["RKNE_X", "LKNE_X", "RANK_X", "LANK_X", "RKNE2_X", "LKNE2_X", "RANK2_X", "LANK2_X"]
        if column in dict_keys:
            mideal_dict[column] = long_missing_indices

    return mideal_dict

def read_3DMC(csv_path, down_hz):
    col_names = range(1,100)  #データの形が汚い場合に対応するためあらかじめ列数(100:適当)を設定
    df = pd.read_csv(csv_path, names=col_names, sep='\t', skiprows=[0,1,2,3,4,5,6,7,8,10])  #Qualisys
    df.columns = df.iloc[0]  # 最初の行をヘッダーに
    df = df.drop(0).reset_index(drop=True)  # ヘッダーにした行をデータから削除し、インデックスをリセット

    if down_hz:
        df_down = df[::4].reset_index()
        sampling_freq = 30
    else:
        df_down = df
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

    marker_set_df = marker_set_df.apply(pd.to_numeric, errors='coerce')  #文字列として読み込まれたデータを数値に変換
    marker_set_df.replace(0, np.nan, inplace=True)  #0をnanに変換

    ineterpolate_frame_dict = culc_interpolate_frame(marker_set_df)  #欠損値のあるフレームを保持した辞書

    df_copy = marker_set_df.copy()
    valid_index_mask = df_copy.notna().all(axis=1)
    valid_index = df_copy[valid_index_mask].index
    valid_index = pd.Index(range(valid_index.min(), valid_index.max() + 1))  #欠損値がない行のインデックスを範囲で取得、この範囲の値を解析に使用する
    marker_set_df = marker_set_df.loc[valid_index, :]  #欠損値のない行のみを抽出
    interpolated_df = marker_set_df.interpolate(method='spline', order=3)  #3次スプライン補間
    marker_set_fin_df = interpolated_df.apply(butter_lowpass_fillter, args=(4, 6, sampling_freq))  #4次のバターワースローパスフィルタ

    output_csv_path = csv_path.with_name(f"marker_set_{csv_path.stem}.csv")
    marker_set_fin_df.to_csv(output_csv_path)

    return marker_set_fin_df, valid_index, ineterpolate_frame_dict

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
        marker_set_df, valid_index, interpolate_frame_dict = read_3DMC(csv_path, down_hz)

        print(f"csv_path = {csv_path}")
        print(f"valid_index = {valid_index}")

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


        for frame_num in valid_index:
            frame_num = frame_num - valid_index.min()
            d_asi = np.linalg.norm(rasi[frame_num,:] - lasi[frame_num,:])
            d_leg = (np.linalg.norm(rank[frame_num,:] - rasi[frame_num,:]) + np.linalg.norm(lank[frame_num, :] - lasi[frame_num,:]) / 2)
            r = 0.012 #使用したマーカー径
            h = 1.75 #被験者の身長
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

            # 欠損している膝、足首の座標を算出
            if frame_num in interpolate_frame_dict["RKNE2_X"] and not frame_num in interpolate_frame_dict["RKNE_X"]:  #RKNEは存在し、RKNE2が欠損している場合
                rknee2[frame_num, :] = rknee[frame_num, :] + (2 * r + 0.1 * k) * e_y0_pelvis
            if frame_num in interpolate_frame_dict["RKNE_X"] and not frame_num in interpolate_frame_dict["RKNE2_X"]:  #RKNE2は存在し、RKNEが欠損している場合
                rknee[frame_num, :] = rknee2[frame_num, :] - (2 * r + 0.1 * k) * e_y0_pelvis

            if frame_num in interpolate_frame_dict["LKNE2_X"] and not frame_num in interpolate_frame_dict["LKNE_X"]:  #LKNEは存在し、LKNE2が欠損している場合
                lknee2[frame_num, :] = lknee[frame_num, :] - (2 * r + 0.1 * k) * e_y0_pelvis
            if frame_num in interpolate_frame_dict["LKNE_X"] and not frame_num in interpolate_frame_dict["LKNE2_X"]:  #LKNE2は存在し、LKNEが欠損している場合
                lknee[frame_num, :] = lknee2[frame_num, :] + (2 * r + 0.1 * k) * e_y0_pelvis

            if frame_num in interpolate_frame_dict["RANK2_X"] and not frame_num in interpolate_frame_dict["RANK_X"]:  #RANKは存在し、RANK2が欠損している場合
                rank2[frame_num, :] = rank[frame_num, :] + (2 * r + 0.1 * k) * e_y0_pelvis
            if frame_num in interpolate_frame_dict["RANK_X"] and not frame_num in interpolate_frame_dict["RANK2_X"]:  #RANK2は存在し、RANKが欠損している場合
                rank[frame_num, :] = rank2[frame_num, :] - (2 * r + 0.1 * k) * e_y0_pelvis

            if frame_num in interpolate_frame_dict["LANK2_X"] and not frame_num in interpolate_frame_dict["LANK_X"]:  #LANKは存在し、LANK2が欠損している場合
                lank2[frame_num, :] = lank[frame_num, :] - (2 * r + 0.1 * k) * e_y0_pelvis
            if frame_num in interpolate_frame_dict["LANK_X"] and not frame_num in interpolate_frame_dict["LANK2_X"]:  #LANK2は存在し、LANKが欠損している場合
                lank[frame_num, :] = lank2[frame_num, :] + (2 * r + 0.1 * k) * e_y0_pelvis

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

            r_hip_angle = r_hip_angle_rot.as_euler('yzx', degrees=True)[0]
            l_hip_angle = l_hip_angle_rot.as_euler('yzx', degrees=True)[0]
            r_knee_angle = r_knee_angle_rot.as_euler('yzx', degrees=True)[0]
            l_knee_angle = l_knee_angle_rot.as_euler('yzx', degrees=True)[0]
            r_ankle_angle = r_ankle_angle_rot.as_euler('yzx', degrees=True)[0]
            l_ankle_angle = l_ankle_angle_rot.as_euler('yzx', degrees=True)[0]

            r_hip_angle = 360 + r_hip_angle if r_hip_angle < 0 else r_hip_angle
            l_hip_angle = 360 + l_hip_angle if l_hip_angle < 0 else l_hip_angle
            r_knee_angle = 360 + r_knee_angle if r_knee_angle < 0 else r_knee_angle
            l_knee_angle = 360 + l_knee_angle if l_knee_angle < 0 else l_knee_angle
            r_ankle_angle = 360 + r_ankle_angle if r_ankle_angle < 0 else r_ankle_angle
            l_ankle_angle = 360 + l_ankle_angle if l_ankle_angle < 0 else l_ankle_angle

            r_hip_angle = 180 - r_hip_angle
            l_hip_angle = 180 - l_hip_angle
            r_knee_angle = 180 - r_knee_angle
            l_knee_angle = 180 - l_knee_angle
            r_ankle_angle = 90 - r_ankle_angle
            l_ankle_angle = 90 - l_ankle_angle

            angles = [r_hip_angle, l_hip_angle, r_knee_angle, l_knee_angle, r_ankle_angle, l_ankle_angle]
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
            heel = lhee[frame_num, :]
            bector = heel - hip[:]
            bector_list.append(bector)
            dist_list.append(np.linalg.norm(bector))

        angle_array = np.array(angle_list)
        angle_df = pd.DataFrame({"r_hip_angle": angle_array[:, 0], "r_knee_angle": angle_array[:, 2], "r_ankle_angle": angle_array[:, 4], "l_hip_angle": angle_array[:, 1], "l_knee_angle": angle_array[:, 3], "l_ankle_angle": angle_array[:, 5]})
        angle_df.index = valid_index
        if down_hz:
            angle_df.to_csv(csv_path.with_name(f"angle_30Hz_{csv_path.stem}.csv"))
        else:
            angle_df.to_csv(csv_path.with_name(f"angle_120Hz_{csv_path.stem}.csv"))

        bector_array = np.array(bector_list)
        lhee_pel_z = bector_array[:, 0]
        # lhee_pel_z = bector_array[:, 2]  #motiveの場合
        df = pd.DataFrame({"lhee_pel_z":lhee_pel_z})
        df.index = valid_index
        df = df.sort_values(by="lhee_pel_z", ascending=True)
        # df = df.sort_values(by="lhee_pel_z", ascending=False)  #motiveの場合
        ic_list = df.index[:120].values


        dist_array = np.array(dist_list)
        df = pd.DataFrame({"dist": dist_array})
        df.index = valid_index

        # plt.figure()
        # plt.plot(df.index, df["dist"])
        # plt.show()

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
        print(f"フィルタリング後のリスト:{filtered_list}\n")

        if down_hz:
            np.save(csv_path.with_name(f"ic_frame_30Hz_{csv_path.stem}.npy"), filtered_list)
        else:
            np.save(csv_path.with_name(f"ic_frame_120Hz_{csv_path.stem}.npy"), filtered_list)

        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6))
        # ax1.plot(angle_df.index, angle_df["r_hip_angle"], label="r_hip_angle")
        # ax1.plot(angle_df.index, angle_df["l_hip_angle"], label="l_hip_angle")
        # ax1.set_ylim(-40, 70)
        # ax2.plot(angle_df.index, angle_df["r_knee_angle"], label="r_knee_angle")
        # ax2.plot(angle_df.index, angle_df["l_knee_angle"], label="l_knee_angle")
        # ax2.set_ylim(-40, 70)
        # ax3.plot(angle_df.index, angle_df["r_ankle_angle"], label="r_ankle_angle")
        # ax3.plot(angle_df.index, angle_df["l_ankle_angle"], label="l_ankle_angle")
        # ax3.set_ylim(-40, 70)
        # plt.show()

if __name__ == "__main__":
    main()
