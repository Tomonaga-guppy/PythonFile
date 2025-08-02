import module_mocap as moc
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

tsv_dir = Path(r"G:\gait_pattern\20250317_fukuyama")
tsv_files = tsv_dir.glob("*.tsv")

def main():
    for itsv, tsv_file in enumerate(tsv_files):
        print(f"Processing {itsv+1}/{len(list(tsv_dir.glob('*.tsv')))}: {tsv_file.name}")
        full_df = moc.read_tsv(tsv_file)  #tsvファイルの読み込み
        target_df = full_df.copy()
        if tsv_file.name == "0317hokou1.tsv":
            target_df = full_df[(full_df.index > 668) & (full_df.index < 1286)]
        elif tsv_file.name == "0317hokou2 r.tsv":
            target_df = full_df[(full_df.index > 119) & (full_df.index < 631)]
        interpolated_df = target_df.copy()
        interpolated_df = interpolated_df.replace(0, np.nan)  #0をNaNに置き換え
        interpolated_df = interpolated_df.interpolate(method='spline', order=3)  #3次スプライン補間
        butter_df = interpolated_df.copy()
        butter_df = moc.butterworth_filter(butter_df, cutoff=6, order=4, fs=100) #4次のバターワースローパスフィルタ

        rasi = butter_df[['RASI X', 'RASI Y', 'RASI Z']].to_numpy()
        lasi = butter_df[['LASI X', 'LASI Y', 'LASI Z']].to_numpy()
        rpsi = butter_df[['RPSI X', 'RPSI Y', 'RPSI Z']].to_numpy()
        lpsi = butter_df[['LPSI X', 'LPSI Y', 'LPSI Z']].to_numpy()
        rknee = butter_df[['RKNE X', 'RKNE Y', 'RKNE Z']].to_numpy()
        lknee = butter_df[['LKNE X', 'LKNE Y', 'LKNE Z']].to_numpy()
        rank = butter_df[['RANK X', 'RANK Y', 'RANK Z']].to_numpy()
        lank = butter_df[['LANK X', 'LANK Y', 'LANK Z']].to_numpy()
        rtoe = butter_df[['RTOE X', 'RTOE Y', 'RTOE Z']].to_numpy()
        ltoe = butter_df[['LTOE X', 'LTOE Y', 'LTOE Z']].to_numpy()
        rhee = butter_df[['RHEE X', 'RHEE Y', 'RHEE Z']].to_numpy()
        lhee = butter_df[['LHEE X', 'LHEE Y', 'LHEE Z']].to_numpy()
        rhip = butter_df[['RHEE X', 'RHEE Y', 'RHEE Z']].to_numpy()
        lhip = butter_df[['LHEE X', 'LHEE Y', 'LHEE Z']].to_numpy()

        angle_list = []
        pel2hee_list = []

        for frame_num in butter_df.index:
            frame_num = frame_num - butter_df.index[0]  # フレーム番号を0から始まるように調整
            # print(f"Frame {frame}:")
            d_asi = np.linalg.norm(rasi[frame_num,:] - lasi[frame_num,:])
            d_leg = (np.linalg.norm(rank[frame_num,:] - rasi[frame_num,:]) + np.linalg.norm(lank[frame_num, :] - lasi[frame_num,:]) / 2)
            r = 0.012 #使用したマーカー径

            h = 1.8 #被験者の身長

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
            rknee2 = rknee.copy()
            rknee2[frame_num, :] = rknee[frame_num, :] + (2 * r + 0.1 * k) * e_y_pelvis
            lknee2 = lknee.copy()
            lknee2[frame_num, :] = lknee[frame_num, :] - (2 * r + 0.1 * k) * e_y_pelvis
            rank2 = rank.copy()
            rank2[frame_num, :] = rank[frame_num, :] + (2 * r + 0.06 * k) * e_y_pelvis
            lank2 = lank.copy()
            lank2[frame_num, :] = lank[frame_num, :] - (2 * r + 0.06 * k) * e_y_pelvis

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

            # 回転行列から回転角を計算
            r_hip_angle = r_hip_angle_rot.as_euler('yzx', degrees=True)[0]
            l_hip_angle = l_hip_angle_rot.as_euler('yzx', degrees=True)[0]
            r_knee_angle = r_knee_angle_rot.as_euler('yzx', degrees=True)[0]
            l_knee_angle = l_knee_angle_rot.as_euler('yzx', degrees=True)[0]
            r_ankle_angle = r_ankle_angle_rot.as_euler('yzx', degrees=True)[0]
            l_ankle_angle = l_ankle_angle_rot.as_euler('yzx', degrees=True)[0]

            # 角度範囲を調整
            # 角度が負の場合は360を足して正の値に変換
            r_hip_angle = 360 + r_hip_angle if r_hip_angle < 0 else r_hip_angle
            l_hip_angle = 360 + l_hip_angle if l_hip_angle < 0 else l_hip_angle
            r_knee_angle = 360 + r_knee_angle if r_knee_angle < 0 else r_knee_angle
            l_knee_angle = 360 + l_knee_angle if l_knee_angle < 0 else l_knee_angle
            r_ankle_angle = 360 + r_ankle_angle if r_ankle_angle < 0 else r_ankle_angle
            l_ankle_angle = 360 + l_ankle_angle if l_ankle_angle < 0 else l_ankle_angle

            # 各角度について特定の範囲に変換
            r_hip_angle = 180 - r_hip_angle if r_hip_angle > 100 else r_hip_angle
            l_hip_angle = 180 - l_hip_angle if l_hip_angle > 100 else l_hip_angle
            r_knee_angle = 180 - r_knee_angle if r_knee_angle < 180 else r_knee_angle - 180
            l_knee_angle = 180 - l_knee_angle if l_knee_angle < 180 else l_knee_angle - 180
            r_ankle_angle = 90 - r_ankle_angle if r_ankle_angle < 180 else 270 - r_ankle_angle
            l_ankle_angle = 90 - l_ankle_angle if l_ankle_angle < 180 else 270 - l_ankle_angle

            angles = [r_hip_angle, l_hip_angle, r_knee_angle, l_knee_angle, r_ankle_angle, l_ankle_angle]
            angle_list.append(angles)

            #骨盤と左足踵のベクトルを計算
            pel2heel = lhee[frame_num, :] - hip[:]
            pel2hee_list.append(pel2heel)

        # 初期接地の算出
        pel2heel_array = np.array(pel2hee_list)
        if tsv_file.name == "0317hokou1.tsv":
            pel2heel = pel2heel_array[:, 1] #y軸正方向が進行方向
        elif tsv_file.name == "0317hokou2 r.tsv":
            pel2heel = pel2heel_array[:, 1] #y軸負方向が進行方向
            pel2heel = -pel2heel
        p2h_df = pd.DataFrame({"pel2heel": pel2heel})
        p2h_df.index = butter_df.index

        """骨盤と踵の距離をプロット
        plt.plot(p2h_df)
        plt.xlabel('Frame')
        plt.ylabel('Pelvis to Heel Vector')
        plt.title(f"Pelvis to Heel Vector for {tsv_file.name}")
        plt.legend()
        plt.grid()
        plt.show()
        """

        #初期接地の算出
        p2h_df.sort_values(by="pel2heel", ascending=False, inplace=True)
        cand_ic_frame = p2h_df.index[:60]

        ic_frame_list = []
        skip_frame = set()
        for frame in cand_ic_frame:
            if frame in skip_frame:
                continue
            ic_frame_list.append(frame)
            skip_frame.update(range(frame-10, frame+10)) # 10フレーム前後をスキップ
        ic_frame_list = sorted(ic_frame_list)
        # print(f"Candidate IC frames: {cand_ic_frame}")
        print(f"IC frames: {ic_frame_list}")

        angle_array = np.array(angle_list)
        angle_df = pd.DataFrame(angle_array, columns=["R_Hip", "L_Hip", "R_Knee", "L_Knee", "R_Ankle", "L_Ankle"])
        angle_df.index = butter_df.index
        # print(f"angle df: {angle_df}")

        r_hip_angle_list, l_hip_angle_list = [], []
        r_knee_angle_list, l_knee_angle_list = [], []
        r_ankle_angle_list, l_ankle_angle_list = [], []

        for icycle, cycle_start_frame in enumerate(ic_frame_list):
            if icycle == len(ic_frame_list)-1:
                continue
            cycle_end_frame = ic_frame_list[icycle+1]
            r_hip_df = angle_df.loc[cycle_start_frame:cycle_end_frame, "R_Hip"]
            l_hip_df = angle_df.loc[cycle_start_frame:cycle_end_frame, "L_Hip"]
            r_knee_df = angle_df.loc[cycle_start_frame:cycle_end_frame, "R_Knee"]
            l_knee_df = angle_df.loc[cycle_start_frame:cycle_end_frame, "L_Knee"]
            r_ankle_df = angle_df.loc[cycle_start_frame:cycle_end_frame, "R_Ankle"]
            l_ankle_df = angle_df.loc[cycle_start_frame:cycle_end_frame, "L_Ankle"]
            r_hip_angle = moc.frame2percent(r_hip_df)
            l_hip_angle = moc.frame2percent(l_hip_df)
            r_knee_angle = moc.frame2percent(r_knee_df)
            l_knee_angle = moc.frame2percent(l_knee_df)
            r_ankle_angle = moc.frame2percent(r_ankle_df)
            l_ankle_angle = moc.frame2percent(l_ankle_df)
            r_hip_angle_list.append(r_hip_angle)
            l_hip_angle_list.append(l_hip_angle)
            r_knee_angle_list.append(r_knee_angle)
            l_knee_angle_list.append(l_knee_angle)
            r_ankle_angle_list.append(r_ankle_angle)
            l_ankle_angle_list.append(l_ankle_angle)
        r_hip_angle_mean = np.mean(r_hip_angle_list, axis=0)
        l_hip_angle_mean = np.mean(l_hip_angle_list, axis=0)
        r_knee_angle_mean = np.mean(r_knee_angle_list, axis=0)
        l_knee_angle_mean = np.mean(l_knee_angle_list, axis=0)
        r_ankle_angle_mean = np.mean(r_ankle_angle_list, axis=0)
        l_ankle_angle_mean = np.mean(l_ankle_angle_list, axis=0)

        r_hip_angle_sd = np.std(r_hip_angle_list, axis=0)
        l_hip_angle_sd = np.std(l_hip_angle_list, axis=0)
        r_knee_angle_sd = np.std(r_knee_angle_list, axis=0)
        l_knee_angle_sd = np.std(l_knee_angle_list, axis=0)
        r_ankle_angle_sd = np.std(r_ankle_angle_list, axis=0)
        l_ankle_angle_sd = np.std(l_ankle_angle_list, axis=0)

        ylim = (-30, 80)
        font_size = 20
        title_size = 20
        fig_size = (10, 8)

        plt.figure(figsize=fig_size)
        # plt.plot(r_hip_angle_mean, label="Right")
        # plt.fill_between(range(len(r_hip_angle_mean)), r_hip_angle_mean - r_hip_angle_sd, r_hip_angle_mean + r_hip_angle_sd, alpha=0.2)
        plt.plot(l_hip_angle_mean, label="Left")
        plt.fill_between(range(len(l_hip_angle_mean)), l_hip_angle_mean - l_hip_angle_sd, l_hip_angle_mean + l_hip_angle_sd, alpha=0.2)
        plt.xlabel('Gait cycle [%]', fontsize=font_size)
        plt.ylabel('Angle [° ]', fontsize=font_size)
        plt.title("Hip Angle L", fontsize=title_size)
        # plt.legend()
        plt.ylim(ylim)
        plt.tick_params(labelsize=font_size)
        plt.grid()
        # plt.show()
        plt.savefig(tsv_dir / f"hip_angle_l_{tsv_file.stem}.png")
        plt.close()

        plt.figure(figsize=fig_size)
        # plt.plot(r_knee_angle_mean, label="Right")
        # plt.fill_between(range(len(r_knee_angle_mean)), r_knee_angle_mean - r_knee_angle_sd, r_knee_angle_mean + r_knee_angle_sd, alpha=0.2)
        plt.plot(l_knee_angle_mean, label="Left")
        plt.fill_between(range(len(l_knee_angle_mean)), l_knee_angle_mean - l_knee_angle_sd, l_knee_angle_mean + l_knee_angle_sd, alpha=0.2)
        plt.xlabel('Gait cycle [%]', fontsize=font_size)
        plt.ylabel('Angle [° ]', fontsize=font_size)
        plt.title("Knee Angle L", fontsize=title_size)
        # plt.legend()
        plt.ylim(ylim)
        plt.tick_params(labelsize=font_size)
        plt.grid()
        # plt.show()
        plt.savefig(tsv_dir / f"knee_angle_l_{tsv_file.stem}.png")
        plt.close()

        plt.figure(figsize=fig_size)
        # plt.plot(r_ankle_angle_mean, label="Right")
        # plt.fill_between(range(len(r_ankle_angle_mean)), r_ankle_angle_mean - r_ankle_angle_sd, r_ankle_angle_mean + r_ankle_angle_sd, alpha=0.2)
        plt.plot(l_ankle_angle_mean, label="Left")
        plt.fill_between(range(len(l_ankle_angle_mean)), l_ankle_angle_mean - l_ankle_angle_sd, l_ankle_angle_mean + l_ankle_angle_sd, alpha=0.2)
        plt.xlabel('Gait cycle [%]', fontsize=font_size)
        plt.ylabel('Angle [° ]', fontsize=font_size)
        plt.title("Ankle Angle L", fontsize=title_size)
        # plt.legend()
        plt.ylim(ylim)
        plt.tick_params(labelsize=font_size)
        plt.grid()
        # plt.show()
        plt.savefig(tsv_dir / f"ankle_angle_l_{tsv_file.stem}.png")
        plt.close()






if __name__ == "__main__":
    main()



    VIDEO_PATH = r"G:\gait_pattern\20250728_br_ledtest\fr\GX010158.MP4"
