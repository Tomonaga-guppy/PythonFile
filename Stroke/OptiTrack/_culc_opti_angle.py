import pandas as pd
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

def read_3d_optitrack(csv_path):
    df = pd.read_csv(csv_path, skiprows=[0, 1, 2, 4], header=[0, 2])

    df_down = df[::4].reset_index(drop=True)

    marker_set = ["RASI", "LASI","RKNE","LKNE", "RANK","LANK","RTOE","LTOE","RHEE","LHEE", "RKNE2", "LKNE2", "RANK2", "LANK2"]

    # marker_set = ["RASI", "LASI","RPSI","LPSI","RKNE","LKNE", "RTHI", "LTHI", "RANK","LANK", "RTIB", "LTIB","RTOE","LTOE","RHEE","LHEE",
    #             "RSHO", "LSHO","C7", "T10", "CLAV", "STRN", "RBAK", "RKNE2", "LKNE2", "RANK2", "LANK2"]

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
                ( 'MarkerSet 01:LTOE', 'X'), 6
                ( 'MarkerSet 01:RANK', 'X'), 7
                ('MarkerSet 01:RANK2', 'X'), 8
                ( 'MarkerSet 01:RASI', 'X'), 9
                ( 'MarkerSet 01:RHEE', 'X'), 10
                ( 'MarkerSet 01:RKNE', 'X'), 11
                ('MarkerSet 01:RKNE2', 'X'), 12
                ( 'MarkerSet 01:RTOE', 'X'), 13
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

        angle_list = []

        for frame_num in full_range:
            #使う関節の選択
            rasi = keypoints_mocap[frame_num, 9, :]
            lasi = keypoints_mocap[frame_num, 3, :]
            rank = keypoints_mocap[frame_num, 7, :]
            lank = keypoints_mocap[frame_num, 0, :]
            rank2 = keypoints_mocap[frame_num, 8, :]
            lank2 = keypoints_mocap[frame_num, 1, :]
            rknee = keypoints_mocap[frame_num, 11, :]
            lknee = keypoints_mocap[frame_num, 4, :]
            rknee2 = keypoints_mocap[frame_num, 12, :]
            lknee2 = keypoints_mocap[frame_num, 5, :]
            rtoe = keypoints_mocap[frame_num, 13, :]
            ltoe = keypoints_mocap[frame_num, 6, :]
            rhee = keypoints_mocap[frame_num, 10, :]
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
            x_rthigh = -(d_asi +r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
            x_lthigh = -(d_asi +r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
            y_rthigh = +(c * np.sin(theta) - d_leg/2)
            y_lthigh = -(c * np.sin(theta)- d_leg/2)
            z_rthigh = -(d_asi + r) * np.sin(beta) + c * np.cos(theta) * np.cos(beta)
            z_lthigh = -(d_asi + r) * np.sin(beta) + c * np.cos(theta) * np.cos(beta)
            rthigh = np.array([x_rthigh, y_rthigh, z_rthigh]).T
            lthigh = np.array([x_lthigh, y_lthigh, z_lthigh]).T
            hip = (rthigh + lthigh) / 2
            lumbar = (0.47 * (rasi + lasi) / 2 + 0.53 * (rank + lank) / 2) + 0.02 * k * np.array([0, 0, 1])

            #骨盤節座標系（原点はhip）
            e_y0_pelvis = lthigh - rthigh
            e_z_pelvis = (lumbar - hip)/np.linalg.norm(lumbar - hip)
            e_x_pelvis = np.cross(e_y0_pelvis, e_z_pelvis)/np.linalg.norm(np.cross(e_y0_pelvis, e_z_pelvis))
            e_y_pelvis = np.cross(e_z_pelvis, e_x_pelvis)
            rot_pelvis = np.array([e_x_pelvis, e_y_pelvis, e_z_pelvis]).T
            euler_angles_pelvis = R.from_matrix(rot_pelvis).as_euler('YZX', degrees=True)  #出力もYZXの順番

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
            euler_angles_rthigh = R.from_matrix(rot_rthigh).as_euler('YZX', degrees=True)

            #左大腿節座標系（原点はlthigh）
            e_y0_lthigh = lknee - lknee2
            e_z_lthigh = (lshank - lthigh)/np.linalg.norm(lshank - lthigh)
            e_x_lthigh = np.cross(e_y0_lthigh, e_z_lthigh)/np.linalg.norm(np.cross(e_y0_lthigh, e_z_lthigh))
            e_y_lthigh = np.cross(e_z_lthigh, e_x_lthigh)
            rot_lthigh = np.array([e_x_lthigh, e_y_lthigh, e_z_lthigh]).T
            euler_angles_lthigh = R.from_matrix(rot_lthigh).as_euler('YZX', degrees=True)

            #右下腿節座標系（原点はrshank）
            e_y0_rshank = rknee2 - rknee
            e_z_rshank = (rfoot - rshank)/np.linalg.norm(rfoot - rshank)
            e_x_rshank = np.cross(e_y0_rshank, e_z_rshank)/np.linalg.norm(np.cross(e_y0_rshank, e_z_rshank))
            e_y_rshank = np.cross(e_z_rshank, e_x_rshank)
            rot_rshank = np.array([e_x_rshank, e_y_rshank, e_z_rshank]).T
            euler_angles_rshank = R.from_matrix(rot_rshank).as_euler('YZX', degrees=True)

            #左下腿節座標系（原点はlshank）
            e_y0_lshank = lknee - lknee2
            e_z_lshank = (lfoot - lshank)/np.linalg.norm(lfoot - lshank)
            e_x_lshank = np.cross(e_y0_lshank, e_z_lshank)/np.linalg.norm(np.cross(e_y0_lshank, e_z_lshank))
            e_y_lshank = np.cross(e_z_lshank, e_x_lshank)
            rot_lshank = np.array([e_x_lshank, e_y_lshank, e_z_lshank]).T
            euler_angles_lshank = R.from_matrix(rot_lshank).as_euler('YZX', degrees=True)

            #右足節座標系（原点はrfoot）
            e_z_rfoot = e_z_pelvis
            e_x0_rfoot = rtoe - rhee
            e_y_rfoot = np.cross(e_z_rfoot, e_x0_rfoot)/np.linalg.norm(np.cross(e_z_rfoot, e_x0_rfoot))
            e_x_rfoot = np.cross(e_y_rfoot, e_z_rfoot)
            rot_rfoot = np.array([e_x_rfoot, e_y_rfoot, e_z_rfoot]).T
            euler_angles_rfoot = R.from_matrix(rot_rfoot).as_euler('YZX', degrees=True)

            #左足節座標系（原点はlfoot）
            e_z_lfoot = e_z_pelvis
            e_x0_lfoot = ltoe - lhee
            e_y_lfoot = np.cross(e_z_lfoot, e_x0_lfoot)/np.linalg.norm(np.cross(e_z_lfoot, e_x0_lfoot))
            e_x_lfoot = np.cross(e_y_lfoot, e_z_lfoot)
            rot_lfoot = np.array([e_x_lfoot, e_y_lfoot, e_z_lfoot]).T
            euler_angles_lfoot = R.from_matrix(rot_lfoot).as_euler('YZX', degrees=True)

            # r_hip_angle = euler_angles_rthigh[0] - euler_angles_pelvis[0]
            # l_hip_angle = euler_angles_lthigh[0] - euler_angles_pelvis[0]
            # r_knee_angle = euler_angles_rshank[0] - euler_angles_rthigh[0]
            # l_knee_angle = euler_angles_lshank[0] - euler_angles_lthigh[0]
            # r_ankle_angle = euler_angles_rfoot[0] - euler_angles_rshank[0]
            # l_ankle_angle = euler_angles_lfoot[0] - euler_angles_lshank[0]

            r_hip_realative_rotation = np.dot(np.linalg.inv(rot_pelvis), rot_rthigh)
            r_hip_angle = R.from_matrix(r_hip_realative_rotation).as_euler('YZX', degrees=True)[0]
            l_hip_realative_rotation = np.dot(np.linalg.inv(rot_pelvis), rot_lthigh)
            l_hip_angle = R.from_matrix(l_hip_realative_rotation).as_euler('YZX', degrees=True)[0]
            r_knee_realative_rotation = np.dot(np.linalg.inv(rot_rthigh), rot_rshank)
            r_knee_angle = R.from_matrix(r_knee_realative_rotation).as_euler('YZX', degrees=True)[0]
            l_knee_realative_rotation = np.dot(np.linalg.inv(rot_lthigh), rot_lshank)
            l_knee_angle = R.from_matrix(l_knee_realative_rotation).as_euler('YZX', degrees=True)[0]
            r_ankle_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rfoot)
            r_ankle_angle = R.from_matrix(r_ankle_realative_rotation).as_euler('YZX', degrees=True)[0]
            l_ankle_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lfoot)
            l_ankle_angle = R.from_matrix(l_ankle_realative_rotation).as_euler('YZX', degrees=True)[0]

            angle_list.append([r_hip_angle, l_hip_angle, r_knee_angle, l_knee_angle, r_ankle_angle, l_ankle_angle])

        angle_array = np.array(angle_list)
        df = pd.DataFrame({"r_hip_angle": angle_array[:, 0], "r_knee_angle": angle_array[:, 2], "r_ankle_angle": angle_array[:, 4], "l_hip_angle": angle_array[:, 1], "l_knee_angle": angle_array[:, 3], "l_ankle_angle": angle_array[:, 5]})
        df.index = df.index + full_range.start
        df.to_csv(os.path.join(os.path.dirname(csv_path), f"angle_{os.path.basename(csv_path)}"))

if __name__ == "__main__":
    main()

