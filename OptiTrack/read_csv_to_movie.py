import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_3d_mocap(all_keypoints, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        keypoints = all_keypoints[frame]
        print(f"keypoints = {keypoints}")
        x_coords = [point[0] for point in keypoints]
        y_coords = [point[1] for point in keypoints]
        z_coords = [point[2] for point in keypoints]
        ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')


        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 2)
        ax.set_zlim(-1, 1)
        ax.invert_yaxis()
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(f"Frame {frame}")

    ani = FuncAnimation(fig, update, frames=len(all_keypoints), repeat=False)
    plt.show()

def main():
    motive_folder = r"F:\Tomson\gait_pattern\20240712\Motive"
    csv_paths = glob.glob(os.path.join(motive_folder, "*.csv"))
    csv_paths = [path for path in csv_paths if "interpolated" not in path] # 既に補間済みのファイルは除外

    for i, csv_path in enumerate(csv_paths):
        # CSVファイルを読み込む(マーカー名称とXYZのみをヘッダーとして取得)
        df = pd.read_csv(csv_path, skiprows= [0,1,2,4], header=[0,2])

        # 4行おきにデータを抽出(ダウンサンプリング)
        df_down = df[::4].reset_index(drop=True)

        #必要なマーカーのみを抽出
        marker_set = ["RASI", "LASI","RPSI","LPSI","RKNE","LKNE", "RTHI", "LTHI", "RANK","LANK", "RTIB", "LTIB","RTOE","LTOE","RHEE","LHEE",
                    "RSHO", "LSHO","C7", "T10", "CLAV", "STRN", "RBAK", "RKNE2", "LKNE2", "RANK2", "LANK2"]

        marker_set_df = df_down[[col for col in df_down.columns if any(marker in col[0] for marker in marker_set)]]
        # print(f"marker_set_df = {marker_set_df}")

        success_frame_list = []

        #すべてのマーカーが検出できているフレームのみを抽出
        for frame in range(0, len(marker_set_df)):
            if not marker_set_df.iloc[frame].isna().any():
                success_frame_list.append(frame)

        full_range = range(min(success_frame_list), max(success_frame_list)+1)
        print(f"full_range = {full_range}")
        success_df = marker_set_df.reindex(full_range)

        interpolate_success_df = success_df.interpolate(method='spline', order = 3) #3次スプライン補間
        interpolate_success_df.to_csv(os.path.join(motive_folder, f"interpolated_{os.path.basename(csv_path)}"))

        columns = interpolate_success_df.columns

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
        '''

        keypoints = interpolate_success_df.values
        # print(f"keypoints = {keypoints}")
        keypoints = keypoints.reshape(-1, len(marker_set), 3)  #xyzで組になるように変形
        # print(f"keypoints = {keypoints}")
        # print(f"keypoints = {keypoints.shape}")

        # animate_3d_mocap(keypoints, save_path=os.path.join(os.path.dirname(motive_folder), f"interpolated_{os.path.basename(csv_path)}.mp4"))

        right_ground_frame = []
        left_ground_frame = []

        mid_psi = (keypoints[:, 20, :] + keypoints[:, 8, :]) / 2  #RPSIとLPSIの中間点
        r_heel = keypoints[:, 17, :] #RHEE
        l_heel = keypoints[:, 5, :] #LHEE

        r_length = np.linalg.norm(mid_psi - r_heel, axis=1)
        l_length = np.linalg.norm(mid_psi - l_heel, axis=1)

        # print(f"r_length = {r_length}")
        # print(f"l_length = {l_length}")

        # r_length_df = pd.DataFrame(r_length, l_length, columns=["R_length" "L_length"])
        # r_length_df.index= full_range

        ground_length_df = pd.DataFrame({"R_length": r_length, "L_length": l_length}, index=full_range)
        # print(f"ground_length_df = {ground_length_df}")

        # # R_lengthが小さい順にソート
        # sorted_r_df = ground_length_df.sort_values(by='R_length')
        # sorted_l_df = ground_length_df.sort_values(by='L_length')
        # print(f"sorted_r_df = {sorted_r_df}")
        # print(f"sorted_l_df = {sorted_l_df}")

        heel_length_df = pd.DataFrame({"R_heel_x": r_heel[:,0], "R_heel_y": r_heel[:,1], "R_heel_z": r_heel[:,2],
                                       "L_heel_x": l_heel[:,0],"L_heel_y": l_heel[:,1], "L_heel_z": l_heel[:,2]}, index=full_range)
        # print(f"heel_length_df = {heel_length_df}")

        sorted_r_heel_length_df = heel_length_df.sort_values(by='R_heel_y')["R_heel_y"]
        sorted_l_heel_length_df = heel_length_df.sort_values(by='L_heel_y')["L_heel_y"]
        print(f"sorted_r_heel_length_df = {sorted_r_heel_length_df.head(20)}")
        print(f"sorted_l_heel_length_df = {sorted_l_heel_length_df.head(20)}")





if __name__ == '__main__':
    main()

