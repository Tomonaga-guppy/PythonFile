import pandas as pd
import numpy as np
import os
import glob
from pyk4a import PyK4A, PyK4APlayback, CalibrationType
import json
import math
import cv2
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_absolute_error

root_dir = r"F:\Tomson\gait_pattern\20240912"
condition = "sub3_normalgait_f"
mkv_files = glob.glob(os.path.join(root_dir, f"*{condition}*.mkv"))


def load_keypoints_for_frame(frame_number, json_folder_path):
    json_file_name = f"original_{frame_number:012d}_keypoints.json"
    json_file_path = os.path.join(json_folder_path, json_file_name)

    if not os.path.exists(json_file_path):
        return None

    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
        if len(json_data['people']) < 1:  #人が検出されなかった場合はnanで埋める
            keypoints_data = np.full((25, 3), np.nan)
        else:
            keypoints_data = np.array(json_data['people'][0]['pose_keypoints_2d']).reshape((25, 3))

    return keypoints_data

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

def cubic_spline_interpolation(keypoints_set, frame_range):
    # 新しい配列を作成して補間結果を保持します
    interpolated_keypoints = np.copy(keypoints_set)

    for axis in range(keypoints_set.shape[1]):
        # 指定されたフレーム範囲のデータを取り出す
        frames = frame_range
        values = np.nan_to_num(keypoints_set[frames, axis])

        # フレーム範囲のフレームを基準に3次スプラインを構築
        spline = CubicSpline(frames, values)

        # 補間した値をその範囲のフレームに再適用
        interpolated_values = spline(frames)
        interpolated_keypoints[frames, axis] = interpolated_values

    return interpolated_keypoints

def read_2d_openpose(mkv_file):
    # print(f"mkv_file = {mkv_file}")
    id = os.path.basename(mkv_file).split('.')[0].split('_')[-1]
    json_folder_path = os.path.join(os.path.dirname(mkv_file), os.path.basename(mkv_file).split('.')[0], 'estimated.json')
    all_keypoints_2d = []  # 各フレームの2Dキーポイントを保持するリスト
    all_keypoints_2d_tf = []
    check_openpose_list = [1, 8, 12, 13, 14, 19, 20, 21]
    valid_frames = []
    json_files = glob.glob(os.path.join(json_folder_path, "*.json"))
    for i, json_file in enumerate(json_files):
        keypoints_data = load_keypoints_for_frame(i, json_folder_path) #[25, 3]
        print(f"i = {i} mkv = {mkv_file} keypoints_data = {keypoints_data}")

        # キーポイント抽出が出来ているフレームを記録
        if all(not np.all(np.isnan(keypoints_data[point, :])) for point in check_openpose_list):
            valid_frames.append(i)
        all_keypoints_2d.append(keypoints_data)

    keypoints_2d_openpose = np.array(all_keypoints_2d)

    return keypoints_2d_openpose, valid_frames

def calculate_angle(vector1, vector2):  #(frame, xyz)または(frame, xy)の配列を入力)
    angle_list = []
    for frame in range(len(vector1)):
        dot_product = np.dot(vector1[frame], vector2[frame])
        cross_product = np.cross(vector1[frame], vector2[frame])
        angle = np.arctan2(np.linalg.norm(cross_product), dot_product)
        angle = angle * 180 / np.pi
        angle_list.append(angle)

    return angle_list

def main():
    for mkv_file in mkv_files:
        mkv_sagittal = mkv_files[0]
        mkv_diagonal_right = mkv_files[1]
        mkv_diagonal_left = mkv_files[2]

        #モーキャプから求めた関節角度データを取得
        angle_csv_files = glob.glob(os.path.join(root_dir, "Motive", f"angle_30Hz_{condition}*.csv"))[0]
        df_mocap_angle = pd.read_csv(angle_csv_files, index_col=0)
        mocap_frame = df_mocap_angle.index.values
        #モーキャプから求めた初期接地時のフレーム
        ic_frame_path = glob.glob(os.path.join(root_dir, "Motive",f"ic_frame_{condition}*.npy"))[0]
        ic_frame_mocap = np.load(ic_frame_path)

        #2d上でのキーポイントを取得 [frame, 25, 3]
        keypoints_sagittal_2d, sagi_frame_2d = read_2d_openpose(mkv_sagittal)
        keypoints_diagonal_right_2d, dia_right_frame_2d = read_2d_openpose(mkv_diagonal_right)
        keypoints_diagonal_left_2d, dia_left_frame_2d = read_2d_openpose(mkv_diagonal_left)

        print(f"keypoints_sagittal_2d = {keypoints_sagittal_2d.shape}")

        #矢状面2d用の処理
        mid_hip_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 8, :], sagi_frame_2d) #[frame, 2]        neck_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 1, :], sagi_frame_2d)
        lhip_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 12, :], sagi_frame_2d)
        lknee_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 13, :], sagi_frame_2d)
        lankle_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 14, :], sagi_frame_2d)
        lbigtoe_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 19, :], sagi_frame_2d)
        lsmalltoe_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 20, :], sagi_frame_2d)
        lheel_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 21, :], sagi_frame_2d)

        mid_hip_sagittal_2d = np.array([butter_lowpass_fillter(mid_hip_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        neck_sagittal_2d = np.array([butter_lowpass_fillter(neck_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lhip_sagittal_2d = np.array([butter_lowpass_fillter(lhip_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lknee_sagittal_2d = np.array([butter_lowpass_fillter(lknee_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lankle_sagittal_2d = np.array([butter_lowpass_fillter(lankle_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lbigtoe_sagittal_2d = np.array([butter_lowpass_fillter(lbigtoe_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lsmalltoe_sagittal_2d = np.array([butter_lowpass_fillter(lsmalltoe_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lheel_sagittal_2d = np.array([butter_lowpass_fillter(lheel_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T

        trunk_vector_sagittal_2d = neck_sagittal_2d - mid_hip_sagittal_2d
        thigh_vector_l_sagittal_2d = lknee_sagittal_2d - lhip_sagittal_2d
        lower_leg_vector_l_sagittal_2d = lknee_sagittal_2d - lankle_sagittal_2d
        foot_vector_l_sagittal_2d = (lbigtoe_sagittal_2d + lsmalltoe_sagittal_2d) / 2 - lheel_sagittal_2d




        # """
        #3次元DLT法の処理
        print("3次元DLT法の処理を開始しました")
        mid_hip_diagonal_right_2d = cubic_spline_interpolation(keypoints_diagonal_right_2d[:, 8, :], dia_right_frame_2d)
        neck_diagonal_right_2d = cubic_spline_interpolation(keypoints_diagonal_right_2d[:, 1, :], dia_right_frame_2d)
        lhip_diagonal_right_2d = cubic_spline_interpolation(keypoints_diagonal_right_2d[:, 12, :], dia_right_frame_2d)
        lknee_diagonal_right_2d = cubic_spline_interpolation(keypoints_diagonal_right_2d[:, 13, :], dia_right_frame_2d)
        lankle_diagonal_right_2d = cubic_spline_interpolation(keypoints_diagonal_right_2d[:, 14, :], dia_right_frame_2d)
        lbigtoe_diagonal_right_2d = cubic_spline_interpolation(keypoints_diagonal_right_2d[:, 19, :], dia_right_frame_2d)
        lsmalltoe_diagonal_right_2d = cubic_spline_interpolation(keypoints_diagonal_right_2d[:, 20, :], dia_right_frame_2d)
        lheel_diagonal_right_2d = cubic_spline_interpolation(keypoints_diagonal_right_2d[:, 21, :], dia_right_frame_2d)

        mid_hip_diagonal_right_2d = np.array([butter_lowpass_fillter(mid_hip_diagonal_right_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_right_frame_2d) for x in range(2)]).T
        neck_diagonal_right_2d = np.array([butter_lowpass_fillter(neck_diagonal_right_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_right_frame_2d) for x in range(2)]).T
        lhip_diagonal_right_2d = np.array([butter_lowpass_fillter(lhip_diagonal_right_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_right_frame_2d) for x in range(2)]).T
        lknee_diagonal_right_2d = np.array([butter_lowpass_fillter(lknee_diagonal_right_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_right_frame_2d) for x in range(2)]).T
        lankle_diagonal_right_2d = np.array([butter_lowpass_fillter(lankle_diagonal_right_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_right_frame_2d) for x in range(2)]).T
        lbigtoe_diagonal_right_2d = np.array([butter_lowpass_fillter(lbigtoe_diagonal_right_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_right_frame_2d) for x in range(2)]).T
        lsmalltoe_diagonal_right_2d = np.array([butter_lowpass_fillter(lsmalltoe_diagonal_right_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_right_frame_2d) for x in range(2)]).T
        lheel_diagonal_right_2d = np.array([butter_lowpass_fillter(lheel_diagonal_right_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_right_frame_2d) for x in range(2)]).T

        mid_hip_diagonal_left_2d = cubic_spline_interpolation(keypoints_diagonal_left_2d[:, 8, :], dia_left_frame_2d)
        neck_diagonal_left_2d = cubic_spline_interpolation(keypoints_diagonal_left_2d[:, 1, :], dia_left_frame_2d)
        lhip_diagonal_left_2d = cubic_spline_interpolation(keypoints_diagonal_left_2d[:, 12, :], dia_left_frame_2d)
        lknee_diagonal_left_2d = cubic_spline_interpolation(keypoints_diagonal_left_2d[:, 13, :], dia_left_frame_2d)
        lankle_diagonal_left_2d = cubic_spline_interpolation(keypoints_diagonal_left_2d[:, 14, :], dia_left_frame_2d)
        lbigtoe_diagonal_left_2d = cubic_spline_interpolation(keypoints_diagonal_left_2d[:, 19, :], dia_left_frame_2d)
        lsmalltoe_diagonal_left_2d = cubic_spline_interpolation(keypoints_diagonal_left_2d[:, 20, :], dia_left_frame_2d)
        lheel_diagonal_left_2d = cubic_spline_interpolation(keypoints_diagonal_left_2d[:, 21, :], dia_left_frame_2d)

        mid_hip_diagonal_left_2d = np.array([butter_lowpass_fillter(mid_hip_diagonal_left_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_left_frame_2d) for x in range(2)]).T
        neck_diagonal_left_2d = np.array([butter_lowpass_fillter(neck_diagonal_left_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_left_frame_2d) for x in range(2)]).T
        lhip_diagonal_left_2d = np.array([butter_lowpass_fillter(lhip_diagonal_left_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_left_frame_2d) for x in range(2)]).T
        lknee_diagonal_left_2d = np.array([butter_lowpass_fillter(lknee_diagonal_left_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_left_frame_2d) for x in range(2)]).T
        lankle_diagonal_left_2d = np.array([butter_lowpass_fillter(lankle_diagonal_left_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_left_frame_2d) for x in range(2)]).T
        lbigtoe_diagonal_left_2d = np.array([butter_lowpass_fillter(lbigtoe_diagonal_left_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_left_frame_2d) for x in range(2)]).T
        lsmalltoe_diagonal_left_2d = np.array([butter_lowpass_fillter(lsmalltoe_diagonal_left_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_left_frame_2d) for x in range(2)]).T
        lheel_diagonal_left_2d = np.array([butter_lowpass_fillter(lheel_diagonal_left_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_left_frame_2d) for x in range(2)]).T


        print("3次元DLT法の処理が終了しました")
        # """







        #すべてで記録できているフレームを抽出
        print(f"sagi_frame_2d = {sagi_frame_2d}")
        # print(f"dia_right_frame_3d = {dia_right_frame_3d}")
        # print(f"dia_left_frame_3d = {dia_left_frame_3d}")
        print(f"mocap_frame = {mocap_frame}")
        # common_frame = sorted(list(set(sagi_frame_2d) & set(dia_right_frame_3d) & set(dia_left_frame_3d) & set(mocap_frame)))
        common_frame = sorted(list(set(sagi_frame_2d) & set(mocap_frame)))

        hip_angle_sagittal_2d_filtered = pd.DataFrame(calculate_angle(trunk_vector_sagittal_2d, thigh_vector_l_sagittal_2d))
        knee_angle_sagittal_2d_filtered = pd.DataFrame(calculate_angle(thigh_vector_l_sagittal_2d, lower_leg_vector_l_sagittal_2d))
        ankle_angle_sagittal_2d_filtered = pd.DataFrame(calculate_angle(lower_leg_vector_l_sagittal_2d, foot_vector_l_sagittal_2d))

        hip_angle_sagittal_2d_filtered = 180 - hip_angle_sagittal_2d_filtered
        knee_angle_sagittal_2d_filtered = 180 - knee_angle_sagittal_2d_filtered
        ankle_angle_sagittal_2d_filtered =  90 - ankle_angle_sagittal_2d_filtered

        # df_mocap_angle.index = df_mocap_angle.index - 5
        # df_mocap_angle = df_mocap_angle.reindex(common_frame)
        # df_mocap_angle.iloc[-5:, :] = 0

        hip_angle_mocap = df_mocap_angle["l_hip_angle"].loc[common_frame]
        knee_angle_mocap = df_mocap_angle["l_knee_angle"].loc[common_frame]
        ankle_angle_mocap = df_mocap_angle["l_ankle_angle"].loc[common_frame]

        # pd.set_option('display.max_rows', 500)
        # print(F"hip_angle_sagittal_2d = {hip_angle_sagittal_2d.loc[common_frame]}")
        # print(f"hip_angle_sagittal_2d_filtered = {hip_angle_sagittal_2d_filtered.loc[common_frame]}")

        if ic_frame_mocap is not None:
            for ic_frame in ic_frame_mocap:
                plt.axvline(x=ic_frame, color='gray', linestyle='--')
        plt.plot(common_frame, hip_angle_sagittal_2d_filtered.loc[common_frame], label="2D sagittal", color='#1f77b4')
        # plt.plot(common_frame, hip_angle_frontal.loc[common_frame], label="3D frontal", color='#ff7f0e')
        plt.plot(common_frame, hip_angle_mocap, label="Mocap", color='#ff7f0e')  #color='#2ca02c' 緑     color='#ff7f0e' オレンジ  color='#1f77b4' 青
        # plt.plot(common_frame, hip_angle_sagittal_2d.loc[common_frame], color='#1f77b4', alpha=0.5)
        # plt.plot(common_frame, hip_angle_frontal_3d_ori, label="3D frontal_ori", color='#ff7f0e', alpha=0.5)
        plt.title("Hip Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition}_hip_angle.png"))
        # plt.show()  #5frame
        plt.cla()

        if ic_frame_mocap is not None:
            for ic_frame in ic_frame_mocap:
                plt.axvline(x=ic_frame, color='gray', linestyle='--')
        plt.plot(common_frame, knee_angle_sagittal_2d_filtered.loc[common_frame], label="2D sagittal", color='#1f77b4')
        # plt.plot(common_frame, knee_angle_frontal.iloc[common_frame], label="3D frontal", color='#ff7f0e')
        plt.plot(common_frame, knee_angle_mocap, label="Mocap", color='#ff7f0e')
        # plt.plot(common_frame, knee_angle_sagittal_2d.loc[common_frame], color='#1f77b4', alpha=0.5)
        # plt.plot(common_frame, knee_angle_frontal_3d_ori, label="3D frontal_ori", color='#ff7f0e', alpha=0.5)
        plt.title("Knee Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition}_knee_angle.png"))
        # plt.show() #3frame
        plt.cla()

        if ic_frame_mocap is not None:
            for ic_frame in ic_frame_mocap:
                plt.axvline(x=ic_frame, color='gray', linestyle='--')
        plt.plot(common_frame, ankle_angle_sagittal_2d_filtered.loc[common_frame], label="2D sagittal", color='#1f77b4')
        # plt.plot(common_frame, ankle_angle_frontal.loc[common_frame], label="3D frontal", color='#ff7f0e')
        plt.plot(common_frame, ankle_angle_mocap, label="Mocap", color='#ff7f0e')
        # plt.plot(common_frame, ankle_angle_sagittal_2d.loc[common_frame], color='#1f77b4', alpha=0.5)
        # plt.plot(common_frame, ankle_angle_frontal_3d_ori, label="3D frontal_ori", color='#ff7f0e', alpha=0.5)
        plt.title("Ankle Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition}_ankle_angle.png"))
        # plt.show() #5frame
        plt.cla()


        hip_absolute_error = abs(hip_angle_sagittal_2d_filtered.loc[common_frame].values.flatten() - hip_angle_mocap.values.flatten())
        mae_hip_sagittal = np.nanmean(hip_absolute_error)
        knee_absolute_error = abs(knee_angle_sagittal_2d_filtered.loc[common_frame].values.flatten() - knee_angle_mocap.values.flatten())
        mae_knee_sagittal = np.nanmean(knee_absolute_error)
        ankle_absolute_error = abs(ankle_angle_sagittal_2d_filtered.loc[common_frame].values.flatten() - ankle_angle_mocap.values.flatten())
        mae_ankle_sagittal = np.nanmean(ankle_absolute_error)

        print(f"mae_hip_sagittal = {mae_hip_sagittal:.3f}")
        print(f"mae_knee_sagittal = {mae_knee_sagittal:.3f}")
        print(f"mae_ankle_sagittal = {mae_ankle_sagittal:.3f}")

        npz_path = os.path.join(os.path.dirname(mkv_files[0]), f"{os.path.basename(mkv_files[0]).split('.')[0].split('_')[0]}_keypoints&frame.npz")
        # np.savez(npz_path, diagonal_right=keypoints_diagonal_right, diagonal_left=keypoints_diagonal_left, frontal=keypoints_frontal, mocap=keypoints_mocap, common_frame=common_frame, sagittal_3d=keypoints_sagittal_3d, sagittal_2d=keypoints_sagittal_2d)
        # np.savez(npz_path, diagonal_right=keypoints_diagonal_right, diagonal_left=keypoints_diagonal_left, frontal=keypoints_frontal, common_frame=common_frame, sagittal_3d=keypoints_sagittal_3d, sagittal_2d=keypoints_sagittal_2d)

if __name__ == "__main__":
    main()
