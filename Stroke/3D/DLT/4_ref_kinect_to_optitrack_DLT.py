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
    # print(f"keypoints_set.shape = {keypoints_set.shape}")  # [frame, 3]

    for axis in range(keypoints_set.shape[1]-1):
        # 指定されたフレーム範囲のデータを取り出す
        frames = frame_range
        values = np.nan_to_num(keypoints_set[frames, axis])

        # フレーム範囲のフレームを基準に3次スプラインを構築
        spline = CubicSpline(frames, values)

        # 補間した値をその範囲のフレームに再適用
        interpolated_values = spline(frames)
        interpolated_keypoints[frames, axis] = interpolated_values

    # print(f"interpolated_keypoints.shape = {interpolated_keypoints.shape}")  # [frame, 3]

    return interpolated_keypoints

def read_2d_openpose(mkv_file):
    # print(f"mkv_file = {mkv_file}")
    id = os.path.basename(mkv_file).split('.')[0].split('_')[-1]
    json_folder_path = os.path.join(os.path.dirname(mkv_file), os.path.basename(mkv_file).split('.')[0], 'estimated.json')
    all_keypoints_2d = []  # 各フレームの2Dキーポイントを保持するリスト
    check_openpose_list = [1, 8, 12, 13, 14, 19, 20, 21]
    valid_frames = []
    json_files = glob.glob(os.path.join(json_folder_path, "*.json"))
    for i, json_file in enumerate(json_files):
        keypoints_data = load_keypoints_for_frame(i, json_folder_path) #[25, 3]
        # print(f"i = {i} mkv = {mkv_file} keypoints_data = {keypoints_data}")

        # キーポイント抽出が出来ているフレームを記録
        if all(not np.all(np.isnan(keypoints_data[point, :])) for point in check_openpose_list):
            valid_frames.append(i)
        all_keypoints_2d.append(keypoints_data)

    keypoints_2d_openpose = np.array(all_keypoints_2d)

    return keypoints_2d_openpose, valid_frames



def main():

    check_side = "right"
    # check_side = "left"

    mkv_sagittal = mkv_files[0]
    mkv_right = mkv_files[1]
    mkv_left = mkv_files[2]

    #モーキャプから求めた関節角度データを取得
    angle_csv_files = glob.glob(os.path.join(root_dir, "qualisys", f"angle_30Hz_{condition}*.csv"))[0]
    df_mocap_angle = pd.read_csv(angle_csv_files, index_col=0)
    mocap_frame = df_mocap_angle.index.values
    #モーキャプから求めた初期接地時のフレーム
    ic_frame_path = glob.glob(os.path.join(root_dir, "qualisys",f"ic_frame_30Hz_{condition}*.npy"))[0]
    ic_frame_mocap = np.load(ic_frame_path)

    #2d上でのキーポイントを取得 [frame, 25, 3]
    keypoints_sagittal_2d, sagi_frame_2d = read_2d_openpose(mkv_sagittal)
    keypoints_right_2d, dia_right_frame_2d = read_2d_openpose(mkv_right)
    keypoints_left_2d, dia_left_frame_2d = read_2d_openpose(mkv_left)

    # print(f"keypoints_sagittal_2d = {keypoints_sagittal_2d}")
    print(f"keypoints_sagittal_2d.shape = {keypoints_sagittal_2d.shape}")

    #矢状面2d用の処理
    neck_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 1, :], sagi_frame_2d)   #[frame, 2]        neck_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 1, :], sagi_frame_2d)
    mid_hip_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 8, :], sagi_frame_2d)
    if check_side == "right":
        hip_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 9, :], sagi_frame_2d)
        knee_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 10, :], sagi_frame_2d)
        ankle_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 11, :], sagi_frame_2d)
        bigtoe_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 22, :], sagi_frame_2d)
        smalltoe_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 23, :], sagi_frame_2d)
        heel_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 24, :], sagi_frame_2d)
    elif check_side == "left":
        hip_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 12, :], sagi_frame_2d)
        knee_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 13, :], sagi_frame_2d)
        ankle_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 14, :], sagi_frame_2d)
        bigtoe_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 19, :], sagi_frame_2d)
        smalltoe_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 20, :], sagi_frame_2d)
        heel_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 21, :], sagi_frame_2d)


    mid_hip_sagittal_2d = np.array([butter_lowpass_fillter(mid_hip_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    neck_sagittal_2d = np.array([butter_lowpass_fillter(neck_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    hip_sagittal_2d = np.array([butter_lowpass_fillter(hip_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    knee_sagittal_2d = np.array([butter_lowpass_fillter(knee_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    ankle_sagittal_2d = np.array([butter_lowpass_fillter(ankle_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    bigtoe_sagittal_2d = np.array([butter_lowpass_fillter(bigtoe_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    smalltoe_sagittal_2d = np.array([butter_lowpass_fillter(smalltoe_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    heel_sagittal_2d = np.array([butter_lowpass_fillter(heel_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T

    trunk_vector_sagittal_2d = neck_sagittal_2d - mid_hip_sagittal_2d
    thigh_vector_sagittal_2d = knee_sagittal_2d - hip_sagittal_2d
    lower_leg_vector_sagittal_2d = knee_sagittal_2d - ankle_sagittal_2d
    foot_vector_sagittal_2d = (bigtoe_sagittal_2d + smalltoe_sagittal_2d) / 2 - heel_sagittal_2d


    # """
    #3次元DLT法の処理
    print("3次元DLT法の処理を開始しました")
    mid_hip_right_2d = cubic_spline_interpolation(keypoints_right_2d[:, 8, :], dia_right_frame_2d)
    neck_right_2d = cubic_spline_interpolation(keypoints_right_2d[:, 1, :], dia_right_frame_2d)
    if check_side == "right":
        hip_right_2d = cubic_spline_interpolation(keypoints_right_2d[:, 9, :], dia_right_frame_2d)
        knee_right_2d = cubic_spline_interpolation(keypoints_right_2d[:, 10, :], dia_right_frame_2d)
        ankle_right_2d = cubic_spline_interpolation(keypoints_right_2d[:, 11, :], dia_right_frame_2d)
        bigtoe_right_2d = cubic_spline_interpolation(keypoints_right_2d[:, 22, :], dia_right_frame_2d)
        smalltoe_right_2d = cubic_spline_interpolation(keypoints_right_2d[:, 23, :], dia_right_frame_2d)
        heel_right_2d = cubic_spline_interpolation(keypoints_right_2d[:, 24, :], dia_right_frame_2d)
    elif check_side == "left":
        hip_right_2d = cubic_spline_interpolation(keypoints_right_2d[:, 12, :], dia_right_frame_2d)
        knee_right_2d = cubic_spline_interpolation(keypoints_right_2d[:, 13, :], dia_right_frame_2d)
        ankle_right_2d = cubic_spline_interpolation(keypoints_right_2d[:, 14, :], dia_right_frame_2d)
        bigtoe_right_2d = cubic_spline_interpolation(keypoints_right_2d[:, 19, :], dia_right_frame_2d)
        smalltoe_right_2d = cubic_spline_interpolation(keypoints_right_2d[:, 20, :], dia_right_frame_2d)
        heel_right_2d = cubic_spline_interpolation(keypoints_right_2d[:, 21, :], dia_right_frame_2d)

    mid_hip_right_2d = np.hstack([np.array([butter_lowpass_fillter(mid_hip_right_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, mid_hip_right_2d[:, 2:3]])
    neck_right_2d = np.hstack([np.array([butter_lowpass_fillter(neck_right_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, neck_right_2d[:, 2:3]])
    hip_right_2d = np.hstack([np.array([butter_lowpass_fillter(hip_right_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, hip_right_2d[:, 2:3]])
    knee_right_2d = np.hstack([np.array([butter_lowpass_fillter(knee_right_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, knee_right_2d[:, 2:3]])
    ankle_right_2d = np.hstack([np.array([butter_lowpass_fillter(ankle_right_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, ankle_right_2d[:, 2:3]])
    bigtoe_right_2d = np.hstack([np.array([butter_lowpass_fillter(bigtoe_right_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, bigtoe_right_2d[:, 2:3]])
    smalltoe_right_2d = np.hstack([np.array([butter_lowpass_fillter(smalltoe_right_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, smalltoe_right_2d[:, 2:3]])
    heel_right_2d = np.hstack([np.array([butter_lowpass_fillter(heel_right_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, heel_right_2d[:, 2:3]])

    mid_hip_left_2d = cubic_spline_interpolation(keypoints_left_2d[:, 8, :], dia_left_frame_2d)
    neck_left_2d = cubic_spline_interpolation(keypoints_left_2d[:, 1, :], dia_left_frame_2d)
    if check_side == "right":
        hip_left_2d = cubic_spline_interpolation(keypoints_left_2d[:, 9, :], dia_left_frame_2d)
        knee_left_2d = cubic_spline_interpolation(keypoints_left_2d[:, 10, :], dia_left_frame_2d)
        ankle_left_2d = cubic_spline_interpolation(keypoints_left_2d[:, 11, :], dia_left_frame_2d)
        bigtoe_left_2d = cubic_spline_interpolation(keypoints_left_2d[:, 22, :], dia_left_frame_2d)
        smalltoe_left_2d = cubic_spline_interpolation(keypoints_left_2d[:, 23, :], dia_left_frame_2d)
        heel_left_2d = cubic_spline_interpolation(keypoints_left_2d[:, 24, :], dia_left_frame_2d)
    elif check_side == "left":
        hip_left_2d = cubic_spline_interpolation(keypoints_left_2d[:, 12, :], dia_left_frame_2d)
        knee_left_2d = cubic_spline_interpolation(keypoints_left_2d[:, 13, :], dia_left_frame_2d)
        ankle_left_2d = cubic_spline_interpolation(keypoints_left_2d[:, 14, :], dia_left_frame_2d)
        bigtoe_left_2d = cubic_spline_interpolation(keypoints_left_2d[:, 19, :], dia_left_frame_2d)
        smalltoe_left_2d = cubic_spline_interpolation(keypoints_left_2d[:, 20, :], dia_left_frame_2d)
        heel_left_2d = cubic_spline_interpolation(keypoints_left_2d[:, 21, :], dia_left_frame_2d)

    mid_hip_left_2d = np.hstack([np.array([butter_lowpass_fillter(mid_hip_left_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, mid_hip_left_2d[:, 2:3]])
    neck_left_2d = np.hstack([np.array([butter_lowpass_fillter(neck_left_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, neck_left_2d[:, 2:3]])
    hip_left_2d = np.hstack([np.array([butter_lowpass_fillter(hip_left_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, hip_left_2d[:, 2:3]])
    knee_left_2d = np.hstack([np.array([butter_lowpass_fillter(knee_left_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, knee_left_2d[:, 2:3]])
    ankle_left_2d = np.hstack([np.array([butter_lowpass_fillter(ankle_left_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, ankle_left_2d[:, 2:3]])
    bigtoe_left_2d = np.hstack([np.array([butter_lowpass_fillter(bigtoe_left_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, bigtoe_left_2d[:, 2:3]])
    smalltoe_left_2d = np.hstack([np.array([butter_lowpass_fillter(smalltoe_left_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, smalltoe_left_2d[:, 2:3]])
    heel_left_2d = np.hstack([np.array([butter_lowpass_fillter(heel_left_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, heel_left_2d[:, 2:3]])

    # 射影行列の読み込み
    P_1 = np.load(os.path.join(root_dir, "calibration" ,"P_1.npy"))
    P_2 = np.load(os.path.join(root_dir, "calibration" ,"P_2.npy"))

    # フレーム数が異なる場合は短い方に合わせる
    openpose_frame = min(len(mid_hip_sagittal_2d), len(mid_hip_right_2d), len(mid_hip_left_2d))
    mid_hip_sagittal_2d = mid_hip_sagittal_2d[:openpose_frame]
    mid_hip_right_2d = mid_hip_right_2d[:openpose_frame]
    mid_hip_left_2d = mid_hip_left_2d[:openpose_frame]

    def reconstruction_dlt(P1, P2, point_array_2d_1, point_array_2d_2):
        X_array = np.array([])
        for frame in range(openpose_frame):
            x1, y1, c1 = point_array_2d_1[frame]
            x2, y2, c2 = point_array_2d_2[frame]
            # print(f"x1 = {x1}, y1 = {y1}, c1 = {c1}")
            # print(f"x2 = {x2}, y2 = {y2}, c2 = {c2}")

            A = np.array([[c1 * (P1[2, 0] * x1 - P1[0, 0]), c1 * (P1[2, 1] * x1 - P1[0, 1]), c1 * (P1[2, 2] * x1 - P1[0, 2])],
                          [c1 * (P1[2, 0] * y1 - P1[1, 0]), c1 * (P1[2, 1] * y1 - P1[1, 1]), c1 * (P1[2, 2] * y1 - P1[1, 2])],
                          [c2 * (P2[2, 0] * x2 - P2[0, 0]), c2 * (P2[2, 1] * x2 - P2[0, 1]), c2 * (P2[2, 2] * x2 - P2[0, 2])],
                          [c2 * (P2[2, 0] * y2 - P2[1, 0]), c2 * (P2[2, 1] * y2 - P2[1, 1]), c2 * (P2[2, 2] * y2 - P2[1, 2])]])

            b = np.array([c1 * (P1[0, 3] - x1),
                            c1 * (P1[1, 3] - y1),
                            c2 * (P2[0, 3] - x2),
                            c2 * (P2[1, 3] - y2)])

            X = np.linalg.pinv(A).dot(b)
            X_array = np.append(X_array, X)

        X_array = X_array.reshape(openpose_frame, 3)
        print(f"X_array = {X_array}")
        print(f"X_array.shape = {X_array.shape}")
        return X_array

    mid_hip_3d = reconstruction_dlt(P_1, P_2, mid_hip_right_2d, mid_hip_left_2d)
    neck_3d = reconstruction_dlt(P_1, P_2, neck_right_2d, neck_left_2d)
    hip_3d = reconstruction_dlt(P_1, P_2, hip_right_2d, hip_left_2d)
    knee_3d = reconstruction_dlt(P_1, P_2, knee_right_2d, knee_left_2d)
    ankle_3d = reconstruction_dlt(P_1, P_2, ankle_right_2d, ankle_left_2d)
    bigtoe_3d = reconstruction_dlt(P_1, P_2, bigtoe_right_2d, bigtoe_left_2d)
    smalltoe_3d = reconstruction_dlt(P_1, P_2, smalltoe_right_2d, smalltoe_left_2d)
    heel_3d = reconstruction_dlt(P_1, P_2, heel_right_2d, heel_left_2d)

    trunk_vector_3d = neck_3d - mid_hip_3d
    thigh_vector_3d = knee_3d - hip_3d
    lower_leg_vector_3d = knee_3d - ankle_3d
    foot_vector_3d = (bigtoe_3d + smalltoe_3d) / 2 - heel_3d

    print("3次元DLT法の処理が終了しました")
    # """







    #すべてで記録できているフレームを抽出
    print(f"sagi_frame_2d = {sagi_frame_2d}")
    print(f"mocap_frame = {mocap_frame}")
    common_frame = sorted(list(set(sagi_frame_2d) & set(mocap_frame)))
    common_frame = range(ic_frame_mocap[3], ic_frame_mocap[9])

    def calculate_angle(vector1, vector2):  #(frame, xyz)または(frame, xy)の配列を入力)
        angle_list = []
        if vector1.shape[1] == 2:  #2Dベクトル
            for frame in range(len(vector1)):
                dot_product = np.dot(vector1[frame,:], vector2[frame,:])
                cross_product = np.cross(vector1[frame,:], vector2[frame,:])
                angle = np.rad2deg(np.arctan2(cross_product, dot_product))
                angle_list.append(angle)

        elif vector1.shape[1] == 3:  #3Dベクトル
            for frame in range(len(vector1)):
                dot_product = np.dot(vector1[frame, :], vector2[frame, :])
                cross_product = np.cross(vector1[frame, :], vector2[frame, :])
                angle = np.rad2deg(np.arctan2(cross_product, dot_product))  # atan2を使って角度を計算
                angle_list.append(angle)

        return angle_list

    hip_angle_sagittal_2d = pd.DataFrame(calculate_angle(thigh_vector_sagittal_2d, trunk_vector_sagittal_2d))
    knee_angle_sagittal_2d = pd.DataFrame(calculate_angle(thigh_vector_sagittal_2d, lower_leg_vector_sagittal_2d))
    ankle_angle_sagittal_2d = pd.DataFrame(calculate_angle(foot_vector_sagittal_2d, lower_leg_vector_sagittal_2d))

    #3D DLT法で求めた関節角度(基準は3DMCの軸)
    hip_angle_3d = - pd.DataFrame(calculate_angle(thigh_vector_3d, trunk_vector_3d)).iloc[:, 1]
    knee_angle_3d = - pd.DataFrame(calculate_angle(thigh_vector_3d, lower_leg_vector_3d)).iloc[:, 1]
    ankle_angle_3d = - pd.DataFrame(calculate_angle(foot_vector_3d, lower_leg_vector_3d)).iloc[:, 1]

    if check_side == "right":
        hip_angle_sagittal_2d = hip_angle_sagittal_2d.where(hip_angle_sagittal_2d <= 100, hip_angle_sagittal_2d - 360)
        knee_angle_sagittal_2d = knee_angle_sagittal_2d.where(knee_angle_sagittal_2d <= 0, knee_angle_sagittal_2d - 360)
        ankle_angle_sagittal_2d = ankle_angle_sagittal_2d.where(ankle_angle_sagittal_2d <= 0, ankle_angle_sagittal_2d - 360)
        hip_angle_sagittal_2d = hip_angle_sagittal_2d + 180
        knee_angle_sagittal_2d = knee_angle_sagittal_2d + 180
        ankle_angle_sagittal_2d = ankle_angle_sagittal_2d + 90

        hip_angle_3d = hip_angle_3d.where(hip_angle_3d <= 100, hip_angle_3d - 360)
        knee_angle_3d = knee_angle_3d.where(knee_angle_3d <= 100, knee_angle_3d - 360)
        ankle_angle_3d = ankle_angle_3d.where(ankle_angle_3d <= 0, ankle_angle_3d - 360)
        hip_angle_3d = hip_angle_3d + 180
        knee_angle_3d = knee_angle_3d + 180
        ankle_angle_3d = ankle_angle_3d + 90

    elif check_side == "left":
        hip_angle_sagittal_2d = 180 - hip_angle_sagittal_2d
        knee_angle_sagittal_2d = 180 - knee_angle_sagittal_2d
        ankle_angle_sagittal_2d = 90 - ankle_angle_sagittal_2d

        hip_angle_sagittal_2d = hip_angle_sagittal_2d.where(hip_angle_sagittal_2d <= 200, hip_angle_sagittal_2d + 360)
        knee_angle_sagittal_2d = knee_angle_sagittal_2d.where(knee_angle_sagittal_2d <= 200, knee_angle_sagittal_2d + 360)
        ankle_angle_sagittal_2d = ankle_angle_sagittal_2d.where(ankle_angle_sagittal_2d < 90, ankle_angle_sagittal_2d + 270)



    if check_side == "right":
        hip_angle_mocap = df_mocap_angle["r_hip_angle"]
        knee_angle_mocap = df_mocap_angle["r_knee_angle"]
        ankle_angle_mocap = df_mocap_angle["r_ankle_angle"]
    elif check_side == "left":
        hip_angle_mocap = df_mocap_angle["l_hip_angle"]
        knee_angle_mocap = df_mocap_angle["l_knee_angle"]
        ankle_angle_mocap = df_mocap_angle["l_ankle_angle"]

    # pd.set_option('display.max_rows', 500)
    # print(F"hip_angle_sagittal_2d = {hip_angle_sagittal_2d.loc[common_frame]}")
    # print(f"hip_angle_sagittal_2d = {hip_angle_sagittal_2d.loc[common_frame]}")


    frame_range = range(ic_frame_mocap[3], ic_frame_mocap[9])

    if ic_frame_mocap is not None:
        for ic_frame in ic_frame_mocap:
            plt.axvline(x=ic_frame, color='gray', linestyle='--')
    plt.plot(common_frame, hip_angle_sagittal_2d.loc[common_frame], label="2D sagittal", color='tab:blue')
    plt.plot(common_frame, hip_angle_3d.loc[common_frame], label="3D DLT", color='tab:green')
    plt.plot(common_frame, hip_angle_mocap.loc[common_frame], label="Mocap", color='tab:orange')
    plt.xlim(common_frame[0], common_frame[-1])
    plt.ylim(-30, 70)
    plt.title("Hip Angle")
    plt.legend()
    plt.xlabel("frame [-]")
    plt.ylabel("angle [°]")
    plt.savefig(os.path.join(root_dir, f"{condition}_hip_angle.png"))
    plt.show()
    plt.cla()

    if ic_frame_mocap is not None:
        for ic_frame in ic_frame_mocap:
            plt.axvline(x=ic_frame, color='gray', linestyle='--')
    plt.plot(common_frame, knee_angle_sagittal_2d.loc[common_frame], label="2D sagittal", color='tab:blue')
    plt.plot(common_frame, knee_angle_3d.loc[common_frame], label="3D DLT", color='tab:green')
    plt.plot(common_frame, knee_angle_mocap.loc[common_frame], label="Mocap", color='tab:orange')
    plt.xlim(common_frame[0], common_frame[-1])
    plt.ylim(-30, 70)
    plt.title("Knee Angle")
    plt.legend()
    plt.xlabel("frame [-]")
    plt.ylabel("angle [°]")
    plt.savefig(os.path.join(root_dir, f"{condition}_knee_angle.png"))
    plt.show()
    plt.cla()

    if ic_frame_mocap is not None:
        for ic_frame in ic_frame_mocap:
            plt.axvline(x=ic_frame, color='gray', linestyle='--')
    plt.plot(common_frame, ankle_angle_sagittal_2d.loc[common_frame], label="2D sagittal", color='tab:blue')
    plt.plot(common_frame, ankle_angle_3d.loc[common_frame], label="3D DLT", color='tab:green')
    plt.plot(common_frame, ankle_angle_mocap.loc[common_frame], label="Mocap", color='tab:orange')
    plt.xlim(common_frame[0], common_frame[-1])
    plt.ylim(-30, 70)
    plt.title("Ankle Angle")
    plt.legend()
    plt.xlabel("frame [-]")
    plt.ylabel("angle [°]")
    plt.savefig(os.path.join(root_dir, f"{condition}_ankle_angle.png"))
    plt.show()
    plt.cla()


    hip_absolute_error_2d = abs(hip_angle_sagittal_2d.loc[common_frame].values.flatten() - hip_angle_mocap.loc[common_frame].values.flatten())
    knee_absolute_error_2d = abs(knee_angle_sagittal_2d.loc[common_frame].values.flatten() - knee_angle_mocap.loc[common_frame].values.flatten())
    ankle_absolute_error_2d = abs(ankle_angle_sagittal_2d.loc[common_frame].values.flatten() - ankle_angle_mocap.loc[common_frame].values.flatten())

    mae_hip_sagittal_2d = np.nanmean(hip_absolute_error_2d)
    mae_knee_sagittal_2d = np.nanmean(knee_absolute_error_2d)
    mae_ankle_sagittal_2d = np.nanmean(ankle_absolute_error_2d)

    print(f"mae_hip_sagittal_2d = {mae_hip_sagittal_2d:.3f}")
    print(f"mae_knee_sagittal_2d = {mae_knee_sagittal_2d:.3f}")
    print(f"mae_ankle_sagittal_2d = {mae_ankle_sagittal_2d:.3f}")

    hip_absolute_error_3d = abs(hip_angle_3d.loc[common_frame].values.flatten() - hip_angle_mocap.loc[common_frame].values.flatten())
    knee_absolute_error_3d = abs(knee_angle_3d.loc[common_frame].values.flatten() - knee_angle_mocap.loc[common_frame].values.flatten())
    ankle_absolute_error_3d = abs(ankle_angle_3d.loc[common_frame].values.flatten() - ankle_angle_mocap.loc[common_frame].values.flatten())
    mae_hip_3d = np.nanmean(hip_absolute_error_3d)
    mae_knee_3d = np.nanmean(knee_absolute_error_3d)
    mae_ankle_3d = np.nanmean(ankle_absolute_error_3d)

    print(f"mae_hip_3d = {mae_hip_3d:.3f}")
    print(f"mae_knee_3d = {mae_knee_3d:.3f}")
    print(f"mae_ankle_3d = {mae_ankle_3d:.3f}")



if __name__ == "__main__":
    main()
