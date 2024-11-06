import pandas as pd
import numpy as np
import os
import glob
from pyk4a import PyK4A, PyK4APlayback, CalibrationType
import json
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline

root_dir = r"G:\gait_pattern\20240912"
condition = "sub3_normal*_f_1"
mkv_files = glob.glob(os.path.join(root_dir, f"*{condition}*.mkv"))
print(f"mkv_files = {mkv_files}")


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

    mkv_sagittal = mkv_files[0]
    mkv_right = mkv_files[1]
    mkv_left = mkv_files[2]

    #2d上でのキーポイントを取得 [frame, 25, 3]
    keypoints_sagittal_2d, sagi_frame_2d = read_2d_openpose(mkv_sagittal)
    keypoints_right_2d, dia_right_frame_2d = read_2d_openpose(mkv_right)
    keypoints_left_2d, dia_left_frame_2d = read_2d_openpose(mkv_left)

    # print(f"keypoints_sagittal_2d = {keypoints_sagittal_2d}")
    print(f"keypoints_sagittal_2d.shape = {keypoints_sagittal_2d.shape}")
    sagi_frame_2d = list([x for x in range(50, 400)])  # テストで適当なフレームを指定
    sagi_start_frame = sagi_frame_2d[0]
    sagi_end_frame = sagi_frame_2d[-1]
    print(f"sagi_start_frame = {sagi_start_frame}, sagi_end_frame = {sagi_end_frame}")

    #矢状面2d用の処理
    neck_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 1, :], sagi_frame_2d)   #[frame, 2]        neck_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 1, :], sagi_frame_2d)
    mid_hip_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 8, :], sagi_frame_2d)

    hip_sagittal_2d_right = cubic_spline_interpolation(keypoints_sagittal_2d[:, 9, :], sagi_frame_2d)
    knee_sagittal_2d_right = cubic_spline_interpolation(keypoints_sagittal_2d[:, 10, :], sagi_frame_2d)
    ankle_sagittal_2d_right = cubic_spline_interpolation(keypoints_sagittal_2d[:, 11, :], sagi_frame_2d)
    bigtoe_sagittal_2d_right = cubic_spline_interpolation(keypoints_sagittal_2d[:, 22, :], sagi_frame_2d)
    smalltoe_sagittal_2d_right = cubic_spline_interpolation(keypoints_sagittal_2d[:, 23, :], sagi_frame_2d)
    heel_sagittal_2d_right = cubic_spline_interpolation(keypoints_sagittal_2d[:, 24, :], sagi_frame_2d)

    hip_sagittal_2d_left = cubic_spline_interpolation(keypoints_sagittal_2d[:, 12, :], sagi_frame_2d)
    knee_sagittal_2d_left  = cubic_spline_interpolation(keypoints_sagittal_2d[:, 13, :], sagi_frame_2d)
    ankle_sagittal_2d_left  = cubic_spline_interpolation(keypoints_sagittal_2d[:, 14, :], sagi_frame_2d)
    bigtoe_sagittal_2d_left  = cubic_spline_interpolation(keypoints_sagittal_2d[:, 19, :], sagi_frame_2d)
    smalltoe_sagittal_2d_left  = cubic_spline_interpolation(keypoints_sagittal_2d[:, 20, :], sagi_frame_2d)
    heel_sagittal_2d_left  = cubic_spline_interpolation(keypoints_sagittal_2d[:, 21, :], sagi_frame_2d)

    mid_hip_sagittal_2d = np.array([butter_lowpass_fillter(mid_hip_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    neck_sagittal_2d = np.array([butter_lowpass_fillter(neck_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T

    hip_sagittal_2d_right = np.array([butter_lowpass_fillter(hip_sagittal_2d_right[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    knee_sagittal_2d_right = np.array([butter_lowpass_fillter(knee_sagittal_2d_right[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    ankle_sagittal_2d_right = np.array([butter_lowpass_fillter(ankle_sagittal_2d_right[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    bigtoe_sagittal_2d_right = np.array([butter_lowpass_fillter(bigtoe_sagittal_2d_right[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    smalltoe_sagittal_2d_right = np.array([butter_lowpass_fillter(smalltoe_sagittal_2d_right[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    heel_sagittal_2d_right = np.array([butter_lowpass_fillter(heel_sagittal_2d_right[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T

    hip_sagittal_2d_left = np.array([butter_lowpass_fillter(hip_sagittal_2d_left[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    knee_sagittal_2d_left = np.array([butter_lowpass_fillter(knee_sagittal_2d_left[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    ankle_sagittal_2d_left = np.array([butter_lowpass_fillter(ankle_sagittal_2d_left[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    bigtoe_sagittal_2d_left = np.array([butter_lowpass_fillter(bigtoe_sagittal_2d_left[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    smalltoe_sagittal_2d_left = np.array([butter_lowpass_fillter(smalltoe_sagittal_2d_left[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
    heel_sagittal_2d_left = np.array([butter_lowpass_fillter(heel_sagittal_2d_left[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T

    trunk_vector_sagittal_2d = neck_sagittal_2d - mid_hip_sagittal_2d

    thigh_vector_sagittal_2d_right = knee_sagittal_2d_right - hip_sagittal_2d_right
    lower_leg_vector_sagittal_2d_right = knee_sagittal_2d_right - ankle_sagittal_2d_right
    foot_vector_sagittal_2d_right = (bigtoe_sagittal_2d_right + smalltoe_sagittal_2d_right) / 2 - heel_sagittal_2d_right

    thigh_vector_sagittal_2d_left = knee_sagittal_2d_left - hip_sagittal_2d_left
    lower_leg_vector_sagittal_2d_left = knee_sagittal_2d_left - ankle_sagittal_2d_left
    foot_vector_sagittal_2d_left = (bigtoe_sagittal_2d_left + smalltoe_sagittal_2d_left) / 2 - heel_sagittal_2d_left

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
                cross_product = np.linalg.norm(cross_product)  #戻り値を3次元にしないように外積の大きさを取得
                angle = np.rad2deg(np.arctan2(cross_product, dot_product))  # atan2を使って角度を計算
                angle_list.append(angle)

        return angle_list

    # hip_angle_sagittal_2d_right = calculate_angle(thigh_vector_sagittal_2d_right, trunk_vector_sagittal_2d)
    # knee_angle_sagittal_2d_right = calculate_angle(thigh_vector_sagittal_2d_right, lower_leg_vector_sagittal_2d_right)
    # ankle_angle_sagittal_2d_right = calculate_angle(foot_vector_sagittal_2d_right, lower_leg_vector_sagittal_2d_right)

    # hip_angle_sagittal_2d_left = calculate_angle(thigh_vector_sagittal_2d_left, trunk_vector_sagittal_2d)
    # knee_angle_sagittal_2d_left = calculate_angle(thigh_vector_sagittal_2d_left, lower_leg_vector_sagittal_2d_left)
    # ankle_angle_sagittal_2d_left = calculate_angle(foot_vector_sagittal_2d_left, lower_leg_vector_sagittal_2d_left)

    hip_angle_sagittal_2d_right = pd.DataFrame(calculate_angle(thigh_vector_sagittal_2d_right, trunk_vector_sagittal_2d)[sagi_start_frame:sagi_end_frame+1], index=sagi_frame_2d)
    knee_angle_sagittal_2d_right = pd.DataFrame(calculate_angle(thigh_vector_sagittal_2d_right, lower_leg_vector_sagittal_2d_right)[sagi_start_frame:sagi_end_frame+1], index=sagi_frame_2d)
    ankle_angle_sagittal_2d_right = pd.DataFrame(calculate_angle(foot_vector_sagittal_2d_right, lower_leg_vector_sagittal_2d_right)[sagi_start_frame:sagi_end_frame+1], index=sagi_frame_2d)

    hip_angle_sagittal_2d_left = pd.DataFrame(calculate_angle(thigh_vector_sagittal_2d_left, trunk_vector_sagittal_2d)[sagi_start_frame:sagi_end_frame+1], index=sagi_frame_2d)
    knee_angle_sagittal_2d_left = pd.DataFrame(calculate_angle(thigh_vector_sagittal_2d_left, lower_leg_vector_sagittal_2d_left)[sagi_start_frame:sagi_end_frame+1], index=sagi_frame_2d)
    ankle_angle_sagittal_2d_left = pd.DataFrame(calculate_angle(foot_vector_sagittal_2d_left, lower_leg_vector_sagittal_2d_left)[sagi_start_frame:sagi_end_frame+1], index=sagi_frame_2d)

    hip_angle_sagittal_2d_right = hip_angle_sagittal_2d_right.where(hip_angle_sagittal_2d_right <= 100, hip_angle_sagittal_2d_right - 360)
    knee_angle_sagittal_2d_right = knee_angle_sagittal_2d_right.where(knee_angle_sagittal_2d_right <= 0, knee_angle_sagittal_2d_right - 360)
    ankle_angle_sagittal_2d_right = ankle_angle_sagittal_2d_right.where(ankle_angle_sagittal_2d_right <= 0, ankle_angle_sagittal_2d_right - 360)

    hip_angle_sagittal_2d_left = hip_angle_sagittal_2d_left.where(hip_angle_sagittal_2d_left <= 100, hip_angle_sagittal_2d_left - 360)
    knee_angle_sagittal_2d_left = knee_angle_sagittal_2d_left.where(knee_angle_sagittal_2d_left <= 0, knee_angle_sagittal_2d_left - 360)
    ankle_angle_sagittal_2d_left = ankle_angle_sagittal_2d_left.where(ankle_angle_sagittal_2d_left <= 0, ankle_angle_sagittal_2d_left - 360)

    hip_angle_sagittal_2d_right = hip_angle_sagittal_2d_right + 180
    knee_angle_sagittal_2d_right = knee_angle_sagittal_2d_right + 180
    ankle_angle_sagittal_2d_right = ankle_angle_sagittal_2d_right + 90

    hip_angle_sagittal_2d_left = hip_angle_sagittal_2d_left + 180
    knee_angle_sagittal_2d_left = knee_angle_sagittal_2d_left + 180
    ankle_angle_sagittal_2d_left = ankle_angle_sagittal_2d_left + 90

    df_sagittal_angle = pd.concat([hip_angle_sagittal_2d_right, knee_angle_sagittal_2d_right, ankle_angle_sagittal_2d_right, hip_angle_sagittal_2d_left, knee_angle_sagittal_2d_left, ankle_angle_sagittal_2d_left], axis=1)
    df_sagittal_angle.index = sagi_frame_2d
    df_sagittal_angle.columns = ["hip_angle_right", "knee_angle_right", "ankle_angle_right", "hip_angle_left", "knee_angle_left", "ankle_angle_left"]
    df_sagittal_angle.to_csv(os.path.join(root_dir, f"{os.path.basename(mkv_sagittal).split('.')[0].split('_dev')[0]}_sagittal_angle.csv"))


    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    ax1.plot(sagi_frame_2d, hip_angle_sagittal_2d_right, label="hip_angle_right")
    ax1.plot(sagi_frame_2d, hip_angle_sagittal_2d_left, label="hip_angle_left")
    ax1.set_title("hip_angle")
    ax1.legend()

    ax2.plot(sagi_frame_2d, knee_angle_sagittal_2d_right, label="knee_angle_right")
    ax2.plot(sagi_frame_2d, knee_angle_sagittal_2d_left, label="knee_angle_left")
    ax2.set_title("knee_angle")
    ax2.legend()

    ax3.plot(sagi_frame_2d, ankle_angle_sagittal_2d_right, label="ankle_angle_right")
    ax3.plot(sagi_frame_2d, ankle_angle_sagittal_2d_left, label="ankle_angle_left")
    ax3.set_title("ankle_angle")
    ax3.legend()

    plt.savefig(os.path.join(root_dir, f"{os.path.basename(mkv_sagittal).split('.')[0].split('_dev')[0]}_sagittal_angle.png"))
    # plt.show()
    plt.close()

    # """
    #3次元DLT法の処理
    print("3次元DLT法の処理を開始しました")

    # 右側カメラ
    mid_hip_right_2d = cubic_spline_interpolation(keypoints_right_2d[:, 8, :], dia_right_frame_2d)
    neck_right_2d = cubic_spline_interpolation(keypoints_right_2d[:, 1, :], dia_right_frame_2d)

    hip_right_2d_right = cubic_spline_interpolation(keypoints_right_2d[:, 9, :], dia_right_frame_2d)
    knee_right_2d_right = cubic_spline_interpolation(keypoints_right_2d[:, 10, :], dia_right_frame_2d)
    ankle_right_2d_right = cubic_spline_interpolation(keypoints_right_2d[:, 11, :], dia_right_frame_2d)
    bigtoe_right_2d_right = cubic_spline_interpolation(keypoints_right_2d[:, 22, :], dia_right_frame_2d)
    smalltoe_right_2d_right = cubic_spline_interpolation(keypoints_right_2d[:, 23, :], dia_right_frame_2d)
    heel_right_2d_right = cubic_spline_interpolation(keypoints_right_2d[:, 24, :], dia_right_frame_2d)

    hip_right_2d_left = cubic_spline_interpolation(keypoints_right_2d[:, 12, :], dia_right_frame_2d)
    knee_right_2d_left = cubic_spline_interpolation(keypoints_right_2d[:, 13, :], dia_right_frame_2d)
    ankle_right_2d_left = cubic_spline_interpolation(keypoints_right_2d[:, 14, :], dia_right_frame_2d)
    bigtoe_right_2d_left = cubic_spline_interpolation(keypoints_right_2d[:, 19, :], dia_right_frame_2d)
    smalltoe_right_2d_left = cubic_spline_interpolation(keypoints_right_2d[:, 20, :], dia_right_frame_2d)
    heel_right_2d_left = cubic_spline_interpolation(keypoints_right_2d[:, 21, :], dia_right_frame_2d)

    mid_hip_right_2d = np.hstack([np.array([butter_lowpass_fillter(mid_hip_right_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, mid_hip_right_2d[:, 2:3]])
    neck_right_2d = np.hstack([np.array([butter_lowpass_fillter(neck_right_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, neck_right_2d[:, 2:3]])

    hip_right_2d_right = np.hstack([np.array([butter_lowpass_fillter(hip_right_2d_right[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, hip_right_2d_right[:, 2:3]])
    knee_right_2d_right = np.hstack([np.array([butter_lowpass_fillter(knee_right_2d_right[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, knee_right_2d_right[:, 2:3]])
    ankle_right_2d_right = np.hstack([np.array([butter_lowpass_fillter(ankle_right_2d_right[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, ankle_right_2d_right[:, 2:3]])
    bigtoe_right_2d_right = np.hstack([np.array([butter_lowpass_fillter(bigtoe_right_2d_right[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, bigtoe_right_2d_right[:, 2:3]])
    smalltoe_right_2d_right = np.hstack([np.array([butter_lowpass_fillter(smalltoe_right_2d_right[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, smalltoe_right_2d_right[:, 2:3]])
    heel_right_2d_right = np.hstack([np.array([butter_lowpass_fillter(heel_right_2d_right[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, heel_right_2d_right[:, 2:3]])

    hip_right_2d_left = np.hstack([np.array([butter_lowpass_fillter(hip_right_2d_left[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, hip_right_2d_left[:, 2:3]])
    knee_right_2d_left = np.hstack([np.array([butter_lowpass_fillter(knee_right_2d_left[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, knee_right_2d_left[:, 2:3]])
    ankle_right_2d_left = np.hstack([np.array([butter_lowpass_fillter(ankle_right_2d_left[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, ankle_right_2d_left[:, 2:3]])
    bigtoe_right_2d_left = np.hstack([np.array([butter_lowpass_fillter(bigtoe_right_2d_left[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, bigtoe_right_2d_left[:, 2:3]])
    smalltoe_right_2d_left = np.hstack([np.array([butter_lowpass_fillter(smalltoe_right_2d_left[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, smalltoe_right_2d_left[:, 2:3]])
    heel_right_2d_left = np.hstack([np.array([butter_lowpass_fillter(heel_right_2d_left[:, x], order=4, cutoff_freq=6, frame_list=dia_right_frame_2d) for x in range(2)]).T, heel_right_2d_left[:, 2:3]])

    # 左側カメラ
    mid_hip_left_2d = cubic_spline_interpolation(keypoints_left_2d[:, 8, :], dia_left_frame_2d)
    neck_left_2d = cubic_spline_interpolation(keypoints_left_2d[:, 1, :], dia_left_frame_2d)

    hip_left_2d_right = cubic_spline_interpolation(keypoints_left_2d[:, 9, :], dia_left_frame_2d)
    knee_left_2d_right = cubic_spline_interpolation(keypoints_left_2d[:, 10, :], dia_left_frame_2d)
    ankle_left_2d_right = cubic_spline_interpolation(keypoints_left_2d[:, 11, :], dia_left_frame_2d)
    bigtoe_left_2d_right = cubic_spline_interpolation(keypoints_left_2d[:, 22, :], dia_left_frame_2d)
    smalltoe_left_2d_right = cubic_spline_interpolation(keypoints_left_2d[:, 23, :], dia_left_frame_2d)
    heel_left_2d_right = cubic_spline_interpolation(keypoints_left_2d[:, 24, :], dia_left_frame_2d)

    hip_left_2d_left = cubic_spline_interpolation(keypoints_left_2d[:, 12, :], dia_left_frame_2d)
    knee_left_2d_left = cubic_spline_interpolation(keypoints_left_2d[:, 13, :], dia_left_frame_2d)
    ankle_left_2d_left = cubic_spline_interpolation(keypoints_left_2d[:, 14, :], dia_left_frame_2d)
    bigtoe_left_2d_left = cubic_spline_interpolation(keypoints_left_2d[:, 19, :], dia_left_frame_2d)
    smalltoe_left_2d_left = cubic_spline_interpolation(keypoints_left_2d[:, 20, :], dia_left_frame_2d)
    heel_left_2d_left = cubic_spline_interpolation(keypoints_left_2d[:, 21, :], dia_left_frame_2d)

    mid_hip_left_2d = np.hstack([np.array([butter_lowpass_fillter(mid_hip_left_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, mid_hip_left_2d[:, 2:3]])
    neck_left_2d = np.hstack([np.array([butter_lowpass_fillter(neck_left_2d[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, neck_left_2d[:, 2:3]])

    hip_left_2d_right = np.hstack([np.array([butter_lowpass_fillter(hip_left_2d_right[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, hip_left_2d_right[:, 2:3]])
    knee_left_2d_right = np.hstack([np.array([butter_lowpass_fillter(knee_left_2d_right[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, knee_left_2d_right[:, 2:3]])
    ankle_left_2d_right = np.hstack([np.array([butter_lowpass_fillter(ankle_left_2d_right[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, ankle_left_2d_right[:, 2:3]])
    bigtoe_left_2d_right = np.hstack([np.array([butter_lowpass_fillter(bigtoe_left_2d_right[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, bigtoe_left_2d_right[:, 2:3]])
    smalltoe_left_2d_right = np.hstack([np.array([butter_lowpass_fillter(smalltoe_left_2d_right[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, smalltoe_left_2d_right[:, 2:3]])
    heel_left_2d_right = np.hstack([np.array([butter_lowpass_fillter(heel_left_2d_right[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, heel_left_2d_right[:, 2:3]])

    hip_left_2d_left = np.hstack([np.array([butter_lowpass_fillter(hip_left_2d_left[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, hip_left_2d_left[:, 2:3]])
    knee_left_2d_left = np.hstack([np.array([butter_lowpass_fillter(knee_left_2d_left[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, knee_left_2d_left[:, 2:3]])
    ankle_left_2d_left = np.hstack([np.array([butter_lowpass_fillter(ankle_left_2d_left[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, ankle_left_2d_left[:, 2:3]])
    bigtoe_left_2d_left = np.hstack([np.array([butter_lowpass_fillter(bigtoe_left_2d_left[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, bigtoe_left_2d_left[:, 2:3]])
    smalltoe_left_2d_left = np.hstack([np.array([butter_lowpass_fillter(smalltoe_left_2d_left[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, smalltoe_left_2d_left[:, 2:3]])
    heel_left_2d_left = np.hstack([np.array([butter_lowpass_fillter(heel_left_2d_left[:, x], order=4, cutoff_freq=6, frame_list=dia_left_frame_2d) for x in range(2)]).T, heel_left_2d_left[:, 2:3]])

    # 射影行列の読み込み
    P_1 = np.load(os.path.join(root_dir, "calibration" ,"P_1.npy"))
    P_2 = np.load(os.path.join(root_dir, "calibration" ,"P_2.npy"))

    # # フレーム数が異なる場合は短い方に合わせる
    openpose_frame = set(dia_right_frame_2d) & set(dia_left_frame_2d) & set(sagi_frame_2d)
    openpose_frame = sorted((openpose_frame))
    print(f"openpose_frame = {openpose_frame}")

    def reconstruction_dlt(P1, P2, point_array_2d_1, point_array_2d_2):
        X_array = np.array([])
        for frame in openpose_frame:
            x1, y1, c1 = point_array_2d_1[frame]
            x2, y2, c2 = point_array_2d_2[frame]
            # print(f"x1 = {x1}, y1 = {y1}, c1 = {c1}")
            # print(f"x2 = {x2}, y2 = {y2}, c2 = {c2}")

            # A = np.array([[c1 * (P1[2, 0] * x1 - P1[0, 0]), c1 * (P1[2, 1] * x1 - P1[0, 1]), c1 * (P1[2, 2] * x1 - P1[0, 2])],
            #               [c1 * (P1[2, 0] * y1 - P1[1, 0]), c1 * (P1[2, 1] * y1 - P1[1, 1]), c1 * (P1[2, 2] * y1 - P1[1, 2])],
            #               [c2 * (P2[2, 0] * x2 - P2[0, 0]), c2 * (P2[2, 1] * x2 - P2[0, 1]), c2 * (P2[2, 2] * x2 - P2[0, 2])],
            #               [c2 * (P2[2, 0] * y2 - P2[1, 0]), c2 * (P2[2, 1] * y2 - P2[1, 1]), c2 * (P2[2, 2] * y2 - P2[1, 2])]])

            # b = np.array([c1 * (P1[0, 3] - x1),
            #                 c1 * (P1[1, 3] - y1),
            #                 c2 * (P2[0, 3] - x2),
            #                 c2 * (P2[1, 3] - y2)])

            # X = np.linalg.pinv(A).dot(b)

            A = np.array([x1*P1[2] - P1[0],
                y1*P1[2] - P1[1],
                x2*P2[2] - P2[0],
                y2*P2[2] - P2[1]])

            Q = A.T.dot(A)
            u, s, vh = np.linalg.svd(Q)
            X = u[:, -1]
            X = X / X[-1]  #実際の3D座標に変換
            X = X[:3]  #最後の要素は不要

            X_array = np.append(X_array, X)

        X_array = X_array.reshape(len(openpose_frame), 3)
        # print(f"X_array = {X_array}")
        # print(f"X_array.shape = {X_array.shape}")
        return X_array

    mid_hip_3d = reconstruction_dlt(P_1, P_2, mid_hip_right_2d, mid_hip_left_2d)
    neck_3d = reconstruction_dlt(P_1, P_2, neck_right_2d, neck_left_2d)

    hip_3d_right = reconstruction_dlt(P_1, P_2, hip_right_2d_right, hip_left_2d_right)
    knee_3d_right = reconstruction_dlt(P_1, P_2, knee_right_2d_right, knee_left_2d_right)
    ankle_3d_right = reconstruction_dlt(P_1, P_2, ankle_right_2d_right, ankle_left_2d_right)
    bigtoe_3d_right = reconstruction_dlt(P_1, P_2, bigtoe_right_2d_right, bigtoe_left_2d_right)
    smalltoe_3d_right = reconstruction_dlt(P_1, P_2, smalltoe_right_2d_right, smalltoe_left_2d_right)
    heel_3d_right = reconstruction_dlt(P_1, P_2, heel_right_2d_right, heel_left_2d_right)

    hip_3d_left = reconstruction_dlt(P_1, P_2, hip_right_2d_left, hip_left_2d_left)
    knee_3d_left = reconstruction_dlt(P_1, P_2, knee_right_2d_left, knee_left_2d_left)
    ankle_3d_left = reconstruction_dlt(P_1, P_2, ankle_right_2d_left, ankle_left_2d_left)
    bigtoe_3d_left = reconstruction_dlt(P_1, P_2, bigtoe_right_2d_left, bigtoe_left_2d_left)
    smalltoe_3d_left = reconstruction_dlt(P_1, P_2, smalltoe_right_2d_left, smalltoe_left_2d_left)
    heel_3d_left = reconstruction_dlt(P_1, P_2, heel_right_2d_left, heel_left_2d_left)

    trunk_vector_3d = neck_3d - mid_hip_3d

    thigh_vector_3d_right = knee_3d_right - hip_3d_right
    lower_leg_vector_3d_right = knee_3d_right - ankle_3d_right
    foot_vector_3d_right = (bigtoe_3d_right + smalltoe_3d_right) / 2 - heel_3d_right

    thigh_vector_3d_left = knee_3d_left - hip_3d_left
    lower_leg_vector_3d_left = knee_3d_left - ankle_3d_left
    foot_vector_3d_left = (bigtoe_3d_left + smalltoe_3d_left) / 2 - heel_3d_left

    hip_l2r_vector_3d = hip_3d_right - hip_3d_left

    print("3次元DLT法の処理が終了しました")
    # """

    #3D DLT法で求めた関節角度(基準は3DMCの軸)
    hip_angle_3d_right = -pd.DataFrame(calculate_angle(thigh_vector_3d_right, trunk_vector_3d))
    knee_angle_3d_right = -pd.DataFrame(calculate_angle(thigh_vector_3d_right, lower_leg_vector_3d_right))
    ankle_angle_3d_right = -pd.DataFrame(calculate_angle(foot_vector_3d_right, lower_leg_vector_3d_right))

    hip_angle_3d_left = -pd.DataFrame(calculate_angle(thigh_vector_3d_left, trunk_vector_3d))
    knee_angle_3d_left = -pd.DataFrame(calculate_angle(thigh_vector_3d_left, lower_leg_vector_3d_left))
    ankle_angle_3d_left = -pd.DataFrame(calculate_angle(foot_vector_3d_left, lower_leg_vector_3d_left))

    hip_angle_abd_add_right = pd.DataFrame(calculate_angle(hip_l2r_vector_3d, thigh_vector_3d_right)) - 90
    hip_angle_abd_add_left = 90 - pd.DataFrame(calculate_angle(hip_l2r_vector_3d, thigh_vector_3d_left))

    # plt.plot(openpose_frame, hip_angle_abd_add_right)
    # plt.plot(openpose_frame, hip_angle_abd_add_left)
    # plt.legend(["hip_angle_abd_add_right", "hip_angle_abd_add_left"])
    # plt.show()
    # plt.close()

    hip_angle_3d_right = hip_angle_3d_right.where(hip_angle_3d_right <= 100, hip_angle_3d_right - 360)
    knee_angle_3d_right = knee_angle_3d_right.where(knee_angle_3d_right <= 100, knee_angle_3d_right - 360)
    ankle_angle_3d_right = ankle_angle_3d_right.where(ankle_angle_3d_right <= 0, ankle_angle_3d_right - 360)
    hip_angle_3d_right = hip_angle_3d_right + 180
    knee_angle_3d_right = knee_angle_3d_right + 180
    ankle_angle_3d_right = ankle_angle_3d_right + 90

    hip_angle_3d_left = hip_angle_3d_left.where(hip_angle_3d_left <= 100, hip_angle_3d_left - 360)
    knee_angle_3d_left = knee_angle_3d_left.where(knee_angle_3d_left <= 100, knee_angle_3d_left - 360)
    ankle_angle_3d_left = ankle_angle_3d_left.where(ankle_angle_3d_left <= 0, ankle_angle_3d_left - 360)
    hip_angle_3d_left = hip_angle_3d_left + 180
    knee_angle_3d_left = knee_angle_3d_left + 180
    ankle_angle_3d_left = ankle_angle_3d_left + 90

    df_3d_angle = pd.concat([hip_angle_3d_right, knee_angle_3d_right, ankle_angle_3d_right, hip_angle_3d_left, knee_angle_3d_left, ankle_angle_3d_left, hip_angle_abd_add_right, hip_angle_abd_add_left], axis=1)
    df_3d_angle.index = openpose_frame
    df_3d_angle.columns = ["hip_angle_right", "knee_angle_right", "ankle_angle_right", "hip_angle_left", "knee_angle_left", "ankle_angle_left", "hip_angle_abd_add_right", "hip_angle_abd_add_left"]
    df_3d_angle.to_csv(os.path.join(root_dir, f"{os.path.basename(mkv_sagittal).split('.')[0].split('_dev')[0]}_3d_angle.csv"))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    ax1.plot(openpose_frame, hip_angle_3d_right, label="hip_angle_right")
    ax1.plot(openpose_frame, hip_angle_3d_left, label="hip_angle_left")
    ax1.set_title("hip_angle")
    ax1.legend()

    ax2.plot(openpose_frame, knee_angle_3d_right, label="knee_angle_right")
    ax2.plot(openpose_frame, knee_angle_3d_left, label="knee_angle_left")
    ax2.set_title("knee_angle")
    ax2.legend()

    ax3.plot(openpose_frame, ankle_angle_3d_right, label="ankle_angle_right")
    ax3.plot(openpose_frame, ankle_angle_3d_left, label="ankle_angle_left")
    ax3.set_title("ankle_angle")
    ax3.legend()

    plt.savefig(os.path.join(root_dir, f"{os.path.basename(mkv_sagittal).split('.')[0].split('_dev')[0]}_3d_angle.png"))
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
