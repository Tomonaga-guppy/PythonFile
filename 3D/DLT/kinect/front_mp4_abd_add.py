from pathlib import Path
import os
import json
import numpy as np
import glob
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
import pandas as pd
import matplotlib.pyplot as plt

root_dir = r"G:\gait_pattern\20241102"
target = "*front_30m*"
ori_mov_path = list(Path(root_dir).glob(f"{target}/original.mp4"))[0]

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

def read_2d_openpose(mp4_file):
    json_folder_path = mp4_file.with_name('estimated.json')
    print(f"json_folder_path = {json_folder_path}")
    all_keypoints_2d = []  # 各フレームの2Dキーポイントを保持するリスト
    check_openpose_list = [1, 8, 12, 13, 14, 19, 20, 21]
    valid_frames = []
    json_files = glob.glob(os.path.join(json_folder_path, "*.json"))
    for i, json_file in enumerate(json_files):
        keypoints_data = load_keypoints_for_frame(i, json_folder_path) #[25, 3]
        # print(f"i = {i} mkv = {mp4_file} keypoints_data = {keypoints_data}")

        # キーポイント抽出が出来ているフレームを記録
        if all(not np.all(np.isnan(keypoints_data[point, :])) for point in check_openpose_list):
            valid_frames.append(i)
        all_keypoints_2d.append(keypoints_data)

    keypoints_2d_openpose = np.array(all_keypoints_2d)

    return keypoints_2d_openpose, valid_frames




def main():
    keypoints, valid_frames = read_2d_openpose(ori_mov_path)
    print(f"keypoints.shape = {keypoints.shape}")

    neck_2d = cubic_spline_interpolation(keypoints[:, 1, :], valid_frames)
    mid_hip_2d = cubic_spline_interpolation(keypoints[:, 8, :], valid_frames)
    rhip_2d = cubic_spline_interpolation(keypoints[:, 9, :], valid_frames)
    rknee_2d = cubic_spline_interpolation(keypoints[:, 10, :], valid_frames)
    rankle_2d = cubic_spline_interpolation(keypoints[:, 11, :], valid_frames)
    rbigtoe_2d = cubic_spline_interpolation(keypoints[:, 22, :], valid_frames)
    rsmalltoe_2d = cubic_spline_interpolation(keypoints[:, 23, :], valid_frames)
    rheel_2d = cubic_spline_interpolation(keypoints[:, 24, :], valid_frames)
    lhip_2d = cubic_spline_interpolation(keypoints[:, 12, :], valid_frames)
    lknee_2d = cubic_spline_interpolation(keypoints[:, 13, :], valid_frames)
    lankle_2d = cubic_spline_interpolation(keypoints[:, 14, :], valid_frames)
    lbigtoe_2d = cubic_spline_interpolation(keypoints[:, 19, :], valid_frames)
    lsmalltoe_2d = cubic_spline_interpolation(keypoints[:, 20, :], valid_frames)
    lheel_2d = cubic_spline_interpolation(keypoints[:, 21, :], valid_frames)

    neck_2d = np.array([butter_lowpass_fillter(neck_2d[:, x], order = 4, cutoff_freq = 6, frame_list = valid_frames) for x in range(2)]).T
    mid_hip_2d = np.array([butter_lowpass_fillter(mid_hip_2d[:, x], order = 4, cutoff_freq = 6, frame_list = valid_frames) for x in range(2)]).T
    rhip_2d = np.array([butter_lowpass_fillter(rhip_2d[:, x], order = 4, cutoff_freq = 6, frame_list = valid_frames) for x in range(2)]).T
    rknee_2d = np.array([butter_lowpass_fillter(rknee_2d[:, x], order = 4, cutoff_freq = 6, frame_list = valid_frames) for x in range(2)]).T
    rankle_2d = np.array([butter_lowpass_fillter(rankle_2d[:, x], order = 4, cutoff_freq = 6, frame_list = valid_frames) for x in range(2)]).T
    rbigtoe_2d = np.array([butter_lowpass_fillter(rbigtoe_2d[:, x], order = 4, cutoff_freq = 6, frame_list = valid_frames) for x in range(2)]).T
    rsmalltoe_2d = np.array([butter_lowpass_fillter(rsmalltoe_2d[:, x], order = 4, cutoff_freq = 6, frame_list = valid_frames) for x in range(2)]).T
    rheel_2d = np.array([butter_lowpass_fillter(rheel_2d[:, x], order = 4, cutoff_freq = 6, frame_list = valid_frames) for x in range(2)]).T
    lhip_2d = np.array([butter_lowpass_fillter(lhip_2d[:, x], order = 4, cutoff_freq = 6, frame_list = valid_frames) for x in range(2)]).T
    lknee_2d = np.array([butter_lowpass_fillter(lknee_2d[:, x], order = 4, cutoff_freq = 6, frame_list = valid_frames) for x in range(2)]).T
    lankle_2d = np.array([butter_lowpass_fillter(lankle_2d[:, x], order = 4, cutoff_freq = 6, frame_list = valid_frames) for x in range(2)]).T
    lbigtoe_2d = np.array([butter_lowpass_fillter(lbigtoe_2d[:, x], order = 4, cutoff_freq = 6, frame_list = valid_frames) for x in range(2)]).T
    lsmalltoe_2d = np.array([butter_lowpass_fillter(lsmalltoe_2d[:, x], order = 4, cutoff_freq = 6, frame_list = valid_frames) for x in range(2)]).T
    lheel_2d = np.array([butter_lowpass_fillter(lheel_2d[:, x], order = 4, cutoff_freq = 6, frame_list = valid_frames) for x in range(2)]).T

    trunk_vector = neck_2d - mid_hip_2d
    rthigh_vector = rknee_2d - rhip_2d
    rlower_leg_vector = rknee_2d - rankle_2d
    rfoot_vector = (rbigtoe_2d + rsmalltoe_2d) / 2  - rheel_2d
    lthigh_vector = lknee_2d - lhip_2d
    llower_leg_vector = lknee_2d - lankle_2d
    lfoot_vector = (lbigtoe_2d + lsmalltoe_2d) / 2 - lheel_2d

    def calculate_angle(vector1, vector2):  #(frame, xyz)または(frame, xy)の配列を入力)
        # vector1からvector2への回転角度を計算
        angle_list = []
        if vector1.shape[1] == 2:  #2Dベクトル
            for frame in range(len(vector1)):
                dot_product = np.dot(vector1[frame,:], vector2[frame,:])
                cross_product = np.cross(vector1[frame,:], vector2[frame,:])
                angle = np.rad2deg(np.arctan2(cross_product, dot_product))
                angle_list.append(angle)

        return angle_list

    rhip_2d = pd.DataFrame(calculate_angle(rthigh_vector, trunk_vector))
    rknee_2d = pd.DataFrame(calculate_angle(rthigh_vector, rlower_leg_vector))
    rankle_2d = pd.DataFrame(calculate_angle(rfoot_vector, rlower_leg_vector))
    lhip_2d = - pd.DataFrame(calculate_angle(lthigh_vector, trunk_vector))
    lknee_2d = - pd.DataFrame(calculate_angle(lthigh_vector, llower_leg_vector))
    lankle_2d = - pd.DataFrame(calculate_angle(lfoot_vector, llower_leg_vector))



    rhip_2d = 180 - rhip_2d
    rknee_2d = 180 - rknee_2d
    rankle_2d = 90 - rankle_2d
    lhip_2d = 180 - lhip_2d
    lknee_2d = 180 - lknee_2d
    lankle_2d = 90 - lankle_2d

    rhip_2d = rhip_2d.where(rhip_2d < 180, rhip_2d-360)
    rknee_2d = rknee_2d.where(rknee_2d < 180, rknee_2d-360)
    rankle_2d = rankle_2d.where(rankle_2d < 180, rankle_2d-360)
    lhip_2d = lhip_2d.where(lhip_2d < 180, lhip_2d-360)
    lknee_2d = lknee_2d.where(lknee_2d < 180, lknee_2d-360)
    lankle_2d = lankle_2d.where(lankle_2d < 180, lankle_2d-360)

    r_ic_frame = [190, 274, 341]  #おおよそ-2m, -0.5m, 0.5m
    l_ic_frame = [155, 236, 306, 376]  #おおよそ-2.5m, -1m, 0m, 1.5m

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    ax1.plot(rhip_2d, label='rhip')
    ax1.plot(lhip_2d, label='lhip')
    ax1.legend()
    ax1.set_ylim(-20, 30)
    ax1.set_title('hip abd_add')
    ax2.plot(rknee_2d, label='rknee')
    ax2.plot(lknee_2d, label='lknee')
    ax2.legend()
    ax2.set_ylim(-20, 30)
    ax2.set_title('knee abd_add')
    ax3.plot(rankle_2d, label='rankle')
    ax3.plot(lankle_2d, label='lankle')
    ax3.legend()
    ax3.set_ylim(-100, 100)
    ax3.set_title('ankle abd_add')
    plt.show()
    plt.close()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    ax1.plot(rhip_2d, label='rhip abd add')
    ax1.set_title('rhip abd add')
    [ax1.axvline(x=frame, color='r', linestyle='--') for frame in r_ic_frame]
    ax1.set_xlim(r_ic_frame[1], r_ic_frame[2])
    ax1.set_ylim(-10, 10)
    ax2.plot(rknee_2d, label='rknee abd add')
    ax2.set_title('rknee abd add')
    [ax2.axvline(x=frame, color='r', linestyle='--') for frame in r_ic_frame]
    ax2.set_xlim(r_ic_frame[1], r_ic_frame[2])
    ax2.set_ylim(-10, 30)
    ax3.plot(rankle_2d, label='rankle abd add')
    ax3.set_title('rankle abd add')
    [ax3.axvline(x=frame, color='r', linestyle='--') for frame in r_ic_frame]
    ax3.set_xlim(r_ic_frame[1], r_ic_frame[2])
    ax3.set_ylim(-100, 0)
    plt.show()
    plt.close()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    ax1.plot(lhip_2d, label='lhip')
    ax1.set_title('lhip abd_add')
    [ax1.axvline(x=frame, color='r', linestyle='--') for frame in l_ic_frame]
    ax1.set_xlim(l_ic_frame[1], l_ic_frame[2])
    ax1.set_ylim(-10, 10)
    ax2.plot(lknee_2d, label='lknee')
    ax2.set_title('lknee abd_add')
    [ax2.axvline(x=frame, color='r', linestyle='--') for frame in l_ic_frame]
    ax2.set_xlim(l_ic_frame[1], l_ic_frame[2])
    ax2.set_ylim(-10, 30)
    ax3.plot(lankle_2d, label='lankle')
    ax3.set_title('lankle abd_add')
    [ax3.axvline(x=frame, color='r', linestyle='--') for frame in l_ic_frame]
    ax3.set_xlim(l_ic_frame[1], l_ic_frame[2])
    ax3.set_ylim(-100, 0)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()



