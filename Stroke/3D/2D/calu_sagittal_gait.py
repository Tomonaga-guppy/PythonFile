import pandas as pd
import numpy as np
import os
import json
import cv2
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
from pathlib import Path
import pickle

root_dir = Path(r"G:\gait_pattern\20241114_ota_test\gopro\sagi")
# condition_list = ["sub0_abngait", "sub0_asgait_1", "sub0_asgait_2"]
# condition_list = ["sub0_asgait_1"]
condition_list = ["sub0_ngait"]

keypoint_names = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow",
                "LWrist","MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle",
                "REye","LEye","REar","LEar","LBigToe","LSmallToe","LHeel","RBigToe",
                    "RSmallToe","RHeel"]

def load_keypoints_for_frame(json_file_path):
    # jsonファイルを読み込んで[25,3]のnumpy配列を返す
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
        if len(json_data['people']) < 1:  #人が検出されなかった場合はnanで埋める
            # keypoints_data = np.zeros((25, 3))
            keypoints_data = np.full((25, 3), np.nan)
        else:
            json_data = np.array(json_data['people'][0]['pose_keypoints_2d'])  #[75,]
            keypoints_data = json_data.reshape((25, 3))
    return keypoints_data

def butter_lowpass_fillter(data, order, cutoff_freq, frame_list):  #4次のバターワースローパスフィルタ
    sampling_freq = 30
    nyquist_freq = sampling_freq / 2
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data[frame_list])
    data_fillter = np.copy(data)
    data_fillter[frame_list] = y
    return data_fillter

def cubic_spline_interpolation(keypoints_set, frame_range):
    # 新しい配列を作成して補間結果を保持
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

def linear_interpolation(x, x0, x1, y0, y1):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def read_2d_openpose(json_folder, camParams_dict):
    all_keypoints_2d = []  # 各フレームの2Dキーポイントを保持するリスト
    check_openpose_list = [1, 8, 12, 13, 14, 19, 20, 21]
    valid_frames = []
    for i, json_file in enumerate(json_folder.glob("*.json")):
        keypoints_data = load_keypoints_for_frame(json_file)
        undistort_points = cv2.undistortPoints(np.array([keypoints_data[:, 0], keypoints_data[:, 1]]).T, camParams_dict["intrinsicMat"], camParams_dict["distortion"])
        # print(f"undistort_points.shape:{undistort_points.shape}")  #(25, 1, 2)
        keypoints_data[:, 0] = undistort_points[:, 0, 0]
        keypoints_data[:, 1] = undistort_points[:, 0, 1]
        p = keypoints_data[:, 2]  #openposeが算出した信頼度

        # キーポイント抽出が出来ているフレームを記録
        if all(not np.all(np.isnan(keypoints_data[point, :])) for point in check_openpose_list):
            valid_frames.append(i)

        #確率が0.5未満のキーポイントをnanに変換
        threshold = 0.
        for j in range(len(p)):
            if p[j] < threshold:
                keypoints_data[j] = [np.nan, np.nan]
        all_keypoints_2d.append(keypoints_data)
        #keypoints_dataの0をnanに変換
        keypoints_data[keypoints_data == 0] = np.nan
        all_keypoints_2d.append(keypoints_data)
    keypoints_2d_openpose = np.array(all_keypoints_2d)
    print(f"keypoints_2d_openpose.shape:{keypoints_2d_openpose.shape}")

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
    for condition in condition_list:
        print(f"condition = {condition}")
        json_folder = root_dir / (condition + "_op.json")

        # 歪みを補正するためのカメラパラメータを読み込む
        camParames_path = root_dir.parent.parent.parent / "int_cali" / "ota" / "Intrinsic_sg.pickle"
        try:
            with open(str(camParames_path), "rb") as f:
                CameraParams_dict = pickle.load(f)
        except:
            print(f"{camParames_path} が読み込めませんでした。")
        #2d上でのキーポイントを取得"
        keypoints_sagittal_2d, sagi_frame_2d = read_2d_openpose(json_folder, CameraParams_dict)  #[frame, 25, 3]

        #矢状面2d用の処理
        mid_hip_sagttal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 8, :], sagi_frame_2d) #[frame, 2]
        neck_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 1, :], sagi_frame_2d)
        lhip_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 12, :], sagi_frame_2d)
        lknee_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 13, :], sagi_frame_2d)
        lankle_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 14, :], sagi_frame_2d)
        lbigtoe_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 19, :], sagi_frame_2d)
        lsmalltoe_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 20, :], sagi_frame_2d)
        lheel_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d[:, 21, :], sagi_frame_2d)

        mid_hip_sagttal_2d_filltered = np.array([butter_lowpass_fillter(mid_hip_sagttal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        neck_sagittal_2d_filltered = np.array([butter_lowpass_fillter(neck_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lhip_sagittal_2d_filltered = np.array([butter_lowpass_fillter(lhip_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lknee_sagittal_2d_filltered = np.array([butter_lowpass_fillter(lknee_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lankle_sagittal_2d_filltered = np.array([butter_lowpass_fillter(lankle_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lbigtoe_sagittal_2d_filltered = np.array([butter_lowpass_fillter(lbigtoe_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lsmalltoe_sagittal_2d_filltered = np.array([butter_lowpass_fillter(lsmalltoe_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lheel_sagittal_2d_filltered = np.array([butter_lowpass_fillter(lheel_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T

        trunk_vector_sagittal_2d = neck_sagittal_2d - mid_hip_sagttal_2d
        thigh_vector_l_sagittal_2d = lknee_sagittal_2d - lhip_sagittal_2d
        lower_leg_vector_l_sagittal_2d = lknee_sagittal_2d - lankle_sagittal_2d
        foot_vector_l_sagittal_2d = (lbigtoe_sagittal_2d + lsmalltoe_sagittal_2d) / 2 - lheel_sagittal_2d

        trunk_vector_sagittal_2d_filtered = neck_sagittal_2d_filltered - mid_hip_sagttal_2d_filltered
        thigh_vector_l_sagittal_2d_filtered = lknee_sagittal_2d_filltered - lhip_sagittal_2d_filltered
        lower_leg_vector_l_sagittal_2d_filtered = lknee_sagittal_2d_filltered - lankle_sagittal_2d_filltered
        foot_vector_l_sagittal_2d_filtered = (lbigtoe_sagittal_2d_filltered + lsmalltoe_sagittal_2d_filltered) / 2 - lheel_sagittal_2d_filltered

        #すべてで記録できているフレームを抽出
        print(f"sagi_frame_2d = {sagi_frame_2d}")

        #ICのフレームを取得
        ic_frame_sg = [sagi_frame_2d[0], sagi_frame_2d[-1]]

        hip_angle_sagittal_2d = pd.DataFrame(calculate_angle(trunk_vector_sagittal_2d, thigh_vector_l_sagittal_2d))
        knee_angle_sagittal_2d = pd.DataFrame(calculate_angle(thigh_vector_l_sagittal_2d, lower_leg_vector_l_sagittal_2d))
        ankle_angle_sagittal_2d = pd.DataFrame(calculate_angle(lower_leg_vector_l_sagittal_2d, foot_vector_l_sagittal_2d))

        hip_angle_sagittal_2d = 180 - hip_angle_sagittal_2d
        knee_angle_sagittal_2d = 180 - knee_angle_sagittal_2d
        ankle_angle_sagittal_2d =  90 - ankle_angle_sagittal_2d

        hip_angle_sagittal_2d_filtered = pd.DataFrame(calculate_angle(trunk_vector_sagittal_2d_filtered, thigh_vector_l_sagittal_2d_filtered))
        knee_angle_sagittal_2d_filtered = pd.DataFrame(calculate_angle(thigh_vector_l_sagittal_2d_filtered, lower_leg_vector_l_sagittal_2d_filtered))
        ankle_angle_sagittal_2d_filtered = pd.DataFrame(calculate_angle(lower_leg_vector_l_sagittal_2d_filtered, foot_vector_l_sagittal_2d_filtered))

        hip_angle_sagittal_2d_filtered = 180 - hip_angle_sagittal_2d_filtered
        knee_angle_sagittal_2d_filtered = 180 - knee_angle_sagittal_2d_filtered
        ankle_angle_sagittal_2d_filtered =  90 - ankle_angle_sagittal_2d_filtered

        for ic_frame in ic_frame_sg:
            plt.axvline(x=ic_frame, color='gray', linestyle='--')
        plt.plot(sagi_frame_2d, hip_angle_sagittal_2d_filtered.loc[sagi_frame_2d], label="2D sagittal", color='#1f77b4')
        plt.plot(sagi_frame_2d, hip_angle_sagittal_2d.loc[sagi_frame_2d], color='#1f77b4', alpha=0.5)
        plt.title("Hip Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition}_hip_angle.png"))
        # plt.show()  #5frame
        plt.cla()

        for ic_frame in ic_frame_sg:
            plt.axvline(x=ic_frame, color='gray', linestyle='--')
        plt.plot(sagi_frame_2d, knee_angle_sagittal_2d_filtered.loc[sagi_frame_2d], label="2D sagittal", color='#1f77b4')
        plt.plot(sagi_frame_2d, knee_angle_sagittal_2d.loc[sagi_frame_2d], color='#1f77b4', alpha=0.5)
        plt.title("Knee Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition}_knee_angle.png"))
        # plt.show() #3frame
        plt.cla()

        for ic_frame in ic_frame_sg:
            plt.axvline(x=ic_frame, color='gray', linestyle='--')
        plt.plot(sagi_frame_2d, ankle_angle_sagittal_2d_filtered.loc[sagi_frame_2d], label="2D sagittal", color='#1f77b4')
        plt.plot(sagi_frame_2d, ankle_angle_sagittal_2d.loc[sagi_frame_2d], color='#1f77b4', alpha=0.5)
        plt.title("Ankle Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition}_ankle_angle.png"))
        # plt.show() #5frame
        plt.cla()

if __name__ == "__main__":
    main()
