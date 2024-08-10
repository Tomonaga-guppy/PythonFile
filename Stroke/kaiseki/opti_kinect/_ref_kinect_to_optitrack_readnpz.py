import pandas as pd
import numpy as np
import os
import glob
from pyk4a import PyK4A, PyK4APlayback, CalibrationType
import json
import math
import cv2
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from sklearn.metrics import mean_absolute_error


root_dir = r"F:\Tomson\gait_pattern\20240712"
condition_list = ["0_Tpose", "1_walk", "2_walk_slow", "3_comp_walk", "4_comp_walk_slow"]
condition_key = condition_list[:]

def butter_lowpass_filter(data, order, cutoff_freq):  #4次のバターワースローパスフィルタ
    sampling_freq = 30
    nyquist_freq = sampling_freq / 2
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # data内にnanがある場合は線形補間してからフィルター
    if np.any(np.isnan(data)):
        nans, x = np.isnan(data), lambda z: z.nonzero()[0]
        data[nans] = np.interp(x(nans), x(~nans), data[~nans])
    y = lfilter(b, a, data, axis=0)
    return y

def calculate_angle(vector1, vector2):  #(frame, xyz)または(frame, xy)の配列を入力
    angle_list = []
    for frame in range(len(vector1)):
        dot_product = np.dot(vector1[frame], vector2[frame])
        cross_product = np.cross(vector1[frame], vector2[frame])
        angle = np.arctan2(np.linalg.norm(cross_product), dot_product)
        angle = angle * 180 / np.pi
        angle_list.append(angle)
    return angle_list

def check_opti_frame(opti_list, keypoints_mocap):
    valid_frames = []
    for frame in range(keypoints_mocap.shape[0]):
        if all(not np.isnan(keypoints_mocap[frame, point, :]).any() for point in opti_list):
            valid_frames.append(frame)
    return valid_frames

def check_openpose_frame(openpose_list, keypoints_openpose):
    valid_frames = []
    for frame in range(keypoints_openpose.shape[0]):
        if all(not np.isnan(keypoints_openpose[frame, point, :]).any() for point in openpose_list):
            valid_frames.append(frame)
    return valid_frames

def main():
    for condition in condition_key:
        condition_num = condition.split("_")[0]
        print(f"\ncondition_num = {condition_num}")
        npz_file_path = glob.glob(os.path.join(root_dir, f"{condition_num}*frame.npz"))[0]
        keypoint_sets = np.load(npz_file_path)
        keypoints_sagittal_2d = keypoint_sets['sagittal_2d']
        keypoints_sagittal_3d = keypoint_sets['sagittal_3d']
        keypoints_diagonal_right = keypoint_sets['diagonal_right']
        keypoints_diagonal_left = keypoint_sets['diagonal_left']
        keypoints_frontal = keypoint_sets['frontal']
        keypoints_mocap = keypoint_sets['mocap']
        common_frame = keypoint_sets['common_frame']

        # keypoints_frontal.shape[0] より大きい要素を除外 openpose_3dでは最後のフレームが取得できていないため
        common_frame = [frame for frame in common_frame if frame < keypoints_frontal.shape[0]]
        # print(f"common_frame = {common_frame}")

        #角度を比較する(今回は左足で比較) 体幹と膝、足首のベクトルを使って角度を計算
        #フィルター前
        trunk_vector_sagittal_2d_ori = keypoints_sagittal_2d[common_frame, 1, :] - keypoints_sagittal_2d[common_frame, 8, :] #neck - mid_hip
        thigh_vector_l_sagittal_2d_ori = keypoints_sagittal_2d[common_frame, 13, :] - keypoints_sagittal_2d[common_frame, 12, :] #knee - hip
        lower_leg_vector_l_sagittal_2d_ori = keypoints_sagittal_2d[common_frame, 14, :] - keypoints_sagittal_2d[common_frame, 13, :] #ankle - knee
        foot_vector_l_sagittal_2d_ori = keypoints_sagittal_2d[common_frame, 21, :]  - (keypoints_sagittal_2d[common_frame, 19, :] + keypoints_sagittal_2d[common_frame, 20, :]) / 2 #heel - (bigtoe + smalltoe) / 2

        trunk_vector_3d_frontal_ori = keypoints_frontal[common_frame, 1, :] - keypoints_frontal[common_frame, 8, :]
        thigh_vector_l_3d_frontal_ori = keypoints_frontal[common_frame, 13, :] - keypoints_frontal[common_frame, 12, :]
        lower_leg_vector_l_3d_frontal_ori = keypoints_frontal[common_frame, 14, :] - keypoints_frontal[common_frame, 13, :]
        foot_vector_l_3d_frontal_ori = keypoints_frontal[common_frame, 21, :]  - (keypoints_frontal[common_frame, 19, :] + keypoints_frontal[common_frame, 20, :]) / 2

        trunk_vector_mocap_ori = (keypoints_mocap[common_frame, 21, :] + keypoints_mocap[common_frame, 9, :]) / 2 - (keypoints_mocap[common_frame, 20, :] + keypoints_mocap[common_frame, 8, :] + keypoints_mocap[common_frame, 15, :] + keypoints_mocap[common_frame, 4, :]) / 4
        thigh_vector_l_mocap_ori = (keypoints_mocap[common_frame, 6, :] + keypoints_mocap[common_frame, 7, :]) / 2 - (keypoints_mocap[common_frame, 4, :] + keypoints_mocap[common_frame, 8, :]) / 2
        lower_vector_l_mocap_ori = (keypoints_mocap[common_frame, 2, :] + keypoints_mocap[common_frame, 3, :]) / 2 - (keypoints_mocap[common_frame, 6, :] + keypoints_mocap[common_frame, 7, :]) / 2
        foot_vector_l_mocap_ori = keypoints_mocap[common_frame, 5, :] - keypoints_mocap[common_frame, 12, :]

        mid_hip_sagttal_2d = np.array([butter_lowpass_filter(keypoints_sagittal_2d[common_frame, 8, x], order = 4, cutoff_freq = 6) for x in range(2)]).T
        neck_sagittal_2d = np.array([butter_lowpass_filter(keypoints_sagittal_2d[common_frame, 1, x], order = 4, cutoff_freq = 6) for x in range(2)]).T
        lhip_sagittal_2d = np.array([butter_lowpass_filter(keypoints_sagittal_2d[common_frame, 12, x], order = 4, cutoff_freq = 6) for x in range(2)]).T
        lknee_sagittal_2d = np.array([butter_lowpass_filter(keypoints_sagittal_2d[common_frame, 13, x], order = 4, cutoff_freq = 6) for x in range(2)]).T
        lankle_sagittal_2d = np.array([butter_lowpass_filter(keypoints_sagittal_2d[common_frame, 14, x], order = 4, cutoff_freq = 6) for x in range(2)]).T
        lbigtoe_sagittal_2d = np.array([butter_lowpass_filter(keypoints_sagittal_2d[common_frame, 19, x], order = 4, cutoff_freq = 6) for x in range(2)]).T
        lsmalltoe_sagittal_2d = np.array([butter_lowpass_filter(keypoints_sagittal_2d[common_frame, 20, x], order = 4, cutoff_freq = 6) for x in range(2)]).T
        lheel_sagittal_2d = np.array([butter_lowpass_filter(keypoints_sagittal_2d[common_frame, 21, x], order = 4, cutoff_freq = 6) for x in range(2)]).T

        mid_hip_diagonal_right = np.array([butter_lowpass_filter(keypoints_diagonal_right[common_frame, 8, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        neck_diagonal_right = np.array([butter_lowpass_filter(keypoints_diagonal_right[common_frame, 1, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lhip_diagonal_right = np.array([butter_lowpass_filter(keypoints_diagonal_right[common_frame, 12, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lknee_diagonal_right = np.array([butter_lowpass_filter(keypoints_diagonal_right[common_frame, 13, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lankle_diagonal_right = np.array([butter_lowpass_filter(keypoints_diagonal_right[common_frame, 14, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lbigtoe_diagonal_right = np.array([butter_lowpass_filter(keypoints_diagonal_right[common_frame, 19, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lsmalltoe_diagonal_right = np.array([butter_lowpass_filter(keypoints_diagonal_right[common_frame, 20, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lheel_diagonal_right = np.array([butter_lowpass_filter(keypoints_diagonal_right[common_frame, 21, x], order = 4, cutoff_freq = 6) for x in range(3)]).T

        mid_hip_diagonal_left = np.array([butter_lowpass_filter(keypoints_diagonal_left[common_frame, 8, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        neck_diagonal_left = np.array([butter_lowpass_filter(keypoints_diagonal_left[common_frame, 1, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lhip_diagonal_left = np.array([butter_lowpass_filter(keypoints_diagonal_left[common_frame, 12, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lknee_diagonal_left = np.array([butter_lowpass_filter(keypoints_diagonal_left[common_frame, 13, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lankle_diagonal_left = np.array([butter_lowpass_filter(keypoints_diagonal_left[common_frame, 14, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lbigtoe_diagonal_left = np.array([butter_lowpass_filter(keypoints_diagonal_left[common_frame, 19, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lsmalltoe_diagonal_left = np.array([butter_lowpass_filter(keypoints_diagonal_left[common_frame, 20, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lheel_diagonal_left = np.array([butter_lowpass_filter(keypoints_diagonal_left[common_frame, 21, x], order = 4, cutoff_freq = 6) for x in range(3)]).T

        rsho = np.array([butter_lowpass_filter(keypoints_mocap[common_frame, 21, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        lsho = np.array([butter_lowpass_filter(keypoints_mocap[common_frame, 9, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        rpsi = np.array([butter_lowpass_filter(keypoints_mocap[common_frame, 20, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        lpsi = np.array([butter_lowpass_filter(keypoints_mocap[common_frame, 8, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        rasi = np.array([butter_lowpass_filter(keypoints_mocap[common_frame, 15, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        lasi = np.array([butter_lowpass_filter(keypoints_mocap[common_frame, 4, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        lknee = np.array([butter_lowpass_filter(keypoints_mocap[common_frame, 6, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        lknee2 = np.array([butter_lowpass_filter(keypoints_mocap[common_frame, 7, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        lank = np.array([butter_lowpass_filter(keypoints_mocap[common_frame, 2, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        lank2 = np.array([butter_lowpass_filter(keypoints_mocap[common_frame, 3, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        ltoe = np.array([butter_lowpass_filter(keypoints_mocap[common_frame, 12, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        lhee = np.array([butter_lowpass_filter(keypoints_mocap[common_frame, 5, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T


        trunk_vector_sagittal_2d = neck_sagittal_2d - mid_hip_sagttal_2d #MidHipaからNeck
        thigh_vector_l_sagittal_2d = lknee_sagittal_2d - lhip_sagittal_2d #LhipからLKnee
        lower_leg_vector_l_sagittal_2d = lankle_sagittal_2d - lknee_sagittal_2d #LKneeからLAnkle
        foot_vector_l_sagittal_2d = lheel_sagittal_2d  - (lbigtoe_sagittal_2d + lsmalltoe_sagittal_2d) / 2 #LBigToeとLSmallToeの中点からLHeel

        trunk_vector_3d_diagonal_right = neck_diagonal_right - mid_hip_diagonal_right
        thigh_vector_3d_diagonal_right = lknee_diagonal_right - lhip_diagonal_right
        lower_leg_vector_3d_diagonal_right = lankle_diagonal_right - lknee_diagonal_right
        foot_vector_3d_diagonal_right = lheel_diagonal_right  - (lbigtoe_diagonal_right + lsmalltoe_diagonal_right) / 2

        trunk_vector_3d_diagonal_left = neck_diagonal_left - mid_hip_diagonal_left
        thigh_vector_3d_diagonal_left = lknee_diagonal_left - lhip_diagonal_left
        lower_leg_vector_3d_diagonal_left = lankle_diagonal_left - lknee_diagonal_left
        foot_vector_3d_diagonal_left = lheel_diagonal_left  - (lbigtoe_diagonal_left + lsmalltoe_diagonal_left) / 2

        trunk_vector_3d_frontal = (trunk_vector_3d_diagonal_right + trunk_vector_3d_diagonal_left) / 2
        thigh_vector_l_3d_frontal = (thigh_vector_3d_diagonal_right + thigh_vector_3d_diagonal_left) / 2
        lower_leg_vector_l_3d_frontal = (lower_leg_vector_3d_diagonal_right + lower_leg_vector_3d_diagonal_left) / 2
        foot_vector_l_3d_frontal = (foot_vector_3d_diagonal_right + foot_vector_3d_diagonal_left) / 2

        trunk_vector_mocap = (rsho + lsho) / 2 - (rasi + lasi + rpsi + lpsi) / 4 #RASIとLASIとRPSIとLPSIの中点→RSHOとLSHOの中点
        thigh_vector_l_mocap = (lknee + lknee2) / 2 - (lasi + lpsi) / 2 #LASIとLPSIの中点→LKNEとLKNE2の中点
        lower_vector_l_mocap = (lank + lank2) / 2 - (lknee + lknee2) / 2 #LKNEとLKNE2の中点→LANKとLANK2の中点
        foot_vector_l_mocap = lhee - ltoe #LTOE→LHEE


        #フィルター前
        hip_angle_sagittal_2d_ori = calculate_angle(trunk_vector_sagittal_2d_ori, thigh_vector_l_sagittal_2d_ori)
        knee_angle_sagittal_2d_ori = calculate_angle(thigh_vector_l_sagittal_2d_ori, lower_leg_vector_l_sagittal_2d_ori)
        ankle_angle_sagittal_2d_ori = calculate_angle(lower_leg_vector_l_sagittal_2d_ori, foot_vector_l_sagittal_2d_ori)

        hip_angle_frontal_3d_ori = calculate_angle(trunk_vector_3d_frontal_ori, thigh_vector_l_3d_frontal_ori)
        knee_angle_frontal_3d_ori = calculate_angle(thigh_vector_l_3d_frontal_ori, lower_leg_vector_l_3d_frontal_ori)
        ankle_angle_frontal_3d_ori = calculate_angle(lower_leg_vector_l_3d_frontal_ori, foot_vector_l_3d_frontal_ori)

        hip_angle_mocap_ori = calculate_angle(trunk_vector_mocap_ori, thigh_vector_l_mocap_ori)
        knee_angle_mocap_ori = calculate_angle(thigh_vector_l_mocap_ori, lower_vector_l_mocap_ori)
        ankle_angle_mocap_ori = calculate_angle(lower_vector_l_mocap_ori, foot_vector_l_mocap_ori)

        #フィルター後
        hip_angle_sagittal_2d = calculate_angle(trunk_vector_sagittal_2d, thigh_vector_l_sagittal_2d)
        knee_angle_sagittal_2d = calculate_angle(thigh_vector_l_sagittal_2d, lower_leg_vector_l_sagittal_2d)
        ankle_angle_sagittal_2d = calculate_angle(lower_leg_vector_l_sagittal_2d, foot_vector_l_sagittal_2d)

        hip_angle_frontal_3d = calculate_angle(trunk_vector_3d_frontal, thigh_vector_l_3d_frontal)
        knee_angle_frontal_3d = calculate_angle(thigh_vector_l_3d_frontal, lower_leg_vector_l_3d_frontal)
        ankle_angle_frontal_3d = calculate_angle(lower_leg_vector_l_3d_frontal, foot_vector_l_3d_frontal)

        hip_angle_mocap = calculate_angle(trunk_vector_mocap, thigh_vector_l_mocap)
        knee_angle_mocap = calculate_angle(thigh_vector_l_mocap, lower_vector_l_mocap)
        ankle_angle_mocap = calculate_angle(lower_vector_l_mocap, foot_vector_l_mocap)

        df_ankle = pd.DataFrame({
            'ankle_angle_sagittal_2d': ankle_angle_sagittal_2d,
            'ankle_angle_sagittal_2d_ori': ankle_angle_sagittal_2d_ori,
            'ankle_angle_frontal_3d': ankle_angle_frontal_3d,
            'ankle_angle_frontal_3d_ori': ankle_angle_frontal_3d_ori,
            'ankle_angle_mocap': ankle_angle_mocap,
            'ankle_angle_mocap_ori': ankle_angle_mocap_ori,
        })
        df_ankle.to_csv(os.path.join(root_dir, f"{condition}_ankle_angle.csv"))


        plt.plot(common_frame, hip_angle_sagittal_2d, label="2D sagittal", color='#1f77b4')
        plt.plot(common_frame, hip_angle_frontal_3d, label="3D frontal", color='#ff7f0e')
        plt.plot(common_frame, hip_angle_mocap, label="Mocap", color='#2ca02c')
        plt.plot(common_frame, hip_angle_sagittal_2d_ori, label="2D sagittal_ori", color='#1f77b4', alpha=0.5)
        plt.plot(common_frame, hip_angle_frontal_3d_ori, label="3D frontal_ori", color='#ff7f0e', alpha=0.5)
        plt.plot(common_frame, hip_angle_mocap_ori, label="Mocap_ori", color='#2ca02c', alpha=0.5)
        plt.title("Hip Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition}_hip_angle.png"))
        # plt.show()
        plt.cla()

        plt.plot(common_frame, knee_angle_sagittal_2d, label="2D sagittal", color='#1f77b4')
        plt.plot(common_frame, knee_angle_frontal_3d, label="3D frontal", color='#ff7f0e')
        plt.plot(common_frame, knee_angle_mocap, label="Mocap", color='#2ca02c')
        plt.plot(common_frame, knee_angle_sagittal_2d_ori, label="2D sagittal_ori", color='#1f77b4', alpha=0.5)
        plt.plot(common_frame, knee_angle_frontal_3d_ori, label="3D frontal_ori", color='#ff7f0e', alpha=0.5)
        plt.plot(common_frame, knee_angle_mocap_ori, label="Mocap_ori", color='#2ca02c', alpha=0.5)
        plt.title("Knee Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition}_knee_angle.png"))
        # plt.show()
        plt.cla()

        plt.plot(common_frame, ankle_angle_sagittal_2d, label="2D sagittal", color='#1f77b4')
        plt.plot(common_frame, ankle_angle_frontal_3d, label="3D frontal", color='#ff7f0e')
        plt.plot(common_frame, ankle_angle_mocap, label="Mocap", color='#2ca02c')
        plt.plot(common_frame, ankle_angle_sagittal_2d_ori, label="2D sagittal_ori", color='#1f77b4', alpha=0.5)
        plt.plot(common_frame, ankle_angle_frontal_3d_ori, label="3D frontal_ori", color='#ff7f0e', alpha=0.5)
        plt.plot(common_frame, ankle_angle_mocap_ori, label="Mocap_ori", color='#2ca02c', alpha=0.5)
        plt.title("Ankle Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition}_ankle_angle.png"))
        # plt.show()
        plt.cla()

        mae_hip_sagittal = mean_absolute_error(hip_angle_sagittal_2d, hip_angle_mocap)
        mae_hip_frontal = mean_absolute_error(hip_angle_frontal_3d, hip_angle_mocap)
        mae_knee_sagittal = mean_absolute_error(knee_angle_sagittal_2d, knee_angle_mocap)
        mae_knee_frontal = mean_absolute_error(knee_angle_frontal_3d, knee_angle_mocap)
        mae_ankle_sagittal = mean_absolute_error(ankle_angle_sagittal_2d, ankle_angle_mocap)
        mae_ankle_frontal = mean_absolute_error(ankle_angle_frontal_3d, ankle_angle_mocap)

        print(f"mae_hip_sagittal = {mae_hip_sagittal}")
        print(f"mae_knee_sagittal = {mae_knee_sagittal}")
        print(f"mae_ankle_sagittal = {mae_ankle_sagittal}")

        print(f"mae_hip_frontal = {mae_hip_frontal}")
        print(f"mae_knee_frontal = {mae_knee_frontal}")
        print(f"mae_ankle_frontal = {mae_ankle_frontal}")

if __name__ == "__main__":
    main()