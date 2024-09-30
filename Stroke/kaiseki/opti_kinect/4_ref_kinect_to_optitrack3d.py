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
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import time

root_dir = r"F:\Tomson\gait_pattern\20240808"

# condition_list = ["0", "1", "2", "3", "4"]  #0がTポーズ,1が通常歩行, 2が通常歩行（遅）, 3が疑似麻痺歩行, 4が疑似麻痺歩行（遅）
# condition_list = ["0", "1"]  #0がTポーズ,1が通常歩行, 2が通常歩行（遅）, 3が疑似麻痺歩行, 4が疑似麻痺歩行（遅）
condition_list = ["1"]  #0がTポーズ,1が通常歩行, 2が通常歩行（遅）, 3が疑似麻痺歩行, 4が疑似麻痺歩行（遅）
condition_keynum = condition_list[:]

def load_keypoints_for_frame(frame_number, json_folder_path):
    json_file_name = f"original_{frame_number:012d}_keypoints.json"
    json_file_path = os.path.join(json_folder_path, json_file_name)

    if not os.path.exists(json_file_path):
        return None

    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
        if len(json_data['people']) < 1:  #人が検出されなかった場合はnanで埋める
            # keypoints_data = np.zeros((25, 3))
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

def linear_interpolation(x, x0, x1, y0, y1):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def get_3d_coordinates(pixel, depth_image, calibration, point_name):
    if np.all(pixel == (0, 0)):
        print(f"    {point_name} Openposeで検出できてない")
        return [0, 0, 0]

    pixel_x, pixel_y = pixel[0], pixel[1]
    x0, x1 = int(math.floor(pixel_x)), int(math.ceil(pixel_x))
    y0, y1 = int(math.floor(pixel_y)), int(math.ceil(pixel_y))

    height, width = depth_image.shape

    if not (0 <= x0 < width and 0 <= x1 < width and 0 <= y0 < height and 0 <= y1 < height):
        print(f"    {(x0, y0)}や{(x1, y1)}は画像の範囲外です (画像サイズ：{width}*{height})")
        return [0, 0, 0]

    depth_value_x0_y0 = depth_image[y0, x0]
    depth_value_x1_y0 = depth_image[y0, x1]
    depth_value_x0_y1 = depth_image[y1, x0]
    depth_value_x1_y1 = depth_image[y1, x1]

    try:
        point_x0_y0 = calibration.convert_2d_to_3d(coordinates=(x0, y0), depth=depth_value_x0_y0, source_camera=CalibrationType.COLOR)
        point_x1_y0 = calibration.convert_2d_to_3d(coordinates=(x1, y0), depth=depth_value_x1_y0, source_camera=CalibrationType.COLOR)
        point_x0_y1 = calibration.convert_2d_to_3d(coordinates=(x0, y1), depth=depth_value_x0_y1, source_camera=CalibrationType.COLOR)
        point_x1_y1 = calibration.convert_2d_to_3d(coordinates=(x1, y1), depth=depth_value_x1_y1, source_camera=CalibrationType.COLOR)
    except ValueError as e:
        print(f"    {point_name}の3D変換時にエラー: {e}")
        return [0, 0, 0]

    point_y0 = [linear_interpolation(pixel_x, x0, x1, point_x0_y0[i], point_x1_y0[i]) for i in range(3)]
    point_y1 = [linear_interpolation(pixel_x, x0, x1, point_x0_y1[i], point_x1_y1[i]) for i in range(3)]

    point = [linear_interpolation(pixel_y, y0, y1, point_y0[i], point_y1[i]) for i in range(3)]
    # print(f"type = {type(point)}")  #<class 'list'>

    return point

def read_2d_openpose(mkv_file):
    # print(f"mkv_file = {mkv_file}")
    id = os.path.basename(mkv_file).split('.')[0].split('_')[-1]
    json_folder_path = os.path.join(os.path.dirname(mkv_file), os.path.basename(mkv_file).split('.')[0], 'estimated.json')
    all_keypoints_2d = []  # 各フレームの2Dキーポイントを保持するリスト
    all_keypoints_2d_tf = []
    check_openpose_list = [1, 8, 12, 13, 14, 19, 20, 21]
    valid_frames = []
    for i, json_file in enumerate(glob.glob(os.path.join(json_folder_path, "*.json"))):
        keypoints_data = load_keypoints_for_frame(i, json_folder_path)[:, :2] #[25, 2]
        kakuritu = load_keypoints_for_frame(i, json_folder_path)[:, 2] #25

        # print(f"i = {i} mkv = {mkv_file} keypoints_data = {keypoints_data}")


        # キーポイント抽出が出来ているフレームを記録
        if all(not np.all(np.isnan(keypoints_data[point, :])) for point in check_openpose_list):
            valid_frames.append(i)
        # if all(np.all(keypoints_data[point, :] != 0) for point in check_openpose_list):
        #     valid_frames.append(i)

        #確率が0.5未満のキーポイントをnanに変換
        threshold = 0.
        for j in range(len(kakuritu)):
            if kakuritu[j] < threshold:
                keypoints_data[j] = [np.nan, np.nan]
        all_keypoints_2d.append(keypoints_data)

        #keypoints_dataの0をnanに変換
        keypoints_data[keypoints_data == 0] = np.nan

        transformation_matrix_path = os.path.join(os.path.dirname(mkv_file), f'tf_matrix_calibration_{id}.npz')
        transformation_matrix = np.load(transformation_matrix_path)['a_2d']

        keypoints_data_tf = [np.dot(np.linalg.inv(transformation_matrix), np.array([keypoints_data[keypoint_num][0], keypoints_data[keypoint_num][1], 1]).T)[:2] for keypoint_num in range(len(keypoints_data))]

        # print(f"keypoints_data = {keypoints_data}")
        # print(f"keypoints_data_tf = {keypoints_data_tf}")

        all_keypoints_2d.append(keypoints_data)
        all_keypoints_2d_tf.append(keypoints_data_tf)
    keypoints_2d_openpose = np.array(all_keypoints_2d)
    keypoints_2d_openpose_tf = np.array(all_keypoints_2d_tf)

    return keypoints_2d_openpose, keypoints_2d_openpose_tf, valid_frames

def read_3d_openpose(keypoint_array_2d, valid_frame, mkv_file):
    print(f"valid_frame = {valid_frame}")
    keypoints_data = np.nan_to_num(keypoint_array_2d)  # [8, frame, 2]←もともと[frame, 2]
    playback = PyK4APlayback(mkv_file)
    playback.open()
    calibration = playback.calibration

    frame_count = 0
    all_keypoints_3d = []  # 各フレームの3Dキーポイントを保持するリスト
    frame_keypoints_3d_all = []

    id = os.path.basename(mkv_file).split('.')[0].split('_')[-1]
    transform_matrix_path = os.path.join(os.path.dirname(mkv_file), f'tf_matrix_calibration_{id}.npz')
    transform_matrix = np.load(transform_matrix_path)
    a1 = transform_matrix['a1']  #Arucoマーカーまでの変換行列
    a2 = transform_matrix['a2']  #Optitrackまでの変換行列(平行移動のみ)

    while True:
        try:
            capture = playback.get_next_capture()
        except:
            print("再生を終了します")
            break

        if frame_count not in valid_frame:
            frame_count += 1
            continue

        if capture.color is None or capture.transformed_depth is None:
            print(f"Frame {frame_count} has no image data.")
            continue

        print(f"frame_count = {frame_count} mkvfile = {mkv_file} ")

        # 画像を取得
        depth_image_path = os.path.join(os.path.dirname(mkv_file), os.path.basename(mkv_file).split('.')[0], "depth_image_original" ,f"{frame_count+1:04d}.png")
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

        interpolated_depth_image_path = os.path.join(os.path.dirname(mkv_file), os.path.basename(mkv_file).split('.')[0], "filled_depth_image" ,f"{frame_count+1:04d}.png")
        interpolated_depth_image = cv2.imread(interpolated_depth_image_path, cv2.IMREAD_UNCHANGED)

        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # axes[0].imshow(depth_image)
        # axes[1].imshow(interpolated_depth_image)
        # plt.show()


        frame_keypoints_3d = []

        #8は[mid_hip, neck, lhip, lknee, lankle, lbigtoe, lsmalltoe, lheel]の順番
        point_names_list = ["mid_hip", "neck", "lhip", "lknee", "lankle", "lbigtoe", "lsmalltoe", "lheel"]
        for i in range(len(keypoints_data)):
            point_name = point_names_list[i]
            pixel = keypoints_data[i, frame_count, :]  #keypoints_data = [8, frame,2]  [frame, 2]
            coordinates_cam = get_3d_coordinates(pixel, interpolated_depth_image, calibration, point_name)  #カメラ座標系での3D座標
            if coordinates_cam == [0, 0, 0]:
                frame_keypoints_3d.append([0, 0, 0])
            else:
                A1_inv = np.linalg.inv(a1)
                coordinates_aruco = np.dot(A1_inv, np.array([coordinates_cam[0], coordinates_cam[1], coordinates_cam[2], 1]).T)[:3]  #Arucoマーカー座標系での3D座標
                A2_inv = np.linalg.inv(a2)
                coordinates = np.dot(A2_inv, np.array([coordinates_aruco[0], coordinates_aruco[1], coordinates_aruco[2], 1]).T)[:3]  #Optitrack座標系での3D座標
                coordinates = [coordinates[0], coordinates[1], coordinates[2]]
                frame_keypoints_3d.append(coordinates)

        frame_keypoints_3d_all.append(frame_keypoints_3d)

        # all_non_zero = all(np.any(frame_keypoints_3d[id][:2] != 0) for id in check_openpose_list)
        # if all_non_zero:
        #     valid_frame_list.append(frame_count)

        frame_count += 1

    keypoints_openpose = np.array(frame_keypoints_3d_all) /1000  #単位をmmからmに変換
    return keypoints_openpose



def main():
    for condition_num in condition_keynum:
        print(f"condition = {condition_num}")
        mkv_files = glob.glob(os.path.join(root_dir, f"{condition_num}*.mkv"))
        mkv_sagittal = mkv_files[0]
        mkv_diagonal_right = mkv_files[1]
        mkv_diagonal_left = mkv_files[2]

        #モーキャプから求めた関節角度データを取得
        angle_csv_files = glob.glob(os.path.join(root_dir, "Motive", f"angle_30Hz_{condition_num}*.csv"))[0]
        df_mocap_angle = pd.read_csv(angle_csv_files, index_col=0)
        mocap_frame = df_mocap_angle.index.values
        #モーキャプから求めた初期接地時のフレーム
        ic_frame_path = glob.glob(os.path.join(root_dir, "Motive",f"ic_frame_{condition_num}*.npy"))[0]
        ic_frame_mocap = np.load(ic_frame_path)
        ic_span_mocap = int(np.diff(ic_frame_mocap).mean()) #初期接地時のフレーム間隔（平均）
        print(f"ic_frame_mocap = {ic_frame_mocap} ic_span_mocap = {ic_span_mocap}")

        #2d上でのキーポイントを取得"
        _, keypoints_sagittal_2d_tf, sagi_frame_2d = read_2d_openpose(mkv_sagittal)  #[frame, 25, 3]
        keypoints_diagonal_right_2d, _, dia_right_frame_2d = read_2d_openpose(mkv_diagonal_right)
        keypoints_diagonal_left_2d, _, dia_left_frame_2d = read_2d_openpose(mkv_diagonal_left)

        print(f"keypoints_sagittal_2d_tf = {keypoints_sagittal_2d_tf.shape}")

        #矢状面2d用の処理
        #3次スプライン補間
        mid_hip_sagttal_2d = cubic_spline_interpolation(keypoints_sagittal_2d_tf[:, 8, :], sagi_frame_2d) #[frame, 2]
        neck_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d_tf[:, 1, :], sagi_frame_2d)
        lhip_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d_tf[:, 12, :], sagi_frame_2d)
        lknee_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d_tf[:, 13, :], sagi_frame_2d)
        lankle_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d_tf[:, 14, :], sagi_frame_2d)
        lbigtoe_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d_tf[:, 19, :], sagi_frame_2d)
        lsmalltoe_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d_tf[:, 20, :], sagi_frame_2d)
        lheel_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d_tf[:, 21, :], sagi_frame_2d)
        # 4次のバターワースローパスフィルタ
        mid_hip_sagttal_2d = np.array([butter_lowpass_fillter(mid_hip_sagttal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        neck_sagittal_2d = np.array([butter_lowpass_fillter(neck_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lhip_sagittal_2d = np.array([butter_lowpass_fillter(lhip_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lknee_sagittal_2d = np.array([butter_lowpass_fillter(lknee_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lankle_sagittal_2d = np.array([butter_lowpass_fillter(lankle_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lbigtoe_sagittal_2d = np.array([butter_lowpass_fillter(lbigtoe_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lsmalltoe_sagittal_2d = np.array([butter_lowpass_fillter(lsmalltoe_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lheel_sagittal_2d = np.array([butter_lowpass_fillter(lheel_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T

        trunk_vector_sagittal_2d = neck_sagittal_2d - mid_hip_sagttal_2d
        thigh_vector_l_sagittal_2d = lknee_sagittal_2d - lhip_sagittal_2d
        lower_leg_vector_l_sagittal_2d = lknee_sagittal_2d - lankle_sagittal_2d
        foot_vector_l_sagittal_2d = (lbigtoe_sagittal_2d + lsmalltoe_sagittal_2d) / 2 - lheel_sagittal_2d





        """
        #前額面3d用の処理
        print("3dようの処理を開始しました")
        mid_hip_diagonal_right_2d = cubic_spline_interpolation(keypoints_diagonal_right_2d[:, 8, :], dia_right_frame_2d)
        neck_diagonal_right_2d = cubic_spline_interpolation(keypoints_diagonal_right_2d[:, 1, :], dia_right_frame_2d)
        lhip_diagonal_right_2d = cubic_spline_interpolation(keypoints_diagonal_right_2d[:, 12, :], dia_right_frame_2d)
        lknee_diagonal_right_2d = cubic_spline_interpolation(keypoints_diagonal_right_2d[:, 13, :], dia_right_frame_2d)
        lankle_diagonal_right_2d = cubic_spline_interpolation(keypoints_diagonal_right_2d[:, 14, :], dia_right_frame_2d)
        lbigtoe_diagonal_right_2d = cubic_spline_interpolation(keypoints_diagonal_right_2d[:, 19, :], dia_right_frame_2d)
        lsmalltoe_diagonal_right_2d = cubic_spline_interpolation(keypoints_diagonal_right_2d[:, 20, :], dia_right_frame_2d)
        lheel_diagonal_right_2d = cubic_spline_interpolation(keypoints_diagonal_right_2d[:, 21, :], dia_right_frame_2d)

        mid_hip_diagonal_right_2d_filltered = np.array([butter_lowpass_fillter(mid_hip_diagonal_right_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_right_frame_2d) for x in range(2)]).T
        neck_diagonal_right_2d_filltered = np.array([butter_lowpass_fillter(neck_diagonal_right_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_right_frame_2d) for x in range(2)]).T
        lhip_diagonal_right_2d_filltered = np.array([butter_lowpass_fillter(lhip_diagonal_right_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_right_frame_2d) for x in range(2)]).T
        lknee_diagonal_right_2d_filltered = np.array([butter_lowpass_fillter(lknee_diagonal_right_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_right_frame_2d) for x in range(2)]).T
        lankle_diagonal_right_2d_filltered = np.array([butter_lowpass_fillter(lankle_diagonal_right_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_right_frame_2d) for x in range(2)]).T
        lbigtoe_diagonal_right_2d_filltered = np.array([butter_lowpass_fillter(lbigtoe_diagonal_right_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_right_frame_2d) for x in range(2)]).T
        lsmalltoe_diagonal_right_2d_filltered = np.array([butter_lowpass_fillter(lsmalltoe_diagonal_right_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_right_frame_2d) for x in range(2)]).T
        lheel_diagonal_right_2d_filltered = np.array([butter_lowpass_fillter(lheel_diagonal_right_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_right_frame_2d) for x in range(2)]).T

        mid_hip_diagonal_left_2d = cubic_spline_interpolation(keypoints_diagonal_left_2d[:, 8, :], dia_left_frame_2d)
        neck_diagonal_left_2d = cubic_spline_interpolation(keypoints_diagonal_left_2d[:, 1, :], dia_left_frame_2d)
        lhip_diagonal_left_2d = cubic_spline_interpolation(keypoints_diagonal_left_2d[:, 12, :], dia_left_frame_2d)
        lknee_diagonal_left_2d = cubic_spline_interpolation(keypoints_diagonal_left_2d[:, 13, :], dia_left_frame_2d)
        lankle_diagonal_left_2d = cubic_spline_interpolation(keypoints_diagonal_left_2d[:, 14, :], dia_left_frame_2d)
        lbigtoe_diagonal_left_2d = cubic_spline_interpolation(keypoints_diagonal_left_2d[:, 19, :], dia_left_frame_2d)
        lsmalltoe_diagonal_left_2d = cubic_spline_interpolation(keypoints_diagonal_left_2d[:, 20, :], dia_left_frame_2d)
        lheel_diagonal_left_2d = cubic_spline_interpolation(keypoints_diagonal_left_2d[:, 21, :], dia_left_frame_2d)

        mid_hip_diagonal_left_2d_filltered = np.array([butter_lowpass_fillter(mid_hip_diagonal_left_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_left_frame_2d) for x in range(2)]).T
        neck_diagonal_left_2d_filltered = np.array([butter_lowpass_fillter(neck_diagonal_left_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_left_frame_2d) for x in range(2)]).T
        lhip_diagonal_left_2d_filltered = np.array([butter_lowpass_fillter(lhip_diagonal_left_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_left_frame_2d) for x in range(2)]).T
        lknee_diagonal_left_2d_filltered = np.array([butter_lowpass_fillter(lknee_diagonal_left_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_left_frame_2d) for x in range(2)]).T
        lankle_diagonal_left_2d_filltered = np.array([butter_lowpass_fillter(lankle_diagonal_left_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_left_frame_2d) for x in range(2)]).T
        lbigtoe_diagonal_left_2d_filltered = np.array([butter_lowpass_fillter(lbigtoe_diagonal_left_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_left_frame_2d) for x in range(2)]).T
        lsmalltoe_diagonal_left_2d_filltered = np.array([butter_lowpass_fillter(lsmalltoe_diagonal_left_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_left_frame_2d) for x in range(2)]).T
        lheel_diagonal_left_2d_filltered = np.array([butter_lowpass_fillter(lheel_diagonal_left_2d[:, x], order = 4, cutoff_freq = 6, frame_list = dia_left_frame_2d) for x in range(2)]).T

        frontal_df_path = os.path.join(root_dir, "frontal_check")
        if not os.path.exists(frontal_df_path):
            os.makedirs(frontal_df_path)

        # rightの関節角度を計算
        point_diagonal_right_2d_array = np.array([mid_hip_diagonal_right_2d_filltered, neck_diagonal_right_2d_filltered, lhip_diagonal_right_2d_filltered, lknee_diagonal_right_2d_filltered, lankle_diagonal_right_2d_filltered, lbigtoe_diagonal_right_2d_filltered, lsmalltoe_diagonal_right_2d_filltered, lheel_diagonal_right_2d_filltered])
        # print(f"point_diagonal_right_2d_array = {point_diagonal_right_2d_array.shape}")  #(8, frame, 2)
        point_diagonal_right_3d_array = read_3d_openpose(point_diagonal_right_2d_array, dia_right_frame_2d, mkv_diagonal_right) #(frame-1, 8, 3)
        # print(f"point_diagonal_right_3d_array = {point_diagonal_right_3d_array.shape}")
        # print(f"dia_right_frame_2d = {dia_right_frame_2d}")
        dict_diagonal_right = {"mid_hip_x": point_diagonal_right_3d_array[:, 0, 0], "mid_hip_y": point_diagonal_right_3d_array[:, 0, 1], "mid_hip_z": point_diagonal_right_3d_array[:, 0, 2], "neck_x": point_diagonal_right_3d_array[:, 1, 0], "neck_y": point_diagonal_right_3d_array[:, 1, 1], "neck_z": point_diagonal_right_3d_array[:, 1, 2], "lhip_x": point_diagonal_right_3d_array[:, 2, 0], "lhip_y": point_diagonal_right_3d_array[:, 2, 1], "lhip_z": point_diagonal_right_3d_array[:, 2, 2], "lknee_x": point_diagonal_right_3d_array[:, 3, 0], "lknee_y": point_diagonal_right_3d_array[:, 3, 1], "lknee_z": point_diagonal_right_3d_array[:, 3, 2], "lankle_x": point_diagonal_right_3d_array[:, 4, 0], "lankle_y": point_diagonal_right_3d_array[:, 4, 1], "lankle_z": point_diagonal_right_3d_array[:, 4, 2], "lbigtoe_x": point_diagonal_right_3d_array[:, 5, 0], "lbigtoe_y": point_diagonal_right_3d_array[:, 5, 1], "lbigtoe_z": point_diagonal_right_3d_array[:, 5, 2], "lsmalltoe_x": point_diagonal_right_3d_array[:, 6, 0], "lsmalltoe_y": point_diagonal_right_3d_array[:, 6, 1], "lsmalltoe_z": point_diagonal_right_3d_array[:, 6, 2], "lheel_x": point_diagonal_right_3d_array[:, 7, 0], "lheel_y": point_diagonal_right_3d_array[:, 7, 1], "lheel_z": point_diagonal_right_3d_array[:, 7, 2]}
        try:
            df_diagonal_right = pd.DataFrame(dict_diagonal_right, index = dia_right_frame_2d)
        except ValueError:
            df_diagonal_right = pd.DataFrame(dict_diagonal_right, index = dia_right_frame_2d[:-1])
        df_diagonal_right.to_csv(f"{frontal_df_path}/{condition_num}_df_diagonal_right.csv")

        # leftの関節角度を計算
        point_diagonal_left_2d_array = np.array([mid_hip_diagonal_left_2d_filltered, neck_diagonal_left_2d_filltered, lhip_diagonal_left_2d_filltered, lknee_diagonal_left_2d_filltered, lankle_diagonal_left_2d_filltered, lbigtoe_diagonal_left_2d_filltered, lsmalltoe_diagonal_left_2d_filltered, lheel_diagonal_left_2d_filltered])
        # print(f"point_diagonal_left_2d_array = {point_diagonal_left_2d_array.shape}")  #(8, frame, 2)
        point_diagonal_left_3d_array = read_3d_openpose(point_diagonal_left_2d_array, dia_left_frame_2d, mkv_diagonal_left) #(frame-1, 8, 3)
        # print(f"point_diagonal_left_3d_array = {point_diagonal_left_3d_array.shape}")
        # print(f"dia_left_frame_2d = {dia_left_frame_2d}")
        dict_diagonal_left = {"mid_hip_x": point_diagonal_left_3d_array[:, 0, 0], "mid_hip_y": point_diagonal_left_3d_array[:, 0, 1], "mid_hip_z": point_diagonal_left_3d_array[:, 0, 2], "neck_x": point_diagonal_left_3d_array[:, 1, 0], "neck_y": point_diagonal_left_3d_array[:, 1, 1], "neck_z": point_diagonal_left_3d_array[:, 1, 2], "lhip_x": point_diagonal_left_3d_array[:, 2, 0], "lhip_y": point_diagonal_left_3d_array[:, 2, 1], "lhip_z": point_diagonal_left_3d_array[:, 2, 2], "lknee_x": point_diagonal_left_3d_array[:, 3, 0], "lknee_y": point_diagonal_left_3d_array[:, 3, 1], "lknee_z": point_diagonal_left_3d_array[:, 3, 2], "lankle_x": point_diagonal_left_3d_array[:, 4, 0], "lankle_y": point_diagonal_left_3d_array[:, 4, 1], "lankle_z": point_diagonal_left_3d_array[:, 4, 2], "lbigtoe_x": point_diagonal_left_3d_array[:, 5, 0], "lbigtoe_y": point_diagonal_left_3d_array[:, 5, 1], "lbigtoe_z": point_diagonal_left_3d_array[:, 5, 2], "lsmalltoe_x": point_diagonal_left_3d_array[:, 6, 0], "lsmalltoe_y": point_diagonal_left_3d_array[:, 6, 1], "lsmalltoe_z": point_diagonal_left_3d_array[:, 6, 2], "lheel_x": point_diagonal_left_3d_array[:, 7, 0], "lheel_y": point_diagonal_left_3d_array[:, 7, 1], "lheel_z": point_diagonal_left_3d_array[:, 7, 2]}
        try:
            df_diagonal_left = pd.DataFrame(dict_diagonal_left, index = dia_left_frame_2d)
        except ValueError:
            df_diagonal_left = pd.DataFrame(dict_diagonal_left, index = dia_left_frame_2d[:-1])
        df_diagonal_left.to_csv(f"{frontal_df_path}/{condition_num}_df_diagonal_left.csv")

        # インデックスを統合して、新しいインデックスセットを作成
        combined_index = df_diagonal_right.index.union(df_diagonal_left.index)
        # 新しいデータフレームを作成
        df_frontal = pd.DataFrame(index=combined_index)
        # 各列について処理を行う
        for col in df_diagonal_right.columns:
            right_col = df_diagonal_right.reindex(combined_index)[col].fillna(0)
            left_col = df_diagonal_left.reindex(combined_index)[col].fillna(0)
            # 両方の値が0以外の場合は平均を取る
            frontal_col = np.where((right_col != 0) & (left_col != 0), (right_col + left_col) / 2,
                                # 片方が0の場合はもう片方の値を使う
                                np.where(right_col != 0, right_col,
                                            np.where(left_col != 0, left_col,
                                                    # 両方0の場合は0を使う
                                                    0)))
            # 新しいデータフレームに結果を追加
            df_frontal[col] = frontal_col

        # 結果を表示、CSVファイルに保存
        print(f"df_frontal = {df_frontal}")
        df_frontal.to_csv(f"{frontal_df_path}/{condition_num}_df_frontal.csv")
        df_frontal = df_frontal.replace(0, np.nan)

        # フレームの範囲を定義
        frame_range = df_frontal.index

        # 3次スプライン補間関数
        def cubic_spline_interpolation_df(column, frame_range):
            # 有効なデータ（NaNでない）でスプライン補間
            valid_index = column.dropna().index

            if len(valid_index) > 1:  # 有効なデータポイントが2つ以上必要
                # 有効なデータでスプラインを作成
                spline = CubicSpline(valid_index, column.dropna())
                # スプラインで補間
                interpolated_values = spline(frame_range)
                return pd.Series(interpolated_values, index=frame_range)
            else:
                # 有効なデータポイントが少ない場合は、そのまま返す
                return column

        # 各カラムに対してスプライン補間を適用
        df_frontal_interpolated = df_frontal.apply(lambda col: cubic_spline_interpolation_df(col, frame_range))

        def butter_lowpass_filter_df(column, order, cutoff_freq, frame_range):
            # フィルタを適用するために、NaNを0に置き換える
            valid_data = column.fillna(0)

            sampling_freq = 30
            nyquist_freq = sampling_freq / 2
            normal_cutoff = cutoff_freq / nyquist_freq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)

            # フィルター適用
            filtered_data = filtfilt(b, a, valid_data)

            # フィルター適用後に元のNaNの位置を保持
            filtered_column = pd.Series(filtered_data, index=frame_range).where(~column.isna(), np.nan)

            return filtered_column

        # 4次バターワースローパスフィルターを適用
        df_frontal_filtered = df_frontal_interpolated.apply(lambda col: butter_lowpass_filter_df(col, order=4, cutoff_freq=6, frame_range=frame_range))

        df_frontal = df_frontal_filtered

        # 補間後のデータを保存
        df_frontal.to_csv(f"{frontal_df_path}/{condition_num}_df_frontal_interpolated.csv")

        # keypoints_frontal = df_frontal.to_numpy()




        midhip_frontal = df_frontal[["mid_hip_x", "mid_hip_y", "mid_hip_z"]]
        neck_frontal = df_frontal[["neck_x", "neck_y", "neck_z"]]
        lhip_frontal = df_frontal[["lhip_x", "lhip_y", "lhip_z"]]
        lknee_frontal = df_frontal[["lknee_x", "lknee_y", "lknee_z"]]
        lankle_frontal = df_frontal[["lankle_x", "lankle_y", "lankle_z"]]
        lbigtoe_frontal = df_frontal[["lbigtoe_x", "lbigtoe_y", "lbigtoe_z"]]
        lsmalltoe_frontal = df_frontal[["lsmalltoe_x", "lsmalltoe_y", "lsmalltoe_z"]]
        lheel_frontal = df_frontal[["lheel_x", "lheel_y", "lheel_z"]]

        print(f"midhip_frontal = {midhip_frontal}")

        # trunk_vector_frontal = neck_frontal - midhip_frontal
        # thigh_vector_l_frontal = lknee_frontal - lhip_frontal
        # lower_leg_vector_l_frontal = lankle_frontal - lknee_frontal
        # foot_vector_l_frontal = (lbigtoe_frontal + lsmalltoe_frontal) / 2 - lheel_frontal

        # 各座標ごとのベクトル計算
        trunk_vector_frontal_x = neck_frontal["neck_x"] - midhip_frontal["mid_hip_x"]
        trunk_vector_frontal_y = neck_frontal["neck_y"] - midhip_frontal["mid_hip_y"]
        trunk_vector_frontal_z = neck_frontal["neck_z"] - midhip_frontal["mid_hip_z"]

        thigh_vector_l_frontal_x = lknee_frontal["lknee_x"] - lhip_frontal["lhip_x"]
        thigh_vector_l_frontal_y = lknee_frontal["lknee_y"] - lhip_frontal["lhip_y"]
        thigh_vector_l_frontal_z = lknee_frontal["lknee_z"] - lhip_frontal["lhip_z"]

        lower_leg_vector_l_frontal_x = lankle_frontal["lankle_x"] - lknee_frontal["lknee_x"]
        lower_leg_vector_l_frontal_y = lankle_frontal["lankle_y"] - lknee_frontal["lknee_y"]
        lower_leg_vector_l_frontal_z = lankle_frontal["lankle_z"] - lknee_frontal["lknee_z"]

        foot_vector_l_frontal_x = (lbigtoe_frontal["lbigtoe_x"] + lsmalltoe_frontal["lsmalltoe_x"]) / 2 - lheel_frontal["lheel_x"]
        foot_vector_l_frontal_y = (lbigtoe_frontal["lbigtoe_y"] + lsmalltoe_frontal["lsmalltoe_y"]) / 2 - lheel_frontal["lheel_y"]
        foot_vector_l_frontal_z = (lbigtoe_frontal["lbigtoe_z"] + lsmalltoe_frontal["lsmalltoe_z"]) / 2 - lheel_frontal["lheel_z"]

        # 差分をDataFrameにまとめる
        trunk_vector_frontal = pd.DataFrame({
            'x': trunk_vector_frontal_x,
            'y': trunk_vector_frontal_y,
            'z': trunk_vector_frontal_z
        })

        thigh_vector_l_frontal = pd.DataFrame({
            'x': thigh_vector_l_frontal_x,
            'y': thigh_vector_l_frontal_y,
            'z': thigh_vector_l_frontal_z
        })

        lower_leg_vector_l_frontal = pd.DataFrame({
            'x': lower_leg_vector_l_frontal_x,
            'y': lower_leg_vector_l_frontal_y,
            'z': lower_leg_vector_l_frontal_z
        })

        foot_vector_l_frontal = pd.DataFrame({
            'x': foot_vector_l_frontal_x,
            'y': foot_vector_l_frontal_y,
            'z': foot_vector_l_frontal_z
        })

        # print(f"trunk_vector_frontal = {trunk_vector_frontal}")


        trunk_vector_frontal = trunk_vector_frontal.to_numpy()
        thigh_vector_l_frontal = thigh_vector_l_frontal.to_numpy()
        lower_leg_vector_l_frontal = lower_leg_vector_l_frontal.to_numpy()
        foot_vector_l_frontal = foot_vector_l_frontal.to_numpy()

        # print(f"trunk_vector_frontal_np = {trunk_vector_frontal}")

        hip_angle_frontal = pd.DataFrame(calculate_angle(trunk_vector_frontal, thigh_vector_l_frontal), index = combined_index)
        knee_angle_frontal = pd.DataFrame(calculate_angle(thigh_vector_l_frontal, lower_leg_vector_l_frontal), index=combined_index)
        ankle_angle_frontal = pd.DataFrame(calculate_angle(lower_leg_vector_l_frontal, foot_vector_l_frontal), index=combined_index)

        # print(f"hip_angle_frontal = {hip_angle_frontal}")







        # 試しで左側のみ使用して3D座標を計算
        df_diagonal_left = df_diagonal_left.apply(lambda col: cubic_spline_interpolation_df(col, frame_range))
        df_diagonal_left = df_diagonal_left.apply(lambda col: butter_lowpass_filter_df(col, order=4, cutoff_freq=6, frame_range=frame_range))

        midhip_left = df_diagonal_left[["mid_hip_x", "mid_hip_y", "mid_hip_z"]]
        neck_left = df_diagonal_left[["neck_x", "neck_y", "neck_z"]]
        lhip_left = df_diagonal_left[["lhip_x", "lhip_y", "lhip_z"]]
        lknee_left = df_diagonal_left[["lknee_x", "lknee_y", "lknee_z"]]
        lankle_left = df_diagonal_left[["lankle_x", "lankle_y", "lankle_z"]]
        lbigtoe_left = df_diagonal_left[["lbigtoe_x", "lbigtoe_y", "lbigtoe_z"]]
        lsmalltoe_left = df_diagonal_left[["lsmalltoe_x", "lsmalltoe_y", "lsmalltoe_z"]]
        lheel_left = df_diagonal_left[["lheel_x", "lheel_y", "lheel_z"]]

        trunk_vector_left_x = neck_left["neck_x"] - midhip_left["mid_hip_x"]
        trunk_vector_left_y = neck_left["neck_y"] - midhip_left["mid_hip_y"]
        trunk_vector_left_z = neck_left["neck_z"] - midhip_left["mid_hip_z"]
        thigh_vector_l_left_x = lknee_left["lknee_x"] - lhip_left["lhip_x"]
        thigh_vector_l_left_y = lknee_left["lknee_y"] - lhip_left["lhip_y"]
        thigh_vector_l_left_z = lknee_left["lknee_z"] - lhip_left["lhip_z"]
        lower_leg_vector_left_x = lankle_left["lankle_x"] - lknee_left["lknee_x"]
        lower_leg_vector_left_y = lankle_left["lankle_y"] - lknee_left["lknee_y"]
        lower_leg_vector_left_z = lankle_left["lankle_z"] - lknee_left["lknee_z"]
        foot_vector_left_x = (lbigtoe_left["lbigtoe_x"] + lsmalltoe_left["lsmalltoe_x"]) / 2 - lheel_left["lheel_x"]
        foot_vector_left_y = (lbigtoe_left["lbigtoe_y"] + lsmalltoe_left["lsmalltoe_y"]) / 2 - lheel_left["lheel_y"]
        foot_vector_left_z = (lbigtoe_left["lbigtoe_z"] + lsmalltoe_left["lsmalltoe_z"]) / 2 - lheel_left["lheel_z"]

        trunk_vector_left = pd.DataFrame({
            'x': trunk_vector_left_x,
            'y': trunk_vector_left_y,
            'z': trunk_vector_left_z
        })
        thigh_vector_left = pd.DataFrame({
            'x': thigh_vector_l_left_x,
            'y': thigh_vector_l_left_y,
            'z': thigh_vector_l_left_z
        })
        lower_leg_vector_left = pd.DataFrame({
            'x': lower_leg_vector_left_x,
            'y': lower_leg_vector_left_y,
            'z': lower_leg_vector_left_z
        })
        foot_vector_left = pd.DataFrame({
            'x': foot_vector_left_x,
            'y': foot_vector_left_y,
            'z': foot_vector_left_z
        })

        trunk_vector_left = trunk_vector_left.to_numpy()
        thigh_vector_left = thigh_vector_left.to_numpy()
        lower_leg_vector_left = lower_leg_vector_left.to_numpy()
        foot_vector_left = foot_vector_left.to_numpy()

        hip_angle_left = pd.DataFrame(calculate_angle(trunk_vector_left, thigh_vector_left), index = combined_index)
        knee_angle_left = pd.DataFrame(calculate_angle(thigh_vector_left, lower_leg_vector_left), index = combined_index)
        ankle_angle_left = pd.DataFrame(calculate_angle(lower_leg_vector_left, foot_vector_left), index = combined_index)






        # 試しで右側のみ使用して3D座標を計算
        df_diagonal_right = df_diagonal_right.apply(lambda col: cubic_spline_interpolation_df(col, frame_range))
        df_diagonal_right = df_diagonal_right.apply(lambda col: butter_lowpass_filter_df(col, order=4, cutoff_freq=6, frame_range=frame_range))

        midhip_right = df_diagonal_right[["mid_hip_x", "mid_hip_y", "mid_hip_z"]]
        neck_right = df_diagonal_right[["neck_x", "neck_y", "neck_z"]]
        lhip_right = df_diagonal_right[["lhip_x", "lhip_y", "lhip_z"]]
        lknee_right = df_diagonal_right[["lknee_x", "lknee_y", "lknee_z"]]
        lankle_right = df_diagonal_right[["lankle_x", "lankle_y", "lankle_z"]]
        lbigtoe_right = df_diagonal_right[["lbigtoe_x", "lbigtoe_y", "lbigtoe_z"]]
        lsmalltoe_right = df_diagonal_right[["lsmalltoe_x", "lsmalltoe_y", "lsmalltoe_z"]]
        lheel_right = df_diagonal_right[["lheel_x", "lheel_y", "lheel_z"]]

        trunk_vector_right_x = neck_right["neck_x"] - midhip_right["mid_hip_x"]
        trunk_vector_right_y = neck_right["neck_y"] - midhip_right["mid_hip_y"]
        trunk_vector_right_z = neck_right["neck_z"] - midhip_right["mid_hip_z"]
        thigh_vector_l_right_x = lknee_right["lknee_x"] - lhip_right["lhip_x"]
        thigh_vector_l_right_y = lknee_right["lknee_y"] - lhip_right["lhip_y"]
        thigh_vector_l_right_z = lknee_right["lknee_z"] - lhip_right["lhip_z"]
        lower_leg_vector_right_x = lankle_right["lankle_x"] - lknee_right["lknee_x"]
        lower_leg_vector_right_y = lankle_right["lankle_y"] - lknee_right["lknee_y"]
        lower_leg_vector_right_z = lankle_right["lankle_z"] - lknee_right["lknee_z"]
        foot_vector_right_x = (lbigtoe_right["lbigtoe_x"] + lsmalltoe_right["lsmalltoe_x"]) / 2 - lheel_right["lheel_x"]
        foot_vector_right_y = (lbigtoe_right["lbigtoe_y"] + lsmalltoe_right["lsmalltoe_y"]) / 2 - lheel_right["lheel_y"]
        foot_vector_right_z = (lbigtoe_right["lbigtoe_z"] + lsmalltoe_right["lsmalltoe_z"]) / 2 - lheel_right["lheel_z"]

        trunk_vector_right = pd.DataFrame({
            'x': trunk_vector_right_x,
            'y': trunk_vector_right_y,
            'z': trunk_vector_right_z
        })
        thigh_vector_right = pd.DataFrame({
            'x': thigh_vector_l_right_x,
            'y': thigh_vector_l_right_y,
            'z': thigh_vector_l_right_z
        })
        lower_leg_vector_right = pd.DataFrame({
            'x': lower_leg_vector_right_x,
            'y': lower_leg_vector_right_y,
            'z': lower_leg_vector_right_z
        })
        foot_vector_right = pd.DataFrame({
            'x': foot_vector_right_x,
            'y': foot_vector_right_y,
            'z': foot_vector_right_z
        })

        trunk_vector_right = trunk_vector_right.to_numpy()
        thigh_vector_right = thigh_vector_right.to_numpy()
        lower_leg_vector_right = lower_leg_vector_right.to_numpy()
        foot_vector_right = foot_vector_right.to_numpy()

        hip_angle_right = pd.DataFrame(calculate_angle(trunk_vector_right, thigh_vector_right), index = combined_index)
        knee_angle_right = pd.DataFrame(calculate_angle(thigh_vector_right, lower_leg_vector_right), index = combined_index)
        ankle_angle_right = pd.DataFrame(calculate_angle(lower_leg_vector_right, foot_vector_right), index = combined_index)


        # print(f"hip_angle_right = {hip_angle_right}")
        # print(f"knee_angle_right = {knee_angle_right}")
        # print(f"ankle_angle_right = {ankle_angle_right}")

        # print(f"hip_angle_left = {hip_angle_left}")
        # print(f"knee_angle_left = {knee_angle_left}")
        # print(f"ankle_angle_left = {ankle_angle_left}")

        print("3dようの処理が終了しました")
        """





        #すべてで記録できているフレームを抽出
        print(f"sagi_frame_2d = {sagi_frame_2d}")
        # print(f"frontal_frame = {combined_index}")
        # print(f"frontal_frame = {list(combined_index)}")
        print(f"mocap_frame = {mocap_frame}")
        # common_frame = sorted(list(set(sagi_frame_2d) & set(list(combined_index)) & set(mocap_frame)))
        common_frame = sorted(list(set(sagi_frame_2d) & set(mocap_frame)))
        print(f"common_frame = {common_frame}")

        def calculate_angle(vector1, vector2):  #(frame, xyz)または(frame, xy)の配列を入力)
            angle_list = []
            for frame in range(len(vector1)):
                dot_product = np.dot(vector1[frame], vector2[frame])
                cross_product = np.cross(vector1[frame], vector2[frame])
                angle = np.arctan2(cross_product, dot_product)
                angle = angle * 180 / np.pi
                angle_list.append(angle)

            return angle_list

        hip_angle_sagittal_2d = pd.DataFrame(calculate_angle(thigh_vector_l_sagittal_2d, trunk_vector_sagittal_2d))
        knee_angle_sagittal_2d = pd.DataFrame(calculate_angle(thigh_vector_l_sagittal_2d, lower_leg_vector_l_sagittal_2d))
        ankle_angle_sagittal_2d = pd.DataFrame(calculate_angle(foot_vector_l_sagittal_2d, lower_leg_vector_l_sagittal_2d))

        hip_angle_sagittal_2d = 180 - hip_angle_sagittal_2d
        knee_angle_sagittal_2d = 180 - knee_angle_sagittal_2d
        ankle_angle_sagittal_2d =  90 - ankle_angle_sagittal_2d

        hip_angle_sagittal_2d = hip_angle_sagittal_2d.where(hip_angle_sagittal_2d <= 200, hip_angle_sagittal_2d - 360)
        knee_angle_sagittal_2d = knee_angle_sagittal_2d.where(knee_angle_sagittal_2d <= 200, knee_angle_sagittal_2d - 360)
        ankle_angle_sagittal_2d = ankle_angle_sagittal_2d.where(ankle_angle_sagittal_2d < 90, ankle_angle_sagittal_2d - 270)



        # hip_angle_frontal = 180 - hip_angle_frontal
        # knee_angle_frontal = 180 - knee_angle_frontal
        # ankle_angle_frontal = 90 - ankle_angle_frontal



        # hip_angle_left = 180 - hip_angle_left
        # knee_angle_left = 180 - knee_angle_left
        # ankle_angle_left = 90 - ankle_angle_left

        # hip_angle_right = 180 - hip_angle_right
        # knee_angle_right = 180 - knee_angle_right
        # ankle_angle_right = 90 - ankle_angle_right




        hip_angle_mocap = df_mocap_angle["l_hip_angle"].loc[common_frame]
        knee_angle_mocap = df_mocap_angle["l_knee_angle"].loc[common_frame]
        ankle_angle_mocap = df_mocap_angle["l_ankle_angle"].loc[common_frame]



        # 相互相関による同期調整
        def culc_lag(angles1, angles2):
            # print(f"angles1 = {angles1}")
            # print(f"angles2 = {angles2}")
            # print(f"angle1_shape = {angles1.shape}")
            # print(f"angle2_shape = {angles2.shape}")

            # angles2をSeriesyに変換
            if isinstance(angles2, pd.DataFrame):
                angles2 = angles2.iloc[:, 0]  # 最初の列を取り出してSeriesに変換

            # angles1とangles2が一次元配列か確認
            angles1_flat = angles1.values.flatten() if isinstance(angles1, pd.Series) else angles1.flatten()
            angles2_flat = angles2.values.flatten() if isinstance(angles2, pd.Series) else angles2.flatten()

            # 相互相関を計算
            cross_corr = np.correlate(angles1_flat - angles1_flat.mean(), angles2_flat - angles2_flat.mean(), mode='full')

            mid_index = len(cross_corr) // 2
            # 中央のインデックスから歩行周期期間以内の範囲で相互号館を抽出
            cross_corr = cross_corr[mid_index - ic_span_mocap : mid_index + ic_span_mocap + 1]

            print(f"cross_corr.argmax() = {cross_corr.argmax()}")
            # 最大相関を持つラグを計算
            lag = cross_corr.argmax() - ic_span_mocap

            return lag

        # 通常歩行の場合はic_frame_mocap[0]からic_frame_mocap[1]--------------------------------------------------------
        # 低速歩行の場合はic_frame_mocap[0]からic_frame_mocap[2]--------------------------------------------------------
        start = ic_frame_mocap[0]
        end = ic_frame_mocap[1]

        lag_2dhip = culc_lag(hip_angle_mocap.loc[start:end], hip_angle_sagittal_2d.loc[start:end])
        lag_2dknee = culc_lag(knee_angle_mocap.loc[start:end], knee_angle_sagittal_2d.loc[start:end])
        lag_2dankle = culc_lag(ankle_angle_mocap.loc[start:end], ankle_angle_sagittal_2d.loc[start:end])
        print(f"mocapと2dのラグ hip = {lag_2dhip}, knee = {lag_2dknee}, ankle = {lag_2dankle}")

        hip_angle_sagittal_2d_shift = hip_angle_sagittal_2d.shift(lag_2dhip).fillna(method='ffill')
        knee_angle_sagittal_2d_shift = knee_angle_sagittal_2d.shift(lag_2dknee).fillna(method='ffill')
        ankle_angle_sagittal_2d_shift = ankle_angle_sagittal_2d.shift(lag_2dankle).fillna(method='ffill')


        # print(f"ankle_angle_frontal = {ankle_angle_frontal}")
        # ankle_angle_frontal.to_csv(os.path.join(root_dir, f"{condition_num}_ankle_angle_frontal.csv"))
        # ankle_angle_sagittal_2d_filtered.to_csv(os.path.join(root_dir, f"{condition_num}_ankle_angle_sagittal_2d_filtered.csv"))




        #角度結果をプロット color='#2ca02c' 緑     color='#ff7f0e' オレンジ  color='#1f77b4' 青
        if condition_num != "0":
            for ic_frame in ic_frame_mocap:
                plt.axvline(x=ic_frame, color='gray', linestyle='--')
        plt.plot(common_frame, hip_angle_sagittal_2d.loc[common_frame], label="2D sagittal", color='tab:blue', alpha=0.5)
        plt.plot(common_frame, hip_angle_sagittal_2d_shift.loc[common_frame], label="2D sagittal_shift", color='tab:blue')
        # plt.plot(common_frame, hip_angle_frontal.loc[common_frame], label="3D frontal", color='tab:green')
        plt.plot(common_frame, hip_angle_mocap, label="Mocap", color='tab:orange')

        # plt.plot(common_frame, hip_angle_left.loc[common_frame], label="3D left", color='tab:red', alpha=0.5)
        # plt.plot(common_frame, hip_angle_right.loc[common_frame], label="3D right", color='tab:purple', alpha=0.5)

        plt.title("Hip Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition_num}_hip_angle.png"))
        plt.show()  #5frame
        plt.cla()

        if condition_num != "0":
            for ic_frame in ic_frame_mocap:
                plt.axvline(x=ic_frame, color='gray', linestyle='--')
        plt.plot(common_frame, knee_angle_sagittal_2d.loc[common_frame], label="2D sagittal", color='tab:blue', alpha=0.5)
        plt.plot(common_frame, knee_angle_sagittal_2d_shift.loc[common_frame], label="2D sagittal_shift", color='tab:blue')
        # plt.plot(common_frame, knee_angle_frontal.loc[common_frame], label="3D frontal", color='tab:green')
        plt.plot(common_frame, knee_angle_mocap, label="Mocap", color='tab:orange')

        # plt.plot(common_frame, knee_angle_left.loc[common_frame], label="3D left", color='tab:red', alpha=0.5)
        # plt.plot(common_frame, knee_angle_right.loc[common_frame], label="3D right", color='tab:purple', alpha=0.5)

        plt.title("Knee Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition_num}_knee_angle.png"))
        plt.show() #3frame
        plt.cla()

        if condition_num != "0":
            for ic_frame in ic_frame_mocap:
                plt.axvline(x=ic_frame, color='gray', linestyle='--')
        plt.plot(common_frame, ankle_angle_sagittal_2d.loc[common_frame], label="2D sagittal", color='tab:blue', alpha=0.5)
        plt.plot(common_frame, ankle_angle_sagittal_2d_shift.loc[common_frame], label="2D sagittal_shift", color='tab:blue')
        # plt.plot(common_frame, ankle_angle_frontal.loc[common_frame], label="3D frontal", color='tab:green')
        plt.plot(common_frame, ankle_angle_mocap, label="Mocap", color='tab:orange')

        # plt.plot(common_frame, ankle_angle_left.loc[common_frame], label="3D left", color='tab:red', alpha=0.5)
        # plt.plot(common_frame, ankle_angle_right.loc[common_frame], label="3D right", color='tab:purple', alpha=0.5)

        plt.title("Ankle Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition_num}_ankle_angle.png"))
        plt.show() #5frame
        plt.cla()

        # #角度データ保存
        # angle_result_df = pd.concat([hip_angle_sagittal_2d_filtered.loc[common_frame], hip_angle_frontal.loc[common_frame], hip_angle_mocap,
        #                              knee_angle_sagittal_2d_filtered.loc[common_frame], knee_angle_frontal.loc[common_frame], knee_angle_mocap,
        #                              ankle_angle_sagittal_2d_filtered.loc[common_frame], ankle_angle_frontal.iloc[common_frame], ankle_angle_mocap], axis=1)
        # angle_result_df.columns = ["hip_angle_sagittal_2d", "hip_angle_frontal", "hip_angle_mocap", "knee_angle_sagittal_2d", "knee_angle_frontal", "knee_angle_mocap", "ankle_angle_sagittal_2d", "ankle_angle_frontal", "ankle_angle_mocap"]
        # print(f"angle_result_df = {angle_result_df}")
        # angle_result_df.to_csv(os.path.join(root_dir, f"{condition_num}_angle_result.csv"))



        hip_absolute_error = abs(hip_angle_sagittal_2d.loc[start:end].values.flatten() - hip_angle_mocap.loc[start:end].values.flatten())
        # hip_absolute_error = abs(hip_angle_sagittal_2d.loc[common_frame].values.flatten() - hip_angle_mocap.values.flatten())
        mae_hip_sagittal = np.nanmean(hip_absolute_error)
        knee_absolute_error = abs(knee_angle_sagittal_2d.loc[start:end].values.flatten() - knee_angle_mocap.loc[start:end].values.flatten())
        # knee_absolute_error = abs(knee_angle_sagittal_2d.loc[common_frame].values.flatten() - knee_angle_mocap.values.flatten())
        mae_knee_sagittal = np.nanmean(knee_absolute_error)
        ankle_absolute_error = abs(ankle_angle_sagittal_2d.loc[start:end].values.flatten() - ankle_angle_mocap.loc[start:end].values.flatten())
        # ankle_absolute_error = abs(ankle_angle_sagittal_2d.loc[common_frame].values.flatten() - ankle_angle_mocap.values.flatten())
        mae_ankle_sagittal = np.nanmean(ankle_absolute_error)

        print(f"mae_hip_sagittal = {mae_hip_sagittal:.3f}")
        print(f"mae_knee_sagittal = {mae_knee_sagittal:.3f}")
        print(f"mae_ankle_sagittal = {mae_ankle_sagittal:.3f}")

        npz_path = os.path.join(os.path.dirname(mkv_files[0]), f"{os.path.basename(mkv_files[0]).split('.')[0].split('_')[0]}_keypoints&frame.npz")
        # np.savez(npz_path, diagonal_right=keypoints_diagonal_right, diagonal_left=keypoints_diagonal_left, frontal=keypoints_frontal, mocap=keypoints_mocap, common_frame=common_frame, sagittal_3d=keypoints_sagittal_3d, sagittal_2d=keypoints_sagittal_2d)
        # np.savez(npz_path, diagonal_right=keypoints_diagonal_right, diagonal_left=keypoints_diagonal_left, frontal=keypoints_frontal, common_frame=common_frame, sagittal_3d=keypoints_sagittal_3d, sagittal_2d=keypoints_sagittal_2d)

if __name__ == "__main__":
    main()
