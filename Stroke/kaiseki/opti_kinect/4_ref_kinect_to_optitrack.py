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

root_dir = r"F:\Tomson\gait_pattern\20240808"

condition_list = ["0", "1", "2", "3", "4"]  #0がTポーズ,1が通常歩行, 2が通常歩行（遅）, 3が疑似麻痺歩行, 4が疑似麻痺歩行（遅）
condition_keynum = condition_list[1]

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

def get_3d_coordinates(pixel, depth_image, calibration):
    if np.all(pixel == (0, 0)):
        print(f"    Openposeで検出できてない")
        return [0, 0, 0]

    pixel_x, pixel_y = pixel[0], pixel[1]
    x0, x1 = int(math.floor(pixel_x)), int(math.ceil(pixel_x))
    y0, y1 = int(math.floor(pixel_y)), int(math.ceil(pixel_y))

    height, width = depth_image.shape

    if not (0 <= x0 < width and 0 <= x1 < width and 0 <= y0 < height and 0 <= y1 < height):
        print(f"    Coordinates {(x0, y0)} or {(x1, y1)} are out of bounds for image of size (width={width}, height={height})")
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
        print(f"    Error converting to 3D coordinates: {e}")
        # print(f"depth_value_x0_y0 = {depth_value_x0_y0}, depth_value_x1_y0 = {depth_value_x1_y0}, depth_value_x0_y1 = {depth_value_x0_y1}, depth_value_x1_y1 = {depth_value_x1_y1}")
        return [0, 0, 0]

    point_y0 = [linear_interpolation(pixel_x, x0, x1, point_x0_y0[i], point_x1_y0[i]) for i in range(3)]
    point_y1 = [linear_interpolation(pixel_x, x0, x1, point_x0_y1[i], point_x1_y1[i]) for i in range(3)]

    point = [linear_interpolation(pixel_y, y0, y1, point_y0[i], point_y1[i]) for i in range(3)]

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
        keypoints_data = load_keypoints_for_frame(i, json_folder_path)[:, :2]
        # print(f"i = {i} mkv = {mkv_file} keypoints_data = {keypoints_data}")

        # キーポイント抽出が出来ているフレームを記録
        if all(not np.all(np.isnan(keypoints_data[point, :])) for point in check_openpose_list):
            valid_frames.append(i)
        # if all(np.all(keypoints_data[point, :] != 0) for point in check_openpose_list):
        #     valid_frames.append(i)

        #keypoints_dataの0をnanに変換
        keypoints_data[keypoints_data == 0] = np.nan

        transformation_matrix_path = os.path.join(os.path.dirname(mkv_file), f'tf_matrix_calibration_{id}.npz')
        transformation_matrix = np.load(transformation_matrix_path)['a_2d']

        keypoints_data_tf = [np.dot(np.linalg.inv(transformation_matrix), np.array([keypoints_data[keypoint_num][0], keypoints_data[keypoint_num][1], 1]).T)[:2] for keypoint_num in range(len(keypoints_data))]

        all_keypoints_2d.append(keypoints_data)
        all_keypoints_2d_tf.append(keypoints_data_tf)
    keypoints_2d_openpose = np.array(all_keypoints_2d)
    keypoints_2d_openpose_tf = np.array(all_keypoints_2d_tf)

    return keypoints_2d_openpose, keypoints_2d_openpose_tf, valid_frames

def read_3d_openpose(keypoint_array_2d, valid_frame, mkv_file):
    print(f"valid_frame = {valid_frame}")
    keypoints_data = np.nan_to_num(keypoint_array_2d)  #[frame, 2]
    playback = PyK4APlayback(mkv_file)
    playback.open()
    calibration = playback.calibration

    frame_count = 0
    all_keypoints_3d = []  # 各フレームの3Dキーポイントを保持するリスト

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
        depth_image = capture.transformed_depth
        # 深度画像の欠損値を近傍の最小値で補間
        mask = depth_image == 0
        interpolated_depth_image = depth_image.copy()
        shifted_list = []
        kernel_size = 3
        # 行方向と列方向にシフトした画像を作成
        for x_shift in range(-int((kernel_size-1)/2),  int((kernel_size-1)/2 + 1)):
            for y_shift in range(int((kernel_size-1)/2), -int((kernel_size-1)/2 + 1), -1):
                if x_shift == 0 and y_shift == 0:
                    continue
                shifted = np.roll(depth_image, (x_shift, y_shift), axis=(0, 1))
                shifted_list.append(shifted)
        # シフトした画像の中で0以外の最小値を取得
        min_values = np.full(depth_image.shape, np.inf)  # 初期値を無限大に設定
        for shifted in shifted_list:
            non_zero_mask = shifted != 0
            min_values[non_zero_mask] = np.minimum(min_values[non_zero_mask], shifted[non_zero_mask])
        # 無限大が残っている場所はすべて0（欠損値）だった場所なので、0に置き換える
        min_values[min_values == np.inf] = 0
        # 欠損値の位置に最小値を代入
        interpolated_depth_image[mask] = min_values[mask]

        depth_map = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_map = cv2.resize(depth_map, (720, 480))
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        interpolated_depth_map = cv2.normalize(interpolated_depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        interpolated_depth_map = cv2.resize(interpolated_depth_map, (720, 480))
        interpolated_depth_map = cv2.applyColorMap(interpolated_depth_map, cv2.COLORMAP_JET)

        # print(f"depthmapの欠損値の数 = {np.sum(depth_image == 0)}")
        # print(f"interpolated_depth_mapの欠損値の数 = {np.sum(interpolated_depth_image == 0)}")
        # depth_map_hstack = cv2.hconcat([depth_map, interpolated_depth_map])
        # cv2.imshow("depth_map", depth_map_hstack)
        # cv2.waitKey(0)

        frame_keypoints_3d = []

        pixel = keypoints_data[frame_count, :]  #keypoints_data = [frame, 2]
        coordinates_cam = get_3d_coordinates(pixel, interpolated_depth_image, calibration)  #カメラ座標系での3D座標
        if coordinates_cam == [0, 0, 0]:
            frame_keypoints_3d.append([0, 0, 0])
        else:
            A1_inv = np.linalg.inv(a1)
            coordinates_aruco = np.dot(A1_inv, np.array([coordinates_cam[0], coordinates_cam[1], coordinates_cam[2], 1]).T)[:3]  #Arucoマーカー座標系での3D座標
            A2_inv = np.linalg.inv(a2)
            coordinates = np.dot(A2_inv, np.array([coordinates_aruco[0], coordinates_aruco[1], coordinates_aruco[2], 1]).T)[:3]  #Optitrack座標系での3D座標
            frame_keypoints_3d.append(coordinates)


        # all_non_zero = all(np.any(frame_keypoints_3d[id][:2] != 0) for id in check_openpose_list)
        # if all_non_zero:
        #     valid_frame_list.append(frame_count)

        frame_count += 1

    keypoints_openpose = np.array(all_keypoints_3d) / 1000  #単位をmmからmに変換

    # return keypoints_openpose, valid_frame_list
    return keypoints_openpose

def read_3d_optitrack(csv_path):
    df = pd.read_csv(csv_path, skiprows=[0, 1, 2, 4], header=[0, 2])

    df_down = df[::4].reset_index(drop=True)

    marker_set = ["RASI", "LASI","RPSI","LPSI","RKNE","LKNE", "RTHI", "LTHI", "RANK","LANK", "RTIB", "LTIB","RTOE","LTOE","RHEE","LHEE",
                "RSHO", "LSHO","C7", "T10", "CLAV", "STRN", "RBAK", "RKNE2", "LKNE2", "RANK2", "LANK2"]

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
    '''
    keypoints = marker_set_df.values
    keypoints_mocap = keypoints.reshape(-1, len(marker_set), 3)  #xyzで組になるように変形

    return keypoints_mocap, full_range

def calculate_angle(vector1, vector2):  #(frame, xyz)または(frame, xy)の配列を入力
    angle_list = []
    for frame in range(len(vector1)):
        dot_product = np.dot(vector1[frame], vector2[frame])
        cross_product = np.cross(vector1[frame], vector2[frame])
        angle = np.arctan2(np.linalg.norm(cross_product), dot_product)
        angle = angle * 180 / np.pi
        angle_list.append(angle)

    return angle_list

def check_opti_frame(keypoints_mocap):
    check_opti_list = [2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 20, 21]
    valid_frames = []
    for frame in range(keypoints_mocap.shape[0]):
        if all(not np.isnan(keypoints_mocap[frame, point, :]).any() for point in check_opti_list):
            valid_frames.append(frame)
    return valid_frames


def main():
    for condition_num in condition_keynum:
        print(f"condition = {condition_num}")
        # mkv_files = glob.glob(os.path.join(root_dir, f"*{condition_num}*.mkv"))
        mkv_files = glob.glob(os.path.join(root_dir, f"{condition_num}*.mkv"))
        mkv_sagittal = mkv_files[0]
        mkv_diagonal_right = mkv_files[1]
        mkv_diagonal_left = mkv_files[2]

        # csv_files = glob.glob(os.path.join(root_dir, f"Motive/{condition_num}*.csv"))
        # print(f"csv_files = {csv_files}")

        #2d上でのキーポイントを取得"
        keypoints_sagittal_2d, keypoints_sagittal_2d_tf, sagi_frame_2d = read_2d_openpose(mkv_sagittal)  #[frame, 25, 3]
        keypoints_diagonal_right_2d, keypoints_diagonal_right_2d_tf, dia_right_frame_2d = read_2d_openpose(mkv_diagonal_right)
        keypoints_diagonal_left_2d, keypoints_diagonal_left_2d_tf, dia_left_frame_2d = read_2d_openpose(mkv_diagonal_left)

        mid_hip_sagttal_2d = cubic_spline_interpolation(keypoints_sagittal_2d_tf[:, 8, :], sagi_frame_2d)
        neck_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d_tf[:, 1, :], sagi_frame_2d)
        lhip_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d_tf[:, 12, :], sagi_frame_2d)
        lknee_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d_tf[:, 13, :], sagi_frame_2d)
        lankle_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d_tf[:, 14, :], sagi_frame_2d)
        lbigtoe_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d_tf[:, 19, :], sagi_frame_2d)
        lsmalltoe_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d_tf[:, 20, :], sagi_frame_2d)
        lheel_sagittal_2d = cubic_spline_interpolation(keypoints_sagittal_2d_tf[:, 21, :], sagi_frame_2d)

        mid_hip_sagttal_2d_filltered = np.array([butter_lowpass_fillter(mid_hip_sagttal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        neck_sagittal_2d_filltered = np.array([butter_lowpass_fillter(neck_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lhip_sagittal_2d_filltered = np.array([butter_lowpass_fillter(lhip_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lknee_sagittal_2d_filltered = np.array([butter_lowpass_fillter(lknee_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lankle_sagittal_2d_filltered = np.array([butter_lowpass_fillter(lankle_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lbigtoe_sagittal_2d_filltered = np.array([butter_lowpass_fillter(lbigtoe_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lsmalltoe_sagittal_2d_filltered = np.array([butter_lowpass_fillter(lsmalltoe_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lheel_sagittal_2d_filltered = np.array([butter_lowpass_fillter(lheel_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T

        trunk_vector_sagittal_2d = neck_sagittal_2d_filltered - mid_hip_sagttal_2d_filltered
        thigh_vector_l_sagittal_2d = lknee_sagittal_2d_filltered - lhip_sagittal_2d_filltered
        lower_leg_vector_l_sagittal_2d = lankle_sagittal_2d_filltered - lknee_sagittal_2d_filltered
        foot_vector_l_sagittal_2d = lheel_sagittal_2d_filltered - (lbigtoe_sagittal_2d_filltered + lsmalltoe_sagittal_2d_filltered) / 2

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

        #3d上でのキーポイントを取得
        # keypoints_sagittal_3d = read_3d_openpose(keypoints_sagittal_2d, sagi_frame_2d, mkv_sagittal)
        # keypoints_diagonal_right = read_3d_openpose(keypoints_diagonal_right_2d, dia_right_frame_2d, mkv_diagonal_right)
        # keypoints_diagonal_left = read_3d_openpose(keypoints_diagonal_left_2d, dia_left_frame_2d, mkv_diagonal_left)

        mid_hip_diagonal_right_3d = read_3d_openpose(mid_hip_diagonal_right_2d, dia_right_frame_2d, mkv_diagonal_right)
        neck_diagonal_right_3d = read_3d_openpose(neck_diagonal_right_2d, dia_right_frame_2d, mkv_diagonal_right)
        lhip_diagonal_right_3d = read_3d_openpose(lhip_diagonal_right_2d, dia_right_frame_2d, mkv_diagonal_right)
        lknee_diagonal_right_3d = read_3d_openpose(lknee_diagonal_right_2d, dia_right_frame_2d, mkv_diagonal_right)
        lankle_diagonal_right_3d = read_3d_openpose(lankle_diagonal_right_2d, dia_right_frame_2d, mkv_diagonal_right)
        lbigtoe_diagonal_right_3d = read_3d_openpose(lbigtoe_diagonal_right_2d, dia_right_frame_2d, mkv_diagonal_right)
        lsmalltoe_diagonal_right_3d = read_3d_openpose(lsmalltoe_diagonal_right_2d, dia_right_frame_2d, mkv_diagonal_right)
        lheel_diagonal_right_3d = read_3d_openpose(lheel_diagonal_right_2d, dia_right_frame_2d, mkv_diagonal_right)
        check_point_diagonal_right = np.array([mid_hip_diagonal_right_3d, neck_diagonal_right_3d, lhip_diagonal_right_3d, lknee_diagonal_right_3d, lankle_diagonal_right_3d, lbigtoe_diagonal_right_3d, lsmalltoe_diagonal_right_3d, lheel_diagonal_right_3d])

        mid_hip_diagonal_left_3d = read_3d_openpose(mid_hip_diagonal_left_2d, dia_left_frame_2d, mkv_diagonal_left)
        neck_diagonal_left_3d = read_3d_openpose(neck_diagonal_left_2d, dia_left_frame_2d, mkv_diagonal_left)
        lhip_diagonal_left_3d = read_3d_openpose(lhip_diagonal_left_2d, dia_left_frame_2d, mkv_diagonal_left)
        lknee_diagonal_left_3d = read_3d_openpose(lknee_diagonal_left_2d, dia_left_frame_2d, mkv_diagonal_left)
        lankle_diagonal_left_3d = read_3d_openpose(lankle_diagonal_left_2d, dia_left_frame_2d, mkv_diagonal_left)
        lbigtoe_diagonal_left_3d = read_3d_openpose(lbigtoe_diagonal_left_2d, dia_left_frame_2d, mkv_diagonal_left)
        lsmalltoe_diagonal_left_3d = read_3d_openpose(lsmalltoe_diagonal_left_2d, dia_left_frame_2d, mkv_diagonal_left)
        lheel_diagonal_left_3d = read_3d_openpose(lheel_diagonal_left_2d, dia_left_frame_2d, mkv_diagonal_left)
        check_point_diagonal_left = np.array([mid_hip_diagonal_left_3d, neck_diagonal_left_3d, lhip_diagonal_left_3d, lknee_diagonal_left_3d, lankle_diagonal_left_3d, lbigtoe_diagonal_left_3d, lsmalltoe_diagonal_left_3d, lheel_diagonal_left_3d])

        # for frame in range()







        if keypoints_diagonal_right.shape[0] > keypoints_diagonal_left.shape[0]:
            keypoints_diagonal_right = keypoints_diagonal_right[:keypoints_diagonal_left.shape[0]]
        elif keypoints_diagonal_right.shape[0] < keypoints_diagonal_left.shape[0]:
            keypoints_diagonal_left = keypoints_diagonal_left[:keypoints_diagonal_right.shape[0]]

        for frame_count in range(keypoints_diagonal_right.shape[0]):
            if keypoints_diagonal_right[frame_count, :, :] != [0, 0, 0] and keypoints_diagonal_left[frame_count, :, :] != [0, 0, 0]:
                keypoint_frontal = (keypoints_diagonal_right[frame_count, :, :] + keypoints_diagonal_left[frame_count, :, :]) / 2
            elif keypoints_diagonal_right[frame_count, :, :] == [0, 0, 0] and keypoints_diagonal_left[frame_count, :, :] != [0, 0, 0]:
                keypoint_frontal = keypoints_diagonal_left[frame_count, :, :]
            elif keypoints_diagonal_right[frame_count, :, :] != [0, 0, 0] and keypoints_diagonal_left[frame_count, :, :] == [0, 0, 0]:
                keypoint_frontal = keypoints_diagonal_right[frame_count, :, :]
            elif keypoints_diagonal_right[frame_count, :, :] == [0, 0, 0] and keypoints_diagonal_left[frame_count, :, :] == [0, 0, 0]:
                keypoint_frontal = [0, 0, 0]

        # keypoints_frontal = (keypoints_diagonal_right + keypoints_diagonal_left) / 2

        # try:
        #     keypoints_mocap, full_range = read_3d_optitrack(csv_files[0])
        # except IndexError:
        #     print(f"Optitrackのデータがないため読み込み終了")
        #     break

        #すべてで記録できているフレームを抽出
        print(f"sagi_frame_2d = {sagi_frame_2d}")
        # print(f"dia_right_frame_3d = {dia_right_frame_3d}")
        # print(f"dia_left_frame_3d = {dia_left_frame_3d}")
        # mocap_frame = check_opti_frame(keypoints_mocap)
        # common_frame = sorted(list(set(sagi_frame_2d) & set(dia_right_frame_3d) & set(dia_left_frame_3d) & set(mocap_frame)))
        common_frame = sorted(list(set(sagi_frame_2d) & set(dia_right_frame_3d) & set(dia_left_frame_3d)))
        # print(f"sagi_frame_2d = {sagi_frame_2d}")
        # print(f"dia_right_frame_2d = {dia_right_frame_2d}")
        # print(f"dia_left_frame_2d = {dia_left_frame_2d}")
        # print(f"mocap_frame = {mocap_frame}")

        # keypoints_frontal.shape[0] より大きい要素を除外 openpose_3dでは最後のフレームが取得できていないため
        common_frame = [frame for frame in common_frame if frame < keypoints_frontal.shape[0]]
        print(f"common_frame = {common_frame}")

        continue

        trunk_vector_sagittal_2d = keypoints_sagittal_2d[:, 1, :] - keypoints_sagittal_2d[:, 8, :] #neck - mid_hip
        thigh_vector_l_sagittal_2d = keypoints_sagittal_2d[:, 13, :] - keypoints_sagittal_2d[:, 12, :] #knee - hip
        lower_leg_vector_l_sagittal_2d = keypoints_sagittal_2d[:, 14, :] - keypoints_sagittal_2d[:, 13, :] #ankle - knee
        foot_vector_l_sagittal_2d = keypoints_sagittal_2d[:, 21, :]  - (keypoints_sagittal_2d[:, 19, :] + keypoints_sagittal_2d[:, 20, :]) / 2 #heel - (bigtoe + smalltoe) / 2

        trunk_vector_3d_frontal_ori = keypoints_frontal[common_frame, 1, :] - keypoints_frontal[common_frame, 8, :]
        thigh_vector_l_3d_frontal_ori = keypoints_frontal[common_frame, 13, :] - keypoints_frontal[common_frame, 12, :]
        lower_leg_vector_l_3d_frontal_ori = keypoints_frontal[common_frame, 14, :] - keypoints_frontal[common_frame, 13, :]
        foot_vector_l_3d_frontal_ori = keypoints_frontal[common_frame, 21, :]  - (keypoints_frontal[common_frame, 19, :] + keypoints_frontal[common_frame, 20, :]) / 2

        # trunk_vector_mocap_ori = (keypoints_mocap[common_frame, 21, :] + keypoints_mocap[common_frame, 9, :]) / 2 - (keypoints_mocap[common_frame, 20, :] + keypoints_mocap[common_frame, 8, :] + keypoints_mocap[common_frame, 15, :] + keypoints_mocap[common_frame, 4, :]) / 4
        # thigh_vector_l_mocap_ori = (keypoints_mocap[common_frame, 6, :] + keypoints_mocap[common_frame, 7, :]) / 2 - (keypoints_mocap[common_frame, 4, :] + keypoints_mocap[common_frame, 8, :]) / 2
        # lower_vector_l_mocap_ori = (keypoints_mocap[common_frame, 2, :] + keypoints_mocap[common_frame, 3, :]) / 2 - (keypoints_mocap[common_frame, 6, :] + keypoints_mocap[common_frame, 7, :]) / 2
        # foot_vector_l_mocap_ori = keypoints_mocap[common_frame, 5, :] - keypoints_mocap[common_frame, 12, :]

        mid_hip_diagonal_right = np.array([butter_lowpass_fillter(keypoints_diagonal_right[common_frame, 8, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(3)]).T
        neck_diagonal_right = np.array([butter_lowpass_fillter(keypoints_diagonal_right[common_frame, 1, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(3)]).T
        lhip_diagonal_right = np.array([butter_lowpass_fillter(keypoints_diagonal_right[common_frame, 12, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(3)]).T
        lknee_diagonal_right = np.array([butter_lowpass_fillter(keypoints_diagonal_right[common_frame, 13, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(3)]).T
        lankle_diagonal_right = np.array([butter_lowpass_fillter(keypoints_diagonal_right[common_frame, 14, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(3)]).T
        lbigtoe_diagonal_right = np.array([butter_lowpass_fillter(keypoints_diagonal_right[common_frame, 19, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(3)]).T
        lsmalltoe_diagonal_right = np.array([butter_lowpass_fillter(keypoints_diagonal_right[common_frame, 20, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(3)]).T
        lheel_diagonal_right = np.array([butter_lowpass_fillter(keypoints_diagonal_right[common_frame, 21, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(3)]).T

        mid_hip_diagonal_left = np.array([butter_lowpass_fillter(keypoints_diagonal_left[common_frame, 8, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(3)]).T
        neck_diagonal_left = np.array([butter_lowpass_fillter(keypoints_diagonal_left[common_frame, 1, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(3)]).T
        lhip_diagonal_left = np.array([butter_lowpass_fillter(keypoints_diagonal_left[common_frame, 12, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(3)]).T
        lknee_diagonal_left = np.array([butter_lowpass_fillter(keypoints_diagonal_left[common_frame, 13, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(3)]).T
        lankle_diagonal_left = np.array([butter_lowpass_fillter(keypoints_diagonal_left[common_frame, 14, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(3)]).T
        lbigtoe_diagonal_left = np.array([butter_lowpass_fillter(keypoints_diagonal_left[common_frame, 19, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(3)]).T
        lsmalltoe_diagonal_left = np.array([butter_lowpass_fillter(keypoints_diagonal_left[common_frame, 20, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(3)]).T
        lheel_diagonal_left = np.array([butter_lowpass_fillter(keypoints_diagonal_left[common_frame, 21, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(3)]).T

        # rsho = np.array([butter_lowpass_fillter(keypoints_mocap[common_frame, 21, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        # lsho = np.array([butter_lowpass_fillter(keypoints_mocap[common_frame, 9, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        # rpsi = np.array([butter_lowpass_fillter(keypoints_mocap[common_frame, 20, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        # lpsi = np.array([butter_lowpass_fillter(keypoints_mocap[common_frame, 8, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        # rasi = np.array([butter_lowpass_fillter(keypoints_mocap[common_frame, 15, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        # lasi = np.array([butter_lowpass_fillter(keypoints_mocap[common_frame, 4, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        # lknee = np.array([butter_lowpass_fillter(keypoints_mocap[common_frame, 6, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        # lknee2 = np.array([butter_lowpass_fillter(keypoints_mocap[common_frame, 7, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        # lank = np.array([butter_lowpass_fillter(keypoints_mocap[common_frame, 2, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        # lank2 = np.array([butter_lowpass_fillter(keypoints_mocap[common_frame, 3, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        # ltoe = np.array([butter_lowpass_fillter(keypoints_mocap[common_frame, 12, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        # lhee = np.array([butter_lowpass_fillter(keypoints_mocap[common_frame, 5, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T


        trunk_vector_sagittal_2d = neck_sagittal_2d - mid_hip_sagttal_2d
        thigh_vector_l_sagittal_2d = lknee_sagittal_2d - lhip_sagittal_2d
        lower_leg_vector_l_sagittal_2d = lankle_sagittal_2d - lknee_sagittal_2d
        foot_vector_l_sagittal_2d = lheel_sagittal_2d  - (lbigtoe_sagittal_2d + lsmalltoe_sagittal_2d) / 2

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

        # trunk_vector_mocap = (rsho + lsho) / 2 - (rasi + lasi + rpsi + lpsi) / 4
        # thigh_vector_l_mocap = (lknee + lknee2) / 2 - (lasi + lpsi) / 2
        # lower_vector_l_mocap = (lank + lank2) / 2 - (lknee + lknee2) / 2
        # foot_vector_l_mocap = lhee - ltoe

        hip_angle_sagittal_2d = calculate_angle(trunk_vector_sagittal_2d_ori, thigh_vector_l_sagittal_2d_ori)
        knee_angle_sagittal_2d = calculate_angle(thigh_vector_l_sagittal_2d_ori, lower_leg_vector_l_sagittal_2d_ori)
        ankle_angle_sagittal_2d = calculate_angle(lower_leg_vector_l_sagittal_2d_ori, foot_vector_l_sagittal_2d_ori)

        hip_angle_frontal_3d_ori = calculate_angle(trunk_vector_3d_frontal_ori, thigh_vector_l_3d_frontal_ori)
        knee_angle_frontal_3d_ori = calculate_angle(thigh_vector_l_3d_frontal_ori, lower_leg_vector_l_3d_frontal_ori)
        ankle_angle_frontal_3d_ori = calculate_angle(lower_leg_vector_l_3d_frontal_ori, foot_vector_l_3d_frontal_ori)

        # hip_angle_mocap_ori = calculate_angle(trunk_vector_mocap_ori, thigh_vector_l_mocap_ori)
        # knee_angle_mocap_ori = calculate_angle(thigh_vector_l_mocap_ori, lower_vector_l_mocap_ori)
        # ankle_angle_mocap_ori = calculate_angle(lower_vector_l_mocap_ori, foot_vector_l_mocap_ori)

        hip_angle_sagittal_2d = calculate_angle(trunk_vector_sagittal_2d, thigh_vector_l_sagittal_2d)
        knee_angle_sagittal_2d = calculate_angle(thigh_vector_l_sagittal_2d, lower_leg_vector_l_sagittal_2d)
        ankle_angle_sagittal_2d = calculate_angle(lower_leg_vector_l_sagittal_2d, foot_vector_l_sagittal_2d)

        hip_angle_frontal_3d = calculate_angle(trunk_vector_3d_frontal, thigh_vector_l_3d_frontal)
        knee_angle_frontal_3d = calculate_angle(thigh_vector_l_3d_frontal, lower_leg_vector_l_3d_frontal)
        ankle_angle_frontal_3d = calculate_angle(lower_leg_vector_l_3d_frontal, foot_vector_l_3d_frontal)

        # hip_angle_mocap = calculate_angle(trunk_vector_mocap, thigh_vector_l_mocap)
        # knee_angle_mocap = calculate_angle(thigh_vector_l_mocap, lower_vector_l_mocap)
        # ankle_angle_mocap = calculate_angle(lower_vector_l_mocap, foot_vector_l_mocap)

        plt.plot(common_frame, hip_angle_sagittal_2d, label="2D sagittal", color='#1f77b4')
        plt.plot(common_frame, hip_angle_frontal_3d, label="3D frontal", color='#ff7f0e')
        # plt.plot(common_frame, hip_angle_mocap, label="Mocap", color='#2ca02c')
        plt.plot(common_frame, hip_angle_sagittal_2d_ori, label="2D sagittal_ori", color='#1f77b4', alpha=0.5)
        plt.plot(common_frame, hip_angle_frontal_3d_ori, label="3D frontal_ori", color='#ff7f0e', alpha=0.5)
        # plt.plot(common_frame, hip_angle_mocap_ori, label="Mocap_ori", color='#2ca02c', alpha=0.5)
        plt.title("Hip Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition_num}_hip_angle.png"))
        plt.cla()

        plt.plot(common_frame, knee_angle_sagittal_2d, label="2D sagittal", color='#1f77b4')
        plt.plot(common_frame, knee_angle_frontal_3d, label="3D frontal", color='#ff7f0e')
        # plt.plot(common_frame, knee_angle_mocap, label="Mocap", color='#2ca02c')
        plt.plot(common_frame, knee_angle_sagittal_2d_ori, label="2D sagittal_ori", color='#1f77b4', alpha=0.5)
        plt.plot(common_frame, knee_angle_frontal_3d_ori, label="3D frontal_ori", color='#ff7f0e', alpha=0.5)
        # plt.plot(common_frame, knee_angle_mocap_ori, label="Mocap_ori", color='#2ca02c', alpha=0.5)
        plt.title("Knee Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition_num}_knee_angle.png"))
        plt.cla()

        plt.plot(common_frame, ankle_angle_sagittal_2d, label="2D sagittal", color='#1f77b4')
        plt.plot(common_frame, ankle_angle_frontal_3d, label="3D frontal", color='#ff7f0e')
        # plt.plot(common_frame, ankle_angle_mocap, label="Mocap", color='#2ca02c')
        plt.plot(common_frame, ankle_angle_sagittal_2d_ori, label="2D sagittal_ori", color='#1f77b4', alpha=0.5)
        plt.plot(common_frame, ankle_angle_frontal_3d_ori, label="3D frontal_ori", color='#ff7f0e', alpha=0.5)
        # plt.plot(common_frame, ankle_angle_mocap_ori, label="Mocap_ori", color='#2ca02c', alpha=0.5)
        plt.title("Ankle Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition_num}_ankle_angle.png"))
        plt.cla()

        # mae_hip_sagittal = mean_absolute_error(hip_angle_sagittal_2d, hip_angle_mocap)
        # mae_hip_frontal = mean_absolute_error(hip_angle_frontal_3d, hip_angle_mocap)
        # mae_knee_sagittal = mean_absolute_error(knee_angle_sagittal_2d, knee_angle_mocap)
        # mae_knee_frontal = mean_absolute_error(knee_angle_frontal_3d, knee_angle_mocap)
        # mae_ankle_sagittal = mean_absolute_error(ankle_angle_sagittal_2d, ankle_angle_mocap)
        # mae_ankle_frontal = mean_absolute_error(ankle_angle_frontal_3d, ankle_angle_mocap)

        # print(f"mae_hip_sagittal = {mae_hip_sagittal}")
        # print(f"mae_knee_sagittal = {mae_knee_sagittal}")
        # print(f"mae_ankle_sagittal = {mae_ankle_sagittal}")

        # print(f"mae_hip_frontal = {mae_hip_frontal}")
        # print(f"mae_knee_frontal = {mae_knee_frontal}")
        # print(f"mae_ankle_frontal = {mae_ankle_frontal}")

        npz_path = os.path.join(os.path.dirname(mkv_files[0]), f"{os.path.basename(mkv_files[0]).split('.')[0].split('_')[0]}_keypoints&frame.npz")
        # np.savez(npz_path, diagonal_right=keypoints_diagonal_right, diagonal_left=keypoints_diagonal_left, frontal=keypoints_frontal, mocap=keypoints_mocap, common_frame=common_frame, sagittal_3d=keypoints_sagittal_3d, sagittal_2d=keypoints_sagittal_2d)
        np.savez(npz_path, diagonal_right=keypoints_diagonal_right, diagonal_left=keypoints_diagonal_left, frontal=keypoints_frontal, common_frame=common_frame, sagittal_3d=keypoints_sagittal_3d, sagittal_2d=keypoints_sagittal_2d)

if __name__ == "__main__":
    main()
