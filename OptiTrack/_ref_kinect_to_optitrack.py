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


root_dir = r"F:\Tomson\gait_pattern\20240712"
condition_list = ["0_Tpose", "1_walk", "2_walk_slow", "3_comp_walk", "4_comp_walk_slow"]
condition_key = condition_list[:]

def load_keypoints_for_frame(frame_number, json_folder_path):
    json_file_name = f"original_{frame_number:012d}_keypoints.json"
    json_file_path = os.path.join(json_folder_path, json_file_name)

    if not os.path.exists(json_file_path):
        return None

    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
        keypoints_data = np.array(json_data['people'][0]['pose_keypoints_2d']).reshape((25, 3))

    return keypoints_data

def is_keypoint_missing(keypoint):
    return keypoint[2] == 0.0 or np.isnan(keypoint[0]) or np.isnan(keypoint[1])

def interpolate_missing_keypoints(current_frame, previous_frame, next_frame):
    interpolated_frame = current_frame.copy()
    for i, keypoint in enumerate(current_frame):
        if is_keypoint_missing(keypoint):
            if not is_keypoint_missing(previous_frame[i]) and not is_keypoint_missing(next_frame[i]):
                interpolated_frame[i][:2] = (previous_frame[i][:2] + next_frame[i][:2]) / 2
                interpolated_frame[i][2] = (previous_frame[i][2] + next_frame[i][2]) / 2
            elif not is_keypoint_missing(previous_frame[i]):
                interpolated_frame[i] = previous_frame[i]
            elif not is_keypoint_missing(next_frame[i]):
                interpolated_frame[i] = next_frame[i]
    return interpolated_frame

def linear_interpolation(x, x0, x1, y0, y1):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def butter_lowpass_filter(data, order, cutoff_freq):  #4次のバターワースローパスフィルタ
    sampling_freq = 30
    nyquist_freq = sampling_freq / 2
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    # y = data
    return y

def get_3d_coordinates(pixel, depth_image, calibration):
    if np.all(pixel == (0, 0)):
        # print(f"OpenPoseで検出できてないよ")
        return [0, 0, 0]

    pixel_x, pixel_y = pixel[0], pixel[1]
    x0, x1 = int(math.floor(pixel_x)), int(math.ceil(pixel_x))
    y0, y1 = int(math.floor(pixel_y)), int(math.ceil(pixel_y))

    # print(f"近傍のピクセル座標: x0={x0}, x1={x1}, y0={y0}, y1={y1}")

    height, width = depth_image.shape
    # print(f"Depth image shape: width={width}, height={height}")

    if not (0 <= x0 < width and 0 <= x1 < width and 0 <= y0 < height and 0 <= y1 < height):
        print(f"Coordinates {(x0, y0)} or {(x1, y1)} are out of bounds for image of size (width={width}, height={height})")
        return [0, 0, 0]

    depth_value_x0_y0 = depth_image[y0, x0]
    depth_value_x1_y0 = depth_image[y0, x1]
    depth_value_x0_y1 = depth_image[y1, x0]
    depth_value_x1_y1 = depth_image[y1, x1]

    if depth_value_x0_y0 <= 0 or np.isnan(depth_value_x0_y0):
        # print(f"Invalid depth value at ({x0}, {y0}): {depth_value_x0_y0}")
        return [0, 0, 0]
    if depth_value_x1_y0 <= 0 or np.isnan(depth_value_x1_y0):
        # print(f"Invalid depth value at ({x1}, {y0}): {depth_value_x1_y0}")
        return [0, 0, 0]
    if depth_value_x0_y1 <= 0 or np.isnan(depth_value_x0_y1):
        # print(f"Invalid depth value at ({x0}, {y1}): {depth_value_x0_y1}")
        return [0, 0, 0]
    if depth_value_x1_y1 <= 0 or np.isnan(depth_value_x1_y1):
        # print(f"Invalid depth value at ({x1}, {y1}): {depth_value_x1_y1}")
        return [0, 0, 0]

    try:
        point_x0_y0 = calibration.convert_2d_to_3d(coordinates=(x0, y0), depth=depth_value_x0_y0, source_camera=CalibrationType.COLOR)
        point_x1_y0 = calibration.convert_2d_to_3d(coordinates=(x1, y0), depth=depth_value_x1_y0, source_camera=CalibrationType.COLOR)
        point_x0_y1 = calibration.convert_2d_to_3d(coordinates=(x0, y1), depth=depth_value_x0_y1, source_camera=CalibrationType.COLOR)
        point_x1_y1 = calibration.convert_2d_to_3d(coordinates=(x1, y1), depth=depth_value_x1_y1, source_camera=CalibrationType.COLOR)
    except ValueError as e:
        print(f"Error converting to 3D coordinates: {e}")
        return [0, 0, 0]

    point_y0 = [linear_interpolation(pixel_x, x0, x1, point_x0_y0[i], point_x1_y0[i]) for i in range(3)]
    point_y1 = [linear_interpolation(pixel_x, x0, x1, point_x0_y1[i], point_x1_y1[i]) for i in range(3)]

    point = [linear_interpolation(pixel_y, y0, y1, point_y0[i], point_y1[i]) for i in range(3)]

    return point

def read_2d_openpose(mkv_file):
    json_foloder_path = os.path.join(os.path.dirname(mkv_file), os.path.basename(mkv_file).split('.')[0], 'estimated.json')
    all_keypoints_2d = []  # 各フレームの2Dキーポイントを保持するリスト
    for i, json_file in enumerate(glob.glob(os.path.join(json_foloder_path, "*.json"))):
        keypoints_data = load_keypoints_for_frame(i, json_foloder_path)[:, :2]
        keypoints_data = [np.zeros(2) if np.all(data == 0) else data for data in keypoints_data]

        # print(f"変換前kyepoints_data = {keypoints_data}\n")

        transformation_matrix_path = os.path.join(os.path.dirname(mkv_file), f'transformation_matrix_0.npz')
        transformation_matrix = np.load(transformation_matrix_path)['a_2d']

        keypoints_data = [np.dot(np.linalg.inv(transformation_matrix), np.array([keypoints_data[keypoint_num][0], keypoints_data[keypoint_num][1], 1]).T)[:2] for keypoint_num in range(len(keypoints_data))]

        # print(f"変換後kyepoints_data = {keypoints_data}\n")
        all_keypoints_2d.append(keypoints_data)
    keypoints_2d_openpose = np.array(all_keypoints_2d)
    return keypoints_2d_openpose

def read_3d_openpose(mkv_file):
    # print(f"mkv_file = {mkv_file}")
    # MKVファイルの再生
    playback = PyK4APlayback(mkv_file)
    playback.open()
    calibration = playback.calibration

    frame_count = 0
    json_foloder_path = os.path.join(os.path.dirname(mkv_file), os.path.basename(mkv_file).split('.')[0], 'estimated.json')
    all_keypoints_3d = []  # 各フレームの3Dキーポイントを保持するリスト

    #モーキャプ座標系に変換するための座標変換行列の読み込み
    transform_matrix_path = os.path.join(os.path.dirname(mkv_file), f'transformation_matrix_{os.path.basename(mkv_file).split(".")[0].split("_")[-1]}.npz')
    transform_matrix = np.load(transform_matrix_path)
    a1 = transform_matrix['a1']  #Arucoマーカーまでの変換行列
    a2 = transform_matrix['a2']  #Optitrackまでの変換行列(平行移動のみ)

    while True:
        try:
            capture = playback.get_next_capture()
        except:
            print("再生を終了します")
            break

        if capture.color is None or capture.transformed_depth is None:
            print(f"Frame {frame_count} has no image data.")
            continue

        print(f"frame_count = {frame_count} mkvfile = {mkv_file} ")

        # 画像を取得
        color_image = capture.color
        depth_image = capture.transformed_depth

        depth_images_folder = os.path.join(os.path.dirname(mkv_file), os.path.basename(mkv_file).split('.')[0], 'depth_images')
        if not os.path.exists(depth_images_folder):
            os.makedirs(depth_images_folder)

        depth_image_path = os.path.join(depth_images_folder, f"depth_{frame_count:012d}.png")
        cv2.imwrite(depth_image_path, depth_image)

        # #depth画像をRGN画像に変換
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # #指定した座標に白い点を描画
        # cv2.circle(depth_colormap, (795, 265), 10, (256, 256, 256), thickness=-1)
        # cv2.imshow("Depth Image", depth_colormap)
        # cv2.imshow("depth_image", depth_image)
        # cv2.imshow("depth_image_filled", depth_image)
        # cv2.waitKey(0)

        # 対応するフレームのJSONファイルを読み込む
        keypoints_data = load_keypoints_for_frame(frame_count, json_foloder_path)
        if keypoints_data is None:
            print(f"Frame {frame_count}: JSON file not found, exiting loop.")
            break

        if any(is_keypoint_missing(keypoint) for keypoint in keypoints_data):
            # 前後のフレームのキーポイントを読み込む
            previous_keypoints_data = load_keypoints_for_frame(frame_count - 1, json_foloder_path) if frame_count > 0 else keypoints_data
            if previous_keypoints_data is None:
                print(f"Previous frame {frame_count - 1}: JSON file not found, exiting loop.")
                break

            try:
                next_keypoints_data = load_keypoints_for_frame(frame_count + 1, json_foloder_path)
                if next_keypoints_data is None:
                    print(f"Next frame {frame_count + 1}: JSON file not found, exiting loop.")
                    break
            except IndexError:
                print(f"Next frame {frame_count + 1}: people not found, exiting loop.")
                break

            # キーポイントの補完を行う
            keypoints_data = interpolate_missing_keypoints(keypoints_data, previous_keypoints_data, next_keypoints_data)

        frame_keypoints_3d = []

        for i, keypoint in enumerate(keypoints_data):
            pixel = np.array(keypoint[:2])
            # print(f"{i}番目のキーポイント 位置 {pixel}")
            coordinates_cam = get_3d_coordinates(pixel, depth_image, calibration)  #カメラ座標系での3D座標
            A1_inv = np.linalg.inv(a1)
            coordinates_aruco = np.dot(A1_inv, np.array([coordinates_cam[0], coordinates_cam[1], coordinates_cam[2], 1]).T)[:3]  #Arucoマーカー座標系での3D座標
            A2_inv = np.linalg.inv(a2)
            coordinates = np.dot(A2_inv, np.array([coordinates_aruco[0], coordinates_aruco[1], coordinates_aruco[2], 1]).T)[:3]  #Optitrack座標系での3D座標
            frame_keypoints_3d.append(coordinates)

        all_keypoints_3d.append(frame_keypoints_3d)

        frame_count += 1

    keypoints_openpose = np.array(all_keypoints_3d)/1000  #単位をmmからmに変換

    # cv2.destroyAllWindows()

    return keypoints_openpose

def read_3d_optitrack(csv_path):
    df = pd.read_csv(csv_path, skiprows= [0,1,2,4], header=[0,2])

    # 4行おきにデータを抽出(ダウンサンプリング)
    df_down = df[::4].reset_index(drop=True)

    #必要なマーカーのみを抽出
    marker_set = ["RASI", "LASI","RPSI","LPSI","RKNE","LKNE", "RTHI", "LTHI", "RANK","LANK", "RTIB", "LTIB","RTOE","LTOE","RHEE","LHEE",
                "RSHO", "LSHO","C7", "T10", "CLAV", "STRN", "RBAK", "RKNE2", "LKNE2", "RANK2", "LANK2"]

    #マーカーセットのみを抽出
    marker_set_df = df_down[[col for col in df_down.columns if any(marker in col[0] for marker in marker_set)]].copy()
    print(f"marker_set_df.shape = {marker_set_df.shape}")
    # marker_set_df.to_csv(os.path.join(os.path.dirname(csv_path), f"marker_set_{os.path.basename(csv_path)}"))
    success_frame_list = []

    #すべてのマーカーが検出できているフレームのみを抽出
    for frame in range(0, len(marker_set_df)):
        if not marker_set_df.iloc[frame].isna().any():
            success_frame_list.append(frame)

    full_range = range(min(success_frame_list), max(success_frame_list)+1)
    success_df = marker_set_df.reindex(full_range)
    interpolate_success_df = success_df.interpolate(method='spline', order = 3) #rangeでとっているため間に欠損値がある場合に対して3次スプライン補間
    print(f"interpolate_success_df.shape = {interpolate_success_df.shape}")
    # full_rangeの各列に対してinterpolate_success_dfの対応する列を設定
    for i, index in enumerate(full_range):
        marker_set_df.loc[index, :] = interpolate_success_df.iloc[i, :]
    marker_set_df.to_csv(os.path.join(os.path.dirname(csv_path), f"marker_set_{os.path.basename(csv_path)}"))
    print(f"marker_set_df.shape = {marker_set_df.shape}")

    # success_df.to_csv(os.path.join(os.path.dirname(csv_path), f"success_{os.path.basename(csv_path)}"))
    # interpolate_success_df.to_csv(os.path.join(os.path.dirname(csv_path), f"interpolated_{os.path.basename(csv_path)}"))

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

def main():

    for condition in condition_key:
        print(f"condition = {condition}")
        #OpnePoseは処理してjsonが出力されている前提, calibrationも終わって座標変換行列つくってる前提
        mkv_files = glob.glob(os.path.join(root_dir, f"*{condition}*.mkv"))  #mkvファイルのパスを取得
        mkv_sagittal = mkv_files[0]  #進行方向から見て左
        mkv_diagonal_right = mkv_files[1]  #進行方向から見て右斜め前
        mkv_diagonal_left = mkv_files[2]  #進行方向から見て左斜め前

        csv_files = glob.glob(os.path.join(root_dir, f"Motive/{condition}*.csv"))  #csvファイルのパスを取得

        #openpose+深度で各キーポイントの3d座標を取得
        keypoints_sagittal_2d = read_2d_openpose(mkv_sagittal)  #(frame, keypoint, xy) (300,25,2)
        keypoints_sagittal_3d = read_3d_openpose(mkv_sagittal)  #(frame, keypoint, xyz) (300,25,3)
        # print(f"keypoints_sagittal_2d = {keypoints_sagittal_2d}")
        # print(f"keypoints_sagittal_2d.shape = {keypoints_sagittal_2d.shape}")

        keypoints_diagonal_right = read_3d_openpose(mkv_diagonal_right) # (frame, keypoint, xyz) (300,25,3)
        keypoints_diagonal_left = read_3d_openpose(mkv_diagonal_left)

        #フレーム数を合わせる(本当はあっているはずだが一応)
        if keypoints_diagonal_right.shape[0] > keypoints_diagonal_left.shape[0]:
            keypoints_diagonal_right = keypoints_diagonal_right[:keypoints_diagonal_left.shape[0]]
        elif keypoints_diagonal_right.shape[0] < keypoints_diagonal_left.shape[0]:
            keypoints_diagonal_left = keypoints_diagonal_left[:keypoints_diagonal_right.shape[0]]

        keypoints_frontal = (keypoints_diagonal_right + keypoints_diagonal_left) / 2

        #optitrackで各キーポイントの3d座標を取得
        print(f"csv_files = {csv_files}")
        try:
            keypoints_mocap, full_range = read_3d_optitrack(csv_files[0])  #(362,27,3)
        except IndexError:
            print(f"Optitrackのデータがないため読み込み終了")
            break


        cam_stop_frame = min(keypoints_sagittal_3d.shape[0], keypoints_diagonal_right.shape[0], keypoints_diagonal_left.shape[0], keypoints_mocap.shape[0])
        # print(f"full_range = {full_range}")
        # print(f"cam_stop_frame = {cam_stop_frame}")
        frame_range = range(full_range.start, min(full_range.stop, cam_stop_frame)) #mocapが上手く取れているフレーム範囲とopenposeのフレーム範囲の共通部分を取得


        #角度を比較する(今回は左足で比較) 体幹と膝、足首のベクトルを使って角度を計算
        #フィルター前
        trunk_vector_sagittal_2d_ori = keypoints_sagittal_2d[:, 1, :] - keypoints_sagittal_2d[:, 8, :] #MidHipaからNeck
        thigh_vector_l_sagittal_2d_ori = keypoints_sagittal_2d[:, 10, :] - keypoints_sagittal_2d[:, 9, :] #LhipからLKnee
        lower_leg_vector_l_sagittal_2d_ori = keypoints_sagittal_2d[:, 11, :] - keypoints_sagittal_2d[:, 10, :] #LKneeからLAnkle
        foot_vector_l_sagittal_2d_ori = keypoints_sagittal_2d[:, 24, :]  - (keypoints_sagittal_2d[:, 22, :] + keypoints_sagittal_2d[:, 23, :]) / 2 #LBigToeとLSmallToeの中点からLHeel

        # trunk_vector_sagittal_3d_ori = keypoints_sagittal_3d[:, 1, :] - keypoints_sagittal_3d[:, 8, :]
        # thigh_vector_l_sagittal_3d_ori = keypoints_sagittal_3d[:, 10, :] - keypoints_sagittal_3d[:, 9, :]
        # lower_leg_vector_l_sagittal_3d_ori = keypoints_sagittal_3d[:, 11, :] - keypoints_sagittal_3d[:, 10, :]
        # foot_vector_l_sagittal_3d_ori = keypoints_sagittal_3d[:, 24, :]  - (keypoints_sagittal_3d[:, 22, :] + keypoints_sagittal_3d[:, 23, :]) / 2

        # trunk_vector_3d_diagonal_right_ori = keypoints_diagonal_right[:, 1, :] - keypoints_diagonal_right[:, 8, :]
        # thigh_vector_3d_diagonal_right_ori = keypoints_diagonal_right[:, 10, :] - keypoints_diagonal_right[:, 9, :]
        # lower_leg_vector_3d_diagonal_right_ori = keypoints_diagonal_right[:, 11, :] - keypoints_diagonal_right[:, 10, :]
        # foot_vector_3d_diagonal_right_ori = keypoints_diagonal_right[:, 24, :]  - (keypoints_diagonal_right[:, 22, :] + keypoints_diagonal_right[:, 23, :]) / 2

        # trunk_vector_3d_diagonal_left_ori = keypoints_diagonal_left[:, 1, :] - keypoints_diagonal_left[:, 8, :]
        # thigh_vector_3d_diagonal_left_ori = keypoints_diagonal_left[:, 10, :] - keypoints_diagonal_left[:, 9, :]
        # lower_leg_vector_3d_diagonal_left_ori = keypoints_diagonal_left[:, 11, :] - keypoints_diagonal_left[:, 10, :]
        # foot_vector_3d_diagonal_left_ori = keypoints_diagonal_left[:, 24, :]  - (keypoints_diagonal_left[:, 22, :] + keypoints_diagonal_left[:, 23, :]) / 2

        trunk_vector_3d_frontal_ori = keypoints_frontal[:, 1, :] - keypoints_frontal[:, 8, :]
        thigh_vector_l_3d_frontal_ori = keypoints_frontal[:, 10, :] - keypoints_frontal[:, 9, :]
        lower_leg_vector_l_3d_frontal_ori = keypoints_frontal[:, 11, :] - keypoints_frontal[:, 10, :]
        foot_vector_l_3d_frontal_ori = keypoints_frontal[:, 24, :]  - (keypoints_frontal[:, 22, :] + keypoints_frontal[:, 23, :]) / 2

        trunk_vector_mocap_ori = (keypoints_mocap[:, 21, :] + keypoints_mocap[:, 9, :]) / 2 - (keypoints_mocap[:, 20, :] + keypoints_mocap[:, 8, :] + keypoints_mocap[:, 15, :] + keypoints_mocap[:, 4, :]) / 4 #RASIとLASIとRPSIとLPSIの中点→RSHOとLSHOの中点
        thigh_vector_l_mocap_ori = (keypoints_mocap[:, 6, :] + keypoints_mocap[:, 7, :]) / 2 - (keypoints_mocap[:, 4, :] + keypoints_mocap[:, 8, :]) / 2 #LASIとLPSIの中点→LKNEとLKNE2の中点
        lower_vector_l_mocap_ori = (keypoints_mocap[:, 2, :] + keypoints_mocap[:, 3, :]) / 2 - (keypoints_mocap[:, 6, :] + keypoints_mocap[:, 7, :]) / 2 #LKNEとLKNE2の中点→LANKとLANK2の中点
        foot_vector_l_mocap_ori = keypoints_mocap[:, 13, :] - keypoints_mocap[:, 12, :] #LTOE→LHEE



        #フィルター後
        mid_hip_sagttal_2d = np.array([butter_lowpass_filter(keypoints_sagittal_2d[:, 8, x], order = 4, cutoff_freq = 6) for x in range(2)]).T
        neck_sagittal_2d = np.array([butter_lowpass_filter(keypoints_sagittal_2d[:, 1, x], order = 4, cutoff_freq = 6) for x in range(2)]).T
        lhip_sagittal_2d = np.array([butter_lowpass_filter(keypoints_sagittal_2d[:, 9, x], order = 4, cutoff_freq = 6) for x in range(2)]).T
        lknee_sagittal_2d = np.array([butter_lowpass_filter(keypoints_sagittal_2d[:, 10, x], order = 4, cutoff_freq = 6) for x in range(2)]).T
        lankle_sagittal_2d = np.array([butter_lowpass_filter(keypoints_sagittal_2d[:, 11, x], order = 4, cutoff_freq = 6) for x in range(2)]).T
        lbigtoe_sagittal_2d = np.array([butter_lowpass_filter(keypoints_sagittal_2d[:, 22, x], order = 4, cutoff_freq = 6) for x in range(2)]).T
        lsmalltoe_sagittal_2d = np.array([butter_lowpass_filter(keypoints_sagittal_2d[:, 23, x], order = 4, cutoff_freq = 6) for x in range(2)]).T
        lheel_sagittal_2d = np.array([butter_lowpass_filter(keypoints_sagittal_2d[:, 24, x], order = 4, cutoff_freq = 6) for x in range(2)]).T

        # mid_hip_sagittal_3d = np.array([butter_lowpass_filter(keypoints_sagittal_3d[:, 8, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        # neck_sagittal_3d = np.array([butter_lowpass_filter(keypoints_sagittal_3d[:, 1, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        # lhip_sagittal_3d = np.array([butter_lowpass_filter(keypoints_sagittal_3d[:, 9, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        # lknee_sagittal_3d = np.array([butter_lowpass_filter(keypoints_sagittal_3d[:, 10, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        # lankle_sagittal_3d = np.array([butter_lowpass_filter(keypoints_sagittal_3d[:, 11, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        # lbigtoe_sagittal_3d = np.array([butter_lowpass_filter(keypoints_sagittal_3d[:, 22, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        # lsmalltoe_sagittal_3d = np.array([butter_lowpass_filter(keypoints_sagittal_3d[:, 23, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        # lheel_sagittal_3d = np.array([butter_lowpass_filter(keypoints_sagittal_3d[:, 24, x], order = 4, cutoff_freq = 6) for x in range(3)]).T

        mid_hip_diagonal_right = np.array([butter_lowpass_filter(keypoints_diagonal_right[:, 8, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        neck_diagonal_right = np.array([butter_lowpass_filter(keypoints_diagonal_right[:, 1, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lhip_diagonal_right = np.array([butter_lowpass_filter(keypoints_diagonal_right[:, 9, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lknee_diagonal_right = np.array([butter_lowpass_filter(keypoints_diagonal_right[:, 10, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lankle_diagonal_right = np.array([butter_lowpass_filter(keypoints_diagonal_right[:, 11, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lbigtoe_diagonal_right = np.array([butter_lowpass_filter(keypoints_diagonal_right[:, 22, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lsmalltoe_diagonal_right = np.array([butter_lowpass_filter(keypoints_diagonal_right[:, 23, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lheel_diagonal_right = np.array([butter_lowpass_filter(keypoints_diagonal_right[:, 24, x], order = 4, cutoff_freq = 6) for x in range(3)]).T

        mid_hip_diagonal_left = np.array([butter_lowpass_filter(keypoints_diagonal_left[:, 8, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        neck_diagonal_left = np.array([butter_lowpass_filter(keypoints_diagonal_left[:, 1, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lhip_diagonal_left = np.array([butter_lowpass_filter(keypoints_diagonal_left[:, 9, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lknee_diagonal_left = np.array([butter_lowpass_filter(keypoints_diagonal_left[:, 10, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lankle_diagonal_left = np.array([butter_lowpass_filter(keypoints_diagonal_left[:, 11, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lbigtoe_diagonal_left = np.array([butter_lowpass_filter(keypoints_diagonal_left[:, 22, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lsmalltoe_diagonal_left = np.array([butter_lowpass_filter(keypoints_diagonal_left[:, 23, x], order = 4, cutoff_freq = 6) for x in range(3)]).T
        lheel_diagonal_left = np.array([butter_lowpass_filter(keypoints_diagonal_left[:, 24, x], order = 4, cutoff_freq = 6) for x in range(3)]).T

        clav = np.array([butter_lowpass_filter(keypoints_mocap[:, 1, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        rsho = np.array([butter_lowpass_filter(keypoints_mocap[:, 21, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        lsho = np.array([butter_lowpass_filter(keypoints_mocap[:, 9, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        rpsi = np.array([butter_lowpass_filter(keypoints_mocap[:, 20, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        lpsi = np.array([butter_lowpass_filter(keypoints_mocap[:, 8, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        rasi = np.array([butter_lowpass_filter(keypoints_mocap[:, 15, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        lasi = np.array([butter_lowpass_filter(keypoints_mocap[:, 4, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        lknee = np.array([butter_lowpass_filter(keypoints_mocap[:, 6, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        lknee2 = np.array([butter_lowpass_filter(keypoints_mocap[:, 7, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        lank = np.array([butter_lowpass_filter(keypoints_mocap[:, 2, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        lank2 = np.array([butter_lowpass_filter(keypoints_mocap[:, 3, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        ltoe = np.array([butter_lowpass_filter(keypoints_mocap[:, 12, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T
        lhee = np.array([butter_lowpass_filter(keypoints_mocap[:, 13, x], order = 4 ,cutoff_freq = 6) for x in range(3)]).T


        trunk_vector_sagittal_2d = neck_sagittal_2d - mid_hip_sagttal_2d #MidHipaからNeck
        thigh_vector_l_sagittal_2d = lknee_sagittal_2d - lhip_sagittal_2d #LhipからLKnee
        lower_leg_vector_l_sagittal_2d = lankle_sagittal_2d - lknee_sagittal_2d #LKneeからLAnkle
        foot_vector_l_sagittal_2d = lheel_sagittal_2d  - (lbigtoe_sagittal_2d + lsmalltoe_sagittal_2d) / 2 #LBigToeとLSmallToeの中点からLHeel

        # # 配列の比較
        # if np.array_equal(trunk_vector_sagittal_2d, trunk_vector_sagittal_2d_ori):
        #     print(f"体幹ベクトルは同じ")
        # else:
        #     print(f"体幹ベクトルは違う")

        # if np.array_equal(thigh_vector_l_sagittal_2d, thigh_vector_l_sagittal_2d_ori):
        #     print(f"大腿ベクトルは同じ")
        # else:
        #     print(f"大腿ベクトルは違う")

        # if np.array_equal(lower_leg_vector_l_sagittal_2d, lower_leg_vector_l_sagittal_2d_ori):
        #     print(f"下脚ベクトルは同じ")
        # else:
        #     print(f"下脚ベクトルは違う")

        # if np.array_equal(foot_vector_l_sagittal_2d, foot_vector_l_sagittal_2d_ori):
        #     print(f"足ベクトルは同じ")
        # else:
        #     print(f"足ベクトルは違う")


        # trunk_vector_sagittal_3d = neck_sagittal_3d - mid_hip_sagittal_3d
        # thigh_vector_l_sagittal_3d = lhip_sagittal_3d - lknee_sagittal_3d
        # lower_leg_vector_l_sagittal_3d = lankle_sagittal_3d - lknee_sagittal_3d
        # foot_vector_l_sagittal_3d = lheel_sagittal_3d  - (lbigtoe_sagittal_3d + lsmalltoe_sagittal_3d) / 2

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





        # print(f"trunk_vector_sagittal_2d.shape = {trunk_vector_sagittal_2d.shape}")
        # print(f"trunk_vector_3d_frontal.shape = {trunk_vector_3d_frontal.shape}")
        # print(f"trunk_vector_mocap.shape = {trunk_vector_mocap.shape}")



        #フィルター前
        hip_angle_sagittal_2d_ori = calculate_angle(trunk_vector_sagittal_2d_ori, thigh_vector_l_sagittal_2d_ori)
        knee_angle_sagittal_2d_ori = calculate_angle(thigh_vector_l_sagittal_2d_ori, lower_leg_vector_l_sagittal_2d_ori)
        ankle_angle_sagittal_2d_ori = calculate_angle(lower_leg_vector_l_sagittal_2d_ori, foot_vector_l_sagittal_2d_ori)

        # hip_angle_sagittal_3d_ori = calculate_angle(trunk_vector_sagittal_3d_ori, thigh_vector_l_sagittal_3d_ori)
        # knee_angle_sagittal_3d_ori = calculate_angle(thigh_vector_l_sagittal_3d_ori, lower_leg_vector_l_sagittal_3d_ori)
        # ankle_angle_sagittal_3d_ori = calculate_angle(lower_leg_vector_l_sagittal_3d_ori, foot_vector_l_sagittal_3d_ori)

        # hip_angle_diagonal_right_3d_ori = calculate_angle(trunk_vector_3d_diagonal_right_ori, thigh_vector_3d_diagonal_right_ori)
        # knee_angle_diagonal_right_3d_ori = calculate_angle(thigh_vector_3d_diagonal_right_ori, lower_leg_vector_3d_diagonal_right_ori)
        # ankle_angle_diagonal_right_3d_ori = calculate_angle(lower_leg_vector_3d_diagonal_right_ori, foot_vector_3d_diagonal_right_ori)

        # hip_angle_diagonal_left_3d_ori = calculate_angle(trunk_vector_3d_diagonal_left_ori, thigh_vector_3d_diagonal_left_ori)
        # knee_angle_diagonal_left_3d_ori = calculate_angle(thigh_vector_3d_diagonal_left_ori, lower_leg_vector_3d_diagonal_left_ori)
        # ankle_angle_diagonal_left_3d_ori = calculate_angle(lower_leg_vector_3d_diagonal_left_ori, foot_vector_3d_diagonal_left_ori)

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

        # hip_angle_sagittal_3d = calculate_angle(trunk_vector_sagittal_3d, thigh_vector_l_sagittal_3d)
        # knee_angle_sagittal_3d = calculate_angle(thigh_vector_l_sagittal_3d, lower_leg_vector_l_sagittal_3d)
        # ankle_angle_sagittal_3d = calculate_angle(lower_leg_vector_l_sagittal_3d, foot_vector_l_sagittal_3d)

        # hip_angle_diagonal_right_3d = calculate_angle(trunk_vector_3d_diagonal_right, thigh_vector_3d_diagonal_right)
        # knee_angle_diagonal_right_3d = calculate_angle(thigh_vector_3d_diagonal_right, lower_leg_vector_3d_diagonal_right)
        # ankle_angle_diagonal_right_3d = calculate_angle(lower_leg_vector_3d_diagonal_right, foot_vector_3d_diagonal_right)

        # hip_angle_diagonal_left_3d = calculate_angle(trunk_vector_3d_diagonal_left, thigh_vector_3d_diagonal_left)
        # knee_angle_diagonal_left_3d = calculate_angle(thigh_vector_3d_diagonal_left, lower_leg_vector_3d_diagonal_left)
        # ankle_angle_diagonal_left_3d = calculate_angle(lower_leg_vector_3d_diagonal_left, foot_vector_3d_diagonal_left)

        hip_angle_frontal_3d = calculate_angle(trunk_vector_3d_frontal, thigh_vector_l_3d_frontal)
        knee_angle_frontal_3d = calculate_angle(thigh_vector_l_3d_frontal, lower_leg_vector_l_3d_frontal)
        ankle_angle_frontal_3d = calculate_angle(lower_leg_vector_l_3d_frontal, foot_vector_l_3d_frontal)

        hip_angle_mocap = calculate_angle(trunk_vector_mocap, thigh_vector_l_mocap)
        knee_angle_mocap = calculate_angle(thigh_vector_l_mocap, lower_vector_l_mocap)
        ankle_angle_mocap = calculate_angle(lower_vector_l_mocap, foot_vector_l_mocap)

        if np.any(np.isnan(hip_angle_frontal_3d)):
            print(f"hip_angle_frontal_3dにnanが含まれている")

        if np.any(np.isnan(hip_angle_mocap)):
            print(f"hip_angle_mocapにnanが含まれている")

        if np.any(np.isnan(hip_angle_sagittal_2d)):
            print(f"hip_angle_sagittal_2dにnanが含まれている")

        if np.any(np.isnan(knee_angle_frontal_3d)):
            print(f"knee_angle_frontal_3dにnanが含まれている")

        if np.any(np.isnan(knee_angle_mocap)):
            print(f"knee_angle_mocapにnanが含まれている")

        if np.any(np.isnan(knee_angle_sagittal_2d)):
            print(f"knee_angle_sagittal_2dにnanが含まれている")

        if np.any(np.isnan(ankle_angle_frontal_3d)):
            print(f"ankle_angle_frontal_3dにnanが含まれている")

        if np.any(np.isnan(ankle_angle_mocap)):
            print(f"ankle_angle_mocapにnanが含まれている")

        if np.any(np.isnan(ankle_angle_sagittal_2d)):
            print(f"ankle_angle_sagittal_2dにnanが含まれている")


        plt.plot(frame_range, [hip_angle_sagittal_2d[i] for i in frame_range], label="2D sagittal", color='#1f77b4')
        # plt.plot(frame_range, [hip_angle_sagittal_3d[i] for i in frame_range], label="3D sagittal", color='#d62728')
        # plt.plot(frame_range, [hip_angle_diagonal_right_3d[i] for i in frame_range], label="3D diagonal right, color='#9467bd')
        # plt.plot(frame_range, [hip_angle_diagonal_left_3d[i] for i in frame_range], label="3D diagonal left", color='#8c564b')
        plt.plot(frame_range, [hip_angle_frontal_3d[i] for i in frame_range], label="3D frontal", color='#ff7f0e')
        plt.plot(frame_range, [hip_angle_mocap[i] for i in frame_range], label="Mocap", color='#2ca02c')

        plt.plot(frame_range, [hip_angle_sagittal_2d_ori[i] for i in frame_range], label="2D sagittal_ori", color='#1f77b4', alpha=0.5)
        # plt.plot(frame_range, [hip_angle_sagittal_3d_ori[i] for i in frame_range], label="3D sagittal_ori", color='#d62728', alpha=0.5)
        # plt.plot(frame_range, [hip_angle_diagonal_right_3d_ori[i] for i in frame_range], label="3D diagonal right_ori", color='#9467bd', alpha=0.5)
        # plt.plot(frame_range, [hip_angle_diagonal_left_3d_ori[i] for i in frame_range], label="3D diagonal left_ori", color='#8c564b', alpha=0.5)
        plt.plot(frame_range, [hip_angle_frontal_3d_ori[i] for i in frame_range], label="3D frontal_ori", color='#ff7f0e', alpha=0.5)
        plt.plot(frame_range, [hip_angle_mocap_ori[i] for i in frame_range], label="Mocap_ori", color='#2ca02c', alpha=0.5)

        plt.title("Hip Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition}_hip_angle.png"))
        # plt.show()
        plt.cla()

        plt.plot(frame_range, [knee_angle_sagittal_2d[i] for i in frame_range], label="2D sagittal", color='#1f77b4')
        # plt.plot(frame_range, [knee_angle_sagittal_3d[i] for i in frame_range], label="3D sagittal", color='#d62728')
        # plt.plot(frame_range, [knee_angle_diagonal_right_3d[i] for i in frame_range], label="3D diagonal right, color='#9467bd')
        # plt.plot(frame_range, [knee_angle_diagonal_left_3d[i] for i in frame_range], label="3D diagonal left", color='#8c564b')
        plt.plot(frame_range, [knee_angle_frontal_3d[i] for i in frame_range], label="3D frontal", color='#ff7f0e')
        plt.plot(frame_range, [knee_angle_mocap[i] for i in frame_range], label="Mocap", color='#2ca02c')

        plt.plot(frame_range, [knee_angle_sagittal_2d_ori[i] for i in frame_range], label="2D sagittal_ori", color='#1f77b4', alpha=0.5)
        # plt.plot(frame_range, [knee_angle_sagittal_3d_ori[i] for i in frame_range], label="3D sagittal_ori", color='#d62728', alpha=0.5)
        # plt.plot(frame_range, [knee_angle_diagonal_right_3d_ori[i] for i in frame_range], label="3D diagonal right_ori", color='#9467bd', alpha=0.5)
        # plt.plot(frame_range, [knee_angle_diagonal_left_3d_ori[i] for i in frame_range], label="3D diagonal left_ori", color='#8c564b', alpha=0.5)
        plt.plot(frame_range, [knee_angle_frontal_3d_ori[i] for i in frame_range], label="3D frontal_ori", color='#ff7f0e', alpha=0.5)
        plt.plot(frame_range, [knee_angle_mocap_ori[i] for i in frame_range], label="Mocap_ori", color='#2ca02c', alpha=0.5)

        plt.title("Knee Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition}_knee_angle.png"))
        # plt.show()
        plt.cla()

        plt.plot(frame_range, [ankle_angle_sagittal_2d[i] for i in frame_range], label="2D sagittal", color='#1f77b4')
        # plt.plot(frame_range, [ankle_angle_sagittal_3d[i] for i in frame_range], label="3D sagittal", color='#d62728')
        # plt.plot(frame_range, [ankle_angle_diagonal_right_3d[i] for i in frame_range], label="3D diagonal right, color='#9467bd')
        # plt.plot(frame_range, [ankle_angle_diagonal_left_3d[i] for i in frame_range], label="3D diagonal left", color='#8c564b')
        plt.plot(frame_range, [ankle_angle_frontal_3d[i] for i in frame_range], label="3D frontal", color='#ff7f0e')
        plt.plot(frame_range, [ankle_angle_mocap[i] for i in frame_range], label="Mocap", color='#2ca02c')

        plt.plot(frame_range, [ankle_angle_sagittal_2d_ori[i] for i in frame_range], label="2D sagittal_ori", color='#1f77b4', alpha=0.5)
        # plt.plot(frame_range, [ankle_angle_sagittal_3d_ori[i] for i in frame_range], label="3D sagittal_ori", color='#d62728', alpha=0.5)
        # plt.plot(frame_range, [ankle_angle_diagonal_right_3d_ori[i] for i in frame_range], label="3D diagonal right_ori", color='#9467bd', alpha=0.5)
        # plt.plot(frame_range, [ankle_angle_diagonal_left_3d_ori[i] for i in frame_range], label="3D diagonal left_ori", color='#8c564b', alpha=0.5)
        plt.plot(frame_range, [ankle_angle_frontal_3d_ori[i] for i in frame_range], label="3D frontal_ori", color='#ff7f0e', alpha=0.5)
        plt.plot(frame_range, [ankle_angle_mocap_ori[i] for i in frame_range], label="Mocap_ori", color='#2ca02c', alpha=0.5)

        plt.title("Ankle Angle")
        plt.legend()
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition}_ankle_angle.png"))
        # plt.show()
        plt.cla()

        npz_path = os.path.join(os.path.dirname(mkv_files[0]), f"{os.path.basename(mkv_files[0]).split('.')[0].split('_')[0]}_keypoints&frame.npz")
        np.savez(npz_path, diagonal_right = keypoints_diagonal_right, diagonal_left = keypoints_diagonal_left, frontal = keypoints_frontal, mocap = keypoints_mocap, frame_range = frame_range, sagittal_3d = keypoints_sagittal_3d, sagittal_2d = keypoints_sagittal_2d)


if __name__ == "__main__":
    main()