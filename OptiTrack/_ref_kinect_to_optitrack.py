import pandas as pd
import numpy as np
import os
import glob
from pyk4a import PyK4A, PyK4APlayback, CalibrationType
import json
import math
import cv2

root_dir = r"F:\Tomson\gait_pattern\20240712"
keywords = "Tpose"

def read_3d_optitrack(csv_path):
    df = pd.read_csv(csv_path, skiprows= [0,1,2,4], header=[0,2])

    # 4行おきにデータを抽出(ダウンサンプリング)
    df_down = df[::4].reset_index(drop=True)

    #必要なマーカーのみを抽出
    marker_set = ["RASI", "LASI","RPSI","LPSI","RKNE","LKNE", "RTHI", "LTHI", "RANK","LANK", "RTIB", "LTIB","RTOE","LTOE","RHEE","LHEE",
                "RSHO", "LSHO","C7", "T10", "CLAV", "STRN", "RBAK", "RKNE2", "LKNE2", "RANK2", "LANK2"]

    #マーカーセットのみを抽出
    marker_set_df = df_down[[col for col in df_down.columns if any(marker in col[0] for marker in marker_set)]]

    success_frame_list = []

    #すべてのマーカーが検出できているフレームのみを抽出
    for frame in range(0, len(marker_set_df)):
        if not marker_set_df.iloc[frame].isna().any():
            success_frame_list.append(frame)

    full_range = range(min(success_frame_list), max(success_frame_list)+1)
    print(f"full_range = {full_range}")
    success_df = marker_set_df.reindex(full_range)

    interpolate_success_df = success_df.interpolate(method='spline', order = 3) #3次スプライン補間
    interpolate_success_df.to_csv(os.path.join(os.path.dirname(csv_path), f"interpolated_{os.path.basename(csv_path)}"))

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
    keypoints_opti = keypoints.reshape(-1, len(marker_set), 3)  #xyzで組になるように変形

    return keypoints_opti


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

def get_3d_coordinates(pixel, depth_image, calibration):
    if np.all(pixel == (0, 0)):
        print(f"0 0 検出できてないよ")
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
        print(f"Invalid depth value at ({x0}, {y0}): {depth_value_x0_y0}")
        return [0, 0, 0]
    if depth_value_x1_y0 <= 0 or np.isnan(depth_value_x1_y0):
        print(f"Invalid depth value at ({x1}, {y0}): {depth_value_x1_y0}")
        return [0, 0, 0]
    if depth_value_x0_y1 <= 0 or np.isnan(depth_value_x0_y1):
        print(f"Invalid depth value at ({x0}, {y1}): {depth_value_x0_y1}")
        return [0, 0, 0]
    if depth_value_x1_y1 <= 0 or np.isnan(depth_value_x1_y1):
        print(f"Invalid depth value at ({x1}, {y1}): {depth_value_x1_y1}")
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

def read_3d_openpose(mkv_file):
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
    a1 = transform_matrix['a1']  #Arucoマーカーまでの変換
    a2 = transform_matrix['a2']  #Optitrackまでの変換(平行移動のみ)
    print(f"mkv_file = {mkv_file}")
    print(f"a1 = {a1}")
    print(f"a2 = {a2}")
    # print(f"transform_matrix_path = {transform_matrix_path}, mkv_file = {mkv_file}")
    # print(f"transform_matrix = {transform_matrix}")

    while True:
        try:
            capture = playback.get_next_capture()
        except:
            print("再生を終了します")
            break

        if capture.color is None:
            print(f"Frame {frame_count} has no RGB image data.")
            continue

        if capture.transformed_depth is None:
            print(f"Frame {frame_count} has no depth image data.")
            continue

        print(f"mkvfile = {mkv_file} frame_count = {frame_count}")

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

            next_keypoints_data = load_keypoints_for_frame(frame_count + 1, json_foloder_path)
            if next_keypoints_data is None:
                print(f"Next frame {frame_count + 1}: JSON file not found, exiting loop.")
                break

            # キーポイントの補完を行う
            keypoints_data = interpolate_missing_keypoints(keypoints_data, previous_keypoints_data, next_keypoints_data)

        frame_keypoints_3d = []

        for i, keypoint in enumerate(keypoints_data):
            pixel = np.array(keypoint[:2])
            # print(f"{i}番目のキーポイント 位置 {pixel}")
            coordinates_cam = get_3d_coordinates(pixel, depth_image, calibration)
            r1 ,t1= a1[:3, :3], a1[:3, 3]
            r1_inv = np.linalg.inv(r1)
            t1_inv = -np.dot(r1_inv, t1)
            A1_inv = np.eye(4)
            A1_inv[:3, :3] = r1_inv
            A1_inv[:3, 3] = t1_inv
            coordinates_aruco = np.dot(A1_inv, np.array([coordinates_cam[0], coordinates_cam[1], coordinates_cam[2], 1]))[:3]
            t2 = a2[:3, 3]
            A2_inv = np.eye(4)
            A2_inv[:3, 3] = -t2
            coordinates = np.dot(A2_inv, np.array([coordinates_aruco[0], coordinates_aruco[1], coordinates_aruco[2], 1]))[:3]
            frame_keypoints_3d.append(coordinates)

        all_keypoints_3d.append(frame_keypoints_3d)

        frame_count += 1

    keypoints_openpose = np.array(all_keypoints_3d)
    print(f"keypoints_openpose = {keypoints_openpose}")
    print(f"keypoints_openpose.shape = {keypoints_openpose.shape}")

    cv2.destroyAllWindows()

        # OpenPoseの出力データを読み込む
        # ここにOpenPoseの出力データを読み込む処理を書く

    return keypoints_openpose

def main():
    #OpnePoseは処理してjsonが出力されている前提, calibrationも終わって座標変換行列つくってる前提
    mkv_files = glob.glob(os.path.join(root_dir, f"*{keywords}*.mkv"))  #mkvファイルのパスを取得
    mkv_sagittal = mkv_files[0]  #進行方向から見て左
    mkv_diagonal_right = mkv_files[1]  #進行方向から見て右斜め前
    mkv_diagonal_left = mkv_files[2]  #進行方向から見て左斜め前

    csv_files = glob.glob(os.path.join(root_dir, f"Motive/[0-9]*{keywords}*.csv"))  #csvファイルのパスを取得

    #openpose+深度で各キーポイントの3d座標を取得
    keypoints_op_sagittal = read_3d_openpose(mkv_sagittal)
    # keypoints_op_diagonal_right = read_3d_openpose(mkv_diagonal_right)
    # keypoints_op_diagonal_left = read_3d_openpose(mkv_diagonal_left)
    #optitrackで各キーポイントの3d座標を取得
    keypoints_opti = read_3d_optitrack(csv_files[0])

    print(f"keypoints_op_sagittal = {keypoints_op_sagittal.shape}")
    # print(f"keypoints_op_diagonal_right = {keypoints_op_diagonal_right.shape}")
    # print(f"keypoints_op_diagonal_left = {keypoints_op_diagonal_left.shape}")
    print(f"keypoints_opti = {keypoints_opti.shape}")







if __name__ == "__main__":
    main()