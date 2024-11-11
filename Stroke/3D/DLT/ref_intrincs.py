from pathlib import Path
import pickle
from pyk4a import PyK4A,  PyK4APlayback, CalibrationType, Calibration
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

helpers_dir = r"C:\Users\pyk4a\example"
os.chdir(helpers_dir)
sys.path.append(helpers_dir)
from helpers import convert_to_bgra_if_required

root_dir = Path(r"G:\gait_pattern\20241106")
intrinsics_ch_paths = root_dir.glob("*Intrinsic*")

for intrinsics_ch_path in intrinsics_ch_paths:
    # チェッカーボードから取得したカメラパラメータを取得
    id = intrinsics_ch_path.stem.split("Intrinsics_")[-1]
    with open(intrinsics_ch_path, "rb") as f:
        CameraParams = pickle.load(f)

    # kinectの機能からカメラパラメータを取得
    mkv_file_paths = list(root_dir.glob(f"*cali_ch_{id}*.mkv"))
    if not mkv_file_paths:
        print(f"mkv_file_paths for {id} is empty.")
        continue
    playback = PyK4APlayback(mkv_file_paths[0])
    playback.open()
    calibration = playback.calibration

    camera_matrix_sum = np.array([])
    dist_coeffs_sum = np.array([])

    frame_num = 0

    while True:
        print(f"id = {id}, frame_num = {frame_num}")
        try:
            capture = playback.get_next_capture()
        except Exception as e:
            print(e)
            break
        if capture.color is None:
            frame_num += 1
            continue

        if frame_num == 0:
            rgb_image = convert_to_bgra_if_required(playback.configuration["color_format"], capture.color)
            camera_matrix = calibration.get_camera_matrix(CalibrationType.COLOR)
            dist_coeffs = calibration.get_distortion_coefficients(CalibrationType.COLOR)
            undistorted_kinect = cv2.undistort(rgb_image, camera_matrix, dist_coeffs)
            undistorted_ch = cv2.undistort(rgb_image, CameraParams['intrinsicMat'], CameraParams['distortion'])
            undistorted_kinect = cv2.resize(undistorted_kinect, (640, 360))
            undistorted_ch = cv2.resize(undistorted_ch, (640, 360))

            spacing = 20
            height, width, channels = undistorted_kinect.shape
            spacer = np.ones((height, spacing, channels), dtype=np.uint8) * 255 # 黒いスペーサー画像


            concatenated_image = np.hstack((undistorted_kinect, spacer, undistorted_ch))
            cv2.imshow("undistorted", concatenated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        frame_num += 1

    print(f"camera_matrix = {camera_matrix}")
    print(f"dist_coeffs = {dist_coeffs}")

    print(f"camera_params = {CameraParams}")