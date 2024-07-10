import numpy as np
from pyk4a import PyK4APlayback, Calibration, CalibrationType, transformation
import cv2
import os
import sys

helpers_dir = r"C:\Users\zutom\pyk4a\example"
os.chdir(helpers_dir)
sys.path.append(helpers_dir)
from helpers import convert_to_bgra_if_required

def main():
    # MKVファイルのパス
    mkv_file_path = r"C:\Users\zutom\aruco_test1.mkv"

    # 再生オブジェクトを作成
    playback = PyK4APlayback(mkv_file_path)
    playback.open()
    frame_count = 1

    # キャリブレーションデータを取得
    calibration = playback.calibration

    # フレームを処理するためのループ
    while True:
        capture = playback.get_next_capture()
        if capture is None:
            break

        # カラーと深度のデータを取得
        color_image = convert_to_bgra_if_required(playback.configuration["color_format"], capture.color)
        depth_image = capture.depth
        aligned_depth_image = capture.transformed_depth

        xpi = 600
        ypi = 500
        z = aligned_depth_image[ypi, xpi]

        try:
            x, y, z = calibration.convert_2d_to_3d(coordinates=(xpi, ypi), depth=z, source_camera=CalibrationType.COLOR)
            print(f"framecount = {frame_count} 3D position of the pixel ({xpi}, {ypi}) is: {x}, {y}, {z}")
        except ValueError as e:
            print(f"Error converting pixel to 3D coordinates: {e}")

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 再生オブジェクトを閉じる
    playback.close()

if __name__ == "__main__":
    main()
