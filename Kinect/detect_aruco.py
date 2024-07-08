import cv2
import numpy as np
from pyk4a import PyK4APlayback
import os
import sys

# OpenCVのバージョンを表示して確認
print("OpenCV version:", cv2.__version__)

# ArUcoのライブラリを導入
aruco = cv2.aruco

helpers_dir = r"C:\Users\zutom\pyk4a\example"
os.chdir(helpers_dir)
sys.path.append(helpers_dir)
from helpers import convert_to_bgra_if_required

def main():
    # 6x6のマーカー, IDは50までの辞書を使用
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
    parameters = aruco.DetectorParameters()

    # 録画したMKVファイルのパス
    mkv_file_path = r"C:\Users\zutom\aruco_demo.mkv"

    # MKVファイルの再生
    playback = PyK4APlayback(mkv_file_path)
    playback.open()

    frame_count = 0

    while True:
        # 画像をキャプチャ
        capture = playback.get_next_capture()

        # キャプチャが有効でない場合（ファイルの終わり）ループを抜ける
        if capture is None:
            break

        if capture.color is None:
            print(f"Frame {frame_count} has no RGB image data.")
            continue

        # RGB画像を取得
        rgb_image = convert_to_bgra_if_required(playback.configuration["color_format"], capture.color)
        depth_image = capture.depth

        # arucoを認識して書き込み
        if hasattr(aruco, 'detectMarkers'):
            corners, ids, rejectedCandidates = aruco.detectMarkers(rgb_image, dictionary, parameters=parameters)
            annotated_frame = aruco.drawDetectedMarkers(rgb_image, corners, ids)

            # RGB画像を表示
            cv2.imshow('RGB Image with ArUco', annotated_frame)
        else:
            print("aruco.detectMarkers is not available in this OpenCV version.")
            break

        # cv2.imshow('Depth Image', depth_image)

        # キーが押されるまで待機
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(f"frame_count = {frame_count}")
        frame_count += 1

    # クリーンアップ
    playback.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
