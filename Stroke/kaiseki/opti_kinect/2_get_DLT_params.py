import cv2
import numpy as np
from pyk4a import PyK4A, connected_device_count,  PyK4APlayback, CalibrationType, Calibration
import os
import sys
import glob
import time
import matplotlib.pyplot as plt

helpers_dir = r"C:\Users\pyk4a\example"
# helpers_dir = r"C:\Users\tus\pyk4a\example"
os.chdir(helpers_dir)
sys.path.append(helpers_dir)
from helpers import convert_to_bgra_if_required

root_dir = r"F:\Tomson\gait_pattern\20240912"
aruco_dir = os.path.dirname(root_dir)
keyward = "calibration*"

def main():
    # ArUcoのライブラリを導入
    aruco = cv2.aruco
    # 6x6のマーカー, IDは50までの辞書を使用
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
    parameters = aruco.DetectorParameters()

    detector = aruco.ArucoDetector(dictionary, parameters)

    if not os.path.exists(aruco_dir+"/aruco_images"):
        os.makedirs(aruco_dir+"/aruco_images")

    # 録画したMKVファイルのパス
    mkv_file_paths = glob.glob(os.path.join(root_dir, f'*{keyward}*.mkv'),recursive=True)

    for i, mkv_file_path in enumerate(mkv_file_paths):
        print(f" {i+1}/{len(mkv_file_paths)} mkv_file_path = {mkv_file_path}")
        device_id = (os.path.basename(mkv_file_path).split('.')[0]).split('_')[-1]

        # MKVファイルの再生
        playback = PyK4APlayback(mkv_file_path)
        playback.open()
        calibration = playback.calibration

        frame_count = 1

        start_frame_count = 60
        target_ids = [0, 1, 2, 3]  # 検出したいマーカーID


        while True:
            # 60フレーム目から60フレーム分のデータから取得
            if frame_count < start_frame_count:
                frame_count += 1
                continue

            if frame_count > 60:
                break

            # 画像をキャプチャ
            capture = playback.get_next_capture()

            # キャプチャが有効でない場合（ファイルの終わり）ループを抜ける
            if capture is None:
                break

            if capture.color is None:
                # print(f"Frame {frame_count} has no RGB image data.")
                continue

            # RGB画像を取得
            rgb_image = convert_to_bgra_if_required(playback.configuration["color_format"], capture.color)

            if not os.path.exists(root_dir + f"/calibration"):
                os.makedirs(root_dir + f"/calibration")
            rgb_img_path = root_dir + f"/calibration/{frame_count}_{device_id}.png"
            corrected_img_path = rgb_img_path.replace(".png", "_corrected.png")

            if not os.path.exists(corrected_img_path):
                cv2.imwrite(rgb_img_path, rgb_image)

            corners, ids, _ = detector.detectMarkers(rgb_image)
            select_corners = []
            select_ids = []
            aruco_detect = True

            if ids is not None:
                for i in range(len(ids)):
                    if ids[i][0] in target_ids:
                        select_corners.append(corners[i])
                        select_ids.append(ids[i])
                select_ids_args = np.argsort(np.array(select_ids).flatten()) #idが小さい方から順に元の配列基準の番号を振る
                select_ids = np.sort(np.array(select_ids).flatten()) #idが昇順になるようにidを並び替える
                select_corners = [select_corners[i] for i in select_ids_args] #idが昇順になるようにcornersを並び替える

                if len(select_ids) < 4:
                    aruco_detect = False
                    if os.path.exists(corrected_img_path):  #補正した画像がある場合はテスト
                        rgb_image = cv2.imread(corrected_img_path)
                        corners, ids, _ = detector.detectMarkers(rgb_image)
                        select_corners = []
                        select_ids = []

                        if ids is not None:
                            for i in range(len(ids)):
                                if ids[i][0] in target_ids:
                                    select_corners.append(corners[i])
                                    select_ids.append(ids[i])
                            select_ids_args = np.argsort(np.array(select_ids).flatten()) #idが小さい方から順に番号を振る
                            select_ids = np.sort(np.array(select_ids).flatten()) #idが昇順になるようにidを並び替える
                            select_corners = [select_corners[i] for i in select_ids_args] #idが昇順になるようにcornersを並び替える
                            if len(select_ids) >= 4 :  # 検出したマーカーが4つ以上の場合は成功
                                aruco_detect = True

                else:
                    aruco_detect = True

                aruco.drawDetectedMarkers(rgb_image, select_corners, select_ids)  #rgb_imageにマーカーを描画

                plt.figure()
                plt.imshow(rgb_image)
                plt.show()

                annotated_img_path = os.path.join(os.path.dirname(rgb_img_path), "annotated" + f"/{frame_count}_{device_id}.png")
                if not os.path.exists(os.path.dirname(annotated_img_path)):
                    os.makedirs(os.path.dirname(annotated_img_path))
                cv2.imwrite(annotated_img_path, rgb_image)

            frame_count += 1

            if aruco_detect == False and not os.path.exists(rgb_img_path.replace(".png", "_False.png")):
                rename_rgb_img_path = rgb_img_path.replace(".png", "_False.png")
                os.rename(rgb_img_path, rename_rgb_img_path)

        if aruco_detect == False:
            print(f"    ArUco markerの検出に失敗しました。\n    画像を補正して{rgb_img_path.replace('.png', '_corrected.png')}で保存してください")
        else:
            pass

        if aruco_detect == True and device_id == "dev1" or device_id == "dev2":  #前額面カメラのDLT法実行
            print(f"select_corners = {select_corners} select_ids_args = {select_ids_args}")


if __name__ == "__main__":
    main()
