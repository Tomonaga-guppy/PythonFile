import cv2
import numpy as np
from pyk4a import PyK4A, connected_device_count,  PyK4APlayback, CalibrationType, Calibration
import os
import sys
import glob

helpers_dir = r"C:\Users\pyk4a\example"
# helpers_dir = r"C:\Users\tus\pyk4a\example"
os.chdir(helpers_dir)
sys.path.append(helpers_dir)
from helpers import convert_to_bgra_if_required

# root_dir = r"F:\Tomson\gait_pattern\20240712"
root_dir = r"F:\Tomson\gait_pattern\20240807test"
aruco_dir = os.path.dirname(root_dir)
keyward = "calibration"

def target_aruco_frame(rgb_image, target_ids, detector, aruco):
    corners, ids, __ = detector.detectMarkers(rgb_image)
    selected_corners = []
    selected_ids = []

    if ids is not None:
        for i in range(len(ids)):
            if ids[i][0] in target_ids:
                selected_corners.append(corners[i])
                selected_ids.append(ids[i])

        if selected_corners:
            selected_ids = np.array(selected_ids)
            annotated_frame = aruco.drawDetectedMarkers(rgb_image, selected_corners, selected_ids)
        else:
            annotated_frame = rgb_image
    else:
        annotated_frame = rgb_image

    return annotated_frame, selected_corners, selected_ids

def calculate_3d_centroid(select_corners, select_ids, depth_image, calibration):
    sorted_indices = np.argsort(select_ids.flatten()) #idが小さい方から順に番号を振る
    sorted_select_corners = [select_corners[i] for i in sorted_indices] #idが昇順になるようにcornersを並び替える
    centroids = {}
    for i, corners in enumerate(sorted_select_corners):
        # 各マーカーのコーナー位置を取得
        marker_corners = corners[0]
        postion3d_list = []

        for xpic in range(int(min(marker_corners[:,0])),int(max(marker_corners[:,0]))):
            for ypic in range(int(min(marker_corners[:,1])),int(max(marker_corners[:,1]))):
                if depth_image[ypic, xpic] == 0:
                    continue
                x,y,z = calibration.convert_2d_to_3d(coordinates=(xpic, ypic), depth=depth_image[ypic, xpic], source_camera=CalibrationType.COLOR)
                postion3d_list.append([x,y,z])

        centroids[i] = np.mean(postion3d_list, axis=0)

    return centroids

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

    for mkv_file_path in mkv_file_paths:
        print(f"mkv_file_path = {mkv_file_path}")
        id = (os.path.basename(mkv_file_path).split('.')[0]).split('_')[-1]
        print(f"id = {id}")

        # MKVファイルの再生
        playback = PyK4APlayback(mkv_file_path)
        playback.open()
        calibration = playback.calibration

        # #デバイスのシリアルナンバーを取得---------------------------------------------------------------
        # device_serial_list = [000767101412, 000723201412, 000795793812] #進行方向を正面として右斜め前、左斜め前、左
        # serial_number = playback.connected_device_count()
        # print(f"serial_number = {serial_number}")

        frame_count = 1

        start_frame_caount = 60
        record_framecount = 60
        transformation_matrix_sum = np.zeros((4,4))
        transformation_matrix_2d_sum = np.zeros((3,3))
        target_ids = [0, 1, 3]  # 検出したいマーカーID
        aruco0_3d_sum, aruco1_3d_sum, aruco3_3d_sum = np.zeros(3), np.zeros(3), np.zeros(3)
        aruco0_2d_sum, aruco1_2d_sum, aruco3_2d_sum = np.zeros(2), np.zeros(2), np.zeros(2)

        while True:
            # 60フレーム目から60フレーム分のデータを取得
            if frame_count < start_frame_caount:
                frame_count += 1
                continue

            if frame_count == start_frame_caount + record_framecount:
                break

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
            depth_image = capture.transformed_depth

            try:
                __, select_corners, select_ids = target_aruco_frame(rgb_image, target_ids, detector, aruco)
                centroid = calculate_3d_centroid(select_corners,select_ids, depth_image, calibration)

                aruco0_3d_sum += centroid[0]
                aruco1_3d_sum += centroid[1]
                aruco3_3d_sum += centroid[2]

                sorted_indices = np.argsort(select_ids.flatten()) #idが小さい方から順に番号を振る
                sorted_select_corners = [select_corners[i] for i in sorted_indices] #idが昇順になるようにcornersを並び替える
                aruco0_2d_sum += np.mean(sorted_select_corners[0][0], axis=0)
                aruco1_2d_sum += np.mean(sorted_select_corners[1][0], axis=0)
                aruco3_2d_sum += np.mean(sorted_select_corners[2][0], axis=0)

            except:
                continue

            print(f"frame_count = {frame_count}")
            frame_count += 1

        aruco0_3d, aruco1_3d, aruco3_3d = aruco0_3d_sum / record_framecount, aruco1_3d_sum / record_framecount, aruco3_3d_sum / record_framecount
        aruco0_2d, aruco1_2d, aruco3_2d = aruco0_2d_sum / record_framecount, aruco1_2d_sum / record_framecount, aruco3_2d_sum / record_framecount

        print(f"aruco0_2d = {aruco0_2d}, aruco1_2d = {aruco1_2d}, aruco3_2d = {aruco3_2d}")

        if id == "0":
            basez = (aruco0_3d - aruco3_3d)/np.linalg.norm(aruco0_3d - aruco3_3d)
            basey = (aruco1_3d - aruco0_3d)/np.linalg.norm(aruco1_3d - aruco0_3d)
            basex = np.cross(basey, basez)/np.linalg.norm(np.cross(basey, basez))
            t1 = aruco0_3d
            t2 = [-410, -446, -55]

            basex_2d = (aruco0_2d - aruco3_2d)/np.linalg.norm(aruco0_2d - aruco3_2d)
            basey_2d = (aruco1_2d - aruco0_2d)/np.linalg.norm(aruco1_2d - aruco0_2d)
            t1_2d = aruco0_2d


        elif id == "1" or id == "2":
            basex = (aruco3_3d - aruco0_3d)/np.linalg.norm(aruco3_3d - aruco0_3d)
            basey = (aruco1_3d - aruco0_3d)/np.linalg.norm(aruco1_3d - aruco0_3d)
            basez = np.cross(basex, basey)/np.linalg.norm(np.cross(basex, basey))
            t1 = aruco0_3d
            t2 = [-245, -446, 0]

        transformation_matrix_1 = np.array([[basex[0], basey[0], basez[0], t1[0]],
                                            [basex[1], basey[1], basez[1], t1[1]],
                                            [basex[2], basey[2], basez[2], t1[2]],
                                            [0,       0,       0,       1]])

        transformation_matrix_2 = np.array([[1, 0, 0, t2[0]],
                                            [0, 1, 0, t2[1]],
                                            [0, 0, 1, t2[2]],
                                            [0, 0, 0, 1]])

        transformation_matrix_2d = np.array([[basex_2d[0], basey_2d[0], t1_2d[0]],
                        [basex_2d[1], basey_2d[1], t1_2d[1]],
                        [0, 0, 1]])

        # クリーンアップ
        playback.close()
        cv2.destroyAllWindows()

        #transformation_matrix_meanを保存
        npy_save_path = os.path.join(os.path.dirname(mkv_file_path) ,f"transformation_matrix_{id}.npz")
        print(f"npy_save_path = {npy_save_path}")
        np.savez(npy_save_path, a1 = transformation_matrix_1, a2 = transformation_matrix_2, a_2d = transformation_matrix_2d)

    #テスト
    print(f"aruco0_2d = {aruco0_2d}")
    aruco0_2d_henkan = np.dot(transformation_matrix_2d, np.append(aruco0_2d, 1))
    print(f"aruco0_2d_henkan = {aruco0_2d_henkan}")
    # aruco1_3d = np.dot(np.linalg.inv(transformation_matrix), np.append(aruco1_3d, 1))
    # aruco5_3d = np.dot(np.linalg.inv(transformation_matrix), np.append(aruco5_3d_cam, 1))
    # aruco1_3d_mean = np.dot(np.linalg.inv(transformation_matrix_mean), np.append(aruco1_3d, 1))
    # aruco5_3d_mean = np.dot(np.linalg.inv(transformation_matrix_mean), np.append(aruco5_3d_cam, 1))

    # # print(f"aruco1_3d = {aruco1_3d}")
    # # print(f"aruco5_3d = {aruco5_3d}")
    # print(f"x= {aruco5_3d[0]} y= {aruco1_3d[1]}")
    # print(f"x= {aruco5_3d_mean[0]} y= {aruco1_3d_mean[1]}")

if __name__ == "__main__":
    main()
