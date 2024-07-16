import cv2
import numpy as np
from pyk4a import PyK4A, connected_device_count,  PyK4APlayback, CalibrationType
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path_to_target = r"..\..\..\PythonDataFile\stroke"
save_dir = os.path.abspath(os.path.join(current_dir, relative_path_to_target))

# ArUcoのライブラリを導入
aruco = cv2.aruco

# helpers_dir = r"C:\Users\zutom\pyk4a\example"
helpers_dir = r"C:\Users\tus\pyk4a\example"
os.chdir(helpers_dir)
sys.path.append(helpers_dir)
from helpers import convert_to_bgra_if_required

def target_aruco_frame(rgb_image, target_ids, detector):
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
    # 6x6のマーカー, IDは50までの辞書を使用
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
    parameters = aruco.DetectorParameters()

    detector = aruco.ArucoDetector(dictionary, parameters)

    if not os.path.exists(save_dir+"/aruco_images"):
        os.makedirs(save_dir+"/aruco_images")

    # 録画したMKVファイルのパス
    mkv_file_path = r"C:\Users\tus\.vscode\python_scripts\calibration_2.mkv"
    # mkv_file_path = r"C:\Users\zutom\aruco_test1.mkv"

    # MKVファイルの再生
    playback = PyK4APlayback(mkv_file_path)
    playback.open()
    calibration = playback.calibration

    #デバイスのシリアルナンバーを取得---------------------------------------------------------------
    serial_number = playback.connected_device_count()
    print(f"serial_number = {serial_number}")

    frame_count = 1
    max_framecount = 100
    transformation_matrix_sum = np.zeros((4,4))
    target_ids = [1, 5, 9]  # 検出したいマーカーID

    while frame_count < max_framecount+1:  #100フレーム分正常にデータを取得したら終了
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
            __, select_corners, select_ids = target_aruco_frame(rgb_image, target_ids, detector)
            centroid = calculate_3d_centroid(select_corners,select_ids, depth_image, calibration)
            aruco1_3d_cam, aruco5_3d_cam, aruco9_3d_cam = centroid[0], centroid[1], centroid[2]
        except:
            continue

        basex = (aruco5_3d_cam - aruco9_3d_cam)/np.linalg.norm(aruco5_3d_cam - aruco9_3d_cam)
        basey = (aruco1_3d_cam - aruco9_3d_cam)/np.linalg.norm(aruco1_3d_cam - aruco9_3d_cam)
        basez = np.cross(basex, basey)/np.linalg.norm(np.cross(basex, basey))
        t = aruco9_3d_cam
        transformation_matrix = np.array([[basex[0], basey[0], basez[0], t[0]],
                                            [basex[1], basey[1], basez[1], t[1]],
                                            [basex[2], basey[2], basez[2], t[2]],
                                            [0,       0,       0,       1]])
        # print(f"basex = {basex} basey = {basey} basez = {basez} t = {t}")
        # print(f"transformation_matrix = {transformation_matrix}")

        transformation_matrix_sum += transformation_matrix

        # save_path = f"{save_dir}/aruco_images/{frame_count}.png"

        # キーが押されるまで待機
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(f"frame_count = {frame_count}")
        frame_count += 1

    # クリーンアップ
    playback.close()
    cv2.destroyAllWindows()

    #transformation_matrixの平均を計算
    transformation_matrix_mean = transformation_matrix_sum / max_framecount
    #transformation_matrix_meanを保存
    np.save(f"{save_dir}/transformation_matrix.npy", transformation_matrix_mean)


    # # #テスト
    # aruco1_3d = np.dot(np.linalg.inv(transformation_matrix), np.append(aruco1_3d_cam, 1))
    # aruco5_3d = np.dot(np.linalg.inv(transformation_matrix), np.append(aruco5_3d_cam, 1))
    # aruco1_3d_mean = np.dot(np.linalg.inv(transformation_matrix_mean), np.append(aruco1_3d_cam, 1))
    # aruco5_3d_mean = np.dot(np.linalg.inv(transformation_matrix_mean), np.append(aruco5_3d_cam, 1))

    # # print(f"aruco1_3d = {aruco1_3d}")
    # # print(f"aruco5_3d = {aruco5_3d}")
    # print(f"x= {aruco5_3d[0]} y= {aruco1_3d[1]}")
    # print(f"x= {aruco5_3d_mean[0]} y= {aruco1_3d_mean[1]}")

if __name__ == "__main__":
    main()
