import cv2
import numpy as np
from pyk4a import PyK4A,  PyK4APlayback, CalibrationType, Calibration
import os
import sys
import glob
import time
import matplotlib.pyplot as plt
import pandas as pd

helpers_dir = r"C:\Users\pyk4a\example"
os.chdir(helpers_dir)
sys.path.append(helpers_dir)
from helpers import convert_to_bgra_if_required

root_dir = r"F:\Tomson\gait_pattern\20241011"
aruco_dir = os.path.dirname(root_dir)
keyward = "*"

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
    print(f"mkv_file_paths = {mkv_file_paths}")

    for i, mkv_file_path in enumerate(mkv_file_paths):
        print(f" {i+1}/{len(mkv_file_paths)} mkv_file_path = {mkv_file_path}")
        base_name = (os.path.basename(mkv_file_path).split('.')[0])

        ###   ArUcoマーカー検出   ############################################################################################################
        # MKVファイルの再生
        playback = PyK4APlayback(mkv_file_path)
        playback.open()
        calibration = playback.calibration
        frame_count = 1
        start_frame_count = 60

        while True:
            # 60フレーム目から60フレーム分のデータから取得
            if frame_count < start_frame_count:
                frame_count += 1
                continue
            if frame_count > start_frame_count:
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

            # kinectの便利機能でパラメータを取得して歪み補正
            camera_matrix = calibration.get_camera_matrix(CalibrationType.COLOR)
            dist_coeffs = calibration.get_distortion_coefficients(CalibrationType.COLOR)

            undistorted_img = cv2.undistort(rgb_image, camera_matrix, dist_coeffs)

            # ax, fig = plt.subplots(1, 2, figsize=(10, 5))
            # fig[0].imshow(rgb_image)
            # fig[1].imshow(undistorted_img)

            if base_name == "f_dev0" or base_name == "f_dev1":
                plt.figure()
                plt.imshow(undistorted_img)
                plt.title(f"file = {base_name}")
                plt.show()




            if not os.path.exists(root_dir + f"/calibration"):
                os.makedirs(root_dir + f"/calibration")
            rgb_img_path = root_dir + f"/calibration/{frame_count}_{base_name}.png"
            cv2.imwrite(rgb_img_path, rgb_image)
            print(f"Saved {rgb_img_path}")



            frame_count += 1



        ###   前額面カメラのDLT法実行   ############################################################################################################

        df_calib = pd.read_csv(root_dir + f"/較正点記録.csv", skiprows=1)
        check_point_list = ['4', '5', '6', '10', '11', '12', '16', '17', '18', '22', '23', '24', '28', '29', '30', '34', '35', '36', "1'", "2'", "3'", "7'", "8'", "9'"]
        df_calib = df_calib[df_calib["較正点"].isin(check_point_list)]

        if base_name == "f_dev0" or base_name == "f_dev1":
            if base_name == "f_dev0":
                cal_points_2d = df_calib[["fr_x", "fr_y"]].values

            elif base_name == "f_dev1":
                cal_points_2d = df_calib[["fl_x", "fl_y"]].values
            print(f"cal_points_2d = {cal_points_2d}")

            cal_points_3d = df_calib[["X", "Y", "Z"]].values

            print(f"cal_points_3d = {cal_points_3d}")

            def dlt(cal_points_2d, cal_points_3d):
                A = []
                A_svd = []
                for i in range(len(cal_points_2d)):
                    x, y = cal_points_2d[i]
                    X, Y, Z = cal_points_3d[i]
                    A.append([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z])  # 疑似逆行列を使った方法の場合
                    A.append([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z])
                A = np.array(A)
                A_svd = np.array(A_svd)

                """ 疑似逆行列を使った方法"""
                b = cal_points_2d.reshape(-1, 1)
                p  = np.linalg.pinv(A).dot(b)  #疑似逆行列を求めてbと掛け合わせる
                P = np.append(p, 1).reshape(3,4)
                print(f"P_疑似逆行列 = {P}")

                return P

            if base_name == "f_dev0":
                P_dev1 = dlt(cal_points_2d, cal_points_3d)
                cal_points_2d_dev1 = cal_points_2d
                np.save(root_dir + f"/calibration/P_1.npy", P_dev1)
            elif base_name == "f_dev1":
                P_dev2 = dlt(cal_points_2d, cal_points_3d)
                cal_points_2d_dev2 = cal_points_2d
                np.save(root_dir + f"/calibration/P_2.npy", P_dev2)


    ###  3D座標の再構成（テスト）   ############################################################################################################

    # print(f"P_dev1 = {P_dev1}")
    # print(f"P_dev2 = {P_dev2}")
    # print(f"cal_points_2d_dev1 = {cal_points_2d_dev1}")
    # print(f"cal_points_2d_dev2 = {cal_points_2d_dev2}")
    # print(f"cal_points_3d = {cal_points_3d}")


    # test
    def reconstruction_3d(P1, P2, point_2d_1, point_2d_2):
        # print(f"P1 = {P1}")
        # print(f"P2 = {P2}")
        # print(f"point_2d_1 = {point_2d_1}")
        # print(f"point_2d_2 = {point_2d_2}")
        A = np.array([[P1[2,0]*point_2d_1[0] - P1[0,0], P1[2,1]*point_2d_1[0] - P1[0,1], P1[2,2]*point_2d_1[0] - P1[0,2]],
                      [P1[2,0]*point_2d_1[1] - P1[1,0], P1[2,1]*point_2d_1[1] - P1[1,1], P1[2,2]*point_2d_1[1] - P1[1,2]],
                      [P2[2,0]*point_2d_2[0] - P2[0,0], P2[2,1]*point_2d_2[0] - P2[0,1], P2[2,2]*point_2d_2[0] - P2[0,2]],
                      [P2[2,0]*point_2d_2[1] - P2[1,0], P2[2,1]*point_2d_2[1] - P2[1,1], P2[2,2]*point_2d_2[1] - P2[1,2]]])
        b  = np.array([P1[0,3] - point_2d_1[0],
                       P1[1,3] - point_2d_1[1],
                       P2[0,3] - point_2d_2[0],
                       P2[1,3] - point_2d_2[1]])
        X = np.linalg.pinv(A).dot(b)
        return X


    test_3d = np.array([])
    for i in range(len(cal_points_2d)):
        point3d = reconstruction_3d(P_dev1, P_dev2, cal_points_2d_dev1[i], cal_points_2d_dev2[i])
        test_3d = np.append(test_3d, point3d)
    test_3d = test_3d.reshape(-1, 3)
    print(f"test_3d = {test_3d}")
    # test_3d = ", ".join([f"{val:.2f}" for val in test_3d])
    # print(f"test_3d = {test_3d}")
    # # 元の配列の形状を保持しつつフォーマットする場合
    # for row in test_3d:
    #     formatted_row = ", ".join([f"{val:.2f}" for val in row])
    #     print(f"[{formatted_row}]")

    # 較正点の3d座標と比較してX,Y,ZごとにMAEを求める
    error = test_3d - cal_points_3d
    print(f"error = {error}")
    error_mae = np.mean(np.abs(error), axis=0)
    # print(f"error_mae = {error_mae}")
    formatted_errors = ", ".join([f"{val:.2f}" for val in error_mae])
    print(f"error_mae = [{formatted_errors}]")

    for i in range(len(cal_points_3d)):
        print(f"座標：{cal_points_3d[i]}, 誤差：{error[i]}")

    threshold_mae = 20
    large_error_indices = np.where(np.linalg.norm(error, axis=1) > threshold_mae)[0]
    print(f"large_error_indices = {large_error_indices}")


if __name__ == "__main__":
    main()
