import cv2
import numpy as np
from pyk4a import PyK4A,  PyK4APlayback, CalibrationType, Calibration
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

        ###   ArUcoマーカー検出   ############################################################################################################
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

            # rgb_show_img = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGB)
            # plt.figure()
            # plt.imshow(rgb_show_img)
            # plt.show()

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

                # plt.figure()
                # plt.imshow(rgb_image)
                # plt.show()

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

        ###   前額面カメラのDLT法実行   ############################################################################################################
        if aruco_detect == True and device_id == "dev1" or device_id == "dev2":
            # print(f"select_corners = {select_corners} select_ids_args = {select_ids_args}")

            cal_points_2d = np.array([corner[0] for corner in select_corners])
            cal_points_2d = cal_points_2d.reshape(-1, 2)
            if device_id == "dev1":  # モーキャプのL字型のマーカーの座標（マニュアル）
                cal_points_2d = np.append(cal_points_2d, [[776.,  804.],[818., 680.], [848., 680.], [891., 677.], [899., 704]], axis=0)
                # cal_points_2d = np.append(cal_points_2d, [[776.,  804.],[818., 680.], [848., 680.], [891., 677.], [899., 704], [844.5, 694.5], [873.5, 692.]], axis=0)
                # cal_points_2d = np.append(cal_points_2d, [[776.,  804.],[818., 680.], [848., 680.], [891., 677.], [899., 704], [873.5, 692.]], axis=0)
            elif device_id == "dev2":
                cal_points_2d = np.append(cal_points_2d, [[1087., 820.], [1128., 701.], [1156., 703.], [1196., 705.], [1149., 732.]], axis=0)
                # cal_points_2d = np.append(cal_points_2d, [[1087., 820.], [1128., 701.], [1156., 703.], [1196., 705.], [1149., 732.], [1119., 717.5], [1152.5, 717.5]], axis=0)
                # cal_points_2d = np.append(cal_points_2d, [[1087., 820.], [1128., 701.], [1156., 703.], [1196., 705.], [1149., 732.], [1152.5, 717.5]], axis=0)
            print(f"cal_points_2d = {cal_points_2d}")
            cal_points_3d = np.array([[-618, 208, 285.5], [-618, 118.5, 285.5], [-618, 118.5, 197], [-618, 208, 197],
                                    [-618, 208, 374], [-618, 118.5, 374], [-618, 118.5, 295.5], [-618, 208, 295.5],
                                    [-618, 108.5, 374], [-618, 19, 374], [-618, 19, 295.5], [-618, 108.5, 295.5],
                                    [-618, 108.5, 285.5], [-618, 19, 285.5], [-618, 19, 197], [-618, 108.5, 197],
                                    [0, 365, 0],
                                    [-10, 235, 393.5], [-30, 141, 393.5], [-30, -10, 393.5], [-598, 141, 393.5]])
            # cal_points_3d = np.array([[-618, 208, 285.5], [-618, 118.5, 285.5], [-618, 118.5, 197], [-618, 208, 197],
            #                         [-618, 208, 374], [-618, 118.5, 374], [-618, 118.5, 295.5], [-618, 208, 295.5],
            #                         [-618, 108.5, 374], [-618, 19, 374], [-618, 19, 295.5], [-618, 108.5, 295.5],
            #                         [-618, 108.5, 285.5], [-618, 19, 285.5], [-618, 19, 197], [-618, 108.5, 197],
            #                         [0, 365, 0],
            #                         [-10, 235, 393.5], [-30, 141, 393.5], [-30, -10, 393.5], [-598, 141, 393.5], [-314, 235, 393.5], [-314, 141, 393.5]])
            # cal_points_3d = np.array([[-618, 208, 285.5], [-618, 118.5, 285.5], [-618, 118.5, 197], [-618, 208, 197],
            #                         [-618, 208, 374], [-618, 118.5, 374], [-618, 118.5, 295.5], [-618, 208, 295.5],
            #                         [-618, 108.5, 374], [-618, 19, 374], [-618, 19, 295.5], [-618, 108.5, 295.5],
            #                         [-618, 108.5, 285.5], [-618, 19, 285.5], [-618, 19, 197], [-618, 108.5, 197],
            #                         [0, 365, 0],
            #                         [-10, 235, 393.5], [-30, 141, 393.5], [-30, -10, 393.5], [-598, 141, 393.5], [-314, 141, 393.5]])

            def dlt(cal_points_2d, cal_points_3d):
                A = []
                A_svd = []
                for i in range(len(cal_points_2d)):
                    x, y = cal_points_2d[i]
                    X, Y, Z = cal_points_3d[i]
                    A_svd.append([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])  # SVDを使った方法の場合？
                    A_svd.append([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])
                    A.append([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z])  # 疑似逆行列を使った方法の場合
                    A.append([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z])
                A = np.array(A)
                A_svd = np.array(A_svd)

                """ 疑似逆行列を使った方法"""
                b = cal_points_2d.reshape(-1, 1)
                p  = np.linalg.pinv(A).dot(b)  #疑似逆行列を求めてbと掛け合わせる
                P = np.append(p, 1).reshape(3,4)
                print(f"P_疑似逆行列 = {P}")

                # """ SVDを使った方法 """
                # U, S, Vt = np.linalg.svd(A_svd)
                # P_svd = Vt[-1,:].reshape(3,4)
                # P_svd = P_svd / P_svd[-1, -1]
                # print(f"P_SVD = {P_svd}")

                return P

            if device_id == "dev1":
                P_dev1 = dlt(cal_points_2d, cal_points_3d)
                cal_points_2d_dev1 = cal_points_2d
                np.save(root_dir + f"/calibration/P_1.npy", P_dev1)
            elif device_id == "dev2":
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
        """ 疑似逆行列を使った方法 """
        A = np.array([[P1[2,0]*point_2d_1[0] - P1[0,0], P1[2,1]*point_2d_1[0] - P1[0,1], P1[2,2]*point_2d_1[0] - P1[0,2]],
                      [P1[2,0]*point_2d_1[1] - P1[1,0], P1[2,1]*point_2d_1[1] - P1[1,1], P1[2,2]*point_2d_1[1] - P1[1,2]],
                      [P2[2,0]*point_2d_2[0] - P2[0,0], P2[2,1]*point_2d_2[0] - P2[0,1], P2[2,2]*point_2d_2[0] - P2[0,2]],
                      [P2[2,0]*point_2d_2[1] - P2[1,0], P2[2,1]*point_2d_2[1] - P2[1,1], P2[2,2]*point_2d_2[1] - P2[1,2]]])
        b  = np.array([P1[0,3] - point_2d_1[0],
                       P1[1,3] - point_2d_1[1],
                       P2[0,3] - point_2d_2[0],
                       P2[1,3] - point_2d_2[1]])
        X = np.linalg.pinv(A).dot(b)

        # w1 = 0.8
        # # w2 = 0.2
        # w2 = 0.1
        # # 2D観測点に基づいて行列Aを作成
        # A = np.array([[w1 * (P1[2,0]*point_2d_1[0] - P1[0,0]), w1 * (P1[2,1]*point_2d_1[0] - P1[0,1]), w1 * (P1[2,2]*point_2d_1[0] - P1[0,2])],
        #             [w1 * (P1[2,0]*point_2d_1[1] - P1[1,0]), w1 * (P1[2,1]*point_2d_1[1] - P1[1,1]), w1 * (P1[2,2]*point_2d_1[1] - P1[1,2])],
        #             [w2 * (P2[2,0]*point_2d_2[0] - P2[0,0]), w2 * (P2[2,1]*point_2d_2[0] - P2[0,1]), w2 * (P2[2,2]*point_2d_2[0] - P2[0,2])],
        #             [w2 * (P2[2,0]*point_2d_2[1] - P2[1,0]), w2 * (P2[2,1]*point_2d_2[1] - P2[1,1]), w2 * (P2[2,2]*point_2d_2[1] - P2[1,2])]])

        # # bベクトルにも重み付けを適用
        # b  = np.array([w1 * (P1[0,3] - point_2d_1[0]),
        #             w1 * (P1[1,3] - point_2d_1[1]),
        #             w2 * (P2[0,3] - point_2d_2[0]),
        #             w2 * (P2[1,3] - point_2d_2[1])])

        # X = np.linalg.pinv(A).dot(b)


        """ ここまで """

        """ SVDを使った方法 意味わからん"""
        # print(f"P1 = {P1}")
        # print(f"point_2d_1 = {point_2d_1}")
        # A = np.array([P1[0] - point_2d_1[0]*P1[2],
        #                P1[1] - point_2d_1[1]*P1[2],
        #                P2[0] - point_2d_2[0]*P2[2],
        #                P2[1] - point_2d_2[1]*P2[2]])

        # # print(f"A0 = {A0}")
        # print(f"A = {A}")
        # U, S, Vt = np.linalg.svd(A)
        # print(f"U = {U}, S = {S}, Vt = {Vt}")
        # V = Vt[:, -1]
        # X = V[:3] / V[-1]
        # print(f"V = {V}")
        # print(f"X = {X}")
        """ ここまで """

        return X


    test_3d = np.array([])
    for i in range(len(cal_points_2d)):
        point3d = reconstruction_3d(P_dev1, P_dev2, cal_points_2d_dev1[i], cal_points_2d_dev2[i])
        test_3d = np.append(test_3d, point3d)
    print(f"test_3d = {test_3d}")
    test_3d = test_3d.reshape(-1, 3)
    # test_3d = ", ".join([f"{val:.2f}" for val in test_3d])
    # print(f"test_3d = {test_3d}")
    # # 元の配列の形状を保持しつつフォーマットする場合
    # for row in test_3d:
    #     formatted_row = ", ".join([f"{val:.2f}" for val in row])
    #     print(f"[{formatted_row}]")


    # 較正点の3d座標と比較してX,Y,ZごとにMAEを求める
    error = test_3d - cal_points_3d
    print(f"error = {error}")
    # error_std = np.std(error, axis=0)
    # print(f"error_std = {error_std}")
    error_mae = np.mean(np.abs(error), axis=0)
    print(f"error_mae = {error_mae}")
    formatted_errors = ", ".join([f"{val:.2f}" for val in error_mae])
    print(f"error_mae = [{formatted_errors}]")





if __name__ == "__main__":
    main()
