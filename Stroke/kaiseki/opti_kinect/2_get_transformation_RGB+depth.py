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
keyward = "calibration*dev2*"

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
        print(f"device id = {device_id}")

        # MKVファイルの再生
        playback = PyK4APlayback(mkv_file_path)
        playback.open()
        calibration = playback.calibration

        frame_count = 1

        start_frame_count = 60
        record_framecount = 60
        target_ids = [0, 1, 3]  # 検出したいマーカーID
        aruco0_3d_sum, aruco1_3d_sum, aruco3_3d_sum = np.zeros(3), np.zeros(3), np.zeros(3)
        aruco0_2d_sum, aruco1_2d_sum, aruco3_2d_sum = np.zeros(2), np.zeros(2), np.zeros(2)
        t1, t2 = np.zeros(3), np.zeros(3)
        t_2d = np.zeros(2)

        mp4file = os.path.dirname(mkv_file_path) + f"/aruco_detection_{device_id}.mp4"
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
        fps = 30.0
        size = (1920,1080)
        writer = cv2.VideoWriter(mp4file, fmt, fps, size) # ライター作成

        hidden_flag = False #マーカーの検出ができなかったフレームがある場合はTrueに

        while True:
            # 60フレーム目から60フレーム分のデータから取得
            if frame_count < start_frame_count:
                frame_count += 1
                continue

            if frame_count == start_frame_count + record_framecount:
                break

            print(f"frame_count = {frame_count} mkvfile = {mkv_file_path}")

            # 画像をキャプチャ
            capture = playback.get_next_capture()

            # キャプチャが有効でない場合（ファイルの終わり）ループを抜ける
            if capture is None:
                break

            if capture.color is None:
                print(f"Frame {frame_count} has no RGB image data.")
                continue

            # if not os.path.exists(補完した後の動画):
            # RGB画像を取得
            rgb_image = convert_to_bgra_if_required(playback.configuration["color_format"], capture.color)
            depth_image = capture.transformed_depth

            def target_aruco_frame(rgb_image, target_ids, detector, aruco):
                corners, ids, rejected_candidates = detector.detectMarkers(rgb_image)
                select_corners = []
                select_ids = []
                id3_points = None

                if ids is not None:
                    for i in range(len(ids)):
                        if ids[i][0] in target_ids:
                            select_corners.append(corners[i])
                            select_ids.append(ids[i])

                    if select_corners:
                        select_ids = np.argsort(np.array(select_ids).flatten()) #idが小さい方から順に番号を振る
                        select_corners = [select_corners[i] for i in select_ids] #idが昇順になるようにcornersを並び替える

                        print(f"select_ids = {select_ids}")
                        print(f"select_corners = {select_corners}")

                        if not 3 in select_ids:   #マーカー3が検出できなかった場合
                            ref_corner0 = select_corners[0][:][0]
                            ref_corner1 = select_corners[1][:][0]
                            ref_vector = ((ref_corner0[1][:] - ref_corner0[0][:]) + (ref_corner0[2][:] - ref_corner0[3][:]) + (ref_corner1[1][:] - ref_corner1[0][:]) + (ref_corner1[2][:] - ref_corner1[3][:])) / 4
                            # id3のコーナー位置を推定
                            id3_upleft = ref_corner0[0][:] + ref_vector*(995/895)
                            id3_upright = ref_corner0[0][:] + ref_vector*(1990/895)
                            id3_downleft = ref_corner0[3][:] + ref_vector*(995/895)
                            id3_downright = ref_corner0[3][:] + ref_vector*(1990/895)
                            id3_upleft = list(map(int, id3_upleft))
                            id3_upright = list(map(int, id3_upright))
                            id3_downleft = list(map(int, id3_downleft))
                            id3_downright = list(map(int, id3_downright))
                            id3_points = np.array([id3_upleft, id3_upright, id3_downright, id3_downleft])

                            select_corners.append(np.array([[id3_upleft, id3_upright, id3_downright, id3_downleft]], dtype=np.float32))
                            select_ids = np.append(select_ids, 3)

                        aruco.drawDetectedMarkers(rgb_image, select_corners, select_ids)  #rgb_imageにマーカーを描画
                        cv2.polylines(rgb_image, [np.array([id3_upleft, id3_upright, id3_downright, id3_downleft])], isClosed= True, color = (0, 0, 255), thickness=1)  #id3の位置を赤色で塗りつぶし(別に要らない)

                return rgb_image, select_corners, select_ids, id3_points

            annotated_frame, select_corners, select_ids, id3_points = target_aruco_frame(rgb_image, target_ids, detector, aruco)

            def calculate_3d_centroid(select_corners, select_ids, depth_image, calibration, id3_points):
                centroids = {}

                for i, corners in enumerate(select_corners):
                    # 各マーカーのコーナー位置を取得
                    corners = corners.astype(np.int32)
                    marker_corners = corners[0]
                    postion3d_list = []

                    mask = np.zeros((1080, 1920), dtype=np.uint8)  # 真っ黒なマスクを作成
                    cv2.fillPoly(mask, [corners], 255)  # マーカーの内部を白く塗りつぶす
                    polygon_pixels = np.column_stack(np.where(mask == 255)) # マーカーの内部のピクセル座標を取得
                    print(f"polygon_pixels = {polygon_pixels}")


                    for polygon_pixel in polygon_pixels:
                        y, x = polygon_pixel
                        # if depth_image[y, x] == 0:
                        #     continue
                        try :
                            x,y,z = calibration.convert_2d_to_3d(coordinates=(x, y), depth=depth_image[y, x], source_camera=CalibrationType.COLOR)
                            print(f"frame = {frame_count} x,y,z = {x,y,z}")
                        except Exception as e:
                            print(f"frame = {frame_count} error = {e}")
                            continue

                        postion3d_list.append([x,y,z])

                    centroids[i] = np.mean(postion3d_list, axis=0)


                # for i, corners in enumerate(select_corners):
                #     # 各マーカーのコーナー位置を取得
                #     marker_corners = corners[0]
                #     postion3d_list = []

                #     for xpic in range(int(min(marker_corners[:,0])),int(max(marker_corners[:,0]))):
                #         for ypic in range(int(min(marker_corners[:,1])),int(max(marker_corners[:,1]))):
                #             if depth_image[ypic, xpic] == 0:
                #                 continue
                #             x,y,z = calibration.convert_2d_to_3d(coordinates=(xpic, ypic), depth=depth_image[ypic, xpic], source_camera=CalibrationType.COLOR)
                #             postion3d_list.append([x,y,z])

                #     centroids[i] = np.mean(postion3d_list, axis=0)

                return centroids

            centroid = calculate_3d_centroid(select_corners,select_ids, depth_image, calibration, id3_points)

            # cv2.imshow("Annotated Frame", rgb_image)

            # plt.figure()
            # plt.imshow(rgb_image)
            # plt.show()

            # # キーが押されるまで待機
            # if cv2.waitKey(0):
            #     break

            try:  #マーカーが検出できなかった場合はKeyErrorが発生する
                aruco0_3d_sum += centroid[0]
                aruco1_3d_sum += centroid[1]
                aruco3_3d_sum += centroid[2]
            except KeyError:
                hidden_flag = True
                annotated_frame = rgb_image
                writer.write(annotated_frame)
                frame_count += 1
                continue

            sorted_indices = np.argsort(select_ids.flatten()) #idが小さい方から順に番号を振る
            sorted_select_corners = [select_corners[i] for i in sorted_indices] #idが昇順になるようにcornersを並び替える
            aruco0_2d_sum += np.mean(sorted_select_corners[0][0], axis=0)
            aruco1_2d_sum += np.mean(sorted_select_corners[1][0], axis=0)
            aruco3_2d_sum += np.mean(sorted_select_corners[2][0], axis=0)

            writer.write(annotated_frame)
            frame_count += 1

        writer.release()

        if hidden_flag:  #マーカーの検出ができなかったフレームがある場合はファイル名を変更
            aruco_video = os.path.dirname(mkv_file_path) + f"/aruco_detection_{device_id}.mp4"
            aruco_hidden_video = os.path.dirname(mkv_file_path) + f"/aruco_detection_{device_id}_hidden.mp4"
            if not os.path.exists(aruco_hidden_video):
                os.rename(aruco_video, aruco_video.replace(".mp4", "_hidden.mp4"))
            if os.path.exists(aruco_video):  #ファイルが残っていた場合削除
                os.remove(aruco_video)

        aruco0_3d, aruco1_3d, aruco3_3d = aruco0_3d_sum / record_framecount, aruco1_3d_sum / record_framecount, aruco3_3d_sum / record_framecount

        # aruco0_2d, aruco1_2d, aruco3_2d = aruco0_2d_sum / record_framecount, aruco1_2d_sum / record_framecount, aruco3_2d_sum / record_framecount
        # print(f"aruco0_2d = {aruco0_2d}, aruco1_2d = {aruco1_2d}, aruco3_2d = {aruco3_2d}")

        # if device_id == "0":
        if device_id == "dev0":
            basez_0 = (aruco0_3d - aruco3_3d)/np.linalg.norm(aruco0_3d - aruco3_3d)
            basey = (aruco1_3d - aruco0_3d)/np.linalg.norm(aruco1_3d - aruco0_3d)
            basex = np.cross(basey, basez_0)/np.linalg.norm(np.cross(basey, basez_0))
            basez = np.cross(basex, basey)/np.linalg.norm(np.cross(basex, basey))
            t1 = aruco0_3d
            t2 = [-410, -446, -55] #8/8
            # t2 = [-466, -221, -240] #8/22

            # rot_90 = np.array([[0, 1], [-1, 0]]).T
            # basex_2d = (aruco0_2d - aruco3_2d)/np.linalg.norm(aruco0_2d - aruco3_2d)
            # basey_2d = np.dot(rot_90, basex_2d)
            # t_2d = aruco0_2d

        # elif device_id == "1" or device_id == "2":
        elif device_id == "dev1" or device_id == "dev2":
            basex = (aruco3_3d - aruco0_3d)/np.linalg.norm(aruco3_3d - aruco0_3d)
            basey = (aruco1_3d - aruco0_3d)/np.linalg.norm(aruco1_3d - aruco0_3d)
            basez = np.cross(basex, basey)/np.linalg.norm(np.cross(basex, basey))
            t1 = aruco0_3d
            t2 = [-245, -446, 0] #8/8
            # t2 = [-618, -163, -240] #8/22

        transformation_matrix_1 = np.array([[basex[0], basey[0], basez[0], t1[0]],
                                            [basex[1], basey[1], basez[1], t1[1]],
                                            [basex[2], basey[2], basez[2], t1[2]],
                                            [0,       0,       0,       1]])

        transformation_matrix_2 = np.array([[1, 0, 0, t2[0]],
                                            [0, 1, 0, t2[1]],
                                            [0, 0, 1, t2[2]],
                                            [0, 0, 0, 1]])

        # transformation_matrix_2d = np.array([[basex_2d[0], basey_2d[0], t_2d[0]],
        #                                     [basex_2d[1], basey_2d[1], t_2d[1]],
        #                                     [0, 0, 1]])

        # クリーンアップ
        playback.close()
        cv2.destroyAllWindows()

        #transformation_matrix_meanを保存
        npy_save_path = os.path.join(os.path.dirname(mkv_file_path) ,f"tf_matrix_calibration_{device_id}.npz")
        print(f"npy_save_path = {npy_save_path}")
        np.savez(npy_save_path, a1 = transformation_matrix_1, a2 = transformation_matrix_2)
        # np.savez(npy_save_path, a1 = transformation_matrix_1, a2 = transformation_matrix_2, a_2d = transformation_matrix_2d)

    #テスト
    # print(f"aruco0_2d = {aruco0_2d}")
    # print(f"transformation_matrix_2d = {transformation_matrix_2d}")
    # aruco0_2d_henkan = np.dot(np.linalg.inv(transformation_matrix_2d), np.append(aruco0_2d, 1))
    # print(f"aruco0_2d_henkan = {aruco0_2d_henkan}")

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
