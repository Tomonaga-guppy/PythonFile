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
        print(f"device id = {device_id}")

        # MKVファイルの再生
        playback = PyK4APlayback(mkv_file_path)
        playback.open()
        calibration = playback.calibration

        frame_count = 1

        start_frame_count = 60
        record_framecount = 60
        target_ids = [0, 1, 2, 3]  # 検出したいマーカーID

        mp4file = os.path.dirname(mkv_file_path) + f"/aruco_detection_{device_id}.mp4"
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
        fps = 30.0
        size = (1920,1080)
        writer = cv2.VideoWriter(mp4file, fmt, fps, size) # ライター作成

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

            if frame_count == start_frame_count:
                if not os.path.exists(root_dir + f"/calibration"):
                    os.makedirs(root_dir + f"/calibration")
                rgb_img_path = root_dir + f"/calibration/{frame_count}_{device_id}.png"
                cv2.imwrite(rgb_img_path, rgb_image)

            depth_image = capture.transformed_depth

            def target_aruco_frame(rgb_image, target_ids, detector, aruco):
                corners, ids, rejected_candidates = detector.detectMarkers(rgb_image)
                select_corners = []
                select_ids = []
                id3_points = None
                aruco_detect = True

                if ids is not None:
                    for i in range(len(ids)):
                        if ids[i][0] in target_ids:
                            select_corners.append(corners[i])
                            select_ids.append(ids[i])

                    if select_corners:
                        select_ids = np.argsort(np.array(select_ids).flatten()) #idが小さい方から順に番号を振る
                        select_corners = [select_corners[i] for i in select_ids] #idが昇順になるようにcornersを並び替える

                        # print(f"select_ids = {select_ids}")
                        # print(f"select_corners = {select_corners}")

                        # plt.figure()
                        # plt.imshow(rgb_image)
                        # plt.show()

                        if len(select_ids) <= 4:
                            aruco_detect = False

                        # if not 2 in select_ids:   #マーカー2が検出できなかった場合
                        #     ref_corner1 = select_corners[1][:][0]
                        #     ref_vector = ref_corner1[1][:] - ref_corner1[0][:]

                        #     ref_corner1_upleft_3d = np.array(calibration.convert_2d_to_3d(ref_corner1[0][:], depth_image[int(ref_corner1[0][1]), int(ref_corner1[0][0])], CalibrationType.COLOR))
                        #     ref_corner1_upright_3d = np.array(calibration.convert_2d_to_3d(ref_corner1[1][:], depth_image[int(ref_corner1[1][1]), int(ref_corner1[1][0])], CalibrationType.COLOR))
                        #     ref_corner1_downlright_3d = np.array(calibration.convert_2d_to_3d(ref_corner1[2][:], depth_image[int(ref_corner1[2][1]), int(ref_corner1[2][0])], CalibrationType.COLOR))
                        #     ref_corner1_downleft_3d = np.array(calibration.convert_2d_to_3d(ref_corner1[3][:], depth_image[int(ref_corner1[3][1]), int(ref_corner1[3][0])], CalibrationType.COLOR))
                        #     ref_vector_up_3d = ref_corner1_upright_3d - ref_corner1_upleft_3d
                        #     ref_vector_down_3d = ref_corner1_downlright_3d - ref_corner1_downleft_3d

                        #     id2_upleft_3d = ref_corner1_upleft_3d + ref_vector_up_3d*(995/895)
                        #     id2_upright_3d = ref_corner1_upleft_3d + ref_vector_up_3d*(1890/895)
                        #     id2_downleft_3d = ref_corner1_downleft_3d + ref_vector_down_3d*(995/895)
                        #     id2_downright_3d = ref_corner1_downleft_3d + ref_vector_down_3d*(1890/895)

                        #     id2_upleft = np.array(calibration.convert_3d_to_2d(id2_upleft_3d, CalibrationType.COLOR)).astype(int)
                        #     id2_upright = np.array(calibration.convert_3d_to_2d(id2_upright_3d, CalibrationType.COLOR)).astype(int)
                        #     id2_downleft = np.array(calibration.convert_3d_to_2d(id2_downleft_3d, CalibrationType.COLOR)).astype(int)
                        #     id2_downright = np.array(calibration.convert_3d_to_2d(id2_downright_3d, CalibrationType.COLOR)).astype(int)

                        #     # # id2のコーナー位置を推定
                        #     id2_upleft_2d = np.round(ref_corner1[0][:] + ref_vector*(995/895)).astype(int)
                        #     id2_upright_2d = np.round(ref_corner1[0][:] + ref_vector*(1890/895)).astype(int)
                        #     id2_downleft_2d = np.round(ref_corner1[3][:] + ref_vector*(995/895)).astype(int)
                        #     id2_downright_2d = np.round(ref_corner1[3][:] + ref_vector*(1890/895)).astype(int)


                        #     # マーカー2の上辺から下数ピクセルを補間して遮蔽部分を補完
                        #     id2_upside_vec = id2_upright - id2_upleft
                        #     id2_upside_vec_length = np.linalg.norm(id2_upside_vec)
                        #     id2_nup_length = np.linalg.norm(np.array([-id2_upside_vec[1], id2_upside_vec[0]]))
                        #     id2_n_up = np.array([-id2_upside_vec[1], id2_upside_vec[0]])/id2_nup_length
                        #     offset = int(id2_upside_vec_length/16) #Arucoマーカー1つが8*8なので1/2の長さを補完
                        #     id2_upleft_offset = (id2_upleft + id2_n_up*offset).astype(np.int32)
                        #     id2_upright_offset = (id2_upright + id2_n_up*offset).astype(np.int32)
                        #     fill_area_up = np.array([id2_upleft, id2_upright, id2_upright_offset, id2_upleft_offset])
                        #     fill_color = rgb_image[int(ref_corner1[0][1]) + offset, int(ref_corner1[0][0]) + offset].tolist()
                        #     cv2.fillPoly(rgb_image, [fill_area_up], fill_color)

                        #     # マーカー2の右辺から左数ピクセルを補間して遮蔽部分を補完
                        #     id2_rightside_vec = id2_downright - id2_upright
                        #     id2_rightside_vec_length = np.linalg.norm(id2_rightside_vec)
                        #     id2_nright_length = np.linalg.norm(np.array([-id2_rightside_vec[1], id2_rightside_vec[0]]))
                        #     id2_n_right = np.array([-id2_rightside_vec[1], id2_rightside_vec[0]])/id2_nright_length
                        #     offset = int(id2_rightside_vec_length/16) #Arucoマーカー1つが8*8なので1/2の長さを補完
                        #     id2_upright_offset = (id2_upright + id2_n_right*offset).astype(np.int32)
                        #     id2_downright_offset = (id2_downright + id2_n_right*offset).astype(np.int32)
                        #     fill_area_right = np.array([id2_upright, id2_downright, id2_downright_offset, id2_upright_offset])
                        #     cv2.fillPoly(rgb_image, [fill_area_right], fill_color)
                        #     points_id2 = np.array([id2_upleft, id2_upright, id2_downright, id2_downleft], dtype=np.int32)

                        # if not 3 in select_ids:   #マーカー3が検出できなかった場合
                        #     ref_corner0 = select_corners[0][:][0]
                        #     ref_vector = ref_corner0[1][:] - ref_corner0[0][:]

                        #     ref_corner0_upleft_3d = np.array(calibration.convert_2d_to_3d(ref_corner0[0][:], depth_image[int(ref_corner0[0][1]), int(ref_corner0[0][0])], CalibrationType.COLOR))
                        #     ref_corner0_upright_3d = np.array(calibration.convert_2d_to_3d(ref_corner0[1][:], depth_image[int(ref_corner0[1][1]), int(ref_corner0[1][0])], CalibrationType.COLOR))
                        #     ref_corner0_downlright_3d = np.array(calibration.convert_2d_to_3d(ref_corner0[2][:], depth_image[int(ref_corner0[2][1]), int(ref_corner0[2][0])], CalibrationType.COLOR))
                        #     ref_corner0_downleft_3d = np.array(calibration.convert_2d_to_3d(ref_corner0[3][:], depth_image[int(ref_corner0[3][1]), int(ref_corner0[3][0])], CalibrationType.COLOR))

                        #     ref_vector_up_3d = ref_corner0_upright_3d - ref_corner0_upleft_3d
                        #     ref_vector_down_3d = ref_corner0_downlright_3d - ref_corner0_downleft_3d

                        #     id3_upleft_3d = ref_corner0_upleft_3d + ref_vector_up_3d*(995/895)
                        #     id3_upright_3d = ref_corner0_upleft_3d + ref_vector_up_3d*(1890/895)
                        #     id3_downleft_3d = ref_corner0_downleft_3d + ref_vector_down_3d*(995/895)
                        #     id3_downright_3d = ref_corner0_downleft_3d + ref_vector_down_3d*(1890/895)

                        #     id3_upleft = np.array(calibration.convert_3d_to_2d(id3_upleft_3d, CalibrationType.COLOR)).astype(int)
                        #     id3_upright = np.array(calibration.convert_3d_to_2d(id3_upright_3d, CalibrationType.COLOR)).astype(int)
                        #     id3_downleft = np.array(calibration.convert_3d_to_2d(id3_downleft_3d, CalibrationType.COLOR)).astype(int)
                        #     id3_downright = np.array(calibration.convert_3d_to_2d(id3_downright_3d, CalibrationType.COLOR)).astype(int)

                        #     # ref_vector = ref_corner0[1][:] - ref_corner0[0][:]
                        #     # # id3のコーナー位置を推定
                        #     id3_upleft_2d = np.round(ref_corner0[0][:] + ref_vector*(995/895)).astype(int)
                        #     id3_upright_2d = np.round(ref_corner0[0][:] + ref_vector*(1890/895)).astype(int)
                        #     id3_downleft_2d = np.round(ref_corner0[3][:] + ref_vector*(995/895)).astype(int)
                        #     id3_downright_2d = np.round(ref_corner0[3][:] + ref_vector*(1890/895)).astype(int)

                        #     # マーカー3の上辺から下数ピクセルを補間して遮蔽部分を補完
                        #     id3_upside_vec = id3_upright - id3_upleft
                        #     id3_n_up = np.array([-id3_upside_vec[1], id3_upside_vec[0]])/np.linalg.norm(np.array([-id3_upside_vec[1], id3_upside_vec[0]]))
                        #     offset = int(np.linalg.norm(id3_upside_vec)/16) #Arucoマーカー1つが8*8なので1/2の長さを補完
                        #     id3_upleft_offset = (id3_upleft + id3_n_up*offset).astype(np.int32)
                        #     id3_upright_offset = (id3_upright + id3_n_up*offset).astype(np.int32)
                        #     fill_area = np.array([id3_upleft, id3_upright, id3_upright_offset, id3_upleft_offset])
                        #     fill_color = rgb_image[int(ref_corner0[0][1]) + offset, int(ref_corner0[0][0]) + offset].tolist()
                        #     cv2.fillPoly(rgb_image, [fill_area], fill_color)

                        #     # マーカー3の右辺から左数ピクセルを補間して遮蔽部分を補完
                        #     id3_rightside_vec = id3_downright - id3_upright
                        #     id3_n_right = np.array([-id3_rightside_vec[1], id3_rightside_vec[0]])/np.linalg.norm(np.array([-id3_rightside_vec[1], id3_rightside_vec[0]]))
                        #     offset = int(np.linalg.norm(id3_rightside_vec)/16) #Arucoマーカー1つが8*8なので1/2の長さを補完
                        #     id3_upright_offset = (id3_upright + id3_n_right*offset).astype(np.int32)
                        #     id3_downright_offset = (id3_downright + id3_n_right*offset).astype(np.int32)
                        #     fill_area = np.array([id3_upright, id3_downright, id3_downright_offset, id3_upright_offset])
                        #     cv2.fillPoly(rgb_image, [fill_area], fill_color)

                        #     points_id3 = np.array([id3_upleft, id3_upright, id3_downright, id3_downleft], dtype=np.int32)
                        #     # cv2.polylines(rgb_image, [points], isClosed= True, color = (0, 0, 255), thickness=1)  #id3の位置を赤色で塗りつぶし

                        #     corners, ids, rejected_candidates = detector.detectMarkers(rgb_image)
                        #     select_ids = np.argsort(np.array(select_ids).flatten()) #idが小さい方から順に番号を振る
                        #     select_corners = [select_corners[i] for i in select_ids] #idが昇順になるようにcornersを並び替える
                        #     print(f"select_ids = {select_ids}")
                        #     print(f"select_corners = {select_corners}")

                        aruco.drawDetectedMarkers(rgb_image, select_corners, select_ids)  #rgb_imageにマーカーを描画
                        # cv2.polylines(rgb_image, [np.array([id2_upleft, id2_upright, id2_downright, id2_downleft])], isClosed= True, color = (0, 0, 255), thickness=1)  #id2の位置を赤色で塗りつぶし
                        # cv2.polylines(rgb_image, [np.array([id3_upleft, id3_upright, id3_downright, id3_downleft])], isClosed= True, color = (0, 0, 255), thickness=1)  #id3の位置を赤色で塗りつぶし(別に要らない)
                        # cv2.polylines(rgb_image, [np.array([id2_upleft_2d, id2_upright_2d, id2_downright_2d, id2_downleft_2d])], isClosed= True, color = (0, 255, 0), thickness=1)
                        # cv2.polylines(rgb_image, [np.array([id3_upleft_2d, id3_upright_2d, id3_downright_2d, id3_downleft_2d])], isClosed= True, color = (0, 255, 0), thickness=1)

                return rgb_image, select_corners, select_ids, id3_points, aruco_detect

            annotated_frame, select_corners, select_ids, id3_points, aruco_detect = target_aruco_frame(rgb_image, target_ids, detector, aruco)

            for i, corners in enumerate(select_corners):
                # 各マーカーのコーナー位置を取得
                corners = corners.astype(np.int32)
                marker_corners = corners[0]

                mask = np.zeros((1080, 1920), dtype=np.uint8)  # 真っ黒なマスクを作成
                cv2.fillPoly(mask, [corners], 255)  # マーカーの内部を白く塗りつぶす
                polygon_pixels = np.column_stack(np.where(mask == 255)) # マーカーの内部のピクセル座標を取得
                # print(f"polygon_pixels = {polygon_pixels}")


                for polygon_pixel in polygon_pixels:
                    y, x = polygon_pixel

            # cv2.imshow("Annotated Frame", rgb_image)

            # plt.figure()
            # plt.imshow(rgb_image)
            # plt.show()

            # # キーが押されるまで待機
            # if cv2.waitKey(0):
            #     break

            writer.write(annotated_frame)
            frame_count += 1

        writer.release()

        if aruco_detect == False and not os.path.exists(rgb_img_path.replace(".png", "_False.png")):
            rename_rgb_img_path = rgb_img_path.replace(".png", "_False.png")
            os.rename(rgb_img_path, rename_rgb_img_path)



        # クリーンアップ
        playback.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
