import cv2
import numpy as np
import matplotlib.pyplot as plt # グラフ描画ライブラリをインポート
from pathlib import Path
import csv
import json

def find_bottom_frame_by_aruco(video_path):
    """
    ArUcoマーカーを用いて、動画内でマーカーが最も下に到達したフレームを特定し、
    y座標の推移をグラフで可視化する。

    Args:
        video_path (str): 解析対象の動画ファイルへのパス。

    Returns:
        int: 最下点のフレーム番号。見つからない場合は-1。
    """
    
    skip_detection = False
    y_csv_path = video_path.parent / f"{video_path.stem}_coordinates.csv"
    
    # グラフ用のデータを保存するリストを初期化
    frame_numbers_list = []
    y_coords_list = []
    
        
    if y_csv_path.exists():
        print(f"既存のCSVファイルが見つかりました: {y_csv_path}")
        print("この動画の処理をスキップします。\n")
        skip_detection = True
    else:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"エラー: 動画ファイル '{video_path}' を開けません。")
            return -1

        # ArUcoマーカーの辞書を定義
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        current_frame_number = 0

        start_frame_skip = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_skip)
        current_frame_number = start_frame_skip
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # フレームからマーカーを検出
            corners, ids, rejected = detector.detectMarkers(frame)

            # マーカーが検出された場合
            if ids is not None:
                # 最初のマーカーを対象とする
                marker_corners = corners[0][0]
                
                # マーカーの中心Y座標を計算
                center_y = np.mean(marker_corners[:, 1])
                center_x = np.mean(marker_corners[:, 0])

                # フレーム番号とy座標をリストに追加
                frame_numbers_list.append(current_frame_number)
                y_coords_list.append(center_y)
                
                # マーカーの中心を描画
                cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                mini_frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))
                cv2.imshow("Frame with ArUco", mini_frame)
                cv2.waitKey(1)
                print(f"検出成功: Frame {current_frame_number}: Marker center Y = {center_y}")

            center_y_pre = center_y
            current_frame_number += 1

        cap.release()
        cv2.destroyAllWindows() # 表示ウィンドウを閉じる処理
    
    if skip_detection:  #既存のCSVがある場合はそこからフレームとy座標を読み込む
        try:
            with open(y_csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # ヘッダー行をスキップ
                for row in reader:
                    frame_numbers_list.append(int(row[0]))
                    y_coords_list.append(float(row[1]))
            print(f"CSVファイル '{y_csv_path}' からデータを読み込みました。")
        except IOError as e:
            print(f"エラー: CSVファイル '{y_csv_path}' の読み込みに失敗しました。 {e}")
            return -1
        
    
    # 地面に到達したフレームを算出
    center_y_pre = None  # 前の座標と比較用
    vy_diff_pre = 0
    vy_threshold = 5  # Y座標の変化量の閾値
    y_coords_threshold = 10  # 一定位置よりも動いていることを確認する閾値
    y_coords_median = np.median(y_coords_list) if y_coords_list else 0
    # print(f"Y座標の中央値: {y_coords_median}")
    
    impact_frame = -1  # 最下点フレームの初期値

    for i in range(len(frame_numbers_list)):
        vy_diff = y_coords_list[i] - center_y_pre if center_y_pre is not None else 0
        center_y_pre = y_coords_list[i]
        # print(f"フレーム {frame_numbers_list[i]}: vy_diff:{vy_diff}, vy_diff_pre:{vy_diff_pre}, y_coords:{y_coords_list[i]}, y_coords_median:{y_coords_median}")
        # print(f"    vy_diff <= 0_bool: {vy_diff <= 0}, vy_diff_pre > 0_bool: {vy_diff_pre > 0},vy_diff_pre > vy_threshold_bool: {vy_diff_pre > vy_threshold},  y_coord_bool: {abs(y_coords_list[i-1]-y_coords_median)<y_coords_threshold}")
        # 最下点の条件: 速度が正から負に変わる、かつ前の速度が閾値以上、かつy座標が中央値付近
        if vy_diff <= 0 and vy_diff_pre > 0 and vy_diff_pre > vy_threshold and abs(y_coords_list[i-1] - y_coords_median) < y_coords_threshold:
            impact_frame = frame_numbers_list[i-1]  #増加から減少に転じた直前のフレームが最下点
            print(f"最下点フレーム: {impact_frame} (Y座標: {y_coords_list[i-1]})")
        
        vy_diff_pre = vy_diff

    # 検出結果をCSVファイルに保存
    if skip_detection == False and frame_numbers_list:
        csv_filename = str(video_path.parent / f"{video_path.stem}_coordinates.csv")
        try:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # ヘッダーを書き込む
                writer.writerow(['Frame', 'CenterY'])
                # データを書き込む
                for i in range(len(frame_numbers_list)):
                    writer.writerow([frame_numbers_list[i], y_coords_list[i]])
            print(f"座標データを '{csv_filename}' として保存しました。")
        except IOError as e:
            print(f"エラー: CSVファイル '{csv_filename}' の書き込みに失敗しました。 {e}")
            
    # グラフを描画して保存する
    if frame_numbers_list:  # データが記録されている場合のみグラフを作成
        plt.figure(figsize=(12, 6))
        plt.plot(frame_numbers_list, y_coords_list, marker='.', linestyle='-', label='Marker Y-coordinate')
        plt.axvline(x=impact_frame, color='r', linestyle='--', label='Impact Frame')
        plt.title('Marker Center Y-coordinate over Frames')
        plt.xlabel('Frame Number')
        plt.ylabel('Y-coordinate')
        plt.grid(True)

        plt.legend()
        graph_filename = str(video_path.parent / "marker_y_coordinate_graph.png")
        plt.savefig(graph_filename)
        print(f"グラフを '{graph_filename}' として保存しました。")
    else:
        print("グラフを描画するためのマーカーデータがありませんでした。")
        

# LED発光を0フレーム目として切り抜いた動画を処理（今は切り抜き前を使用）
video_file = Path(r"G:\gait_pattern\20250915_synctest\1.MP4")
find_bottom_frame_by_aruco(video_file)

gopro_trim_info_path = video_file.parent / "gopro_trimming_info.json"
with open(gopro_trim_info_path, 'r') as f:
    gopro_trim_info = json.load(f)
