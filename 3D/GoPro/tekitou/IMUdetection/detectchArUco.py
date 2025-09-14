"""
IMUに貼ったArUcoマーカーを検出する
"""

import cv2
import numpy as np

def find_bottom_frame_by_aruco(video_path):
    """
    ArUcoマーカーを用いて、動画内でマーカーが最も下に到達したフレームを特定する。

    Args:
        video_path (str): 解析対象の動画ファイルへのパス。

    Returns:
        int: 最下点のフレーム番号。見つからない場合は-1。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイル '{video_path}' を開けません。")
        return -1

    # 1. ArUcoマーカーの辞書を定義 (例: 4x4ピクセルのマーカーセット)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    bottom_y = -1
    bottom_frame_number = -1
    current_frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 2. フレームからマーカーを検出
        corners, ids, rejected = detector.detectMarkers(frame)

        # 3. マーカーが検出された場合
        if ids is not None:
            # cornersは検出されたマーカーの角の座標リスト
            # 最初のマーカーを対象とする
            marker_corners = corners[0][0]
            
            # 4. マーカーの中心Y座標を計算
            center_y = np.mean(marker_corners[:, 1])

            # 5. 最下点を更新
            if center_y > bottom_y:
                bottom_y = center_y
                bottom_frame_number = current_frame_number
        
        current_frame_number += 1

    cap.release()

    # 結果を可視化
    if bottom_frame_number != -1:
        print(f"ArUcoマーカー法による最下点フレーム: {bottom_frame_number}")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, bottom_frame_number)
        ret, result_frame = cap.read()
        if ret:
            # マーカーを再検出して描画
            corners, ids, _ = detector.detectMarkers(result_frame)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(result_frame, corners, ids)
            cv2.putText(result_frame, f"Bottom Frame: {bottom_frame_number}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite("result_aruco_marker.png", result_frame)
            print("結果画像を 'result_aruco_marker.png' として保存しました。")
        cap.release()
    else:
        print("ArUcoマーカーを検出できませんでした。")
        
    return bottom_frame_number

# --- 実行 ---
# ここにあなたの動画ファイルパスを指定してください
video_file = 'your_video.mp4'
find_bottom_frame_by_aruco(video_file)