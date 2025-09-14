"""
背景差分でIMUを検知できるかどうか
"""

import cv2
import numpy as np

def find_bottom_frame_by_bounding_box(video_path):
    """
    背景差分とバウンディングボックスを用いて、物体が最も下に到達したフレームを特定する。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイル '{video_path}' を開けません。")
        return -1

    subtractor = cv2.createBackgroundSubtractoMOG2(history=100, varThreshold=50, detectShadows=False)

    max_y = -1 # Y座標は下に行くほど値が大きくなる
    bottom_frame_number = -1
    current_frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = subtractor.apply(frame)
        _, thresh_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        morphed_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, kernel)
        morphed_mask = cv2.morphologyEx(morphed_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(morphed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 500:
                # --- 変更点 Start ---
                # 輪郭の重心を計算する代わりに、バウンディングボックスを取得
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # ボックスの底辺のY座標を計算
                bottom_most_y = y + h

                # 最も下の位置を更新
                if bottom_most_y > max_y:
                    max_y = bottom_most_y
                    bottom_frame_number = current_frame_number
                # --- 変更点 End ---

        current_frame_number += 1

    cap.release()
    
    # 結果を可視化
    if bottom_frame_number != -1:
        print(f"背景差分法による最下点フレーム: {bottom_frame_number}")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, bottom_frame_number)
        ret, result_frame = cap.read()
        if ret:
            # 枠で囲むなどの処理を追加しても良い
            cv2.putText(result_frame, f"Bottom Frame: {bottom_frame_number}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite("result_background_subtraction.png", result_frame)
            print("結果画像を 'result_background_subtraction.png' として保存しました。")
        cap.release()
    else:
        print("動体を検出できませんでした。")
        
    return bottom_frame_number

# --- 実行 ---
video_file = 'your_video.mp4' 
find_bottom_frame_by_bounding_box(video_file)