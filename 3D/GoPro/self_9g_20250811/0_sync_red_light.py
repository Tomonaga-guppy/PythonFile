import cv2
import numpy as np

# --- グローバル変数 ---
# マウスで選択された矩形領域の座標 (x1, y1, x2, y2)
roi_box = None
# ROI選択中かどうかを判定するフラグ
selecting = False
# ROI選択の始点
start_point = (-1, -1)

def select_roi_with_mouse(event, x, y, flags, param):
    """
    マウスイベントを処理してROI（関心領域）を選択するコールバック関数
    """
    global roi_box, selecting, start_point
    original_display_frame = param

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        selecting = True
        roi_box = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            frame_copy = original_display_frame.copy()
            cv2.rectangle(frame_copy, start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow("Select ROI", frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        end_point = (x, y)
        x1 = min(start_point[0], end_point[0])
        y1 = min(start_point[1], end_point[1])
        x2 = max(start_point[0], end_point[0])
        y2 = max(start_point[1], end_point[1])

        if x2 > x1 and y2 > y1:
            roi_box = (x1, y1, x2, y2)
            frame_copy = original_display_frame.copy()
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow("Select ROI", frame_copy)


def main():
    """
    メイン処理
    """
    global roi_box

    # --- 設定項目 ---
    VIDEO_PATH = r"G:\gait_pattern\20250811_br\gopro\fl\0-0-14.MP4"

    DISPLAY_SCALE = 0.25

    # ★★★ R値の増加量のしきい値 ★★★
    # 前のフレームと比較して、Rの平均値がこの値以上増加したら「点灯」とみなす
    R_INCREASE_THRESH = 20.0

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けませんでした。パスを確認してください: {VIDEO_PATH}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("エラー: 動画からフレームを読み込めませんでした。")
        cap.release()
        return

    display_height = int(first_frame.shape[0] * DISPLAY_SCALE)
    display_width = int(first_frame.shape[1] * DISPLAY_SCALE)
    display_first_frame = cv2.resize(first_frame, (display_width, display_height))

    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", select_roi_with_mouse, display_first_frame)

    print("--- 範囲選択 ---")
    print("マウスをドラッグして矩形範囲を選択後、何かキーを押してください。")
    cv2.imshow("Select ROI", display_first_frame)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if roi_box is not None or key != 255 or cv2.getWindowProperty("Select ROI", cv2.WND_PROP_VISIBLE) < 1:
            break

    if roi_box is None:
        print("\n範囲が選択されませんでした。処理を終了します。")
        cap.release()
        cv2.destroyAllWindows()
        return

    cv2.destroyWindow("Select ROI")
    print(f"選択された範囲（縮小座標）: {roi_box}")

    rx1, ry1, rx2, ry2 = roi_box
    orig_x1 = int(rx1 / DISPLAY_SCALE)
    orig_y1 = int(ry1 / DISPLAY_SCALE)
    orig_x2 = int(rx2 / DISPLAY_SCALE)
    orig_y2 = int(ry2 / DISPLAY_SCALE)

    print("\n--- 動画処理中 ---")
    print(f"R値が前のフレームより {R_INCREASE_THRESH} 以上増加したら検出します。")
    print("'q' を押すと終了します。")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ★★★ 前のフレームのR値を保存する変数を初期化 ★★★
    prev_avg_r = -1.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        roi_area = frame[orig_y1:orig_y2, orig_x1:orig_x2]
        if roi_area.size == 0: continue

        # ★★★ 現在のフレームの平均BGR値を計算 ★★★
        avg_b, avg_g, avg_r, _ = cv2.mean(roi_area)

        r_diff = 0.0
        is_detected = False

        # ★★★ 最初のフレームでない場合、差分を計算して判定 ★★★
        if prev_avg_r >= 0:
            r_diff = avg_r - prev_avg_r
            if r_diff > R_INCREASE_THRESH:
                is_detected = True

        # ★★★ 現在のR値を次のフレームのために保存 ★★★
        prev_avg_r = avg_r

        display_frame = cv2.resize(frame, (display_width, display_height))

        if is_detected:
            box_color = (0, 0, 255)
            font_scale = 0.8
            detected_frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            box_color = (0, 255, 0)
            font_scale = 0.5


        cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), box_color, 2)

        # ★★★ R値とその増加量を表示 ★★★
        text_r_value = f"R:{avg_r:.1f}"
        text_r_diff = f"R-Inc: {r_diff:+.1f}" # 増加量が分かりやすいよう符号(+)も表示

        cv2.putText(display_frame, text_r_value, (rx1, ry1 - 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)
        cv2.putText(display_frame, text_r_diff, (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)

        cv2.imshow("Processing Video (R-Increase)", display_frame)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\n処理が完了しました。")
    if detected_frame_num:
        print(f"R値が増加したフレーム番号: {int(detected_frame_num)}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
