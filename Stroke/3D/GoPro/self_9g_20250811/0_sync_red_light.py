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


def show_detection_frames(cap, detected_frame_num, roi_box, display_scale, total_frames, total_duration):
    """
    検出されたフレームとその前後3フレームずつの計7枚を1行で表示する
    """
    rx1, ry1, rx2, ry2 = roi_box
    orig_x1 = int(rx1 / display_scale)
    orig_y1 = int(ry1 / display_scale)
    orig_x2 = int(rx2 / display_scale)
    orig_y2 = int(ry2 / display_scale)

    # 動画のFPSを取得
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    frame_info = []

    # 前後3フレームずつ、計7フレームを取得
    for offset in range(-3, 4):  # -3, -2, -1, 0, 1, 2, 3
        frame_num = int(detected_frame_num) + offset
        if frame_num < 0:
            frame_num = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        # 時間を計算
        time_sec = frame_num / fps if fps > 0 else 0

        if ret:
            # ROI範囲を描画
            frame_with_roi = frame.copy()

            # 検出フレームは赤色、その他は緑色で描画
            if offset == 0:
                roi_color = (0, 0, 255)  # 赤色（検出フレーム）
                border_color = (0, 0, 255)
                border_thickness = 5
            else:
                roi_color = (0, 255, 0)  # 緑色
                border_color = (0, 255, 0)
                border_thickness = 2

            cv2.rectangle(frame_with_roi, (orig_x1, orig_y1), (orig_x2, orig_y2), roi_color, 3)

            # ROI内の平均R値を計算
            roi_area = frame[orig_y1:orig_y2, orig_x1:orig_x2]
            avg_r = 0
            if roi_area.size > 0:
                avg_b, avg_g, avg_r, _ = cv2.mean(roi_area)

            frames.append(frame_with_roi)
            frame_info.append((frame_num, time_sec, avg_r, border_color, border_thickness))
        else:
            # フレームが読み込めない場合は黒い画像を作成
            if frames:
                black_frame = np.zeros_like(frames[0], dtype=np.uint8)
            else:
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frames.append(black_frame)
            frame_info.append((frame_num, time_sec, 0, (255, 255, 255), 2))

    if len(frames) == 0:
        print("エラー: フレームを取得できませんでした。")
        return

    # 全フレームのサイズを統一するため、まず基準サイズを決定
    if frames:
        base_height, base_width = frames[0].shape[:2]
    else:
        base_height, base_width = 480, 640

    # 各フレームをリサイズ（7枚を横に並べるので少し大きめにする）
    frame_scale = 0.3  # 7枚なので少し大きくする
    target_height = int(base_height * frame_scale)
    target_width = int(base_width * frame_scale)

    # フレームにテキスト情報を追加してリサイズ
    processed_frames = []
    for i, (frame, (frame_num, time_sec, avg_r, border_color, border_thickness)) in enumerate(zip(frames, frame_info)):
        # フレームサイズを基準サイズに統一
        if frame.shape[:2] != (base_height, base_width):
            frame = cv2.resize(frame, (base_width, base_height))

        # フレームを枠で囲む
        frame_with_border = cv2.copyMakeBorder(frame, border_thickness, border_thickness,
                                              border_thickness, border_thickness,
                                              cv2.BORDER_CONSTANT, value=border_color)

        # リサイズ
        border_width = target_width + border_thickness * 2
        border_height = target_height + border_thickness * 2
        resized_frame = cv2.resize(frame_with_border, (border_width, border_height))

        # テキスト用の領域を下に追加
        text_height = 70
        text_area = np.zeros((text_height, border_width, 3), dtype=np.uint8)

        # フレーム番号と時間を表示
        cv2.putText(text_area, f"Frame: {frame_num}", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(text_area, f"Time: {time_sec:.2f}s", (5, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(text_area, f"R: {avg_r:.0f}", (5, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # フレーム位置の表示（中央のフレームを強調）
        if i == 3:  # 中央のフレーム（検出フレーム）
            cv2.putText(text_area, "DETECTED", (border_width//2-40, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

        # フレームとテキストを結合
        final_frame = np.vstack([resized_frame, text_area])
        processed_frames.append(final_frame)

    # 7枚のフレームを1行に並べる
    final_grid = np.hstack(processed_frames)

    # タイトルを追加
    title_height = 100
    title_frame = np.zeros((title_height, final_grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_frame, f"Detection Confirmation - Frame {int(detected_frame_num)}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(title_frame, f"Red border = Detected frame  |  FPS: {fps:.1f}",
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(title_frame, f"Showing frames {frame_info[0][0]} to {frame_info[-1][0]}",
               (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    final_display = np.vstack([title_frame, final_grid])

    cv2.imshow("Detection Frames (7 Frames View)", final_display)
    print(f"\n=== 検出確認 ===")
    print(f"検出フレーム: {int(detected_frame_num)}")
    print(f"検出時間: {detected_frame_num/fps:.2f}秒")
    print(f"表示フレーム範囲: {frame_info[0][0]} ～ {frame_info[-1][0]}")
    print(f"動画情報: FPS={fps:.1f}, 総フレーム数={total_frames}, 総時間={total_duration:.1f}秒")
    print("何かキーを押すと終了します。")
    cv2.waitKey(0)
    cv2.destroyWindow("Detection Frames (7 Frames View)")


def main():
    """
    メイン処理
    """
    global roi_box

    # --- 設定項目 ---
    VIDEO_PATH = r"G:\gait_pattern\20250811_br\sub0\thera0-14\fl\0-0-14.MP4"

    DISPLAY_SCALE = 0.25

    # ★★★ R値の増加量のしきい値 ★★★
    # 前のフレームと比較して、Rの平均値がこの値以上増加したら「点灯」とみなす
    R_INCREASE_THRESH = 10.0

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

    # 動画の総フレーム数とFPSを取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / fps if fps > 0 else 0

    # ★★★ 前のフレームのR値を保存する変数を初期化 ★★★
    prev_avg_r = -1.0
    detected_frame_num = None  # 検出されたフレーム番号を保存

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 現在のフレーム番号と時間を取得
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = current_frame / fps if fps > 0 else 0

        roi_area = frame[orig_y1:orig_y2, orig_x1:orig_x2]
        if roi_area.size == 0: continue

        # ★★★ 現在のフレームの平均BGR値を計算 ★★★
        avg_b, avg_g, avg_r, _ = cv2.mean(roi_area)

        r_diff = 0.0
        is_detected = False

        # ★★★ 最初のフレームでない場合、差分を計算して判定 ★★★
        if prev_avg_r >= 0:
            r_diff = avg_r - prev_avg_r
            if r_diff > R_INCREASE_THRESH and detected_frame_num is None:  # 最初の検出のみ記録
                is_detected = True
                detected_frame_num = current_frame

        # ★★★ 現在のR値を次のフレームのために保存 ★★★
        prev_avg_r = avg_r

        display_frame = cv2.resize(frame, (display_width, display_height))

        if is_detected:
            box_color = (0, 0, 255)
            font_scale = 0.8
        else:
            box_color = (0, 255, 0)
            font_scale = 0.5

        cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), box_color, 2)

        # ★★★ R値とその増加量を表示 ★★★
        text_r_value = f"R:{avg_r:.1f}"
        text_r_diff = f"R-Inc: {r_diff:+.1f}" # 増加量が分かりやすいよう符号(+)も表示

        # ★★★ 時間情報を表示 ★★★
        text_time = f"Time: {current_time:.2f}s / {total_duration:.2f}s"
        text_frame = f"Frame: {current_frame} / {total_frames}"

        cv2.putText(display_frame, text_r_value, (rx1, ry1 - 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)
        cv2.putText(display_frame, text_r_diff, (rx1, ry1 - 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)
        cv2.putText(display_frame, text_time, (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(display_frame, text_frame, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Processing Video (R-Increase)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\n処理が完了しました。")

    # ★★★ 検出されたフレームがある場合、前後のフレームを表示 ★★★
    if detected_frame_num is not None:
        print(f"R値が増加したフレーム番号: {int(detected_frame_num)}")
        show_detection_frames(cap, detected_frame_num, roi_box, DISPLAY_SCALE, total_frames, total_duration)
    else:
        print("検出されたフレームはありませんでした。")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()