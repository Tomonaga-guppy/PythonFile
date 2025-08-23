import cv2
import numpy as np
import os
from pathlib import Path

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
            cv2.imshow("Select ROI for Blackout", frame_copy)

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
            # 黒塗り範囲をプレビュー表示
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame_copy, "Preview: Black area", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Select ROI for Blackout", frame_copy)


def main():
    """
    メイン処理
    """
    global roi_box

    # --- 設定項目 ---
    INPUT_VIDEO_PATH = r"G:\gait_pattern\int_cali\tkrzk\sagi\cali.MP4"

    # 出力ファイル名を自動生成（元のファイル名に "_blackfill" を追加）
    input_path = Path(INPUT_VIDEO_PATH)
    OUTPUT_VIDEO_PATH = str(input_path.parent / f"{input_path.stem}_blackfill{input_path.suffix}")

    # 表示用の縮小率
    DISPLAY_SCALE = 0.25

    print(f"入力動画: {INPUT_VIDEO_PATH}")
    print(f"出力動画: {OUTPUT_VIDEO_PATH}")

    # 動画ファイルを開く
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けませんでした。パスを確認してください: {INPUT_VIDEO_PATH}")
        return

    # 最初のフレームを取得
    ret, first_frame = cap.read()
    if not ret:
        print("エラー: 動画からフレームを読み込めませんでした。")
        cap.release()
        return

    # 動画の情報を取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"動画情報: {width}x{height}, {fps:.2f}fps, {total_frames}フレーム")

    # 表示用にフレームを縮小
    display_height = int(height * DISPLAY_SCALE)
    display_width = int(width * DISPLAY_SCALE)
    display_first_frame = cv2.resize(first_frame, (display_width, display_height))

    # ROI選択のセットアップ
    cv2.namedWindow("Select ROI for Blackout")
    cv2.setMouseCallback("Select ROI for Blackout", select_roi_with_mouse, display_first_frame)

    print("\n--- 黒塗り範囲選択 ---")
    print("マウスをドラッグして黒塗りしたい矩形範囲を選択後、何かキーを押してください。")
    cv2.imshow("Select ROI for Blackout", display_first_frame)

    # ROI選択待ち
    while True:
        key = cv2.waitKey(1) & 0xFF
        if roi_box is not None or key != 255 or cv2.getWindowProperty("Select ROI for Blackout", cv2.WND_PROP_VISIBLE) < 1:
            break

    if roi_box is None:
        print("\n範囲が選択されませんでした。処理を終了します。")
        cap.release()
        cv2.destroyAllWindows()
        return

    cv2.destroyWindow("Select ROI for Blackout")
    print(f"選択された範囲（縮小座標）: {roi_box}")

    # 元の動画サイズに座標を変換
    rx1, ry1, rx2, ry2 = roi_box
    orig_x1 = int(rx1 / DISPLAY_SCALE)
    orig_y1 = int(ry1 / DISPLAY_SCALE)
    orig_x2 = int(rx2 / DISPLAY_SCALE)
    orig_y2 = int(ry2 / DISPLAY_SCALE)

    print(f"実際の黒塗り範囲: ({orig_x1}, {orig_y1}) to ({orig_x2}, {orig_y2})")

    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4コーデック
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    if not out.isOpened():
        print("エラー: 出力動画ファイルを作成できませんでした。")
        cap.release()
        return

    print("\n--- 動画処理中 ---")
    print("全フレームに黒塗り処理を適用しています...")
    print("'q' を押すと途中で中断できます。")

    # 動画を最初から再読み込み
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 指定範囲を黒塗り
        frame[orig_y1:orig_y2, orig_x1:orig_x2] = [0, 0, 0]

        # 出力動画に書き込み
        out.write(frame)

        frame_count += 1

        # 進捗表示（100フレームごと）
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"進捗: {frame_count}/{total_frames} フレーム ({progress:.1f}%)")

        # プレビュー表示（表示用に縮小）
        display_frame = cv2.resize(frame, (display_width, display_height))
        cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 1)
        cv2.putText(display_frame, f"Processing: {frame_count}/{total_frames}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Processing Video", display_frame)

        # 'q'キーで中断
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n処理が中断されました。")
            break

    # リソースを解放
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n処理が完了しました！")
    print(f"処理済みフレーム数: {frame_count}")
    print(f"出力ファイル: {OUTPUT_VIDEO_PATH}")

    # 出力ファイルが存在するか確認
    if os.path.exists(OUTPUT_VIDEO_PATH):
        file_size = os.path.getsize(OUTPUT_VIDEO_PATH) / (1024 * 1024)  # MB
        print(f"ファイルサイズ: {file_size:.2f} MB")
    else:
        print("警告: 出力ファイルが作成されませんでした。")


if __name__ == '__main__':
    main()