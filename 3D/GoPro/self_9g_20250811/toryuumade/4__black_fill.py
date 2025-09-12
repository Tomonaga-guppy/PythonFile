import cv2
import numpy as np
import os
from pathlib import Path

"""
竹村研で撮影したデータは右に設置したカメラに計測者が映っている
OpenPoseでPAとPTを検出したい際に計測者に検出が向いてしまうことがあるので計測者がいる範囲のピクセルを黒塗りする必要がある
PAやPTの検出を妨害しないようにフレーム範囲を指定して黒塗り
"""

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


def select_frame_ranges(cap, total_frames, fps):
    """
    黒塗りするフレーム範囲を複数選択する
    """
    print("\n=== フレーム範囲選択 ===")
    print("矢印キー: フレーム移動 | スペース: 再生/停止")
    print("'s': 開始フレーム設定 | 'e': 終了フレーム設定 | 'r': 範囲を削除")
    print("'c': 範囲選択完了 | 'q': 終了")
    
    frame_ranges = []  # [(start, end), (start, end), ...]
    current_frame = 0
    is_playing = False
    temp_start = -1
    
    window_name = "Frame Range Selector"
    cv2.namedWindow(window_name)
    
    # トラックバーのコールバック関数
    def on_trackbar(val):
        nonlocal current_frame, is_playing
        current_frame = val
        is_playing = False
    
    # トラックバーを作成
    cv2.createTrackbar("Position", window_name, 0, total_frames - 1, on_trackbar)
    
    while True:
        # 自動再生
        if is_playing:
            current_frame += 1
            if current_frame >= total_frames:
                current_frame = total_frames - 1
                is_playing = False
        
        # トラックバー位置を更新
        cv2.setTrackbarPos("Position", window_name, current_frame)
        
        # フレームを読み込み
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        if not ret:
            current_frame = max(0, current_frame - 1)
            is_playing = False
            continue
        
        # 現在時刻を計算
        current_time = current_frame / fps if fps > 0 else 0
        
        # フレームをリサイズ
        display_frame = cv2.resize(frame, (1280, 720))
        
        # UIオーバーレイを描画
        overlay = display_frame.copy()
        margin = 20
        overlay_height = 300
        cv2.rectangle(overlay, (margin, 10), (display_frame.shape[1] - margin, overlay_height), (0, 0, 0), -1)
        alpha = 0.7
        display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)
        
        # 現在のフレーム情報を表示
        frame_text = f"Frame: {current_frame} / {total_frames-1}"
        cv2.putText(display_frame, frame_text, (margin + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        time_text = f"Time: {current_time:.2f}s"
        cv2.putText(display_frame, time_text, (margin + 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # 一時的な開始フレームを表示
        if temp_start >= 0:
            temp_text = f"Temp Start: {temp_start}"
            cv2.putText(display_frame, temp_text, (margin + 10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # 設定済み範囲を表示
        ranges_text = f"Ranges: {len(frame_ranges)}"
        cv2.putText(display_frame, ranges_text, (margin + 10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # 範囲の詳細を表示
        y_offset = 210
        for i, (start, end) in enumerate(frame_ranges):
            range_text = f"  {i+1}: {start}-{end} ({end-start+1} frames)"
            cv2.putText(display_frame, range_text, (margin + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            y_offset += 30
        
        # 操作説明を表示
        controls_text = "'s': Start | 'e': End | 'r': Remove last | 'c': Complete | 'q': Quit"
        cv2.putText(display_frame, controls_text, (margin + 10, overlay_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        # 現在のフレームが設定済み範囲に含まれているかチェック
        in_range = any(start <= current_frame <= end for start, end in frame_ranges)
        if in_range:
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 255, 0), 8)
            cv2.putText(display_frame, "IN BLACKOUT RANGE", (50, display_frame.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
        
        # 一時的な開始フレームの場合
        if current_frame == temp_start:
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 255, 255), 8)
            cv2.putText(display_frame, "TEMP START FRAME", (50, display_frame.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
        
        cv2.imshow(window_name, display_frame)
        
        # キー入力処理
        wait_time = 33 if is_playing else 0
        key = cv2.waitKeyEx(wait_time)
        
        if key == -1:
            continue
        
        if key == ord('q'):
            print("フレーム範囲選択を中断しました")
            break
        
        elif key == ord('c'):
            if frame_ranges:
                print(f"フレーム範囲選択完了: {frame_ranges}")
                cv2.destroyWindow(window_name)
                return frame_ranges
            else:
                print("範囲が設定されていません")
        
        elif key == ord('s'):  # 開始フレーム設定
            temp_start = current_frame
            print(f"一時開始フレーム: {temp_start}")
        
        elif key == ord('e'):  # 終了フレーム設定
            if temp_start >= 0:
                if current_frame >= temp_start:
                    new_range = (temp_start, current_frame)
                    frame_ranges.append(new_range)
                    print(f"範囲追加: {new_range} ({current_frame - temp_start + 1}フレーム)")
                    temp_start = -1
                else:
                    print("エラー: 終了フレームは開始フレーム以降に設定してください")
            else:
                print("エラー: まず開始フレーム('s')を設定してください")
        
        elif key == ord('r'):  # 最後の範囲を削除
            if frame_ranges:
                removed = frame_ranges.pop()
                print(f"範囲削除: {removed}")
            else:
                print("削除する範囲がありません")
        
        elif key == ord(' '):  # スペースキー
            is_playing = not is_playing
        
        elif key == 2424832:  # 左矢印キー
            is_playing = False
            current_frame = max(0, current_frame - 1)
        
        elif key == 2555904:  # 右矢印キー
            is_playing = False
            current_frame = min(total_frames - 1, current_frame + 1)
        
        # ウィンドウが閉じられたかチェック
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cv2.destroyWindow(window_name)
    return []


def main():
    """
    メイン処理
    """
    global roi_box

    # --- 設定項目 ---
    INPUT_VIDEO_PATH = r"G:\gait_pattern\20250811_br\sub1\thera1-1\fl\undistorted_ori.mp4"

    # 出力ファイル名を設定
    input_path = Path(INPUT_VIDEO_PATH)
    OUTPUT_VIDEO_PATH = str(input_path.parent / "undistorted.mp4")
    OUTPUT_IMAGE_DIR = input_path.parent / "undistorted"

    # 表示用の縮小率
    DISPLAY_SCALE = 0.25

    print(f"入力動画: {INPUT_VIDEO_PATH}")
    print(f"出力動画: {OUTPUT_VIDEO_PATH}")
    print(f"出力画像ディレクトリ: {OUTPUT_IMAGE_DIR}")

    # ★★★ 出力ディレクトリが存在しない場合は作成 ★★★
    OUTPUT_IMAGE_DIR.mkdir(exist_ok=True)
    print(f"画像出力ディレクトリを作成/確認: {OUTPUT_IMAGE_DIR}")

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

    # ★★★ フレーム範囲を選択 ★★★
    frame_ranges = select_frame_ranges(cap, total_frames, fps)
    
    if not frame_ranges:
        print("フレーム範囲が選択されませんでした。処理を終了します。")
        cap.release()
        cv2.destroyAllWindows()
        return

    print(f"\n設定されたフレーム範囲: {frame_ranges}")
    
    # 総黒塗りフレーム数を計算
    total_blackout_frames = sum(end - start + 1 for start, end in frame_ranges)
    print(f"黒塗り対象フレーム数: {total_blackout_frames} / {total_frames}")

    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4コーデック
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    if not out.isOpened():
        print("エラー: 出力動画ファイルを作成できませんでした。")
        cap.release()
        return

    print("\n--- 動画・画像処理中 ---")
    print("指定されたフレーム範囲に黒塗り処理を適用し、動画と画像を同時出力しています...")
    print("'q' を押すと途中で中断できます。")

    # 動画を最初から再読み込み
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = 0
    saved_image_count = 0
    blackout_applied_count = 0

    # ★★★ フレーム範囲をセットに変換して高速検索 ★★★
    blackout_frames = set()
    for start, end in frame_ranges:
        blackout_frames.update(range(start, end + 1))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ★★★ 現在のフレームが黒塗り対象かチェック ★★★
        if frame_count in blackout_frames:
            # 指定範囲を黒塗り
            frame[orig_y1:orig_y2, orig_x1:orig_x2] = [0, 0, 0]
            blackout_applied_count += 1

        # 出力動画に書き込み
        out.write(frame)

        # ★★★ 画像として保存（フレーム番号でファイル名を設定）★★★
        image_filename = f"frame_{frame_count:05d}.png"  # 5桁のゼロパディング
        image_path = OUTPUT_IMAGE_DIR / image_filename
        
        # 画像保存
        success = cv2.imwrite(str(image_path), frame)
        if success:
            saved_image_count += 1
        else:
            print(f"警告: 画像保存に失敗しました: {image_path}")

        frame_count += 1

        # 進捗表示（100フレームごと）
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"進捗: {frame_count}/{total_frames} フレーム ({progress:.1f}%) - 黒塗り適用: {blackout_applied_count} - 画像保存: {saved_image_count}")

        # プレビュー表示（表示用に縮小）
        display_frame = cv2.resize(frame, (display_width, display_height))
        
        # 黒塗り範囲を表示
        if frame_count - 1 in blackout_frames:  # frame_countは次のフレームを指しているので-1
            cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
        else:
            cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 1)
        
        cv2.putText(display_frame, f"Processing: {frame_count}/{total_frames}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Blackout applied: {blackout_applied_count}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Saved images: {saved_image_count}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
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
    print(f"黒塗り適用フレーム数: {blackout_applied_count}")
    print(f"保存された画像数: {saved_image_count}")
    print(f"出力動画ファイル: {OUTPUT_VIDEO_PATH}")
    print(f"出力画像ディレクトリ: {OUTPUT_IMAGE_DIR}")

    # フレーム範囲の詳細を表示
    print(f"\n=== 適用されたフレーム範囲 ===")
    for i, (start, end) in enumerate(frame_ranges):
        duration = (end - start + 1) / fps if fps > 0 else 0
        print(f"範囲 {i+1}: フレーム {start}-{end} ({end-start+1}フレーム, {duration:.2f}秒)")

    # 出力ファイルが存在するか確認
    if os.path.exists(OUTPUT_VIDEO_PATH):
        file_size = os.path.getsize(OUTPUT_VIDEO_PATH) / (1024 * 1024)  # MB
        print(f"動画ファイルサイズ: {file_size:.2f} MB")
    else:
        print("警告: 出力動画ファイルが作成されませんでした。")

    # ★★★ 画像ディレクトリの情報を表示 ★★★
    if OUTPUT_IMAGE_DIR.exists():
        image_files = list(OUTPUT_IMAGE_DIR.glob("*.jpg"))
        print(f"保存された画像ファイル数: {len(image_files)}")
        
        if image_files:
            # ディレクトリサイズを計算
            total_size = sum(f.stat().st_size for f in image_files) / (1024 * 1024)  # MB
            print(f"画像ディレクトリの総サイズ: {total_size:.2f} MB")
            print(f"最初の画像: {image_files[0].name}")
            print(f"最後の画像: {image_files[-1].name}")
    else:
        print("警告: 出力画像ディレクトリが作成されませんでした。")


if __name__ == '__main__':
    main()