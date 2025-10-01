import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

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


def select_video_range(cap, detected_frame_num, video_path):
    """
    detected_frameを0フレーム目として開始・終了フレームを選択し、動画を切り出す
    既存のJSONファイルがある場合は、相対的な開始・終了フレームを自動適用
    """
    # 既存のJSONファイルをチェック
    json_filename = "gopro_trimming_info.json"
    json_path = video_path.parent.with_name("gopro_trimming_info.json")
    print(f"JSONファイルパス: {json_path}")

    existing_start_rel = None
    existing_end_rel = None
    existing_trim_length = None

    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            trimming_settings = existing_data.get("trimming_settings", {})
            existing_start_rel = trimming_settings.get("start_frame_relative", None)
            existing_end_rel = trimming_settings.get("end_frame_relative", None)
            existing_trim_length = trimming_settings.get("trimmed_frame_count", None)

            if existing_start_rel is not None and existing_end_rel is not None:
                print(f"\n既存のJSONファイルを検出: {json_path}")
                print(f"既存の相対開始フレーム: {existing_start_rel}")
                print(f"既存の相対終了フレーム: {existing_end_rel}")
                print(f"既存の切り出し長: {existing_trim_length}フレーム")

                return existing_start_rel, existing_end_rel

                # print("既存の設定を自動適用しますか？ (y/n) (デフォルト: y)")

                # # ★★★ 標準入力を使用してユーザー入力を取得 ★★★
                # try:
                #     user_input = input("入力してください (y/n): ").strip().lower()
                #     if user_input == '' or user_input == 'y' or user_input == 'yes':
                #         print("既存の設定を適用します")
                #         return existing_start_rel, existing_end_rel
                #     elif user_input == 'n' or user_input == 'no':
                #         print("新しい設定を行います")
                #     else:
                #         print("無効な入力です。新しい設定を行います")
                # except (EOFError, KeyboardInterrupt):
                #     print("\n入力がキャンセルされました。新しい設定を行います")

        except Exception as e:
            print(f"既存JSONファイルの読み込みに失敗: {e}")
            existing_start_rel = None
            existing_end_rel = None

    print("\n=== 動画切り出し範囲選択 ===")
    print("detected_frameを0フレーム目として、その後の範囲を選択してください")
    print("矢印キー: フレーム移動 | スペース: 再生/停止 | Enter: 決定 | 'q': 終了")

    # 動画情報を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # detected_frame以降のフレーム数を計算
    available_frames = total_frames - int(detected_frame_num)

    if available_frames <= 0:
        print("エラー: detected_frame以降にフレームが存在しません")
        return None, None

    # 選択状態の管理
    current_relative_frame = 0  # detected_frameからの相対位置
    is_playing = False
    start_frame_rel = -1  # 開始フレーム（相対位置）
    end_frame_rel = -1    # 終了フレーム（相対位置）
    selection_mode = "start"  # "start" or "end"

    window_name = "Video Range Selector"
    cv2.namedWindow(window_name)

    # トラックバーのコールバック関数
    def on_trackbar(val):
        nonlocal current_relative_frame, is_playing
        current_relative_frame = val
        is_playing = False

    # トラックバーを作成（0からavailable_frames-1まで）
    cv2.createTrackbar("Position", window_name, 0, available_frames - 1, on_trackbar)

    while True:
        # 自動再生
        if is_playing:
            current_relative_frame += 1
            if current_relative_frame >= available_frames:
                current_relative_frame = available_frames - 1
                is_playing = False

        # トラックバー位置を更新
        cv2.setTrackbarPos("Position", window_name, current_relative_frame)

        # 実際のフレーム番号を計算
        actual_frame = int(detected_frame_num) + current_relative_frame

        # フレームを読み込み
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
        ret, frame = cap.read()

        if not ret:
            current_relative_frame = max(0, current_relative_frame - 1)
            is_playing = False
            continue

        # 現在時刻を計算
        current_time = actual_frame / fps if fps > 0 else 0
        relative_time = current_relative_frame / fps if fps > 0 else 0

        # UIオーバーレイを描画（余白を持たせて横幅いっぱいに使用）
        overlay = frame.copy()
        margin = 20  # 左右の余白
        overlay_height = 250
        cv2.rectangle(overlay, (margin, 10), (frame.shape[1] - margin, overlay_height), (0, 0, 0), -1)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # 現在のモードを表示
        mode_text = f"Selecting: {'START frame' if selection_mode == 'start' else 'END frame'}"
        cv2.putText(frame, mode_text, (margin + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        # フレーム情報を表示
        frame_text = f"Relative Frame: {current_relative_frame} / Actual Frame: {actual_frame}"
        cv2.putText(frame, frame_text, (margin + 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # 時間情報を表示
        time_text = f"Relative Time: {relative_time:.2f}s / Actual Time: {current_time:.2f}s"
        cv2.putText(frame, time_text, (margin + 10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # 選択済みフレーム情報を表示
        selection_text = f"Start: {start_frame_rel if start_frame_rel >= 0 else 'Not Set'} | End: {end_frame_rel if end_frame_rel >= 0 else 'Not Set'}"
        cv2.putText(frame, selection_text, (margin + 10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # 操作説明を表示
        controls_text = "Arrow Keys: Move | Space: Play/Pause | Enter: Confirm | 'q': Quit"
        cv2.putText(frame, controls_text, (margin + 10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        # 選択済みフレームをマーク
        if start_frame_rel >= 0 and current_relative_frame == start_frame_rel:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 8)
            cv2.putText(frame, "START FRAME", (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

        if end_frame_rel >= 0 and current_relative_frame == end_frame_rel:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 8)
            cv2.putText(frame, "END FRAME", (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

        # フレームをリサイズして表示
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow(window_name, display_frame)

        # キー入力処理
        wait_time = 33 if is_playing else 0
        key = cv2.waitKeyEx(wait_time)

        if key == -1:
            continue

        if key == ord('q'):
            print("範囲選択を中断しました")
            cv2.destroyWindow(window_name)
            return None, None

        elif key == 13:  # Enterキー
            if selection_mode == "start":
                start_frame_rel = current_relative_frame
                selection_mode = "end"
                print(f"開始フレーム（相対）: {start_frame_rel} を設定しました")
                print("次に終了フレームを選択してください")

            elif selection_mode == "end":
                if current_relative_frame <= start_frame_rel:
                    print("エラー: 終了フレームは開始フレームより後に設定してください")
                    continue

                end_frame_rel = current_relative_frame
                print(f"終了フレーム（相対）: {end_frame_rel} を設定しました")

                # 確認
                duration_frames = end_frame_rel - start_frame_rel + 1
                duration_seconds = duration_frames / fps if fps > 0 else 0
                print(f"切り出し範囲: {start_frame_rel} ～ {end_frame_rel} ({duration_frames}フレーム, {duration_seconds:.2f}秒)")

                cv2.destroyWindow(window_name)
                return start_frame_rel, end_frame_rel


        elif key == ord(' '):  # スペースキー
            is_playing = not is_playing

        elif key == 2424832:  # 左矢印キー
            is_playing = False
            current_relative_frame = max(0, current_relative_frame - 1)

        elif key == 2555904:  # 右矢印キー
            is_playing = False
            current_relative_frame = min(available_frames - 1, current_relative_frame + 1)

        # ウィンドウが閉じられたかチェック
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow(window_name)
    return None, None


def clip_video_segment(video_path, detected_frame_num, start_frame_rel, end_frame_rel):
    """
    指定された範囲で動画を切り出して保存する
    """
    print("\n=== 動画切り出し処理 ===")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けませんでした: {video_path}")
        return None

    # 動画情報を取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 実際のフレーム番号を計算
    actual_start_frame = int(detected_frame_num) + start_frame_rel
    actual_end_frame = int(detected_frame_num) + end_frame_rel
    total_output_frames = actual_end_frame - actual_start_frame + 1

    # 出力ファイル名を生成
    output_filename = f"{video_path.stem}_trimed_f{actual_start_frame}-{actual_end_frame}.mp4"
    output_path = video_path.parent / output_filename

    # VideoWriterを初期化
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"エラー: 出力ファイルを作成できませんでした: {output_path}")
        cap.release()
        return None

    # 開始フレームに移動
    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start_frame)

    print(f"動画切り出し中... ({total_output_frames}フレーム)")

    # フレームを書き込み
    written_frames = 0
    for i in range(total_output_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"警告: フレーム {actual_start_frame + i} を読み込めませんでした")
            break

        writer.write(frame)
        written_frames += 1

        # 進捗表示
        if (i + 1) % 30 == 0 or i == total_output_frames - 1:
            progress = (i + 1) / total_output_frames * 100
            print(f"進捗: {progress:.1f}% ({i + 1}/{total_output_frames})")

    cap.release()
    writer.release()

    print(f"完了！切り出した動画を保存しました: {output_path}")
    print(f"出力フレーム数: {written_frames}")

    return output_path


def save_trimming_info(video_path, detected_frame_num, start_frame_rel, end_frame_rel, output_video_path):
    """
    動画切り出し情報をJSONファイルで保存する
    """
    # JSONファイル名を生成
    json_filename = f"{video_path.stem}_gopro_trimming_info.json"
    json_path = video_path.with_name(json_filename)

    # 動画情報を取得
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 実際のフレーム番号を計算
    actual_start_frame = int(detected_frame_num) + start_frame_rel
    actual_end_frame = int(detected_frame_num) + end_frame_rel

    # 時間情報を計算
    detected_time = detected_frame_num / fps if fps > 0 else 0
    start_time = actual_start_frame / fps if fps > 0 else 0
    end_time = actual_end_frame / fps if fps > 0 else 0
    trimmed_duration = (actual_end_frame - actual_start_frame + 1) / fps if fps > 0 else 0

    # JSON情報を作成
    trimming_info = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "original_video": str(video_path),
            "output_video": str(output_video_path) if output_video_path else None
        },
        "original_video_info": {
            "fps": fps,
            "total_frames": total_frames,
            "total_duration_seconds": total_duration,
            "width": width,
            "height": height
        },
        "detection_info": {
            "detected_frame_number": int(detected_frame_num),
            "detected_time_seconds": detected_time,
            "note": "LED light detection frame - used as reference point (frame 0) for trimming"
        },
        "trimming_settings": {
            "reference_frame": int(detected_frame_num),
            "start_frame_relative": start_frame_rel,
            "end_frame_relative": end_frame_rel,
            "start_frame_absolute": actual_start_frame,
            "end_frame_absolute": actual_end_frame,
            "trimmed_frame_count": actual_end_frame - actual_start_frame + 1
        },
        "timing_info": {
            "detected_time_seconds": detected_time,
            "start_time_seconds": start_time,
            "end_time_seconds": end_time,
            "trimmed_duration_seconds": trimmed_duration
        }
    }

    # JSONファイルに保存
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(trimming_info, f, indent=2, ensure_ascii=False)

        print(f"切り出し情報をJSONファイルに保存しました: {json_path}")
        print(f"基準フレーム: {int(detected_frame_num)}")
        print(f"切り出し範囲（相対）: {start_frame_rel} ～ {end_frame_rel}")
        print(f"切り出し範囲（絶対）: {actual_start_frame} ～ {actual_end_frame}")
        print(f"切り出し総フレーム数: {actual_end_frame - actual_start_frame + 1}")
        print(f"切り出し時間: {trimmed_duration:.2f}秒")

        return json_path

    except Exception as e:
        print(f"エラー: JSONファイルの保存に失敗しました: {e}")
        return None


def show_detection_frames(cap, detected_frame_num, roi_box, display_scale, total_frames, total_duration, video_path):
    """
    検出されたフレームとその前後3フレームずつを3x3の配置で表示する
    上段：前の3フレーム（-3, -2, -1）
    中段：左右は黒、中央に検出フレーム（0）
    下段：後の3フレーム（+1, +2, +3）
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
            else:
                roi_color = (0, 255, 0)  # 緑色
                border_color = (0, 255, 0)

            cv2.rectangle(frame_with_roi, (orig_x1, orig_y1), (orig_x2, orig_y2), roi_color, 5)

            # ROI内の平均R値を計算
            roi_area = frame[orig_y1:orig_y2, orig_x1:orig_x2]
            avg_r = 0
            if roi_area.size > 0:
                avg_b, avg_g, avg_r, _ = cv2.mean(roi_area)

            # R値の差分を計算（前のフレームとの比較）
            r_diff = 0.0
            if len(frame_info) > 0:
                prev_avg_r = frame_info[-1][2]  # 前のフレームのR値
                r_diff = avg_r - prev_avg_r

            frames.append(frame_with_roi)
            frame_info.append((frame_num, time_sec, avg_r, border_color, r_diff, offset))
        else:
            # フレームが読み込めない場合は黒い画像を作成
            if frames:
                black_frame = np.zeros_like(frames[0], dtype=np.uint8)
            else:
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frames.append(black_frame)
            frame_info.append((frame_num, time_sec, 0, (255, 255, 255), 0.0, offset))

    if len(frames) == 0:
        print("エラー: フレームを取得できませんでした。")
        return

    # 全フレームのサイズを統一するため、まず基準サイズを決定
    if frames:
        base_height, base_width = frames[0].shape[:2]
    else:
        base_height, base_width = 480, 640

    # 各フレームのサイズを設定（少し小さくする）
    frame_scale = 0.3  # 0.4から0.3に変更して小さくする
    target_height = int(base_height * frame_scale)
    target_width = int(base_width * frame_scale)

    # 枠の厚さ
    border_thickness = 4  # 少し小さくする

    # 3x3グリッドの配置を作成
    grid_frames = []

    # 上段：前の3フレーム（offset -3, -2, -1）
    top_row = []
    for i in range(3):
        frame = frames[i]
        frame_num, time_sec, avg_r, border_color, r_diff, offset = frame_info[i]

        processed_frame = create_frame_with_info(frame, frame_num, time_sec, avg_r, r_diff,
                                               border_color, border_thickness, target_width,
                                               target_height, base_width, base_height, False)
        top_row.append(processed_frame)

    # 中段：左右は黒、中央に検出フレーム（offset 0）
    middle_row = []
    # 左側：黒フレーム
    black_frame = create_black_frame_with_info(target_width, target_height, border_thickness)
    middle_row.append(black_frame)

    # 中央：検出フレーム
    frame = frames[3]  # offset 0のフレーム
    frame_num, time_sec, avg_r, border_color, r_diff, offset = frame_info[3]
    detected_frame = create_frame_with_info(frame, frame_num, time_sec, avg_r, r_diff,
                                          border_color, border_thickness, target_width,
                                          target_height, base_width, base_height, True)
    middle_row.append(detected_frame)

    # 右側：黒フレーム
    middle_row.append(black_frame.copy())

    # 下段：後の3フレーム（offset +1, +2, +3）
    bottom_row = []
    for i in range(4, 7):
        frame = frames[i]
        frame_num, time_sec, avg_r, border_color, r_diff, offset = frame_info[i]

        processed_frame = create_frame_with_info(frame, frame_num, time_sec, avg_r, r_diff,
                                               border_color, border_thickness, target_width,
                                               target_height, base_width, base_height, False)
        bottom_row.append(processed_frame)

    # 各行を結合
    top_combined = np.hstack(top_row)
    middle_combined = np.hstack(middle_row)
    bottom_combined = np.hstack(bottom_row)

    # 3行を縦に結合
    final_grid = np.vstack([top_combined, middle_combined, bottom_combined])

    # タイトルを追加（フォントサイズを大きくする）
    title_height = 140  # タイトル領域を少し大きくする
    title_frame = np.zeros((title_height, final_grid.shape[1], 3), dtype=np.uint8)

    # ★★★ Pathlibを使用してファイル名を取得 ★★★
    video_filename = video_path.name

    cv2.putText(title_frame, f"Detection: {video_filename} - Frame {int(detected_frame_num)}",
               (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)  # ファイル名を表示
    cv2.putText(title_frame, f"Red border = Detected frame  |  FPS: {fps:.1f}",
               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    cv2.putText(title_frame, f"Top: Frames {frame_info[0][0]}-{frame_info[2][0]} | Center: Frame {frame_info[3][0]} | Bottom: Frames {frame_info[4][0]}-{frame_info[6][0]} | Press any key to continue",
               (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    final_display = np.vstack([title_frame, final_grid])

    # ウィンドウサイズを調整して画面に収まるようにする
    screen_width = 1920
    screen_height = 1080
    if final_display.shape[1] > screen_width or final_display.shape[0] > screen_height:
        scale_w = screen_width / final_display.shape[1]
        scale_h = screen_height / final_display.shape[0]
        scale_factor = min(scale_w, scale_h) * 0.9  # 少し余裕を持たせる
        new_width = int(final_display.shape[1] * scale_factor)
        new_height = int(final_display.shape[0] * scale_factor)
        final_display = cv2.resize(final_display, (new_width, new_height))

    # ウィンドウ名を定義
    window_name = "Detection Frames (3x3 Grid View)"

    # cv2.imshow(window_name, final_display)
    print(f"\n=== 検出確認 ===")
    print(f"動画ファイル: {video_filename}")
    print(f"検出フレーム: {int(detected_frame_num)}")
    print(f"検出時間: {detected_frame_num/fps:.2f}秒")
    print(f"上段フレーム範囲: {frame_info[0][0]} ～ {frame_info[2][0]}")
    print(f"検出フレーム: {frame_info[3][0]}")
    print(f"下段フレーム範囲: {frame_info[4][0]} ～ {frame_info[6][0]}")
    print(f"動画情報: FPS={fps:.1f}, 総フレーム数={total_frames}, 総時間={total_duration:.1f}秒")
    print("何かキーを押すと動画切り出し設定に進みます。")

    output_filename = f"{video_filename}_LED_detection_frame_{int(detected_frame_num)}_time_{detected_frame_num/fps:.2f}s.jpg"
    output_path = video_path.parent / output_filename
    # Pathオブジェクトを文字列に変換してcv2.imwriteに渡す
    cv2.imwrite(str(output_path), final_display)

    # キー入力を待つ
    cv2.waitKey(0)

    # ★★★ ウィンドウが存在するかチェックしてから削除 ★★★
    try:
        # ウィンドウが存在するかチェック
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(window_name)
    except cv2.error:
        # ウィンドウが既に閉じられている場合はエラーを無視
        pass


def create_frame_with_info(frame, frame_num, time_sec, avg_r, r_diff, border_color, border_thickness,
                          target_width, target_height, base_width, base_height, is_detected):
    """
    フレームに情報テキストを追加して処理する
    """
    # フレームサイズを基準サイズに統一
    if frame.shape[:2] != (base_height, base_width):
        frame = cv2.resize(frame, (base_width, base_height))

    # フレームをターゲットサイズにリサイズ
    resized_frame = cv2.resize(frame, (target_width, target_height))

    # 枠を追加
    frame_with_border = cv2.copyMakeBorder(resized_frame, border_thickness, border_thickness,
                                          border_thickness, border_thickness,
                                          cv2.BORDER_CONSTANT, value=border_color)

    # テキスト用の領域を下に追加（高さを増やす）
    text_height = 140  # 120から140に増加
    border_width = target_width + border_thickness * 2
    text_area = np.zeros((text_height, border_width, 3), dtype=np.uint8)

    # フレーム番号と情報を表示（フォントサイズを大きくする）
    cv2.putText(text_area, f"Frame: {frame_num}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)  # 0.7→1.0, 太さ2→3
    cv2.putText(text_area, f"Time: {time_sec:.2f}s", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)  # 0.6→0.9, 太さ2→3
    cv2.putText(text_area, f"R: {avg_r:.0f}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)  # 0.6→0.9, 太さ2→3
    cv2.putText(text_area, f"R-diff: {r_diff:+.1f}", (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 128, 255), 3)  # 0.6→0.9, 太さ2→3

    # 検出フレームの場合は強調表示
    if is_detected:
        cv2.putText(text_area, "DETECTED", (border_width//2-60, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 4)  # 0.6→0.8, 太さ3→4

    # フレームとテキストを結合
    final_frame = np.vstack([frame_with_border, text_area])
    return final_frame


def create_black_frame_with_info(target_width, target_height, border_thickness):
    """
    黒いフレームを作成する
    """
    # 黒いフレームを作成
    black_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # 枠を追加（グレー）
    frame_with_border = cv2.copyMakeBorder(black_frame, border_thickness, border_thickness,
                                          border_thickness, border_thickness,
                                          cv2.BORDER_CONSTANT, value=(64, 64, 64))

    # テキスト用の領域を下に追加
    text_height = 140  # 120から140に増加
    border_width = target_width + border_thickness * 2
    text_area = np.zeros((text_height, border_width, 3), dtype=np.uint8)

    # 空白表示（フォントサイズを大きくする）
    cv2.putText(text_area, "---", (border_width//2-40, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 128, 128), 3)  # 1.0→1.5, 太さ2→3

    # フレームとテキストを結合
    final_frame = np.vstack([frame_with_border, text_area])
    return final_frame


def main():
    """
    メイン処理
    """
    VIDEO_PATH = Path(r"G:\gait_pattern\20250915_synctest\GoPro\6.MP4")

    # ★★★ 既にtrimedファイルが存在するかチェック ★★★
    parent_dir = VIDEO_PATH.parent
    existing_trimed_files = list(parent_dir.glob("trimed*.mp4"))

    if existing_trimed_files:
        print(f"既存のtrimedファイルが見つかりました: {[f.name for f in existing_trimed_files]}")
        print(f"ディレクトリ: {parent_dir}")
        print("このディレクトリの処理をスキップします。\n")
        return

    print(f"動画ファイル: {VIDEO_PATH}")

    global roi_box
    roi_box = None

    # --- 設定項目 ---
    DISPLAY_SCALE = 0.25

    # ★★★ R値の増加量のしきい値 ★★★
    # 前のフレームと比較して、Rの平均値がこの値以上増加したら「点灯」とみなす
    R_INCREASE_THRESH = 10.0

    # ★★★ ファイルの存在チェック ★★★
    if not VIDEO_PATH.exists():
        print(f"エラー: 動画ファイルが存在しません: {VIDEO_PATH}")
        return
    
    # ★★★ ファイル情報を表示 ★★★
    print(f"--- 動画情報 ---")
    print(f"ファイル名: {VIDEO_PATH.name}")
    print(f"ファイルパス: {VIDEO_PATH}")
    print(f"親ディレクトリ: {VIDEO_PATH.parent}")
    print(f"拡張子: {VIDEO_PATH.suffix}")
    print(f"ファイルサイズ: {VIDEO_PATH.stat().st_size / (1024**2):.1f} MB")

    cap = cv2.VideoCapture(str(VIDEO_PATH))  # Pathオブジェクトを文字列に変換
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
    print("'q' を押すか、検出が完了すると自動的に終了します。")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 動画の総フレーム数とFPSを取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / fps if fps > 0 else 0

    # ★★★ 前のフレームのR値を保存する変数を初期化 ★★★
    prev_avg_r = -1.0
    detected_frame_num = None  # 検出されたフレーム番号を保存
    processing_window_exists = True

    while cap.isOpened() and processing_window_exists:
        # ★★★ フレーム読み込み前に現在のフレーム番号を取得 ★★★
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        ret, frame = cap.read()
        if not ret:
            break

        # 時間を計算
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
                print(f"\n★ 検出完了！フレーム番号: {current_frame}, R値増加: {r_diff:.1f}")

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

        # ★★★ 検出完了の場合は状況を表示 ★★★
        if detected_frame_num is not None:
            cv2.putText(display_frame, "DETECTION COMPLETED!", (10, display_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        cv2.imshow("Processing Video (R-Increase)", display_frame)

        # ★★★ 検出完了の場合、自動的に処理ウィンドウを閉じる ★★★
        if detected_frame_num is not None:
            # 検出完了後、少し待って自動的にウィンドウを閉じる
            cv2.waitKey(1500)  # 1.5秒待機して結果を表示
            cv2.destroyWindow("Processing Video (R-Increase)")
            processing_window_exists = False
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # qが押されたら処理ウィンドウを閉じる
            cv2.destroyWindow("Processing Video (R-Increase)")
            processing_window_exists = False
            break

        # ウィンドウが閉じられたかチェック
        if cv2.getWindowProperty("Processing Video (R-Increase)", cv2.WND_PROP_VISIBLE) < 1:
            processing_window_exists = False
            break

    print("\n処理が完了しました。")

    # ★★★ 検出されたフレームがある場合、前後のフレームを表示し、動画切り出しを実行 ★★★
    if detected_frame_num is not None:
        print(f"R値が増加したフレーム番号: {int(detected_frame_num)}")

        # 検出フレーム確認画面を表示
        show_detection_frames(cap, detected_frame_num, roi_box, DISPLAY_SCALE, total_frames, total_duration, VIDEO_PATH)

        # 動画切り出し範囲を選択
        start_frame_rel, end_frame_rel = select_video_range(cap, detected_frame_num, VIDEO_PATH)

        if start_frame_rel is not None and end_frame_rel is not None:
            # 動画を切り出し
            output_video_path = clip_video_segment(VIDEO_PATH, detected_frame_num, start_frame_rel, end_frame_rel)

            # JSON情報を保存
            if output_video_path:
                save_trimming_info(VIDEO_PATH, detected_frame_num, start_frame_rel, end_frame_rel, output_video_path)

            print(f"\n=== 処理完了 ===")
            print(f"検出フレーム: {int(detected_frame_num)}")
            print(f"切り出し動画: {output_video_path}")
        else:
            print("動画切り出しがキャンセルされました。")
    else:
        print("検出されたフレームはありませんでした。")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()