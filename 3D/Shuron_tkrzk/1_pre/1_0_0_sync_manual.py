#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEDの点灯フレーム（基準フレーム）を「目視で」選び、そのフレームを0として動画を切り出すスクリプト。

変更点（元スクリプトから）:
- 赤色閾値（R差分）による自動検出を廃止（基準フレームは必ず手動選択）
- ROI選択を廃止（ROIキャッシュも不要）
- 検出確認用の3x3表示は「前後フレームの目視確認」用途に簡略化（R値表示なし）

操作（基準フレーム選択）:
- ←/→ : 1フレーム移動
- Space: 再生/停止
- Enter: このフレームを基準フレームとして確定
- q    : キャンセル

操作（切り出し範囲選択）:
- 既存JSON(trimming_info.json)があれば自動適用
- なければ、基準フレームを0として開始→終了をEnterで指定
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime


def select_reference_frame(cap, video_path, start_frame=0):
    """
    目視で「基準フレーム（LED点灯など）」を1つ選ぶ。
    返り値: (reference_frame_abs or None)
    """
    print("\n=== 基準フレーム（LED点灯フレーム）を目視で選択 ===")
    print("←/→: フレーム移動 | Space: 再生/停止 | Enter: 確定 | 'q': キャンセル")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 開始位置へ
    start_frame = max(0, min(int(start_frame), total_frames - 1))
    current_frame = start_frame
    is_playing = False
    need_frame_update = True

    window_name = "Reference Frame Selector"
    cv2.namedWindow(window_name)

    def on_trackbar(val):
        nonlocal current_frame, is_playing, need_frame_update
        current_frame = val
        is_playing = False
        need_frame_update = True

    cv2.createTrackbar("Position", window_name, current_frame, max(0, total_frames - 1), on_trackbar)

    while True:
        if is_playing:
            current_frame += 1
            if current_frame >= total_frames:
                current_frame = total_frames - 1
                is_playing = False
            need_frame_update = True

        if need_frame_update:
            cv2.setTrackbarPos("Position", window_name, current_frame)

            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                current_frame = max(0, current_frame - 1)
                is_playing = False
                need_frame_update = False
                continue

            t = current_frame / fps if fps > 0 else 0.0

            overlay = frame.copy()
            margin = 20
            overlay_height = 160
            cv2.rectangle(overlay, (margin, 10), (frame.shape[1] - margin, overlay_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            cv2.putText(frame, f"File: {video_path.name}", (margin + 10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {current_frame} / {total_frames-1}", (margin + 10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.putText(frame, f"Time: {t:.2f}s", (margin + 10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

            # 表示リサイズ（画面が小さい環境向け）
            display = cv2.resize(frame, (1280, 720))
            cv2.imshow(window_name, display)

            need_frame_update = False

        wait_time = 1 if is_playing else 30
        key = cv2.waitKeyEx(wait_time)

        if key == -1:
            continue

        if key == ord('q'):
            print("基準フレーム選択をキャンセルしました。")
            cv2.destroyWindow(window_name)
            return None

        if key == 13:  # Enter
            print(f"基準フレームを確定しました: {current_frame}")
            cv2.destroyWindow(window_name)
            return int(current_frame)

        if key == ord(' '):
            is_playing = not is_playing
            need_frame_update = True

        elif key == 2424832:  # ←
            is_playing = False
            current_frame = max(0, current_frame - 1)
            need_frame_update = True

        elif key == 2555904:  # →
            is_playing = False
            current_frame = min(total_frames - 1, current_frame + 1)
            need_frame_update = True

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("ウィンドウが閉じられたため中断します。")
            break

    cv2.destroyWindow(window_name)
    return None


def select_video_range(cap, reference_frame_abs, video_path, tpose_flag=False):
    """
    reference_frame_abs を0フレーム目として開始・終了フレーム（相対）を選択し、動画を切り出す。
    既存JSONファイルがある場合は、相対開始・終了フレームを自動適用。
    tposeの場合は、開始を0、終了を59フレーム（1秒）に固定。
    """
    json_path = video_path.parent.with_name("trimming_info.json")
    print(f"JSONファイルパス: {json_path}")

    if tpose_flag:
        print("\nTポーズモードが有効です。開始フレームを0、終了フレームを59に固定します。")
        return 0, 59

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            trimming_settings = existing_data.get("trimming_settings", {})
            start_rel = trimming_settings.get("start_frame_relative", None)
            end_rel = trimming_settings.get("end_frame_relative", None)
            if start_rel is not None and end_rel is not None:
                print(f"\n既存のJSONファイルを検出: {json_path}")
                print(f"既存の相対開始フレーム: {start_rel}")
                print(f"既存の相対終了フレーム: {end_rel}")
                return int(start_rel), int(end_rel)
        except Exception as e:
            print(f"既存JSONファイルの読み込みに失敗: {e}")

    print("\n=== 動画切り出し範囲選択 ===")
    print("基準フレームを0として、その後の範囲を選択してください")
    print("矢印キー: フレーム移動 | Space: 再生/停止 | Enter: 決定 | 'q': 終了")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    available_frames = total_frames - int(reference_frame_abs)
    if available_frames <= 0:
        print("エラー: 基準フレーム以降にフレームが存在しません")
        return None, None

    current_rel = 0
    is_playing = False
    start_rel = -1
    end_rel = -1
    selection_mode = "start"
    need_frame_update = True

    window_name = "Video Range Selector"
    cv2.namedWindow(window_name)

    def on_trackbar(val):
        nonlocal current_rel, is_playing, need_frame_update
        current_rel = val
        is_playing = False
        need_frame_update = True

    cv2.createTrackbar("Position", window_name, 0, available_frames - 1, on_trackbar)

    while True:
        if is_playing:
            current_rel += 1
            if current_rel >= available_frames:
                current_rel = available_frames - 1
                is_playing = False
            need_frame_update = True

        if need_frame_update:
            cv2.setTrackbarPos("Position", window_name, current_rel)

            actual_frame = int(reference_frame_abs) + current_rel

            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
            ret, frame = cap.read()
            if not ret:
                current_rel = max(0, current_rel - 1)
                is_playing = False
                need_frame_update = False
                continue

            current_time = actual_frame / fps if fps > 0 else 0
            rel_time = current_rel / fps if fps > 0 else 0

            overlay = frame.copy()
            margin = 20
            overlay_height = 250
            cv2.rectangle(overlay, (margin, 10), (frame.shape[1] - margin, overlay_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            mode_text = f"Selecting: {'START frame' if selection_mode == 'start' else 'END frame'}"
            cv2.putText(frame, mode_text, (margin + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv2.putText(frame, f"Relative Frame: {current_rel} / Actual Frame: {actual_frame}",
                        (margin + 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, f"Relative Time: {rel_time:.2f}s / Actual Time: {current_time:.2f}s",
                        (margin + 10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame,
                        f"Start: {start_rel if start_rel >= 0 else 'Not Set'} | End: {end_rel if end_rel >= 0 else 'Not Set'}",
                        (margin + 10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, "Arrow Keys: Move | Space: Play/Pause | Enter: Confirm | 'q': Quit",
                        (margin + 10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

            if start_rel >= 0 and current_rel == start_rel:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 8)
                cv2.putText(frame, "START FRAME", (50, frame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

            if end_rel >= 0 and current_rel == end_rel:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 8)
                cv2.putText(frame, "END FRAME", (50, frame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

            display = cv2.resize(frame, (1280, 720))
            cv2.imshow(window_name, display)
            need_frame_update = False

        wait_time = 1 if is_playing else 30
        key = cv2.waitKeyEx(wait_time)

        if key == -1:
            continue

        if key == ord('q'):
            print("範囲選択を中断しました")
            cv2.destroyWindow(window_name)
            return None, None

        if key == 13:  # Enter
            if selection_mode == "start":
                start_rel = current_rel
                selection_mode = "end"
                print(f"開始フレーム（相対）: {start_rel} を設定しました。次に終了フレームを選択してください。")
                need_frame_update = True
            else:
                if current_rel <= start_rel:
                    print("エラー: 終了フレームは開始フレームより後に設定してください")
                    continue
                end_rel = current_rel
                duration_frames = end_rel - start_rel + 1
                duration_seconds = duration_frames / fps if fps > 0 else 0
                print(f"終了フレーム（相対）: {end_rel}")
                print(f"切り出し範囲: {start_rel} ～ {end_rel} ({duration_frames}フレーム, {duration_seconds:.2f}秒)")
                cv2.destroyWindow(window_name)
                return int(start_rel), int(end_rel)

        if key == ord(' '):
            is_playing = not is_playing
            need_frame_update = True

        elif key == 2424832:  # ←
            is_playing = False
            current_rel = max(0, current_rel - 1)
            need_frame_update = True

        elif key == 2555904:  # →
            is_playing = False
            current_rel = min(available_frames - 1, current_rel + 1)
            need_frame_update = True

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow(window_name)
    return None, None


def clip_video_segment(video_path, reference_frame_abs, start_frame_rel, end_frame_rel):
    print("\n=== 動画切り出し処理 ===")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けませんでした: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    actual_start_frame = int(reference_frame_abs) + int(start_frame_rel)
    actual_end_frame = int(reference_frame_abs) + int(end_frame_rel)
    total_output_frames = actual_end_frame - actual_start_frame + 1

    output_path = video_path.parent / "trimed.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"エラー: 出力ファイルを作成できませんでした: {output_path}")
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start_frame)
    print(f"動画切り出し中... ({total_output_frames}フレーム)")

    written_frames = 0
    for i in range(total_output_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"警告: フレーム {actual_start_frame + i} を読み込めませんでした")
            break
        writer.write(frame)
        written_frames += 1

        if (i + 1) % 30 == 0 or i == total_output_frames - 1:
            progress = (i + 1) / total_output_frames * 100
            print(f"進捗: {progress:.1f}% ({i + 1}/{total_output_frames})")

    cap.release()
    writer.release()

    print(f"完了！切り出した動画を保存しました: {output_path}")
    print(f"出力フレーム数: {written_frames}")
    return output_path


def save_trimming_info(video_path, reference_frame_abs, start_frame_rel, end_frame_rel, output_video_path):
    json_path = video_path.parent.with_name("trimming_info.json")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    actual_start_frame = int(reference_frame_abs) + int(start_frame_rel)
    actual_end_frame = int(reference_frame_abs) + int(end_frame_rel)

    reference_time = reference_frame_abs / fps if fps > 0 else 0
    start_time = actual_start_frame / fps if fps > 0 else 0
    end_time = actual_end_frame / fps if fps > 0 else 0
    trimmed_duration = (actual_end_frame - actual_start_frame + 1) / fps if fps > 0 else 0

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
        "reference_info": {
            "reference_frame_number": int(reference_frame_abs),
            "reference_time_seconds": reference_time,
            "note": "Manually selected reference frame (LED ON etc.) used as frame 0 for trimming"
        },
        "trimming_settings": {
            "reference_frame": int(reference_frame_abs),
            "start_frame_relative": int(start_frame_rel),
            "end_frame_relative": int(end_frame_rel),
            "start_frame_absolute": int(actual_start_frame),
            "end_frame_absolute": int(actual_end_frame),
            "trimmed_frame_count": int(actual_end_frame - actual_start_frame + 1)
        },
        "timing_info": {
            "reference_time_seconds": reference_time,
            "start_time_seconds": start_time,
            "end_time_seconds": end_time,
            "trimmed_duration_seconds": trimmed_duration
        }
    }

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(trimming_info, f, indent=2, ensure_ascii=False)

        print(f"切り出し情報をJSONファイルに保存しました: {json_path}")
        print(f"基準フレーム: {int(reference_frame_abs)}")
        print(f"切り出し範囲（相対）: {start_frame_rel} ～ {end_frame_rel}")
        print(f"切り出し範囲（絶対）: {actual_start_frame} ～ {actual_end_frame}")
        return json_path
    except Exception as e:
        print(f"エラー: JSONファイルの保存に失敗しました: {e}")
        return None


def show_reference_context_grid(cap, reference_frame_abs, video_path):
    """
    基準フレームとその前後3フレーム（計7枚）を3x3で保存して、目視確認しやすくする。
    （中央が基準フレーム。左右は黒で埋める。）
    """
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    info = []
    for offset in range(-3, 4):
        f = int(reference_frame_abs) + offset
        f = max(0, f)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        t = f / fps if fps > 0 else 0.0

        if ret:
            frames.append(frame)
        else:
            # 失敗時は黒
            if frames:
                frames.append(np.zeros_like(frames[0]))
            else:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
        info.append((f, t, offset))
        
        trim_trame_dir = video_path.parent / "LED_reference_check"
        trim_trame_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(trim_trame_dir / f"frame_{f}_offset_{offset}.jpg"), frames[-1])

    base_h, base_w = frames[0].shape[:2]
    scale = 0.3
    th, tw = int(base_h * scale), int(base_w * scale)
    border = 4

    def add_border_and_text(frm, fnum, tsec, is_ref):
        frm = cv2.resize(frm, (tw, th))
        color = (0, 0, 255) if is_ref else (0, 255, 0)
        frm = cv2.copyMakeBorder(frm, border, border, border, border, cv2.BORDER_CONSTANT, value=color)
        text_h = 90
        text = np.zeros((text_h, frm.shape[1], 3), dtype=np.uint8)
        cv2.putText(text, f"Frame: {fnum}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(text, f"Time: {tsec:.2f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if is_ref:
            cv2.putText(text, "REFERENCE", (frm.shape[1]//2 - 85, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return np.vstack([frm, text])

    def black_cell():
        frm = np.zeros((th, tw, 3), dtype=np.uint8)
        frm = cv2.copyMakeBorder(frm, border, border, border, border, cv2.BORDER_CONSTANT, value=(64, 64, 64))
        text = np.zeros((90, frm.shape[1], 3), dtype=np.uint8)
        cv2.putText(text, "---", (frm.shape[1]//2 - 25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 128, 128), 2)
        return np.vstack([frm, text])

    cells = []
    for (fnum, tsec, offset), frm in zip(info, frames):
        cells.append(add_border_and_text(frm, fnum, tsec, offset == 0))

    top = np.hstack(cells[0:3])
    mid = np.hstack([black_cell(), cells[3], black_cell()])
    bot = np.hstack(cells[4:7])

    grid = np.vstack([top, mid, bot])

    title_h = 100
    title = np.zeros((title_h, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(title, f"Reference Check: {video_path.name}  |  Frame {int(reference_frame_abs)}",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

    final = np.vstack([title, grid])

    out_path = video_path.parent / "LED_reference_check.jpg"
    cv2.imwrite(str(out_path), final)
    print(f"基準フレーム周辺の確認画像を保存しました: {out_path}")


def main():
    root_dir = Path(r"G:\gait_pattern\2025_shuron_tkrzk")

    for sub_id in range(2, 4): #被験者2,3
        print(f"\n########## 被験者{sub_id} ##########")
        sub_dir = root_dir / f"sub{sub_id}"

        session_dirs = [d for d in sub_dir.iterdir() if d.is_dir()]
        session_dirs.sort()
        session_dirs = [d for d in session_dirs if "cali" not in d.name.lower()]
        print(f"セッションディレクトリ数: {len(session_dirs)}")

        for session_dir in session_dirs:
            print(f"\n---- セッションディレクトリ: {session_dir.name} ----")
            tpose_flag = "thera0-0" in session_dir.name.lower()

            # 手動でも、最初は少し飛ばしたい場合があるので残す（不要なら 0 に）
            skip_frame = 100
            if sub_id == 1 and "thera0-0_0" in session_dir.name.lower():
                skip_frame = 2400
            elif sub_id == 1 and "thera0-0_1" in session_dir.name.lower():
                skip_frame = 1200

            # 方向ごとに処理
            reference_frame_for_other_views = None

            for direction in ["fl", "fr", "sagi"]:
                video_dir = session_dir / "gopro" / direction
                video_list = list(video_dir.glob("GX*.MP4"))
                if not video_list:
                    print(f"[WARN] 動画が見つかりません: {video_dir}")
                    continue
                video_path = video_list[0]

                parent_dir = video_path.parent
                if (parent_dir / "trimed.mp4").exists():
                    print(f"\n=== {direction}方向をスキップ ===")
                    print(f"既存のtrimed.mp4が見つかりました: {parent_dir}")
                    continue

                print(f"\n=== {direction}方向の処理を開始 ===")
                print(f"動画ファイル: {video_path}")

                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"エラー: 動画ファイルを開けませんでした: {video_path}")
                    continue

                # 基準フレーム選択開始位置
                # - 1本目は skip_frame
                # - 2本目以降は 1本目で選んだ基準の少し前へジャンプ（目視の手間削減）
                start_for_reference = skip_frame
                if reference_frame_for_other_views is not None:
                    start_for_reference = max(0, reference_frame_for_other_views - 120)

                reference_frame_abs = select_reference_frame(cap, video_path, start_frame=start_for_reference)
                if reference_frame_abs is None:
                    cap.release()
                    cv2.destroyAllWindows()
                    print("基準フレームが未設定のため、この動画をスキップします。")
                    continue

                # 他視点の開始ジャンプ用に保存
                if reference_frame_for_other_views is None:
                    reference_frame_for_other_views = reference_frame_abs

                # 周辺フレームの確認画像を保存（任意）
                show_reference_context_grid(cap, reference_frame_abs, video_path)

                # 切り出し範囲を選択
                start_rel, end_rel = select_video_range(cap, reference_frame_abs, video_path, tpose_flag=tpose_flag)
                cap.release()
                cv2.destroyAllWindows()

                if start_rel is None or end_rel is None:
                    print("動画切り出しがキャンセルされました。")
                    continue

                # 切り出し
                out_path = clip_video_segment(video_path, reference_frame_abs, start_rel, end_rel)

                # JSON保存
                if out_path:
                    save_trimming_info(video_path, reference_frame_abs, start_rel, end_rel, out_path)

                print(f"\n=== 完了 ({direction}) ===")
                print(f"基準フレーム: {reference_frame_abs}")
                print(f"切り出し動画: {out_path}")

    print("\n全処理が完了しました。")


if __name__ == "__main__":
    main()
