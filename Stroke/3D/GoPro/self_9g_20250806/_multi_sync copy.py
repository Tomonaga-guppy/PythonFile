import cv2
import numpy as np
import os
from tqdm import tqdm

"""
3つのタイミングを順番に設定する動画切り出しツール
f0: 基準点 (Reference Point)
f1: 開始点 (Start Point)
f2: 終了点 (End Point)
"""

# --- 設定項目 ---
# ユーザー指定のファイルパスを維持
INPUT_VIDEO_PATH = r"G:\gait_pattern\20250807_br\Tpose\fr\GX010183.MP4"
OUTPUT_VIDEO_PATH = r"G:\gait_pattern\20250807_br\Tpose\fr\trim.mp4"

class TimingSelector:
    def __init__(self):
        self.f0_reference = -1  # 基準点
        self.f1_start = -1      # 開始点
        self.f2_end = -1        # 終了点

    def play_and_select_timing(self):
        """3つのタイミングを順番に選択する"""
        cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
        if not cap.isOpened():
            print(f"エラー: 動画ファイル '{INPUT_VIDEO_PATH}' を開けませんでした。")
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame_pos = 0
        is_playing = False
        window_name = "Video Frame Navigator - Timing Selector"

        # 現在選択中のポイント (0: f0, 1: f1, 2: f2)
        current_selection_stage = 0
        stage_names = ["f0 (基準点)", "f1 (開始点)", "f2 (終了点)"]

        print("\n--- 3点タイミング選択モード ---")
        print("順番: f0(基準点) → f1(開始点) → f2(終了点)")
        print("再生バー: マウスでドラッグして移動")
        print("右矢印キー: 1フレーム進む")
        print("左矢印キー: 1フレーム戻る")
        print("スペースキー: 再生/一時停止")
        print("'s' キー: 現在のフレームを設定")
        print("'r' キー: 現在の選択段階をリセット")
        print("'q' キー: 終了")

        cv2.namedWindow(window_name)

        def on_trackbar(val):
            nonlocal current_frame_pos, is_playing
            current_frame_pos = val
            is_playing = False

        cv2.createTrackbar("Seek", window_name, 0, total_frames - 1, on_trackbar)

        while current_selection_stage < 3:
            if is_playing:
                current_frame_pos += 1
                if current_frame_pos >= total_frames:
                    current_frame_pos = total_frames - 1
                    is_playing = False

            cv2.setTrackbarPos("Seek", window_name, current_frame_pos)

            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
            ret, frame = cap.read()
            if not ret:
                current_frame_pos = max(0, current_frame_pos - 1)
                is_playing = False
                continue

            # UIオーバーレイの描画
            self._draw_ui_overlay(frame, current_frame_pos, total_frames,
                                current_selection_stage, stage_names)

            frame = cv2.resize(frame, (1280, 720))
            cv2.imshow(window_name, frame)

            wait_time = 33 if is_playing else 0
            key = cv2.waitKeyEx(wait_time)

            if key == -1:
                continue

            if key == ord('q'):
                print("\n選択を中断しました。")
                cap.release()
                cv2.destroyAllWindows()
                return False

            elif key == ord('s'):
                self._set_current_point(current_frame_pos, current_selection_stage)
                current_selection_stage += 1

            elif key == ord('r'):
                self._reset_current_stage(current_selection_stage)

            elif key == ord(' '):
                is_playing = not is_playing

            elif key == 2424832:  # 左矢印キー
                is_playing = False
                current_frame_pos = max(0, current_frame_pos - 1)

            elif key == 2555904:  # 右矢印キー
                is_playing = False
                current_frame_pos = min(total_frames - 1, current_frame_pos + 1)

        cap.release()
        cv2.destroyAllWindows()

        # 最終確認画面
        self._show_final_confirmation()
        return True

    def _draw_ui_overlay(self, frame, current_frame, total_frames, stage, stage_names):
        """UIオーバーレイを描画"""
        font_scale_large = 1.2
        font_scale_medium = 1.0
        font_scale_small = 0.8
        font_thickness = 2

        # 半透明の背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (1250, 180), (0, 0, 0), -1)
        alpha = 0.7
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)[:]

        # 現在のフレーム情報
        info_text = f"Frame: {current_frame}/{total_frames-1}"
        cv2.putText(frame, info_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale_large, (0, 255, 0), font_thickness)

        # 現在の選択段階
        stage_text = f"現在の選択: {stage_names[stage]}"
        cv2.putText(frame, stage_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale_medium, (0, 255, 255), 2)

        # 設定済みポイントの表示
        points_text = f"f0: {self.f0_reference if self.f0_reference >= 0 else '未設定'} | " \
                     f"f1: {self.f1_start if self.f1_start >= 0 else '未設定'} | " \
                     f"f2: {self.f2_end if self.f2_end >= 0 else '未設定'}"
        cv2.putText(frame, points_text, (20, 115), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale_medium, (255, 255, 0), 2)

        # 操作説明
        controls_text = "Seekbar | <- -> | Space: Play/Pause | 's': Set | 'r': Reset | 'q': Quit"
        cv2.putText(frame, controls_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale_small, (255, 255, 255), 1)

    def _set_current_point(self, frame_pos, stage):
        """現在の段階に応じてポイントを設定"""
        if stage == 0:  # f0設定
            self.f0_reference = frame_pos
            print(f"\nf0 (基準点) を設定しました: フレーム {frame_pos}")
        elif stage == 1:  # f1設定
            self.f1_start = frame_pos
            print(f"f1 (開始点) を設定しました: フレーム {frame_pos}")
        elif stage == 2:  # f2設定
            self.f2_end = frame_pos
            print(f"f2 (終了点) を設定しました: フレーム {frame_pos}")

    def _reset_current_stage(self, stage):
        """現在の段階をリセット"""
        if stage == 0:
            self.f0_reference = -1
            print("\nf0 (基準点) をリセットしました")
        elif stage == 1:
            self.f1_start = -1
            print("f1 (開始点) をリセットしました")
        elif stage == 2:
            self.f2_end = -1
            print("f2 (終了点) をリセットしました")

    def _show_final_confirmation(self):
        """最終確認を表示"""
        print("\n" + "="*50)
        print("設定完了！以下のタイミングが設定されました:")
        print("="*50)
        print(f"f0 (基準点): フレーム {self.f0_reference}")
        print(f"f1 (開始点): フレーム {self.f1_start}")
        print(f"f2 (終了点): フレーム {self.f2_end}")
        print("="*50)

        # フレーム間の関係も表示
        if self.f0_reference >= 0 and self.f1_start >= 0:
            print(f"f0 → f1: {self.f1_start - self.f0_reference} フレーム")
        if self.f1_start >= 0 and self.f2_end >= 0:
            print(f"f1 → f2: {self.f2_end - self.f1_start} フレーム (切り出し長)")
        if self.f0_reference >= 0 and self.f2_end >= 0:
            print(f"f0 → f2: {self.f2_end - self.f0_reference} フレーム")
        print()

    def get_timing_points(self):
        """設定されたタイミングポイントを取得"""
        return self.f0_reference, self.f1_start, self.f2_end

    def is_valid(self):
        """設定が有効かチェック"""
        return (self.f0_reference >= 0 and
                self.f1_start >= 0 and
                self.f2_end >= 0 and
                self.f1_start <= self.f2_end)


def clip_video_from_timing(f0, f1, f2):
    """設定されたタイミングに基づいて動画を切り出す"""
    if f1 < 0 or f2 < 0 or f1 > f2:
        print("エラー: 無効なタイミング設定です。")
        return False

    print(f"\n--- 動画の切り出し開始 (フレーム {f1} ～ {f2}) ---")

    cap_orig = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap_orig.isOpened():
        print(f"エラー: 動画ファイル '{INPUT_VIDEO_PATH}' を開けませんでした。")
        return False

    width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_orig.get(cv2.CAP_PROP_FPS)

    # 書き込むべき総フレーム数を計算
    frames_to_write = f2 - f1 + 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"エラー: 出力ファイル '{OUTPUT_VIDEO_PATH}' を開けませんでした。")
        cap_orig.release()
        return False

    print(f"フレーム {f1} から {f2} まで ({frames_to_write} フレーム) を書き込み中...")

    # 開始フレームに移動
    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, f1)

    # tqdmを使ってプログレスバーを表示
    for _ in tqdm(range(frames_to_write), desc="Writing Video", unit="frame"):
        ret, frame = cap_orig.read()
        if not ret:
            print("\n警告: 動画の途中でフレームが読み込めなくなりました。")
            break
        writer.write(frame)

    cap_orig.release()
    writer.release()

    # 切り出し結果の保存
    save_timing_info(f0, f1, f2, fps)

    print(f"\n完了！切り出した動画を '{OUTPUT_VIDEO_PATH}' に保存しました。")
    return True


def save_timing_info(f0, f1, f2, fps):
    """タイミング情報をテキストファイルに保存"""
    info_file = OUTPUT_VIDEO_PATH.replace('.mp4', '_timing_info.txt')

    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("=== 動画切り出しタイミング情報 ===\n")
        f.write(f"入力動画: {INPUT_VIDEO_PATH}\n")
        f.write(f"出力動画: {OUTPUT_VIDEO_PATH}\n")
        f.write(f"FPS: {fps:.2f}\n\n")

        f.write("タイミング設定 (フレーム番号):\n")
        f.write(f"f0 (基準点): {f0}\n")
        f.write(f"f1 (開始点): {f1}\n")
        f.write(f"f2 (終了点): {f2}\n\n")

        f.write("時間換算 (秒):\n")
        f.write(f"f0 (基準点): {f0/fps:.3f}s\n")
        f.write(f"f1 (開始点): {f1/fps:.3f}s\n")
        f.write(f"f2 (終了点): {f2/fps:.3f}s\n\n")

        f.write("フレーム間隔:\n")
        f.write(f"f0 → f1: {f1-f0} フレーム ({(f1-f0)/fps:.3f}s)\n")
        f.write(f"f1 → f2: {f2-f1} フレーム ({(f2-f1)/fps:.3f}s)\n")
        f.write(f"f0 → f2: {f2-f0} フレーム ({(f2-f0)/fps:.3f}s)\n")

        f.write(f"\n切り出し動画長: {(f2-f1+1)/fps:.3f}s ({f2-f1+1} フレーム)\n")

    print(f"タイミング情報を '{info_file}' に保存しました。")


if __name__ == '__main__':
    selector = TimingSelector()

    # タイミング選択
    if selector.play_and_select_timing():
        f0, f1, f2 = selector.get_timing_points()

        if selector.is_valid():
            # 動画切り出し実行
            clip_video_from_timing(f0, f1, f2)
        else:
            print("エラー: タイミング設定が無効です。")
    else:
        print("タイミング選択がキャンセルされました。")