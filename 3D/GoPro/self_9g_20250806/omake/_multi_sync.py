import cv2
import numpy as np
import os
from tqdm import tqdm

"""
手をたたいたタイミング（目視）から動画を切り出す
"""


# --- 設定項目 ---
# ユーザー指定のファイルパスを維持
INPUT_VIDEO_PATH = r"G:\gait_pattern\stero_cali\9g_20250806\fr\GX010153.MP4"

OUTPUT_VIDEO_PATH = r"G:\gait_pattern\stero_cali\9g_20250806\fr\cali.mp4"

def play_and_select_timing():
    """【修正版】再生バーとフレーム操作機能付きのウィンドウでタイミングを指定する"""
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"エラー: 動画ファイル '{INPUT_VIDEO_PATH}' を開けませんでした。")
        return -1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_pos = 0
    start_frame = -1
    is_playing = False
    window_name = "Video Frame Navigator"

    print("\n--- フレーム選択 ---")
    print("再生バー: マウスでドラッグして移動")
    print("右矢印キー: 1フレーム進む")
    print("左矢印キー: 1フレーム戻る")
    print("スペースキー: 再生/一時停止")
    print("'s' キー: 現在のフレームを切り出し開始点として決定")
    print("'q' キー: 終了")

    cv2.namedWindow(window_name)

    # --- 修正点 1: コールバック関数が直接フレーム位置を更新するように変更 ---
    # 'nonlocal' を使い、この関数の中から親関数の変数を変更できるようにします。
    def on_trackbar(val):
        nonlocal current_frame_pos, is_playing
        # この関数はユーザーがマウスでシークバーを操作した時だけ呼ばれます。
        # valにはシークバーの新しい位置が入っています。
        current_frame_pos = val
        is_playing = False  # 手動でシークした場合は再生を止める

    cv2.createTrackbar("Seek", window_name, 0, total_frames - 1, on_trackbar)

    while True:
        # --- 修正点 2: メインループからシークバー位置のポーリングを削除 ---
        # これにより、キーボード入力とマウス入力の競合を防ぎます。

        if is_playing:
            current_frame_pos += 1
            if current_frame_pos >= total_frames:
                current_frame_pos = total_frames - 1
                is_playing = False

        # 現在のフレーム位置に合わせてシークバーの位置を更新
        # (キー操作や再生によってフレーム位置が変わった場合のため)
        cv2.setTrackbarPos("Seek", window_name, current_frame_pos)

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
        ret, frame = cap.read()
        if not ret:
            current_frame_pos = max(0, current_frame_pos - 1)
            is_playing = False
            continue

        font_scale_large = 1.2
        font_scale_small = 0.8
        font_thickness = 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (1250, 110), (0,0,0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        info_text = f"Frame: {current_frame_pos}/{total_frames-1}"
        controls_text = "Seekbar | <- Left | Right -> | Space: Play/Pause | 's': Set Point | 'q': Quit"
        cv2.putText(frame, info_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, (0, 255, 0), font_thickness)
        cv2.putText(frame, controls_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), 1)

        # ユーザーが追加したリサイズ処理を維持
        frame = cv2.resize(frame, (1280, 720))

        cv2.imshow(window_name, frame)

        wait_time = 33 if is_playing else 0
        key = cv2.waitKeyEx(wait_time)

        if key == -1: continue

        if key == ord('q'):
            start_frame = -1
            break
        elif key == ord('s'):
            start_frame = current_frame_pos
            print(f"\n切り出し開始ポイントを設定しました: フレーム {start_frame}")
            break
        elif key == ord(' '):
            is_playing = not is_playing
        elif key == 2424832: # 左矢印キー
            is_playing = False
            current_frame_pos = max(0, current_frame_pos - 1)
        elif key == 2555904: # 右矢印キー
            is_playing = False
            current_frame_pos = min(total_frames - 1, current_frame_pos + 1)

    cap.release()
    cv2.destroyAllWindows()
    return start_frame


def clip_video_from_point(start_frame):
    """【★更新】指定されたフレーム以降の動画をプログレスバー付きで切り出して保存する"""
    if start_frame < 0:
        print("開始ポイントが指定されなかったため、動画生成をスキップします。")
        return

    print("\n--- 動画の切り出し開始 ---")

    cap_orig = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap_orig.isOpened():
        print(f"エラー: 動画ファイル '{INPUT_VIDEO_PATH}' を開けませんでした。")
        return

    width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))

    # 書き込むべき総フレーム数を計算
    frames_to_write = total_frames - start_frame

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"エラー: 出力ファイル '{OUTPUT_VIDEO_PATH}' を開けませんでした。")
        cap_orig.release()
        return

    print(f"元の動画のフレーム {start_frame} から最後までを書き込み中...")

    # 開始フレームに移動
    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # tqdmを使ってプログレスバーを表示
    for _ in tqdm(range(frames_to_write), desc="Writing Video", unit="frame"):
        ret, frame = cap_orig.read()
        if not ret:
            print("\n警告: 動画の途中でフレームが読み込めなくなりました。")
            break
        writer.write(frame)

    cap_orig.release()
    writer.release()
    print(f"\n完了！切り出した動画を '{OUTPUT_VIDEO_PATH}' に保存しました。")


if __name__ == '__main__':
    selected_frame = play_and_select_timing()
    clip_video_from_point(selected_frame)
