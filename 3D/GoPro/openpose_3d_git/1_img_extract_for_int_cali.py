import cv2
import os
import numpy as np

# --- 設定項目 ---

SIDE = "fr"

# 1. 入力する動画ファイルのパス
# ご自身の動画ファイルのパスに変更してください
INPUT_VIDEO_PATH = fr"G:\gait_pattern\20250717_br\ext_cali\{SIDE}\8x6\cali_trim.mp4"

# 2. 画像を保存するフォルダのパス (このフォルダは自動で作成されます)
OUTPUT_DIR = fr"G:\gait_pattern\20250717_br\ext_cali\{SIDE}\8x6\cali_frames"

# 3. 保存する画像の枚数
NUM_IMAGES_TO_SAVE = 40

# 4. 保存する画像のファイル名の接頭辞（連番の前に付きます）
IMAGE_BASENAME = "frame"

# 5. 保存する画像の形式 ('png' または 'jpg' を推奨)
IMAGE_EXTENSION = "png"


def extract_frames_evenly(video_path, output_dir, num_frames, basename, extension):
    """
    動画全体から指定した枚数の画像を、時系列に沿って均等に抽出し保存する。
    """
    # --- 1. 動画ファイルの読み込み ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイル '{video_path}' を開けませんでした。")
        return

    # --- 2. 動画の総フレーム数を取得 ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"動画の総フレーム数: {total_frames}")

    if total_frames < num_frames:
        print(f"エラー: 動画の総フレーム数({total_frames})が、保存したい枚数({num_frames})より少ないです。")
        cap.release()
        return

    # --- 3. 保存先フォルダの作成 ---
    # exist_ok=True を指定すると、フォルダが既に存在していてもエラーにならない
    os.makedirs(output_dir, exist_ok=True)
    print(f"画像を '{output_dir}' フォルダに保存します。")

    # --- 4. 抽出するフレームのインデックスを計算 ---
    # np.linspace を使うと、開始点と終了点の間で均等な間隔の数値を指定個数だけ生成できます。
    # これにより、動画全体にわたって均等なタイミングのフレーム番号リストが得られます。
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    print(f"抽出対象のフレーム番号: {frame_indices}")

    # --- 5. フレームを1枚ずつ抽出して保存 ---
    for i, frame_index in enumerate(frame_indices):
        # cap.set() を使って、目的のフレームに直接移動します（シーケンシャルに読み込むより高速です）。
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ret, frame = cap.read()
        if not ret:
            print(f"警告: フレーム {frame_index} の読み込みに失敗しました。このフレームはスキップします。")
            continue

        # 保存するファイル名を生成します (例: frame_00.png, frame_01.png, ...)
        # str(i).zfill(2) は、数値を2桁のゼロ埋め文字列にします (例: 1 -> "01")
        output_filename = f"{basename}_{str(i).zfill(2)}.{extension}"
        output_filepath = os.path.join(output_dir, output_filename)

        # cv2.imwrite() で現在のフレームを画像ファイルとして保存
        cv2.imwrite(output_filepath, frame)
        print(f"保存しました: {output_filepath}")

    # --- 6. 後処理 ---
    cap.release()
    print("\nすべての処理が完了しました。")


if __name__ == '__main__':
    # スクリプトのメイン処理を実行
    extract_frames_evenly(
        video_path=INPUT_VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        num_frames=NUM_IMAGES_TO_SAVE,
        basename=IMAGE_BASENAME,
        extension=IMAGE_EXTENSION
    )
