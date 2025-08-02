from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
from pathlib import Path
from tqdm import tqdm

# 1. 学習済みモデルをロード
model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
# model = YOLO('yolo11n.pt')
model = YOLO(model_path)

condition_list = ["thera0-3", "thera1-1", "thera2-1"]
direction_list = ["sagi", "fr", "fl"]

for direction in direction_list:
    for condition in condition_list:
        # 2. 画像ファイルを読み込む
        input_dir = Path(fr"g:\gait_pattern\20250228_ota\data\20250221\sub0\{condition}\{direction}")
        all_input_videos = input_dir.glob("*GX*.MP4")  #mp4ファイルも拾ってしまう
        input_videos = [p for p in all_input_videos if p.suffix == '.MP4']  # MP4ファイルのみを対象
        print(f"処理する動画ファイル候補: {input_videos}")
        input_video = input_videos[0]  # 動画ファイルを取得(MP4ファイルが2個以上ある場合は何か対応必要)
        print(f"処理する動画ファイル: {input_video}")

        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            print(f"動画の読み込みに失敗しました: {input_video}")
            continue
        # 元動画の情報を取得
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 動画の総フレーム数
        size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 動画のサイズ
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 出力する動画の設定
        output_video_dir = input_video.with_name(f"yolo_face_full.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 動画のエンコーディング方式を指定
        writer = cv2.VideoWriter(str(output_video_dir), fourcc, fps, size)  # 動画ファイルの作成
        frame_count = 0

        for _ in tqdm(range(total_frames), desc=f"Processing {input_video.name}"):
            ret, img = cap.read()
            if not ret:
                print(f"動画の読み込みを終了しました. フレーム数{frame_count}")
                break
            # 3. モデルで推論を実行
            results = model(img, verbose=False)  # verbose=False で出力を抑制
            # 4. 検出結果を画像に描画
            annotated_img = results[0].plot()
            # 5. 動画ファイルに書き込む
            writer.write(annotated_img)
        # 動画ファイルを閉じる
        writer.release()