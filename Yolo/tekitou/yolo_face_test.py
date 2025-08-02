# 矢状面のみお試しで回すよう

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

for condition in condition_list:
    # 2. 画像ファイルを読み込む
    img_dir = Path(fr"g:\gait_pattern\20250228_ota\data\20250221\sub0\{condition}\sagi\Undistort")
    img_files = sorted(img_dir.glob("*.png"))  # 画像ファイルを取得
    video_dir = img_dir.with_name(f"yolo_face.mp4")

    test_tmg = img_files[0]
    size = cv2.imread(str(test_tmg)).shape[:2][::-1]  # 画像のサイズを取得 (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 動画のエンコーディング方式を指定
    writer = cv2.VideoWriter(str(video_dir), fourcc, 60.0, size)  # 動画ファイルの作成

    for img_path in tqdm(img_files, desc=f"Processing {condition}"):
        img = cv2.imread(str(img_path))

        # 3. モデルで推論を実行
        results = model(img, verbose=False)  # verbose=False で出力を抑制

        # 4. 検出結果を画像に描画
        annotated_img = results[0].plot()

        # 5. 動画ファイルに書き込む
        writer.write(annotated_img)
    # 動画ファイルを閉じる
    writer.release()