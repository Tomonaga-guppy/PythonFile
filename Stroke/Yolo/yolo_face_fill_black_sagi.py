from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
from pathlib import Path
from tqdm import tqdm

# 1. 学習済みモデルをロード
model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
# model = YOLO('yolo11n.pt')
model = YOLO(model_path)


def apply_black_fill(image, x1, y1, x2, y2):
    # 指定範囲を黒で塗りつぶす関数
    color = [0, 0, 0]  # 黒色のBGR値
    width = x2 - x1
    height = y2 - y1
    # 中心を基準にして矩形のサイズを2倍に
    new_x1 = x1 - width // 2
    new_y1 = y1 - height // 2
    new_x2 = x2 + width // 2
    new_y2 = y2 + height // 2
    # 画像の範囲を超えないように調整
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image.shape[1], new_x2)
    new_y2 = min(image.shape[0], new_y2)
    cv2.rectangle(image, (new_x1, new_y1), (new_x2, new_y2), color, -1)
    return image


# 2. 画像ファイルを読み込む
# input_video_dir = Path(fr"g:\gait_pattern\20250228_ota\data\20250221\sub0\thera0-3\sagi\Undistort.mp4")
input_video_dir = Path(fr"g:\gait_pattern\20250228_ota\data\20250221\sub0")
input_videos = list(input_video_dir.glob(r"**\*\Undistort.mp4"))  # mp4ファイルも拾ってしまう

print(f"処理する動画ファイル: {input_videos}")

for input_video in input_videos:
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        print(f"動画の読み込みに失敗しました: {input_video}")
        exit(1)


    # 元動画の情報を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 動画の総フレーム数
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 動画のサイズ
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 出力する動画の設定
    file_name = input_video.stem
    output_video_dir = input_video.with_name(f"{file_name}_black_fill.mp4")
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

        annotated_img = img.copy()
        # 4. 検出結果を画像に描画
        for result in results:  #検出した顔の数だけに塗りつぶし
            for box in result.boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                annotated_img = apply_black_fill(annotated_img, x1, y1, x2, y2)

        # 5. 動画ファイルに書き込む
        writer.write(annotated_img)
    # 動画ファイルを閉じる
    writer.release()