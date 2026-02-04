from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
from pathlib import Path
from tqdm import tqdm

# 動画ではなく画像フォルダから処理

# 1. 学習済みモデルをロード
model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
# model = YOLO('yolo11n.pt')
model = YOLO(model_path)

condition_list = ["thera0-3", "thera1-1", "thera2-1"]
direction_list = ["sagi", "fr", "fl"]

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


for direction in direction_list:
    for condition in condition_list:
        # 2. 画像ファイルを読み込む
        input_dir = Path(fr"g:\gait_pattern\20250228_ota\data\20250221\sub0\{condition}\{direction}")
        imgs_folder = input_dir / "Undistort"  # Undistortフォルダ内の画像を対象
        imgs = sorted(imgs_folder.glob("*.png"))  # PNG画像ファイルを取得

        # 出力する動画の設定
        output_video_dir = imgs_folder.with_name(f"face_fill.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 動画のエンコーディング方式を指定
        size = cv2.imread(str(imgs[0])).shape[:2][::-1]  # 画像のサイズを取得 (width, height)
        fps = 60.0  # フレームレートを指定
        writer = cv2.VideoWriter(str(output_video_dir), fourcc, fps, size)  # 動画ファイルの作成

        for img_path in tqdm(imgs, desc=f"Processing {condition}"):
            # 3. モデルで推論を実行
            img = cv2.imread(str(img_path))
            results = model(img, verbose=False)  # verbose=False で出力を抑制

            # 4. 検出結果を画像に描画
            for result in results:  #検出した顔の数だけに塗りつぶし
                for box in result.boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                    img = apply_black_fill(img, x1, y1, x2, y2)

            save_img_folder = img_path.parent / "Filled"
            save_img_folder.mkdir(exist_ok=True)  # フォルダが存在しない場合は作成
            save_img_path  = save_img_folder / img_path.name
            cv2.imwrite(str(save_img_path), img)

            # 5. 動画ファイルに書き込む
            writer.write(img)
        # 動画ファイルを閉じる
        writer.release()