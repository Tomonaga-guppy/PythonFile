

from pathlib import Path
import cv2
from tqdm import tqdm

condition_list = ["thera0-3", "thera1-1", "thera2-1"]

for condition in condition_list: # 各条件に対して処理を行う
    img_dir = Path(r"g:\gait_pattern\20250228_ota\data\20250221\sub0") / condition / "sagi" / "Undistort"
    if not img_dir.exists():
        print(f"Directory does not exist: {img_dir}")
        continue

    img_files = sorted(img_dir.glob("*.png"))  #画像ファイルを取得

    # 最初の画像からサイズを取得
    first_img = cv2.imread(str(img_files[0]))
    height, width = first_img.shape[:2]

    # 動画ライターを作成
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = img_dir.with_name("Undistort.mp4")
    out = cv2.VideoWriter(str(video_path), fourcc, 60, (width, height))

    for img_file in tqdm(img_files, desc=f"Processing {condition}"):
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Failed to read image: {img_file}")
            continue
        # サイズが異なる場合はリサイズ
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))
        out.write(img)

    out.release()
    print(f"Saved video: {img_dir / 'output.mp4'}")

