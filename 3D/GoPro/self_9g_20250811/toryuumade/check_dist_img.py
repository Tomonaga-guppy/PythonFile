import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def visualize_distortion_map(camera_params_path: Path):
    """
    カメラパラメータのJSONファイルを読み込み、画像を補正して表示・保存する

    Args:
        camera_params_path (Path): camera_params.json ファイルへのパス
    """
    if not camera_params_path.exists():
        print(f"エラー: ファイルが見つかりません: {camera_params_path}")
        return

    # JSONファイルの親ディレクトリを取得
    base_dir = camera_params_path.parent

    # 1. JSONファイルからパラメータを読み込む
    print(f"'{camera_params_path}' を読み込んでいます...")
    with open(camera_params_path, 'r') as f:
        params = json.load(f)

    mtx = np.array(params['intrinsics'])
    dist = np.array(params['distortion'])
    
    # 2. 'cali_imgs' フォルダから最初の画像を取得
    cali_imgs_dir = base_dir / 'cali_imgs'
    if not cali_imgs_dir.is_dir():
        print(f"エラー: '{cali_imgs_dir}' フォルダが見つかりません。")
        return

    print(f"'{cali_imgs_dir}' から画像を検索しています...")
    # 対応する画像形式を検索し、リストを結合してソートする
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(cali_imgs_dir.glob(ext)))
    
    # 念のためソートして順序を固定
    image_files.sort()

    if not image_files:
        print(f"エラー: '{cali_imgs_dir}' 内に画像ファイルが見つかりません。")
        return

    # 最初の画像ファイルを選択
    image_path = image_files[0]
    print(f"処理する画像: '{image_path}'")

    # 3. 画像を読み込み、歪み補正を実行
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"エラー: 画像 '{image_path}' を読み込めませんでした。")
        return

    h, w = img.shape[:2]

    # 歪み補正
    dst = cv2.undistort(img, mtx, dist)

    # 4. 補正画像を保存
    output_filename = f"{image_path.stem}_undistorted.png"
    output_path = base_dir / output_filename
    cv2.imwrite(str(output_path), dst)
    print(f"補正画像を '{output_path}' に保存しました。")

    # 5. 元の画像と補正画像を並べて表示
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # OpenCVはBGR、MatplotlibはRGBなので色チャンネルを変換
    original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undistorted_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(undistorted_rgb)
    axes[1].set_title('Undistorted Image')
    axes[1].axis('off')

    plt.suptitle(f"Image Distortion Correction: {image_path.name}")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # --- ここに使用したいJSONファイルへのパスを指定してください ---
    params_file = Path(r"G:\gait_pattern\int_cali\9g_20250807_6x5\sagi\camera_params.json")
    
    visualize_distortion_map(params_file)

