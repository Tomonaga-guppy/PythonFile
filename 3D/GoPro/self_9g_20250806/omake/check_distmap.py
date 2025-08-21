import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def visualize_distortion_map(camera_params_path: Path):
    """
    カメラパラメータのJSONファイルを読み込み、歪みマップを可視化する。

    Args:
        camera_params_path (Path): camera_params.json ファイルへのパス
    """
    if not camera_params_path.exists():
        print(f"エラー: ファイルが見つかりません: {camera_params_path}")
        return

    # 1. JSONファイルからパラメータを読み込む
    print(f"'{camera_params_path}' を読み込んでいます...")
    with open(camera_params_path, 'r') as f:
        params = json.load(f)

    mtx = np.array(params['intrinsics'])
    dist = np.array(params['distortion'])
    width = params['image_width']
    height = params['image_height']
    image_size = (width, height)

    # 歪み係数の次元を (1, N) or (N, 1) の形に整える
    if dist.ndim > 1 and dist.shape[0] > 1 and dist.shape[1] > 1:
        dist = dist[0]

    print("パラメータの読み込み完了。")

    # 2. 補正前の元となる格子画像を生成
    grid_spacing = 100  # 格子の間隔 (ピクセル)
    original_grid_img = np.full((height, width, 3), 255, dtype=np.uint8)
    for y in range(0, height, grid_spacing):
        cv2.line(original_grid_img, (0, y), (width, y), (200, 200, 200), 2)
    for x in range(0, width, grid_spacing):
        cv2.line(original_grid_img, (x, 0), (x, height), (200, 200, 200), 2)
    cv2.putText(original_grid_img, 'Original Grid', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)

    # 3. 歪み補正を実行
    # getOptimalNewCameraMatrixのalpha=1は、元画像の全ピクセルが残るように調整するオプション
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, image_size, 1, image_size)
    undistorted_img = cv2.undistort(original_grid_img, mtx, dist, None, new_camera_mtx)
    cv2.putText(undistorted_img, 'Undistorted Grid', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)

    # 4. 結果を並べて表示・保存
    # Matplotlibで表示するためにBGRからRGBに変換
    original_grid_rgb = cv2.cvtColor(original_grid_img, cv2.COLOR_BGR2RGB)
    undistorted_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(original_grid_rgb)
    axes[0].set_title("Original Grid (Ideal)")
    axes[0].axis('off')
    axes[1].imshow(undistorted_rgb)
    axes[1].set_title("Undistorted Grid (Shows Lens Effect)")
    axes[1].axis('off')
    plt.suptitle(f"Distortion Visualization for {camera_params_path.name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 歪みマップ画像をファイルに保存
    output_filename = camera_params_path.stem + "_distortion_map.png"
    plt.savefig(output_filename)
    print(f"\n歪みマップの比較画像を保存しました: {output_filename}")

    # 画面に表示
    plt.show()


if __name__ == '__main__':
    # --- ここに使用したいJSONファイルへのパスを指定してください ---
    params_file = Path(r"G:\gait_pattern\int_cali\9g_20250807_6x5_35\fl\camera_params_sb.json")

    visualize_distortion_map(params_file)