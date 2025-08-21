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
    output_filename = camera_params_path.with_name("distortion_map.png")
    plt.savefig(output_filename)
    print(f"\n歪みマップの比較画像を保存しました: {output_filename}")

    # 画面に表示
    plt.show()

def undistort_and_display_frame_remap(camera_params_path: Path, video_path: Path):
    """
    指定された動画の最初のフレームを読み込み、高速なremap方式で歪み補正を行い、
    補正前後の画像を並べて表示・保存する。

    Args:
        camera_params_path (Path): camera_params.json ファイルへのパス
        video_path (Path):         歪み補正を適用するテスト動画ファイルへのパス
    """
    # --- 1. パラメータと動画の存在チェック ---
    if not camera_params_path.exists():
        print(f"エラー: カメラパラメータファイルが見つかりません: {camera_params_path}")
        return
    if not video_path.exists():
        print(f"エラー: 動画ファイルが見つかりません: {video_path}")
        return

    # --- 2. JSONファイルからカメラパラメータを読み込む ---
    print(f"'{camera_params_path}' を読み込んでいます...")
    with open(camera_params_path, 'r') as f:
        params = json.load(f)

    # JSONファイルのキーが異なる場合に対応
    if 'intrinsics' in params:
        mtx = np.array(params['intrinsics'])
        dist = np.array(params['distortion'])
    elif 'camera_matrix' in params:
        mtx = np.array(params['camera_matrix'])
        dist = np.array(params['distortion_coefficients'])
    else:
        print("エラー: JSONファイル内にカメラ行列が見つかりません。")
        return

    image_size = (params['image_width'], params['image_height'])
    print("パラメータの読み込み完了。")

    # --- 3. 動画ファイルから最初のフレームを読み込む ---
    print(f"'{video_path}' からフレームを読み込んでいます...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けませんでした: {video_path}")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("エラー: 動画からフレームを読み込めませんでした。")
        return
    # フレームのサイズがパラメータと一致しているか確認
    frame_height, frame_width = frame.shape[:2]
    if (frame_width, frame_height) != image_size:
        print(f"警告: 動画の解像度({frame_width}x{frame_height})が、"
              f"カメラパラメータの解像度({image_size[0]}x{image_size[1]})と異なります。")

    print("フレームの読み込み完了。")

    # --- 4. 歪み補正を実行 (高速なremap方式) ---
    print("undistortマップを作成中...")
    # 歪み補正のための変換マップを事前に計算
    # 第4引数に新しいカメラ行列を指定できるが、ここでは元と同じmtxを指定
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, image_size, cv2.CV_32FC1)

    print("remapを使用して歪み補正を実行...")
    # 変換マップを使って高速にピクセルを再配置（補正）
    undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    # --- 5. 結果を並べて表示・保存 ---
    original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    undistorted_rgb = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(original_rgb)
    axes[0].set_title(f"Original Frame from {video_path.name}")
    axes[0].axis('off')

    axes[1].imshow(undistorted_rgb)
    axes[1].set_title("Undistorted Frame (using remap)")
    axes[1].axis('off')

    plt.suptitle("Frame Undistortion Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_filename = video_path.with_name("undistorted_frame.png")
    plt.savefig(output_filename)
    print(f"\n歪み補正したフレームの比較画像を保存しました: {output_filename}")

    plt.show()

if __name__ == '__main__':
    # --- ここに使用したいJSONファイルへのパスを指定してください ---
    params_file = Path(r"G:\gait_pattern\int_cali\tkrzk\sagi\camera_params.json")

    # --- 2. 歪み補正をテストしたい動画ファイルへのパスを指定してください ---
    video_file = Path(r"G:\gait_pattern\int_cali\tkrzk\sagi\cali.MP4")

    # #歪みマップを可視化する場合
    visualize_distortion_map(params_file)

    # 実際に歪み補正を行い、フレームを表示する場合
    undistort_and_display_frame_remap(params_file, video_file)