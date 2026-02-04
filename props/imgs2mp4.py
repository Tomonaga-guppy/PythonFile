"""
画像フォルダからHD動画を作成するスクリプト
"""

from pathlib import Path
import cv2
from tqdm import tqdm
import numpy as np

# =========================
# 設定
# =========================
# 画像フォルダのパス（ここを変更）
IMAGE_DIR = Path(r"G:\gait_pattern\2025_shuron_BR9G\sub1\thera1-0\gopro\sagi\undistorted")

# 出力動画のパス（指定しない場合は画像フォルダと同じ場所に作成）
OUTPUT_VIDEO = IMAGE_DIR.parent / f"{IMAGE_DIR.name}.mp4"

# 動画設定
FPS = 120.0  # フレームレート
OUTPUT_RESOLUTION = (1280, 720)  # HD (width, height)

# 画像拡張子（複数指定可）
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]

# リサイズ方法（"fit": アスペクト比維持、"stretch": 引き伸ばし、"crop": 中央クロップ）
RESIZE_MODE = "fit"

# 背景色（"fit"モードでレターボックス使用時、BGR形式）
BACKGROUND_COLOR = (0, 0, 0)  # 黒


def get_image_files(image_dir: Path, extensions: list) -> list[Path]:
    """
    画像ファイルを取得してソート
    
    Parameters
    ----------
    image_dir : Path
        画像フォルダのパス
    extensions : list
        対象とする拡張子のリスト
    
    Returns
    -------
    list[Path]
        ソートされた画像ファイルのリスト
    """
    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))
    
    # ファイル名でソート
    return sorted(image_files)


def resize_image(img: np.ndarray, target_size: tuple, mode: str, bg_color: tuple) -> np.ndarray:
    """
    画像をリサイズ
    
    Parameters
    ----------
    img : np.ndarray
        入力画像
    target_size : tuple
        目標サイズ (width, height)
    mode : str
        リサイズモード ("fit", "stretch", "crop")
    bg_color : tuple
        背景色 (B, G, R)
    
    Returns
    -------
    np.ndarray
        リサイズ後の画像
    """
    target_w, target_h = target_size
    h, w = img.shape[:2]
    
    if mode == "stretch":
        # 引き伸ばし
        return cv2.resize(img, (target_w, target_h))
    
    elif mode == "crop":
        # 中央クロップ
        aspect_target = target_w / target_h
        aspect_src = w / h
        
        if aspect_src > aspect_target:
            # 横に広い：高さを合わせて中央を切り取り
            new_h = h
            new_w = int(h * aspect_target)
            x_offset = (w - new_w) // 2
            cropped = img[:, x_offset:x_offset + new_w]
        else:
            # 縦に広い：幅を合わせて中央を切り取り
            new_w = w
            new_h = int(w / aspect_target)
            y_offset = (h - new_h) // 2
            cropped = img[y_offset:y_offset + new_h, :]
        
        return cv2.resize(cropped, (target_w, target_h))
    
    else:  # "fit"
        # アスペクト比維持（レターボックス）
        aspect_target = target_w / target_h
        aspect_src = w / h
        
        if aspect_src > aspect_target:
            # 横長：幅を基準
            new_w = target_w
            new_h = int(target_w / aspect_src)
        else:
            # 縦長：高さを基準
            new_h = target_h
            new_w = int(target_h * aspect_src)
        
        resized = cv2.resize(img, (new_w, new_h))
        
        # 背景画像を作成
        canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
        
        # 中央に配置
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return canvas


def create_video(image_dir: Path, output_video: Path, fps: float,
                resolution: tuple, resize_mode: str, bg_color: tuple):
    """
    画像フォルダから動画を作成
    
    Parameters
    ----------
    image_dir : Path
        画像フォルダのパス
    output_video : Path
        出力動画のパス
    fps : float
        フレームレート
    resolution : tuple
        出力解像度 (width, height)
    resize_mode : str
        リサイズモード
    bg_color : tuple
        背景色
    """
    
    # 画像フォルダの存在確認
    if not image_dir.exists() or not image_dir.is_dir():
        print(f"[ERROR] 画像フォルダが見つかりません: {image_dir}")
        return False
    
    # 画像ファイル取得
    image_files = get_image_files(image_dir, IMAGE_EXTENSIONS)
    
    if len(image_files) == 0:
        print(f"[ERROR] 画像が見つかりません: {image_dir}")
        return False
    
    print(f"\n{'='*70}")
    print(f"画像フォルダ: {image_dir}")
    print(f"画像数: {len(image_files)}")
    print(f"出力動画: {output_video}")
    print(f"解像度: {resolution[0]}x{resolution[1]}")
    print(f"FPS: {fps}")
    print(f"リサイズモード: {resize_mode}")
    print(f"再生時間: {len(image_files)/fps:.2f} 秒")
    print(f"{'='*70}\n")
    
    # 動画ライター作成
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, resolution)
    
    if not out.isOpened():
        print(f"[ERROR] 動画ライターを開けません")
        return False
    
    # 画像を読み込んで書き込み
    with tqdm(total=len(image_files), desc="動画作成", unit="frame") as pbar:
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"\n[WARNING] 読み込み失敗: {img_path.name}")
                continue
            
            # リサイズ
            resized = resize_image(img, resolution, resize_mode, bg_color)
            
            # 書き込み
            out.write(resized)
            pbar.update(1)
    
    out.release()
    print(f"\n[SUCCESS] 動画を保存しました: {output_video}")
    return True


def main():
    """メイン処理"""
    print("=" * 70)
    print("画像フォルダから動画作成ツール")
    print("=" * 70)
    
    success = create_video(
        image_dir=IMAGE_DIR,
        output_video=OUTPUT_VIDEO,
        fps=FPS,
        resolution=OUTPUT_RESOLUTION,
        resize_mode=RESIZE_MODE,
        bg_color=BACKGROUND_COLOR
    )
    
    if success:
        print("\n処理が正常に完了しました。")
    else:
        print("\n処理が失敗しました。")


if __name__ == "__main__":
    main()