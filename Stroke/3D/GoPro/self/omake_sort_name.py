#指定したフォルダの画像をソートして4桁の数字で名前を付けるスクリプト

import cv2
from pathlib import Path

def sort_images_in_folder(folder_path):
    """
    指定されたフォルダ内の画像をソートし、4桁の数字で名前を付ける

    Args:
        folder_path: 画像が保存されているフォルダのパス
    """

    if not folder_path.exists() or not folder_path.is_dir():
        print(f"エラー: {folder_path} は存在しないか、ディレクトリではありません。")
        return

    # フォルダ内の画像ファイルを取得
    image_files = sorted(folder_path.glob("*.png"))  # png形式の画像を対象とする

    if not image_files:
        print(f"警告: {folder_path} 内に画像ファイルが見つかりません。")
        return

    for i, img_path in enumerate(image_files):
        new_name = f"{i+1:04d}.png"  # 4桁の数字で名前を付ける
        new_path = folder_path / new_name

        # 既存のファイル名と重複しないように確認
        if new_path.exists():
            print(f"警告: {new_path} はすでに存在します。スキップします。")
            continue

        img_path.rename(new_path)
        print(f"{img_path.name} を {new_name} に変更しました。")

if __name__ == "__main__":
    # 使用例
    folder_path = Path(r"G:\gait_pattern\stero_cali\9g_6x5\fl\cali_imgs")  # ここに対象のフォルダパスを指定
    sort_images_in_folder(folder_path)