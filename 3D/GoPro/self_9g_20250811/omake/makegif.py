import os
import glob
from PIL import Image
import re

def create_gif_from_pngs(folder_path, output_path="output.gif",
                        max_width=800, max_height=600,
                        duration=100, optimize=True):
    """
    フォルダ内のPNG画像をGIFに変換する

    Parameters:
    - folder_path: PNG画像があるフォルダのパス
    - output_path: 出力するGIFファイルのパス
    - max_width, max_height: リサイズする最大サイズ（軽量化のため）
    - duration: フレーム間隔（ミリ秒）
    - optimize: 最適化を行うかどうか
    """

    # PNGファイルを取得し、ファイル名でソート
    png_files = glob.glob(os.path.join(folder_path, "*.png"))

    if not png_files:
        print("PNGファイルが見つかりません。")
        return

    # ファイル名の辞書順（アルファベット順）でソート
    png_files.sort(key=lambda x: os.path.basename(x))

    print(f"見つかったPNGファイル数: {len(png_files)}")

    # 画像を読み込み、リサイズしてリストに追加
    images = []

    for i, png_file in enumerate(png_files):
        try:
            img = Image.open(png_file)

            # RGBA → RGB変換（GIF用）
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # リサイズして軽量化
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

            images.append(img)

            if (i + 1) % 10 == 0:
                print(f"処理済み: {i + 1}/{len(png_files)}")

        except Exception as e:
            print(f"エラー: {png_file} - {e}")

    if not images:
        print("有効な画像がありません。")
        return

    # GIF作成
    print("GIFを作成中...")

    try:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=optimize,
            quality=85  # 品質を少し下げて軽量化
        )

        # ファイルサイズを表示
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"GIF作成完了: {output_path}")
        print(f"ファイルサイズ: {file_size:.2f} MB")
        print(f"フレーム数: {len(images)}")

    except Exception as e:
        print(f"GIF作成エラー: {e}")

# 使用例
if __name__ == "__main__":
    # フォルダパスを設定
    folder_path = r"G:\gait_pattern\stereo_cali\9g_20250807_6x5_35\fr\cali_imgs"
    output_path = "gait_animation.gif"

    # GIF作成（軽量化設定）
    create_gif_from_pngs(
        folder_path=folder_path,
        output_path=output_path,
        max_width=600,      # 幅を600pxに制限
        max_height=450,     # 高さを450pxに制限
        duration=150,       # フレーム間隔を150msに（少しゆっくり）
        optimize=True       # 最適化有効
    )

    print("\n=== さらに軽量化したい場合は以下の設定を試してください ===")
    print("1. max_width, max_heightをさらに小さく（例：400x300）")
    print("2. durationを大きく（例：200ms以上）")
    print("3. フレーム数を減らす（例：2フレームに1フレーム使用）")