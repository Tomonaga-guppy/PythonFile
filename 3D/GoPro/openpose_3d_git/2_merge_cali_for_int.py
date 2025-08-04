import os
import shutil

"""
内部キャリブレーション用の画像は近くで撮影したものもあるのでキャリブレーション時につかいやすくするため一つのフォルダにまとめる
"""

# --- 設定項目 ---

ROOT_DIR = r"G:\gait_pattern\20250717_br\int_cali\fr\8x6"
# 1. 統合元のフォルダ パス (1つ目)
# ご自身の環境に合わせて変更してください
SOURCE_DIR_A = os.path.join(ROOT_DIR, "cali_frames")

# 2. 統合元のフォルダ パス (2つ目)
# ご自身の環境に合わせて変更してください
SOURCE_DIR_B = os.path.join(ROOT_DIR, "cali_frames_near")

# 3. 統合先の新しいフォルダ パス (このフォルダは自動で作成されます)
# ご自身の環境に合わせて変更してください
DESTINATION_DIR = os.path.join(ROOT_DIR, "cali_frames_all")


def merge_image_folders(source_a, source_b, dest_dir):
    """
    2つのソースフォルダから画像を1つのフォルダに統合する。
    ファイル名が重複した場合は、リネームして両方保存する。
    """
    # --- 1. 統合先フォルダの作成 ---
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"フォルダを作成しました: {dest_dir}")
    else:
        print(f"フォルダは既に存在します: {dest_dir}")

    # --- 2. 処理するフォルダのリスト ---
    source_dirs = [source_a, source_b]

    # --- 3. 各フォルダからファイルをコピー ---
    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            print(f"警告: ソースフォルダが見つかりません。スキップします: {source_dir}")
            continue

        print(f"\n--- '{source_dir}' からのコピーを開始 ---")

        # フォルダ内のすべてのファイル/フォルダ名を取得
        for filename in os.listdir(source_dir):
            source_path = os.path.join(source_dir, filename)

            # ファイルでなければスキップ (サブフォルダはコピーしない)
            if not os.path.isfile(source_path):
                continue

            # コピー先のパスを決定
            dest_path = os.path.join(dest_dir, filename)

            # --- 4. ファイル名の重複チェックとリネーム処理 ---
            if os.path.exists(dest_path):
                print(f"ファイル名の重複を検出: {filename}")

                # 新しいファイル名を探す
                base, ext = os.path.splitext(filename)
                counter = 1
                while True:
                    # 新しいファイル名を生成 (例: my_image_1.png)
                    new_filename = f"{base}_{counter}{ext}"
                    new_dest_path = os.path.join(dest_dir, new_filename)
                    if not os.path.exists(new_dest_path):
                        # この名前が空いていたら、これに決定
                        dest_path = new_dest_path
                        print(f"  -> '{new_filename}' としてリネームします。")
                        break
                    counter += 1

            # --- 5. ファイルのコピー ---
            shutil.copy2(source_path, dest_path)
            # copy2はメタデータ(作成日時など)も一緒にコピーします
            print(f"コピーしました: {os.path.basename(dest_path)}")

    print("\nすべての処理が完了しました。")


if __name__ == '__main__':
    # スクリプトのメイン処理を実行
    merge_image_folders(
        source_a=SOURCE_DIR_A,
        source_b=SOURCE_DIR_B,
        dest_dir=DESTINATION_DIR
    )
