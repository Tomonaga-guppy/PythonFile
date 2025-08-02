import os
import shutil
from tqdm import tqdm

def organize_calibration_images():
    """
    指定された2つのフォルダからキャリブレーション画像を1つのフォルダにまとめ、
    元のフォルダに応じてファイル名にプレフィックスを付けます。
    """
    # --- 設定項目 ---
    # ベースとなるパス
    base_path = r'G:\gait_pattern\20250717_br\ext_cali'

    # 画像が格納されているソースフォルダのリスト
    # (プレフィックス, フォルダパス) のタプルで指定
    source_folders = [
        ('fl', os.path.join(base_path, r'fl\8x6\cali_frames')),
        ('fr', os.path.join(base_path, r'fr\8x6\cali_frames'))
    ]

    # 画像をまとめる先のフォルダ
    destination_folder = os.path.join(base_path, 'cali_merge')
    # --- 設定はここまで ---

    # 1. 保存先フォルダを作成する
    try:
        os.makedirs(destination_folder, exist_ok=True)
        print(f"保存先フォルダを作成または確認しました: {destination_folder}")
    except OSError as e:
        print(f"エラー: 保存先フォルダの作成に失敗しました。 {e}")
        return

    # 2. 各ソースフォルダからファイルを処理する
    total_files_copied = 0
    for prefix, source_path in source_folders:
        # ソースフォルダが存在するかチェック
        if not os.path.isdir(source_path):
            print(f"警告: ソースフォルダが見つかりません。スキップします: {source_path}")
            continue

        # フォルダ内の全ファイルを取得
        try:
            files = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]
            print(f"\n'{source_path}' から {len(files)} 個のファイルを処理します...")
            print(f"files = {files}")

            # 各ファイルをコピーしてリネーム
            for filename in tqdm(files, desc=f"Processing {prefix} files"):

                # コピー元とコピー先のフルパスを定義
                source_file_path = os.path.join(source_path, filename)
                destination_file_path = os.path.join(destination_folder, filename)

                # ファイルをコピー
                shutil.copy2(source_file_path, destination_file_path)
                # print(f"  コピー完了: {filename} -> {new_filename}")
                total_files_copied += 1

        except Exception as e:
            print(f"エラー: '{source_path}' の処理中に問題が発生しました。 {e}")

    print(f"\n処理が完了しました。合計 {total_files_copied} 個のファイルを {destination_folder} にコピーしました。")

if __name__ == '__main__':
    organize_calibration_images()
