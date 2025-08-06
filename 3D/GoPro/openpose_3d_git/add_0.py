import os
from pathlib import Path

# --- 設定 ---
# ファイル名を変更したいフォルダのパスを指定してください
TARGET_FOLDER = r"G:\gait_pattern\20250717_br\ext_cali\fr\6x5\cali_frames"
# --- 設定ここまで ---


def rename_files_in_folder(folder_path_str: str):
    """
    指定されたフォルダ内のすべてのファイル名の末尾に "_0" を追加する。
    （例: "image.png" -> "image_0.png"）
    """

    folder_path = Path(folder_path_str)

    # 1. フォルダが存在するか確認
    if not folder_path.is_dir():
        print(f"エラー: 指定されたフォルダが見つかりません。")
        print(f"パス: {folder_path}")
        return

    print(f"対象フォルダ: {folder_path}")

    # 処理対象となるファイルリストを作成
    files_to_rename = [
        p for p in folder_path.iterdir()
        if p.is_file() and not p.stem.endswith('_1')
    ]

    if not files_to_rename:
        print("すべてのファイルは既にリネーム済みか、対象ファイルがありません。")
        return

    print("-" * 30)
    print(f"{len(files_to_rename)} 個のファイルがリネーム対象です。例:")
    # 最初の5件をサンプルとして表示
    for f in files_to_rename[:5]:
        print(f"  '{f.name}'  ->  '{f.stem}_0{f.suffix}'")
    print("-" * 30)

    # 2. 実行前にユーザーに最終確認を求める
    try:
        confirm = input("これらのファイル名を変更しますか？ (y/n): ")
    except KeyboardInterrupt:
        print("\n処理を中断しました。")
        return

    if confirm.lower() != 'y':
        print("処理をキャンセルしました。")
        return

    # 3. ファイル名の変更処理を実行
    print("\nリネーム処理を開始します...")
    renamed_count = 0
    skipped_count = 0

    # iterdir()は順不同なので、sorted()で名前順に処理する
    for file_path in sorted(folder_path.iterdir()):
        # ファイルであり、かつ名前に「_1」がまだ付いていないものを対象
        if file_path.is_file() and not file_path.stem.endswith('_1'):
            try:
                # 新しいファイル名を構築 (例: image.png -> image_1.png)
                new_name = f"{file_path.stem}_1{file_path.suffix}"
                new_path = file_path.with_name(new_name)

                # ファイル名を変更
                file_path.rename(new_path)
                print(f"  変更: '{file_path.name}' -> '{new_name}'")
                renamed_count += 1
            except Exception as e:
                print(f"エラー: '{file_path.name}' の変更中にエラーが発生しました - {e}")
        elif file_path.is_file():
            # スキップ対象のファイル
            skipped_count += 1

    print("\n--- 処理完了 ---")
    print(f"変更されたファイル数: {renamed_count} 件")
    if skipped_count > 0:
        print(f"スキップされたファイル数: {skipped_count} 件 (既にリネーム済み)")


if __name__ == '__main__':
    rename_files_in_folder(TARGET_FOLDER)