from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


def select_directory():
    """
    ファイルダイアログを表示してディレクトリを選択する
    Returns:
        Path: 選択されたディレクトリのパス、キャンセルされた場合はNone
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    folder_path = filedialog.askdirectory(
        title="画像を削除するフォルダを選択してください"
    )
    
    root.destroy()
    
    if folder_path:
        return Path(folder_path)
    else:
        return None


def delete_images(folder_path, extensions=None, recursive=False):
    """
    指定されたフォルダ内の画像ファイルを削除する
    Args:
        folder_path (Path): 対象フォルダのパス
        extensions (list): 削除する拡張子のリスト（デフォルト: jpg, jpeg, png, gif, bmp, tiff）
        recursive (bool): サブフォルダも含めて削除するかどうか
    Returns:
        int: 削除したファイル数
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    deleted_count = 0
    
    # 再帰的に検索するかどうかで処理を分岐
    if recursive:
        pattern = '**/*'
    else:
        pattern = '*'
    
    for ext in extensions:
        for file in folder_path.glob(f"{pattern}{ext}"):
            if file.is_file():
                try:
                    file.unlink()
                    print(f"削除: {file}")
                    deleted_count += 1
                except Exception as e:
                    print(f"削除失敗: {file} - {e}")
    
    return deleted_count


def main():
    # フォルダを選択
    folder_path = select_directory()
    
    if folder_path is None:
        print("フォルダが選択されませんでした。終了します。")
        return
    
    print(f"選択されたフォルダ: {folder_path}")
    
    # 削除対象の画像ファイルを先にリストアップ
    extensions = ['.jpg', '.png']
    files_to_delete = []
    
    for ext in extensions:
        files_to_delete.extend(folder_path.glob(f"*{ext}"))
    
    if len(files_to_delete) == 0:
        print("削除対象の画像ファイルが見つかりませんでした。")
        return
    
    print(f"\n削除対象のファイル数: {len(files_to_delete)}")
    print("削除対象ファイル一覧:")
    for f in files_to_delete[:10]:  # 最初の10件のみ表示
        print(f"  - {f.name}")
    if len(files_to_delete) > 10:
        print(f"  ... 他 {len(files_to_delete) - 10} 件")
    
    # 確認ダイアログ
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    confirm = messagebox.askyesno(
        "確認",
        f"{len(files_to_delete)} 件の画像ファイルを削除しますか？\n\nフォルダ: {folder_path}"
    )
    
    root.destroy()
    
    if not confirm:
        print("キャンセルされました。")
        return
    
    # 画像を削除
    deleted_count = delete_images(folder_path, recursive=False)
    
    print(f"\n完了: {deleted_count} 件の画像ファイルを削除しました。")


if __name__ == "__main__":
    main()