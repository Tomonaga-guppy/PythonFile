"""
mocapのファイル名変更(被験者番号修正用)
"""
import os
import re
from pathlib import Path
from tkinter import Tk, filedialog

def rename_files(folder_path):
    """
    フォルダ内のファイル名を 'A-B-C' または 'A-B-C_...' 形式で処理し、
    A と B が 0 でない場合は 1 を引く
    全てのサブディレクトリを再帰的に処理
    """
    folder = Path(folder_path)
    renamed_count = 0
    total_files = 0
    
    # サブディレクトリを含めて全てのファイルを取得（再帰的）
    for file_path in folder.rglob('*'):
        if file_path.is_file():
            
            total_files += 1
            # ファイル名（拡張子含む）を取得
            filename = file_path.name
            
            # パターン: 数字-数字-数字 の形式を検索（その後に _ または . が続く）
            pattern = r'^(\d+)-(\d+)-(\d+)'
            match = re.match(pattern, filename)
            
            if match:
                a = int(match.group(1))
                b = int(match.group(2))
                c = int(match.group(3))
                
                # A と B が 0 でない場合は 1 を引く
                new_a = a - 1 if a > 0 else a
                new_b = b - 1 if b > 0 else b
                new_c = c  # C はそのまま
                
                # 新しいファイル名を作成
                new_filename = filename.replace(
                    f"{a}-{b}-{c}",
                    f"{new_a}-{new_b}-{new_c}",
                    1  # 最初の1つだけ置換
                )
                
                # リネーム実行
                new_path = file_path.parent / new_filename
                
                if new_path != file_path:
                    # 相対パスを表示
                    rel_path = file_path.relative_to(folder)
                    rel_new_path = new_path.relative_to(folder)
                    print(f"リネーム: {rel_path} -> {rel_new_path}")
                    file_path.rename(new_path)
                    renamed_count += 1
    
    print(f"\n処理完了:")
    print(f"  総ファイル数: {total_files}")
    print(f"  リネーム数: {renamed_count}")

if __name__ == "__main__":
    # Tkinterのルートウィンドウを非表示
    root = Tk()
    root.withdraw()
    
    # フォルダ選択ダイアログを表示
    target_folder = filedialog.askdirectory(title="対象フォルダを選択してください")
    
    if target_folder:
        print(f"選択されたフォルダ: {target_folder}")
        print("全てのサブディレクトリを再帰的に処理します...\n")
        rename_files(target_folder)
    else:
        print("フォルダが選択されませんでした。")