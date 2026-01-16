"""
mocapフォルダ変な位置に移動してしまったので修正用(間違っていたフォルダは削除したのでもう機能しない)
"""

import os
import shutil
from pathlib import Path
from tkinter import Tk, filedialog

def copy_mocap_files(root_folder):
    """
    ルートフォルダの mocap フォルダから、
    thera で始まる各サブフォルダの mocap フォルダへファイルをコピー
    """
    root_path = Path(root_folder)
    
    # コピー元: ルートフォルダ/mocap
    source_mocap = root_path / "mocap"
    
    if not source_mocap.exists():
        print(f"エラー: {source_mocap} が見つかりません")
        return
    
    # コピー元の CSV ファイルを取得（名前順にソート）
    source_files = sorted(source_mocap.glob("*.csv"), key=lambda x: x.name.lower())
    
    if len(source_files) == 0:
        print("エラー: コピー元の mocap フォルダに CSV ファイルがありません")
        return
    
    print(f"コピー元ファイル数: {len(source_files)}")
    for i, f in enumerate(source_files, 1):
        print(f"  {i}. {f.name}")
    
    # thera で始まるサブフォルダを取得（名前順にソート）
    thera_folders = sorted(
        [d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("thera")],
        key=lambda x: x.name.lower()
    )
    
    if len(thera_folders) == 0:
        print("エラー: thera で始まるフォルダが見つかりません")
        return
    
    print(f"\nコピー先フォルダ数: {len(thera_folders)}")
    for i, folder in enumerate(thera_folders, 1):
        print(f"  {i}. {folder.name}")
    
    # ファイル数とフォルダ数が一致するか確認
    if len(source_files) != len(thera_folders):
        print(f"\n警告: ファイル数({len(source_files)})とフォルダ数({len(thera_folders)})が一致しません")
        response = input("続行しますか? (y/n): ")
        if response.lower() != 'y':
            print("処理を中止しました")
            return
    
    # コピー処理
    print("\n--- コピー開始 ---")
    copy_count = 0
    
    for i, (source_file, thera_folder) in enumerate(zip(source_files, thera_folders), 1):
        # コピー先: theraフォルダ/mocap
        dest_mocap = thera_folder / "mocap"
        
        # mocap フォルダが存在しない場合は作成
        dest_mocap.mkdir(exist_ok=True)
        
        # コピー先のファイルパス
        dest_file = dest_mocap / source_file.name
        
        try:
            shutil.copy2(source_file, dest_file)
            print(f"{i}. {source_file.name} -> {thera_folder.name}/mocap/")
            copy_count += 1
        except Exception as e:
            print(f"エラー: {source_file.name} のコピーに失敗しました: {e}")
    
    print(f"\n--- 処理完了 ---")
    print(f"コピー成功: {copy_count}/{len(source_files)} ファイル")

if __name__ == "__main__":
    # Tkinterのルートウィンドウを非表示
    root = Tk()
    root.withdraw()
    
    # フォルダ選択ダイアログを表示
    root_folder = filedialog.askdirectory(title="ルートフォルダを選択してください")
    
    if root_folder:
        print(f"選択されたルートフォルダ: {root_folder}\n")
        copy_mocap_files(root_folder)
    else:
        print("フォルダが選択されませんでした。")