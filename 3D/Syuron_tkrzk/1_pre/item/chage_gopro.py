"""
set_goproの後に実施(フォルダ移動するだけ)
"""
import shutil
from pathlib import Path
from tkinter import Tk, filedialog

def move_to_gopro(root_folder):
    """
    cali以外の各サブフォルダ内のfl/fr/sagiフォルダをgoproフォルダに移動
    移動先に既に存在する場合は上書き
    """
    root_path = Path(root_folder)
    
    # cali以外のサブフォルダを取得（名前順にソート）
    subfolders = sorted(
        [d for d in root_path.iterdir() if d.is_dir() and d.name != "cali"],
        key=lambda x: x.name.lower()
    )
    
    if len(subfolders) == 0:
        print("エラー: 処理対象のサブフォルダが見つかりません")
        return
    
    print(f"処理対象サブフォルダ数: {len(subfolders)}")
    for i, folder in enumerate(subfolders, 1):
        print(f"  {i}. {folder.name}")
    
    if len(subfolders) != 8:
        print(f"\n警告: サブフォルダ数が {len(subfolders)} です（8つ想定）")
        response = input("続行しますか? (y/n): ")
        if response.lower() != 'y':
            print("処理を中止しました")
            return
    
    # 移動するフォルダ名
    target_folders = ["fl", "fr", "sagi"]
    
    total_moved = 0
    total_errors = 0
    
    print("\n=== 移動処理開始 ===")
    
    for subfolder in subfolders:
        print(f"\n--- {subfolder.name} の処理 ---")
        
        # goproフォルダのパス（存在しない場合は作成）
        gopro_folder = subfolder / "gopro"
        gopro_folder.mkdir(exist_ok=True)
        
        for target_name in target_folders:
            source_folder = subfolder / target_name
            dest_folder = gopro_folder / target_name
            
            # フォルダが存在するか確認
            if not source_folder.exists():
                print(f"  警告: {target_name} フォルダが見つかりません。スキップします。")
                continue
            
            try:
                # 移動先に既に存在する場合は削除
                if dest_folder.exists():
                    print(f"  既存の gopro/{target_name} を削除します")
                    shutil.rmtree(dest_folder)
                
                # フォルダを移動
                shutil.move(str(source_folder), str(dest_folder))
                print(f"  ✓ {target_name} -> gopro/{target_name}")
                total_moved += 1
            except Exception as e:
                print(f"  エラー: {target_name} の移動に失敗: {e}")
                total_errors += 1
    
    print(f"\n=== 処理完了 ===")
    print(f"移動成功: {total_moved} フォルダ")
    print(f"エラー: {total_errors} 件")

if __name__ == "__main__":
    # # Tkinterのルートウィンドウを非表示
    # root = Tk()
    # root.withdraw()
    
    # フォルダ選択ダイアログを表示
    # root_folder = filedialog.askdirectory(title="ルートフォルダを選択してください")
    
    # 対象フォルダのパスを直接指定
    root_folder = r"G:\gait_pattern\BR9G_shuron\sub10"
    
    if root_folder:
        print(f"選択されたフォルダ: {root_folder}\n")
        move_to_gopro(root_folder)
    else:
        print("フォルダが選択されませんでした")