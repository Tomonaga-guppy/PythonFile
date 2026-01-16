"""
goproデータをsub-theraに振り分けるスクリプト
*想定した生成と階層少し違うので注意!chage_goproでなおせるが以降使う場合は本スクリプト変更するほうが絶対楽
"""

import shutil
from pathlib import Path
from tkinter import Tk, filedialog

# 使用する動画のインデックス（1スタート）を9つ定義
# 固定キャリ，Tpose, Tpose, 通常歩行，通常歩行，疑似麻痺歩行(右)，歩行，通常歩行，通常歩行
VIDEO_INDICES = [1, 11, 12, 13, 14, 15, 16, 17, 18]

def copy_gopro_videos(source_root, dest_root):
    """
    コピー元のfl/fr/sagiフォルダから指定インデックスの動画を
    コピー先の対応するサブフォルダにコピー
    コピー元参考：G:\gait_pattern\BR9G_shuron\ori_data\20251127_sub5-6\gopro
    コピー先参考：G:\gait_pattern\BR9G_shuron\sub5
    """
    source_path = Path(source_root)
    dest_path = Path(dest_root)
    
    # コピー先のサブフォルダを取得（名前順にソート）
    dest_subfolders = sorted(
        [d for d in dest_path.iterdir() if d.is_dir()],
        key=lambda x: x.name.lower()
    )
    
    if len(dest_subfolders) != 9:
        print(f"警告: コピー先のサブフォルダ数が {len(dest_subfolders)} です（9つ必要）")
        if len(dest_subfolders) < 9:
            print("処理を中止します")
            return
    
    print(f"コピー先サブフォルダ:")
    for i, folder in enumerate(dest_subfolders[:9], 1):
        print(f"  {i}. {folder.name}")
    
    # 処理するフォルダ名
    folder_names = ["fl", "fr", "sagi"]
    
    total_copied = 0
    total_errors = 0
    
    for folder_name in folder_names:
        print(f"\n--- {folder_name} フォルダの処理 ---")
        
        # コピー元フォルダ
        source_folder = source_path / folder_name
        
        if not source_folder.exists():
            print(f"警告: {source_folder} が見つかりません。スキップします。")
            continue
        
        # コピー元の動画ファイルを取得（名前順にソート）
        video_extensions = ['.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI']
        source_videos = sorted(
            [f for f in source_folder.iterdir() 
             if f.is_file() and f.suffix in video_extensions],
            key=lambda x: x.name.lower()
        )
        
        if len(source_videos) == 0:
            print(f"警告: {folder_name} フォルダに動画ファイルがありません。スキップします。")
            continue
        
        print(f"動画ファイル数: {len(source_videos)}")
        
        # 各インデックスに対応する動画をコピー
        for i, (video_index, dest_subfolder) in enumerate(zip(VIDEO_INDICES[:9], dest_subfolders[:9]), 1):
            # インデックスが範囲内か確認（1スタートなので-1）
            if video_index < 1 or video_index > len(source_videos):
                print(f"  {i}. エラー: インデックス {video_index} は範囲外です（1-{len(source_videos)}）")
                total_errors += 1
                continue
            
            # コピー元ファイル（1スタートなので-1）
            source_video = source_videos[video_index - 1]
            
            # コピー先フォルダ
            dest_folder = dest_subfolder / "gopro" / folder_name
            dest_folder.mkdir(exist_ok=True)
            
            # コピー先ファイルパス
            dest_file = dest_folder / source_video.name
            
            try:
                print(f"  {i}. {source_video.name} (インデックス:{video_index}) -> {dest_subfolder.name}/{folder_name}/")
                shutil.copy2(source_video, dest_file)
                total_copied += 1
            except Exception as e:
                print(f"  エラー: {source_video.name} のコピーに失敗: {e}")
                total_errors += 1
    
    print(f"\n=== 処理完了 ===")
    print(f"コピー成功: {total_copied} ファイル")
    print(f"エラー: {total_errors} 件")

if __name__ == "__main__":
    # Tkinterのルートウィンドウを非表示
    root = Tk()
    root.withdraw()
    
    # コピー元フォルダを選択
    print("コピー元のフォルダ（GoPro データ）を選択してください")
    source_folder = filedialog.askdirectory(title="コピー元フォルダを選択")
    
    if not source_folder:
        print("コピー元フォルダが選択されませんでした")
        exit()
    
    # コピー先ルートフォルダを選択
    print("\nコピー先のルートフォルダを選択してください")
    dest_folder = filedialog.askdirectory(title="コピー先ルートフォルダを選択")
    
    if not dest_folder:
        print("コピー先フォルダが選択されませんでした")
        exit()
    
    print(f"\n=== 設定 ===")
    print(f"コピー元: {source_folder}")
    print(f"コピー先: {dest_folder}")
    print(f"使用インデックス: {VIDEO_INDICES}")
    
    response = input("\n処理を開始しますか? (y/n): ")
    if response.lower() == 'y':
        copy_gopro_videos(source_folder, dest_folder)
    else:
        print("処理を中止しました")