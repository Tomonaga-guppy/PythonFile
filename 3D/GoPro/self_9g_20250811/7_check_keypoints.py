import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import glob

def load_3d_pose_json(json_file_path):
    """
    3Dポーズ結果のJSONファイルを読み込む
    
    Args:
        json_file_path: JSONファイルパス
    
    Returns:
        points_3d: 3D座標 (N, 3)
        keypoint_indices: キーポイントインデックス
        keypoint_names: キーポイント名
        metadata: その他のメタデータ
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    points_3d = np.array(data['points_3d'])
    keypoint_indices = data['keypoint_indices']
    keypoint_names = data['keypoint_names']
    
    # メタデータ
    metadata = {
        'frame_file': data.get('frame_file', ''),
        'subject': data.get('subject', ''),
        'therapist': data.get('therapist', ''),
        'num_points': data.get('num_points', len(points_3d)),
        'statistics': data.get('statistics', {})
    }
    
    return points_3d, keypoint_indices, keypoint_names, metadata

def get_skeleton_connections():
    """
    OpenPose COCO形式のスケルトン接続を取得
    
    Returns:
        skeleton_connections: (start_idx, end_idx)のタプルのリスト
        connection_names: 各接続の名前
    """
    skeleton_connections = [
        # 頭部と首
        (0, 1),   # Nose -> Neck
        (1, 2),   # Neck -> RShoulder
        (1, 5),   # Neck -> LShoulder
        (0, 15),  # Nose -> REye
        (0, 16),  # Nose -> LEye
        (15, 17), # REye -> REar
        (16, 18), # LEye -> LEar
        
        # 右腕
        (2, 3),   # RShoulder -> RElbow
        (3, 4),   # RElbow -> RWrist
        
        # 左腕
        (5, 6),   # LShoulder -> LElbow
        (6, 7),   # LElbow -> LWrist
        
        # 体幹
        (1, 8),   # Neck -> MidHip
        (8, 9),   # MidHip -> RHip
        (8, 12),  # MidHip -> LHip
        
        # 右脚
        (9, 10),  # RHip -> RKnee
        (10, 11), # RKnee -> RAnkle
        (11, 22), # RAnkle -> RBigToe
        (11, 24), # RAnkle -> RHeel
        (22, 23), # RBigToe -> RSmallToe
        
        # 左脚
        (12, 13), # LHip -> LKnee
        (13, 14), # LKnee -> LAnkle
        (14, 19), # LAnkle -> LBigToe
        (14, 21), # LAnkle -> LHeel
        (19, 20), # LBigToe -> LSmallToe
    ]
    
    connection_names = [
        "Nose-Neck", "Neck-RShoulder", "Neck-LShoulder", "Nose-REye", "Nose-LEye",
        "REye-REar", "LEye-LEar", "RShoulder-RElbow", "RElbow-RWrist",
        "LShoulder-LElbow", "LElbow-LWrist", "Neck-MidHip", "MidHip-RHip",
        "MidHip-LHip", "RHip-RKnee", "RKnee-RAnkle", "RAnkle-RBigToe",
        "RAnkle-RHeel", "RBigToe-RSmallToe", "LHip-LKnee", "LKnee-LAnkle",
        "LAnkle-LBigToe", "LAnkle-LHeel", "LBigToe-LSmallToe"
    ]
    
    return skeleton_connections, connection_names

def plot_3d_stick_figure(points_3d, keypoint_indices, keypoint_names, metadata, 
                         show_keypoint_labels=True, show_keypoint_numbers=False,
                         save_path=None, figsize=(12, 9)):
    """
    3Dスティックフィギュアをプロット
    
    Args:
        points_3d: 3D座標 (N, 3)
        keypoint_indices: キーポイントインデックス
        keypoint_names: キーポイント名
        metadata: メタデータ
        show_keypoint_labels: キーポイント名を表示するか
        show_keypoint_numbers: キーポイント番号を表示するか
        save_path: 保存パス（Noneの場合は保存しない）
        figsize: 図のサイズ
    """
    # 図の作成
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # スケルトン接続を取得
    skeleton_connections, connection_names = get_skeleton_connections()
    
    # キーポイントをプロット
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
               c='red', s=80, alpha=0.8, label='Keypoints', zorder=5)
    
    # スケルトンの線を描画
    connection_count = 0
    for i, (start_idx, end_idx) in enumerate(skeleton_connections):
        if start_idx in keypoint_indices and end_idx in keypoint_indices:
            try:
                start_pos = keypoint_indices.index(start_idx)
                end_pos = keypoint_indices.index(end_idx)
                start_point = points_3d[start_pos]
                end_point = points_3d[end_pos]
                
                # 身体の部位によって色を変える
                if i < 7:  # 頭部・首
                    color = 'blue'
                    linewidth = 2
                elif i < 11:  # 腕
                    color = 'green'
                    linewidth = 2.5
                elif i < 14:  # 体幹
                    color = 'purple'
                    linewidth = 3
                else:  # 脚
                    color = 'orange'
                    linewidth = 2.5
                
                ax.plot([start_point[0], end_point[0]], 
                       [start_point[1], end_point[1]], 
                       [start_point[2], end_point[2]], 
                       color=color, alpha=0.8, linewidth=linewidth, zorder=3)
                connection_count += 1
                
            except ValueError:
                continue
    
    # キーポイントラベルを表示
    if show_keypoint_labels or show_keypoint_numbers:
        for i, (point, kp_idx, kp_name) in enumerate(zip(points_3d, keypoint_indices, keypoint_names)):
            label_text = ""
            if show_keypoint_numbers:
                label_text += f"{kp_idx}"
            if show_keypoint_labels:
                if show_keypoint_numbers:
                    label_text += f":{kp_name}"
                else:
                    label_text = kp_name
            
            ax.text(point[0], point[1], point[2], label_text, 
                   fontsize=8, alpha=0.9, zorder=6)
    
    # 軸の設定
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    
    # タイトル設定
    title = f"3D Stick Figure - {metadata.get('subject', 'Unknown')}/{metadata.get('therapist', 'Unknown')}"
    frame_info = metadata.get('frame_file', '').replace('_keypoints.json', '')
    if frame_info:
        title += f"\nFrame: {frame_info}"
    title += f" ({len(points_3d)} keypoints, {connection_count} connections)"
    ax.set_title(title, fontsize=14, pad=20)
    
    # 軸の範囲を調整
    if len(points_3d) > 0:
        # 各軸の範囲を計算
        x_range = points_3d[:, 0].max() - points_3d[:, 0].min()
        y_range = points_3d[:, 1].max() - points_3d[:, 1].min()
        z_range = points_3d[:, 2].max() - points_3d[:, 2].min()
        max_range = max(x_range, y_range, z_range) / 2.0
        
        # 中心点を計算
        mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
        mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
        mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
        
        # 軸の範囲を設定（正方形になるように）
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 統計情報を表示
    stats = metadata.get('statistics', {})
    if stats:
        stats_text = f"Mean: ({stats.get('mean', [0,0,0])[0]:.1f}, {stats.get('mean', [0,0,0])[1]:.1f}, {stats.get('mean', [0,0,0])[2]:.1f}) mm"
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # レジェンド
    ax.legend(loc='upper right')
    
    # レイアウト調整
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"スティックフィギュアを保存: {save_path}")
    
    # 表示
    plt.show()
    
    return fig, ax

def find_3d_pose_files(search_dir):
    """
    指定ディレクトリから3Dポーズ結果ファイルを検索
    
    Args:
        search_dir: 検索ディレクトリ
    
    Returns:
        pose_files: 見つかったJSONファイルのリスト
    """
    search_path = Path(search_dir)
    pose_files = []
    
    # 3d_pose_results ディレクトリを再帰的に検索
    for pose_dir in search_path.rglob("3d_pose_results"):
        json_files = list(pose_dir.glob("3d_pose_*.json"))
        pose_files.extend(json_files)
    
    return sorted(pose_files)

def interactive_frame_selector():
    """
    インタラクティブにフレームを選択して可視化
    """
    # --- パラメータ設定 ---
    root_dir = Path(r"G:\gait_pattern\20250811_br")
    
    print("3Dポーズスティックフィギュア可視化プログラム")
    print(f"検索ディレクトリ: {root_dir}")
    print("=" * 60)
    
    # 3Dポーズファイルを検索
    pose_files = find_3d_pose_files(root_dir)
    
    if not pose_files:
        print("3Dポーズ結果ファイルが見つかりません。")
        return
    
    print(f"見つかった3Dポーズファイル数: {len(pose_files)}")
    
    # ファイルをグループ化（被験者/セラピスト別）
    file_groups = {}
    for file_path in pose_files:
        # パスから被験者とセラピスト情報を抽出
        parts = file_path.parts
        subject = None
        therapist = None
        
        for part in parts:
            if part.startswith('sub'):
                subject = part
            elif part.startswith('thera'):
                therapist = part
        
        if subject and therapist:
            group_key = f"{subject}/{therapist}"
            if group_key not in file_groups:
                file_groups[group_key] = []
            file_groups[group_key].append(file_path)
    
    # グループを表示
    print("\n利用可能な被験者/セラピストの組み合わせ:")
    for i, group_key in enumerate(sorted(file_groups.keys())):
        file_count = len(file_groups[group_key])
        print(f"  {i+1}. {group_key} ({file_count} フレーム)")
    
    # グループ選択
    while True:
        try:
            group_choice = int(input(f"\n組み合わせを選択してください (1-{len(file_groups)}): ")) - 1
            if 0 <= group_choice < len(file_groups):
                break
            else:
                print("無効な選択です。")
        except ValueError:
            print("数字を入力してください。")
    
    selected_group = sorted(file_groups.keys())[group_choice]
    selected_files = file_groups[selected_group]
    
    print(f"\n選択されたグループ: {selected_group}")
    print(f"利用可能なフレーム数: {len(selected_files)}")
    
    # フレーム一覧表示（最初の10個）
    print("\nフレーム一覧（最初の10個）:")
    for i, file_path in enumerate(selected_files[:10]):
        frame_name = file_path.name.replace('3d_pose_', '').replace('.json', '')
        print(f"  {i+1}. {frame_name}")
    
    if len(selected_files) > 10:
        print(f"  ... (他 {len(selected_files) - 10} フレーム)")
    
    # フレーム選択オプション
    print("\nオプション:")
    print("  1. 特定のフレーム番号を指定")
    print("  2. ランダムに5フレーム表示")
    print("  3. 最初の5フレームを表示")
    print("  4. 最後の5フレームを表示")
    
    while True:
        try:
            option = int(input("オプションを選択してください (1-4): "))
            if 1 <= option <= 4:
                break
            else:
                print("無効な選択です。")
        except ValueError:
            print("数字を入力してください。")
    
    # 可視化オプション
    show_labels = input("\nキーポイント名を表示しますか？ (y/n): ").lower() == 'y'
    show_numbers = input("キーポイント番号を表示しますか？ (y/n): ").lower() == 'y'
    save_images = input("画像を保存しますか？ (y/n): ").lower() == 'y'
    
    # 保存ディレクトリ
    if save_images:
        save_dir = root_dir / "3d_stick_figures"
        save_dir.mkdir(exist_ok=True)
    else:
        save_dir = None
    
    # 選択されたオプションに応じて処理
    files_to_process = []
    
    if option == 1:
        # 特定のフレーム番号を指定
        while True:
            try:
                frame_num = int(input(f"フレーム番号を入力してください (1-{len(selected_files)}): ")) - 1
                if 0 <= frame_num < len(selected_files):
                    files_to_process = [selected_files[frame_num]]
                    break
                else:
                    print("無効なフレーム番号です。")
            except ValueError:
                print("数字を入力してください。")
    
    elif option == 2:
        # ランダムに5フレーム
        import random
        files_to_process = random.sample(selected_files, min(5, len(selected_files)))
    
    elif option == 3:
        # 最初の5フレーム
        files_to_process = selected_files[:5]
    
    elif option == 4:
        # 最後の5フレーム
        files_to_process = selected_files[-5:]
    
    # 可視化実行
    print(f"\n{len(files_to_process)} フレームを可視化中...")
    
    for i, file_path in enumerate(files_to_process):
        print(f"\n処理中 ({i+1}/{len(files_to_process)}): {file_path.name}")
        
        try:
            # 3Dポーズデータを読み込み
            points_3d, keypoint_indices, keypoint_names, metadata = load_3d_pose_json(file_path)
            
            # 保存パスを設定
            save_path = None
            if save_dir:
                save_path = save_dir / f"{file_path.stem}_stick_figure.png"
            
            # スティックフィギュアをプロット
            fig, ax = plot_3d_stick_figure(
                points_3d, keypoint_indices, keypoint_names, metadata,
                show_keypoint_labels=show_labels,
                show_keypoint_numbers=show_numbers,
                save_path=save_path
            )
            
            print(f"  キーポイント数: {len(points_3d)}")
            print(f"  検出されたキーポイント: {', '.join(keypoint_names[:5])}{'...' if len(keypoint_names) > 5 else ''}")
            
        except Exception as e:
            print(f"  エラー: {e}")
    
    print(f"\n可視化完了！")
    if save_dir:
        print(f"画像保存先: {save_dir}")

def main():
    """
    メイン処理
    """
    print("3Dポーズスティックフィギュア可視化プログラム")
    print("=" * 60)
    
    # インタラクティブモードで実行
    interactive_frame_selector()

if __name__ == '__main__':
    main()