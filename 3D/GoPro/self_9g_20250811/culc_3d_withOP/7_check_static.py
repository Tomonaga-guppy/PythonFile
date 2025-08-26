import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# OpenPoseのCOCO形式のキーポイント接続情報
COCO_SKELETON = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
    [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17],
    [8, 12], [11, 5], [2, 8], [5, 11], [8, 11], [12, 8]
]

# キーポイント名の定義
KEYPOINT_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
    "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye",
    "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
]

def load_keypoints_3d(file_path):
    """3Dキーポイントデータを読み込む"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    frames_3d = []
    metadata = data.get('metadata', {})
    
    for frame_data in data['frames']:
        points_3d = []
        for person in frame_data['people']:
            keypoints_3d = person['pose_keypoints_3d']
            
            # 3Dキーポイントを25個のポイント×3座標（x,y,z）+ 信頼度に分割
            points = []
            for i in range(0, len(keypoints_3d), 4):
                if i + 3 < len(keypoints_3d):
                    x, y, z, confidence = keypoints_3d[i:i+4]
                    points.append([x, y, z, confidence])
                else:
                    points.append([0, 0, 0, 0])
            
            if len(points) < 25:
                points.extend([[0, 0, 0, 0]] * (25 - len(points)))
            
            points_3d.append(points[:25])
        
        frames_3d.append(points_3d)
    
    return frames_3d, metadata

def find_valid_frame(frames_3d):
    """有効な人物データを持つフレームを見つける"""
    for frame_idx, frame in enumerate(frames_3d):
        for person_idx, person in enumerate(frame):
            # MidHip（インデックス8）の信頼度をチェック
            if len(person) > 8 and person[8][3] > 0.5:  # 信頼度が0.5以上
                return frame_idx, person_idx
    return 0, 0  # デフォルト

def plot_3d_stick_figure(points_3d, keypoint_indices, keypoint_names, metadata, 
                        save_path=None, show_labels=True, show_ground=True, show_coordinates=True):
    """3Dスティックフィギュアをプロット"""
    
    # 座標変換: (x,y,z) -> (z,x,y) でy軸が上向きになるように調整
    transformed_points = []
    valid_points = []
    
    for i, point in enumerate(points_3d):
        x, y, z, confidence = point
        if confidence > 0.5:  # 信頼度の閾値
            # 座標変換: x→Y軸、y→Z軸、z→X軸
            transformed_x = z    # 元のz座標をX軸に
            transformed_y = x    # 元のx座標をY軸に  
            transformed_z = y    # 元のy座標をZ軸に
            transformed_points.append([transformed_x, transformed_y, transformed_z])
            valid_points.append(i)
    
    if not transformed_points:
        print("有効な3Dポイントが見つかりませんでした")
        return
    
    transformed_points = np.array(transformed_points)
    
    # 3Dプロットの設定
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 軸の設定
    x_coords = transformed_points[:, 0]
    y_coords = transformed_points[:, 1] 
    z_coords = transformed_points[:, 2]
    
    # 軸の範囲を設定（少し余裕を持たせる）
    margin = 200
    x_range = [x_coords.min() - margin, x_coords.max() + margin]
    y_range = [y_coords.min() - margin, y_coords.max() + margin]
    z_range = [z_coords.min() - margin, z_coords.max() + margin]
    
    # Y軸の最小値を-300に固定（地面の表現のため）
    y_range_min = int(np.floor(y_coords.min() / 100) * 100)
    if y_range_min > -300:
        y_range_min = -300
    
    ax.set_xlim(x_range)
    ax.set_ylim(y_range_min, y_range[1])
    ax.set_zlim(z_range)
    
    # 地面の表示
    if show_ground:
        xx, zz = np.meshgrid(np.linspace(x_range[0], x_range[1], 10),
                           np.linspace(z_range[0], z_range[1], 10))
        yy = np.full_like(xx, y_range_min)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='lightgray')
    
    # キーポイントをプロット
    for i, (original_idx, point) in enumerate(zip(valid_points, transformed_points)):
        ax.scatter(point[0], point[1], point[2], c='red', s=50)
        if show_labels and original_idx < len(keypoint_names):
            ax.text(point[0], point[1], point[2], 
                   f'{keypoint_names[original_idx]}({original_idx})', fontsize=8)
    
    # スケルトンの線を描画
    for connection in COCO_SKELETON:
        idx1, idx2 = connection
        if idx1 in valid_points and idx2 in valid_points:
            point1_idx = valid_points.index(idx1)
            point2_idx = valid_points.index(idx2)
            
            point1 = transformed_points[point1_idx]
            point2 = transformed_points[point2_idx]
            
            ax.plot([point1[0], point2[0]], 
                   [point1[1], point2[1]], 
                   [point1[2], point2[2]], 'b-', linewidth=2)
    
    # MidHip座標の表示
    if show_coordinates:
        midhip_idx = 8  # MidHipのインデックス
        if midhip_idx in valid_points:
            midhip_point_idx = valid_points.index(midhip_idx)
            midhip_point = transformed_points[midhip_point_idx]
            coord_text = f'MidHip: ({midhip_point[0]:.1f}, {midhip_point[1]:.1f}, {midhip_point[2]:.1f})'
            ax.text2D(0.02, 0.98, coord_text, transform=ax.transAxes, 
                     fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ビューアングルの設定（矢状面ビュー）
    ax.view_init(elev=0, azim=90)
    
    # 軸ラベルの設定
    ax.set_xlabel('X軸 (前後)', fontsize=12)
    ax.set_ylabel('Y軸 (上下)', fontsize=12)
    ax.set_zlabel('Z軸 (左右)', fontsize=12)
    
    # タイトルの設定
    title = "3D Pose Visualization (Sagittal View)"
    if 'frame_rate' in metadata:
        title += f" - Frame Rate: {metadata['frame_rate']} fps"
    ax.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"画像を保存しました: {save_path}")
    
    plt.show()

def process_static_visualization():
    """静止画可視化の処理"""
    print("=== 3D静止画可視化プログラム ===")
    
    # 現在のディレクトリからthera系フォルダを検索
    current_dir = Path.cwd()
    thera_folders = []
    
    for item in current_dir.iterdir():
        if item.is_dir() and item.name.startswith('thera'):
            thera_folders.append(item)
    
    if not thera_folders:
        print("thera系フォルダが見つかりませんでした")
        return
    
    print(f"\n見つかったthera系フォルダ:")
    for i, folder in enumerate(thera_folders):
        print(f"{i+1}: {folder.name}")
    
    if len(thera_folders) == 1:
        selected_folder = thera_folders[0]
        print(f"\n自動選択: {selected_folder.name}")
    else:
        choice = int(input(f"\nフォルダを選択してください (1-{len(thera_folders)}): ")) - 1
        selected_folder = thera_folders[choice]
    
    # JSONファイルを検索
    json_files = list(selected_folder.glob("*.json"))
    
    if not json_files:
        print(f"{selected_folder.name}内にJSONファイルが見つかりませんでした")
        return
    
    print(f"\n見つかったJSONファイル:")
    for i, file_path in enumerate(json_files):
        print(f"{i+1}: {file_path.name}")
    
    # ファイル選択
    if len(json_files) == 1:
        selected_file = json_files[0]
        print(f"\n自動選択: {selected_file.name}")
    else:
        choice = int(input(f"\nファイルを選択してください (1-{len(json_files)}): ")) - 1
        selected_file = json_files[choice]
    
    # 3Dキーポイントデータを読み込み
    try:
        frames_3d, metadata = load_keypoints_3d(selected_file)
        print(f"読み込み完了: {len(frames_3d)} フレーム")
        
        # 有効なフレームを見つける
        frame_idx, person_idx = find_valid_frame(frames_3d)
        print(f"使用フレーム: {frame_idx}, 人物: {person_idx}")
        
        # 保存先ディレクトリを作成
        save_dir = selected_folder / "3d_static_images"
        save_dir.mkdir(exist_ok=True)
        
        # 静止画を生成
        points_3d = frames_3d[frame_idx][person_idx]
        save_path = save_dir / f"{selected_file.stem}_static_pose.png"
        
        plot_3d_stick_figure(
            points_3d, 
            list(range(25)), 
            KEYPOINT_NAMES,
            metadata,
            save_path=save_path,
            show_labels=True,
            show_ground=True,
            show_coordinates=True
        )
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def main():
    """メイン関数"""
    print("=== 3Dポーズ静止画可視化ツール ===")
    print("JSONファイルから3Dポーズデータの静止画を生成します")
    
    try:
        process_static_visualization()
    except KeyboardInterrupt:
        print("\n\n処理が中断されました")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")

if __name__ == "__main__":
    main()