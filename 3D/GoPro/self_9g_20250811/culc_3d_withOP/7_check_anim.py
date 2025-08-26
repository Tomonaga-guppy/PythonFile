import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import glob
from tqdm import tqdm

def load_3d_pose_json(json_file_path):
    """
    3Dポーズ結果のJSONファイルを読み込む
    
    Args:
        json_file_path: JSONファイルパス
    
    Returns:
        points_3d: 3Dポーズの3D座標配列
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
        'statistics': data.get('statistics', {}),
        'triangulation_method': data.get('triangulation_method', 'unknown'),
        'point_confidences': data.get('point_confidences', [])
    }
    
    return points_3d, keypoint_indices, keypoint_names, metadata

def get_skeleton_connections():
    """
    OpenPose COCO形式のスケルトン接続を取得
    
    Returns:
        skeleton_connections: (start_idx, end_idx)のタプルのリスト
        connection_names: 各接続の名前
        connection_colors: 各接続の色
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
    
    # 接続の色分け
    connection_colors = []
    for i in range(len(skeleton_connections)):
        if i < 7:  # 頭部・首
            connection_colors.append('blue')
        elif i < 11:  # 腕
            connection_colors.append('green')
        elif i < 14:  # 体幹
            connection_colors.append('purple')
        else:  # 脚
            connection_colors.append('orange')
    
    return skeleton_connections, connection_names, connection_colors

def load_animation_sequence(pose_files, max_frames=None):
    """
    アニメーション用に複数フレームのデータを読み込む
    
    Args:
        pose_files: 3Dポーズファイルのリスト
        max_frames: 最大フレーム数（Noneの場合は全て）
    
    Returns:
        animation_data: フレームごとのデータのリスト
        global_bounds: 全フレームでの座標範囲
    """
    animation_data = []
    all_points = []
    
    # フレーム数を制限
    if max_frames:
        pose_files = pose_files[:max_frames]
    
    print(f"アニメーション用データを読み込み中... ({len(pose_files)} フレーム)")
    
    for file_path in tqdm(pose_files):
        try:
            points_3d, keypoint_indices, keypoint_names, metadata = load_3d_pose_json(file_path)
            
            frame_data = {
                'points_3d': points_3d,
                'keypoint_indices': keypoint_indices,
                'keypoint_names': keypoint_names,
                'metadata': metadata,
                'file_path': file_path
            }
            
            animation_data.append(frame_data)
            
            # 全体の座標範囲計算用
            if len(points_3d) > 0:
                all_points.append(points_3d)
                
        except Exception as e:
            print(f"エラー: {file_path} - {e}")
            continue
    
    # 全フレームでの座標範囲を計算（各軸独立で最適化）
    if all_points:
        all_points_concat = np.vstack(all_points)
        # 座標を変換: (x,y,z) -> (z,x,y)
        transformed_all_points = np.column_stack([all_points_concat[:, 2], all_points_concat[:, 0], all_points_concat[:, 1]])
        
        # 各軸のデータ範囲を取得
        x_min, x_max = transformed_all_points[:, 0].min(), transformed_all_points[:, 0].max()
        y_min, y_max = transformed_all_points[:, 1].min(), transformed_all_points[:, 1].max()
        z_min, z_max = transformed_all_points[:, 2].min(), transformed_all_points[:, 2].max()
        
        # マージンを追加（データ範囲の10%または最低500mm）
        x_margin = max((x_max - x_min) * 0.1, 500)
        y_margin = max((y_max - y_min) * 0.1, 500)
        z_margin = max((z_max - z_min) * 0.1, 500)
        
        # マージンを考慮した範囲を計算
        x_range_min = x_min - x_margin
        x_range_max = x_max + x_margin
        y_range_min = y_min - y_margin
        y_range_max = y_max + y_margin
        z_range_min = z_min - z_margin
        z_range_max = z_max + z_margin
        
        # 1000mmの倍数に丸める（下向きと上向き）
        x_range_min = np.floor(x_range_min / 1000) * 1000
        x_range_max = np.ceil(x_range_max / 1000) * 1000
        y_range_min = -300
        y_range_max = np.ceil(y_range_max / 1000) * 1000
        z_range_min = np.floor(z_range_min / 1000) * 1000
        z_range_max = np.ceil(z_range_max / 1000) * 1000
        
        global_bounds = {
            'x': [x_range_min, x_range_max],  # z座標（元のz）
            'y': [y_range_min, y_range_max],  # x座標（元のx）
            'z': [z_range_min, z_range_max]   # y座標（元のy）
        }
    else:
        global_bounds = {
            'x': [-1000, 1000],
            'y': [-1000, 1000],
            'z': [-1000, 1000]
        }
    
    print(f"読み込み完了: {len(animation_data)} フレーム")
    return animation_data, global_bounds

def plot_3d_stick_figure(points_3d, keypoint_indices, keypoint_names, metadata, 
                         show_keypoint_labels=True, show_keypoint_numbers=False,
                         save_path=None, figsize=(12, 9)):
    """
    3Dスティックフィギュアをプロット（歩行向け座標系）
    
    Args:
        points_3d: 3D座標 (N, 3) - 元の座標系
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
    skeleton_connections, connection_names, connection_colors = get_skeleton_connections()
    
    # 座標を変換: (x,y,z) -> (z,x,y) つまり x座標をY軸、y座標をZ軸、z座標をX軸に表示
    transformed_points = np.column_stack([points_3d[:, 2], points_3d[:, 0], points_3d[:, 1]])
    
    # キーポイントをプロット
    ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], 
               c='red', s=80, alpha=0.8, label='Keypoints', zorder=5)
    
    # スケルトンの線を描画
    connection_count = 0
    for i, (start_idx, end_idx) in enumerate(skeleton_connections):
        if start_idx in keypoint_indices and end_idx in keypoint_indices:
            try:
                start_pos = keypoint_indices.index(start_idx)
                end_pos = keypoint_indices.index(end_idx)
                start_point = transformed_points[start_pos]
                end_point = transformed_points[end_pos]
                
                # 身体の部位によって色を変える
                color = connection_colors[i]
                if i < 7:  # 頭部・首
                    linewidth = 2
                elif i < 11:  # 腕
                    linewidth = 2.5
                elif i < 14:  # 体幹
                    linewidth = 3
                else:  # 脚
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
        for i, (point, kp_idx, kp_name) in enumerate(zip(transformed_points, keypoint_indices, keypoint_names)):
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
    
    # 軸の設定（変換後の座標系：z座標をX軸、x座標をY軸、y座標をZ軸）
    ax.set_xlabel('Z coordinate (mm)', fontsize=12)
    ax.set_ylabel('X coordinate (mm)', fontsize=12)
    ax.set_zlabel('Y coordinate (mm)', fontsize=12)
    
    # グリッド線の間隔を1000に設定
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1000))
    
    # タイトル設定
    title = f"3D Stick Figure - {metadata.get('subject', 'Unknown')}/{metadata.get('therapist', 'Unknown')}"
    frame_info = metadata.get('frame_file', '').replace('_keypoints.json', '')
    if frame_info:
        title += f"\nFrame: {frame_info}"
    title += f" ({len(transformed_points)} keypoints, {connection_count} connections)"
    if metadata.get('triangulation_method'):
        title += f"\nMethod: {metadata['triangulation_method']}"
    ax.set_title(title, fontsize=14, pad=20)
    
    # 軸の範囲を調整（各軸で独立して1000mmの倍数に丸める）
    if len(transformed_points) > 0:
        # 各軸のデータ範囲を取得
        x_min, x_max = transformed_points[:, 0].min(), transformed_points[:, 0].max()
        y_min, y_max = transformed_points[:, 1].min(), transformed_points[:, 1].max()
        z_min, z_max = transformed_points[:, 2].min(), transformed_points[:, 2].max()
        
        # マージンを追加（データ範囲の10%または最低500mm）
        x_margin = max((x_max - x_min) * 0.1, 500)
        y_margin = max((y_max - y_min) * 0.1, 500)
        z_margin = max((z_max - z_min) * 0.1, 500)
        
        # マージンを考慮した範囲を計算
        x_range_min = x_min - x_margin
        x_range_max = x_max + x_margin
        y_range_min = y_min - y_margin
        y_range_max = y_max + y_margin
        z_range_min = z_min - z_margin
        z_range_max = z_max + z_margin
        
        # 1000mmの倍数に丸める（下向きと上向き）
        x_range_min = np.floor(x_range_min / 1000) * 1000
        x_range_max = np.ceil(x_range_max / 1000) * 1000
        y_range_min = -300
        y_range_max = np.ceil(y_range_max / 1000) * 1000
        z_range_min = np.floor(z_range_min / 1000) * 1000
        z_range_max = np.ceil(z_range_max / 1000) * 1000
        
        # 軸の範囲を設定（各軸独立）
        ax.set_xlim(x_range_min, x_range_max)
        ax.set_ylim(y_range_min, y_range_max)
        ax.set_zlim(z_range_min, z_range_max)
        
        # 地面をXZ方向（変換後座標でX-Y平面）に描画
        ground_z = 0  # 地面のZ座標（Y座標方向での高さ）
        
        # 平面の範囲を設定
        x_plane_start = np.floor(x_range_min / 1000) * 1000
        x_plane_end = np.ceil(x_range_max / 1000) * 1000
        y_plane_start = y_range_min
        # y_plane_start = np.floor(y_range_min / 1000) * 1000
        y_plane_end = np.ceil(y_range_max / 1000) * 1000
        
        # グレーの平面を描画
        X_plane, Y_plane = np.meshgrid(
            [x_plane_start, x_plane_end], 
            [y_plane_start, y_plane_end]
        )
        Z_plane = np.full_like(X_plane, ground_z)
        
        ax.plot_surface(X_plane, Y_plane, Z_plane, 
                       color='gray', alpha=0.2, zorder=1, 
                       linewidth=0, antialiased=True)
    
    # 地面にX軸とZ軸の基準線を描画（アニメーション用）
    # X軸方向の線（変換後座標でX軸、元のz座標方向）
    ax.plot([x_plane_start, x_plane_end], [0, 0], [ground_z, ground_z],
            color='red', linewidth=3, alpha=0.8, label='Z-axis (forward)', zorder=2)
    
    # Z軸方向の線（変換後座標でY軸、元のx座標方向）  
    ax.plot([0, 0], [y_plane_start, y_plane_end], [ground_z, ground_z],
            color='blue', linewidth=3, alpha=0.8, label='X-axis (sideways)', zorder=2)
    
    # ビューアングルを歩行観察に適した角度に設定
    ax.view_init(elev=10, azim=45)  # 少し上から、斜め横から見た角度
    
    # 統計情報を表示（変換後の座標で）
    if len(transformed_points) > 0:
        stats_text = f"Center: Z: {np.mean(transformed_points[:, 0]):.1f}, X: {np.mean(transformed_points[:, 1]):.1f}, Y: {np.mean(transformed_points[:, 2]):.1f} mm"
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 信頼度情報を表示
    if 'point_confidences' in metadata and metadata['point_confidences']:
        avg_conf = np.mean(metadata['point_confidences'])
        conf_text = f"Avg Confidence: {avg_conf:.3f}"
        ax.text2D(0.02, 0.02, conf_text, transform=ax.transAxes, 
                 fontsize=10, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
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

def create_3d_animation(animation_data, global_bounds, 
                       show_keypoint_labels=False, show_keypoint_numbers=False,
                       save_path=None, figsize=(12, 9), view_name="sagittal", elev=0, azim=90):
    """
    3Dスティックフィギュアのアニメーションを作成（歩行向け座標系）
    
    Args:
        animation_data: フレームごとのデータのリスト
        global_bounds: 全フレームでの座標範囲
        show_keypoint_labels: キーポイント名を表示するか
        show_keypoint_numbers: キーポイント番号を表示するか
        save_path: 保存パス（Noneの場合は表示のみ）
        figsize: 図のサイズ
        view_name: 視点の名前
        elev: elevation角度
        azim: azimuth角度
    
    Returns:
        ani: アニメーションオブジェクト
    """
    if not animation_data:
        print("アニメーションデータがありません。")
        return None
    
    # 図の作成
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # スケルトン接続を取得
    skeleton_connections, connection_names, connection_colors = get_skeleton_connections()
    
    # 軸の範囲を設定（歩行向け座標系）
    ax.set_xlim(global_bounds['x'])  # 歩行方向（前後）
    ax.set_ylim(global_bounds['y'])  # 左右
    ax.set_zlim(global_bounds['z'])  # 上下
    
    # 地面をXZ方向（変換後座標でX-Y平面）に描画（アニメーション用）
    ground_z = 0  # 地面のZ座標（Y座標方向での高さ）
    x_min, x_max = global_bounds['x']
    y_min, y_max = global_bounds['y']
    z_min, z_max = global_bounds['z']
    
    # 平面の範囲を設定
    x_plane_start = np.floor(x_min / 1000) * 1000
    x_plane_end = np.ceil(x_max / 1000) * 1000
    y_plane_start = y_min
    y_plane_end = np.ceil(y_max / 1000) * 1000
    
    # グレーの平面を描画
    X_plane, Y_plane = np.meshgrid(
        [x_plane_start, x_plane_end], 
        [y_plane_start, y_plane_end]
    )
    Z_plane = np.full_like(X_plane, ground_z)
    
    ax.plot_surface(X_plane, Y_plane, Z_plane, 
                   color='gray', alpha=0.2, zorder=1, 
                   linewidth=0, antialiased=True)
    
    # 地面にX軸とZ軸の基準線を描画（アニメーション用）
    # X軸方向の線（変換後座標でX軸、元のz座標方向）
    ax.plot([x_plane_start, x_plane_end], [0, 0], [ground_z, ground_z],
            color='blue', linewidth=3, alpha=0.8, label='Z-axis (forward)', zorder=2)
    
    # Z軸方向の線（変換後座標でY軸、元のx座標方向）  
    ax.plot([0, 0], [y_plane_start, y_plane_end], [ground_z, ground_z],
            color='red', linewidth=3, alpha=0.8, label='X-axis (sideways)', zorder=2)
    
    # ビューアングルを設定
    ax.view_init(elev=elev, azim=azim)

    # 軸ラベルを設定
    ax.set_xlabel('Z coordinate (mm)', fontsize=12)
    ax.set_ylabel('X coordinate (mm)', fontsize=12)
    ax.set_zlabel('Y coordinate (mm)', fontsize=12)
    
    # グリッド線の間隔を1000mmに設定
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1000))
    
    # アニメーション用のプロット要素を初期化
    keypoint_scatter = ax.scatter([], [], [], c='red', s=80, alpha=0.8, zorder=5)
    skeleton_lines = []
    keypoint_texts = []
    
    # スケルトンライン用のオブジェクトを作成
    for i, color in enumerate(connection_colors):
        linewidth = 3 if 11 <= i < 14 else 2.5  # 体幹は太く
        line, = ax.plot([], [], [], color=color, alpha=0.8, linewidth=linewidth, zorder=3)
        skeleton_lines.append(line)
    
    # タイトルテキスト
    title_text = ax.text2D(0.5, 0.95, "", transform=ax.transAxes, 
                          fontsize=14, horizontalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # フレーム情報テキスト
    frame_info_text = ax.text2D(0.02, 0.02, "", transform=ax.transAxes, 
                               fontsize=10, verticalalignment='bottom',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def update_frame(frame_idx):
        """
        アニメーションフレーム更新関数
        """
        if frame_idx >= len(animation_data):
            return []
        
        frame_data = animation_data[frame_idx]
        points_3d = frame_data['points_3d']
        keypoint_indices = frame_data['keypoint_indices']
        keypoint_names = frame_data['keypoint_names']
        metadata = frame_data['metadata']
        
        # 座標を変換: (x,y,z) -> (z,x,y)
        if len(points_3d) > 0:
            transformed_points = np.column_stack([points_3d[:, 2], points_3d[:, 0], points_3d[:, 1]])
        else:
            transformed_points = points_3d
        
        # 既存のテキストをクリア
        for text in keypoint_texts:
            text.remove()
        keypoint_texts.clear()
        
        # キーポイントをプロット
        if len(transformed_points) > 0:
            keypoint_scatter._offsets3d = (transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2])
            
            # スケルトンの線を更新
            for i, (start_idx, end_idx) in enumerate(skeleton_connections):
                if i < len(skeleton_lines) and start_idx in keypoint_indices and end_idx in keypoint_indices:
                    try:
                        start_pos = keypoint_indices.index(start_idx)
                        end_pos = keypoint_indices.index(end_idx)
                        start_point = transformed_points[start_pos]
                        end_point = transformed_points[end_pos]
                        
                        skeleton_lines[i].set_data_3d([start_point[0], end_point[0]], 
                                                     [start_point[1], end_point[1]], 
                                                     [start_point[2], end_point[2]])
                    except (ValueError, IndexError):
                        skeleton_lines[i].set_data_3d([], [], [])
                else:
                    if i < len(skeleton_lines):
                        skeleton_lines[i].set_data_3d([], [], [])
            
            # キーポイントラベルを表示
            if show_keypoint_labels or show_keypoint_numbers:
                for i, (point, kp_idx, kp_name) in enumerate(zip(transformed_points, keypoint_indices, keypoint_names)):
                    label_text = ""
                    if show_keypoint_numbers:
                        label_text += f"{kp_idx}"
                    if show_keypoint_labels:
                        if show_keypoint_numbers:
                            label_text += f":{kp_name}"
                        else:
                            label_text = kp_name
                    
                    text = ax.text(point[0], point[1], point[2], label_text, 
                                  fontsize=8, alpha=0.9, zorder=6)
                    keypoint_texts.append(text)
        else:
            keypoint_scatter._offsets3d = ([], [], [])
            for line in skeleton_lines:
                line.set_data_3d([], [], [])
        
        # タイトル更新
        title = f"3D Walking Animation ({view_name}) - {metadata.get('subject', 'Unknown')}/{metadata.get('therapist', 'Unknown')}"
        if 'triangulation_method' in metadata:
            title += f" ({metadata['triangulation_method']})"
        title_text.set_text(title)
        
        # フレーム情報更新
        frame_file = metadata.get('frame_file', '').replace('_keypoints.json', '')
        confidence_info = ""
        if 'point_confidences' in metadata and metadata['point_confidences']:
            avg_conf = np.mean(metadata['point_confidences'])
            confidence_info = f", Avg Conf: {avg_conf:.3f}"
        
        # MidHip（キーポイント8番）の座標を取得
        midhip_info = ""
        if len(transformed_points) > 0 and 8 in keypoint_indices:
            try:
                midhip_pos = keypoint_indices.index(8)
                midhip_coord = transformed_points[midhip_pos]
                midhip_info = f"\nMidHip: Z:{midhip_coord[0]:.1f}, X:{midhip_coord[1]:.1f}, Y:{midhip_coord[2]:.1f}"
            except (ValueError, IndexError):
                midhip_info = "\nMidHip: Not detected"
        else:
            midhip_info = "\nMidHip: Not detected"
        
        frame_info = f"Frame: {frame_idx+1}/{len(animation_data)} ({frame_file})\nKeypoints: {len(transformed_points)}{confidence_info}{midhip_info}"
        frame_info_text.set_text(frame_info)
        
        return [keypoint_scatter, title_text, frame_info_text] + skeleton_lines + keypoint_texts
    
    # アニメーション作成
    total_frames = len(animation_data)
    fps = 60
    interval = 1000 / fps  # ミリ秒
    
    print(f"歩行アニメーション作成中... ({total_frames} フレーム, {view_name}視点)")
    
    ani = animation.FuncAnimation(
        fig, update_frame, frames=total_frames,
        interval=interval, blit=False, repeat=True
    )
    
    # 保存
    if save_path:
        print(f"アニメーション保存中: {save_path}")
        # MP4で保存（ffmpegが必要）
        try:
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='3D Walking Animation'), bitrate=1800)
            ani.save(save_path, writer=writer, progress_callback=lambda i, n: print(f'保存進捗 ({view_name}): {i}/{n}'))
            print(f"アニメーション保存完了: {save_path}")
        except Exception as e:
            print(f"MP4保存エラー: {e}")
            # GIFで保存を試行
            gif_path = str(save_path).replace('.mp4', '.gif')
            print(f"GIFで保存を試行: {gif_path}")
            try:
                ani.save(gif_path, writer='pillow', fps=fps)
                print(f"GIF保存完了: {gif_path}")
            except Exception as e2:
                print(f"GIF保存エラー: {e2}")
    
    return ani

def find_3d_pose_files(search_dir, pattern="3d_pose_results_weighted_linear_OC"):
    """
    指定ディレクトリから3Dポーズ結果ファイルを検索
    
    Args:
        search_dir: 検索ディレクトリ
        pattern: 検索パターン
    
    Returns:
        pose_files: 見つかったJSONファイルのリスト
    """
    search_path = Path(search_dir)
    pose_files = []
    
    # 指定パターンのディレクトリを再帰的に検索
    for pose_dir in search_path.rglob(pattern):
        json_files = list(pose_dir.glob("3d_pose_*.json"))
        pose_files.extend(json_files)
    
    return sorted(pose_files)

def extract_frame_number(file_path):
    """
    ファイル名からフレーム番号を抽出
    """
    import re
    match = re.search(r'frame_(\d+)', file_path.name)
    if match:
        return int(match.group(1))
    return 0

def parse_selection(input_str, max_num):
    """
    ユーザー入力を解析して選択された番号のリストを返す
    
    Args:
        input_str: ユーザー入力文字列（例: "1,3,5" または "1-3" または "all"）
        max_num: 最大番号
    
    Returns:
        selected_numbers: 選択された番号のリスト（0ベース）
    """
    input_str = input_str.strip().lower()
    
    # "all" の場合は全て選択
    if input_str == "all":
        return list(range(max_num))
    
    selected = set()
    parts = input_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # 範囲指定（例: "1-3"）
            try:
                start, end = part.split('-')
                start_num = int(start) - 1  # 0ベースに変換
                end_num = int(end) - 1      # 0ベースに変換
                for i in range(start_num, end_num + 1):
                    if 0 <= i < max_num:
                        selected.add(i)
            except ValueError:
                continue
        else:
            # 単一選択（例: "1"）
            try:
                num = int(part) - 1  # 0ベースに変換
                if 0 <= num < max_num:
                    selected.add(num)
            except ValueError:
                continue
    
    return sorted(list(selected))

def interactive_animation_creator():
    """
    インタラクティブにアニメーションを作成
    """
    # --- パラメータ設定 ---
    root_dir = Path(r"G:\gait_pattern\20250811_br")
    
    print("3D歩行アニメーション作成プログラム")
    print(f"検索ディレクトリ: {root_dir}")
    print("=" * 60)
    
    # 3Dポーズファイルを検索
    print("3Dポーズファイルを検索中...")
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
    
    # 各グループ内でフレーム順にソート
    for group_key in file_groups:
        file_groups[group_key] = sorted(file_groups[group_key], key=extract_frame_number)
    
    # グループを表示
    sorted_groups = sorted(file_groups.keys())
    print("\n利用可能な被験者/セラピストの組み合わせ:")
    for i, group_key in enumerate(sorted_groups):
        file_count = len(file_groups[group_key])
        print(f"  {i+1}. {group_key} ({file_count} フレーム)")
    
    # 複数グループ選択
    print("\n選択方法:")
    print("  - 単一選択: 1")
    print("  - 複数選択: 1,3,5")
    print("  - 範囲選択: 1-3")
    print("  - 組み合わせ: 1-3,5,7")
    print("  - 全て選択: all")
    
    while True:
        try:
            selection_input = input(f"\n組み合わせを選択してください: ").strip()
            selected_indices = parse_selection(selection_input, len(sorted_groups))
            
            if selected_indices:
                break
            else:
                print("無効な選択です。再度入力してください。")
        except Exception as e:
            print(f"入力エラー: {e}")
    
    # 選択されたグループを表示
    selected_groups = [sorted_groups[i] for i in selected_indices]
    print(f"\n選択されたグループ ({len(selected_groups)}個):")
    total_files = 0
    
    for group in selected_groups:
        frame_count = len(file_groups[group])
        print(f"  - {group} ({frame_count} フレーム)")
        total_files += frame_count
    
    print(f"\n合計ファイル数: {total_files}")
    print("※各グループごとに個別の動画ファイルを作成します")
    
    # アニメーション設定
    print("\nアニメーション設定:")
    print("- キーポイント名: 表示しません")
    print("- キーポイント番号: 表示しません")
    print("- アニメーション: 保存します")
    
    show_labels = False
    show_numbers = False
    save_animation = True
    
    # 視点選択
    view_options = {
        1: {"name": "sagittal", "elev": 0, "azim": 90, "description": "矢状面（横から）"},
        2: {"name": "frontal", "elev": 0, "azim": 0, "description": "前額面（正面から）"},
        3: {"name": "oblique", "elev": 10, "azim": 45, "description": "斜め視点（デフォルト）"}
    }
    
    print("\n視点選択:")
    for key, view in view_options.items():
        print(f"  {key}. {view['description']}")
    print("  4. 全ての視点（1,2,3すべて）")
    
    while True:
        try:
            view_input = input("\n視点を選択してください（複数選択の場合はカンマ区切り、例: 1,3）: ")
            if view_input.strip() == "4":
                selected_views = [1, 2, 3]
                break
            else:
                view_selections = parse_selection(view_input, 3)
                if view_selections:
                    selected_views = [i + 1 for i in view_selections]  # 1ベースに戻す
                    break
                else:
                    print("無効な選択です。1-4の数字を入力してください。")
        except Exception as e:
            print(f"入力エラー: {e}")
    
    print(f"選択された視点: {[view_options[v]['description'] for v in selected_views]}")
    
    # 各グループごとにアニメーション作成
    all_animations = []
    total_combinations = len(selected_groups) * len(selected_views)
    current_combination = 0
    
    print(f"\n歩行アニメーション作成開始")
    print(f"作成予定: {len(selected_groups)} グループ × {len(selected_views)} 視点 = {total_combinations} 個の動画")
    print("=" * 60)
    
    for group_idx, group in enumerate(selected_groups):
        print(f"\nグループ {group_idx+1}/{len(selected_groups)}: {group}")
        
        # グループのファイルリストを取得
        group_files = sorted(file_groups[group], key=extract_frame_number)
        print(f"フレーム数: {len(group_files)}")
        
        # アニメーションデータを読み込み
        print(f"データ読み込み中...")
        animation_data, global_bounds = load_animation_sequence(group_files)
        
        if not animation_data:
            print(f"有効なアニメーションデータがありません: {group}")
            continue
        
        # 保存ディレクトリを設定
        first_file_path = group_files[0]
        thera_folder = None
        for part in first_file_path.parts:
            if part.startswith('thera'):
                thera_index = first_file_path.parts.index(part)
                thera_folder = Path(*first_file_path.parts[:thera_index+1])
                break
        
        if thera_folder:
            save_dir = thera_folder
        else:
            save_dir = root_dir / "3d_walking_animations"
        
        save_dir.mkdir(exist_ok=True)
        
        # 各視点でアニメーション作成
        for view_id in selected_views:
            current_combination += 1
            view_config = view_options[view_id]
            
            print(f"\n進捗 {current_combination}/{total_combinations}: {group} - {view_config['description']}")
            
            # ファイル名生成（グループごと）
            safe_group_name = group.replace('/', '_')
            filename = f"3d_walking_animation_{safe_group_name}_{view_config['name']}_frames{len(group_files)}.mp4"
            save_path = save_dir / filename
            
            print(f"保存先: {save_path}")
            
            try:
                ani = create_3d_animation(
                    animation_data, global_bounds,
                    show_keypoint_labels=show_labels,
                    show_keypoint_numbers=show_numbers,
                    save_path=save_path,
                    view_name=view_config['name'],
                    elev=view_config['elev'],
                    azim=view_config['azim']
                )
                
                if ani:
                    all_animations.append(ani)
                    print(f"✓ 完了: {save_path.name}")
                
            except Exception as e:
                print(f"✗ エラー: {save_path.name} - {e}")
                continue
    
    print(f"\n歩行アニメーション作成完了！")
    print(f"作成された動画数: {len(all_animations)}")
    print("=" * 60)
    
    return all_animations

def main():
    """
    メイン処理
    """
    print("3D歩行ポーズスティックフィギュア可視化プログラム")
    print("=" * 60)
    
    # アニメーションモード
    interactive_animation_creator()

if __name__ == "__main__":
    main()