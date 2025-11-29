import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tqdm import tqdm
import warnings

from m_triangulation import triangulate_and_rotate
from camera_utils import (
    undistort_points, 
    triangulate_with_checks,
    bundle_adjustment_simple,
    calculate_reprojection_error
)
from postprocessing_utils import (
    filter_nans_advanced,
    butterworth_filter_advanced,
    interpolate_markers_temporal,
    calculate_3d_confidence_metrics
)

# =============================================================================
# 設定と定数
# =============================================================================
ROOT_DIR = Path(r"G:\gait_pattern\20250811_br")
STEREO_CALI_DIR = Path(r"G:\gait_pattern\stereo_cali\9g_20250811")
BUTTERWORTH_CUTOFF = 6.0
FRAME_RATE = 60

# OpenCap風の処理設定
OPENCAP_CONFIG = {
    'min_confidence': 0.1,
    'min_parallax_angle': 15.0,  # 度
    'max_reprojection_error': 10.0,  # ピクセル
    'use_bundle_adjustment': True,
    'interpolation_method': 'cubic',  # 'cubic', 'pchip', 'linear'
    'filter_order': 4,
    'max_gap_frames': 10
}

# =============================================================================
# 座標変換関数
# =============================================================================

def rotate_coordinates_x_axis(points_3d, angle_degrees=180, translation=None):
    """
    3D座標をX軸周りに回転させた後、平行移動を適用する
    
    Args:
        points_3d: (num_frames, num_keypoints, 3) または (num_keypoints, 3)
        angle_degrees: 回転角度(度)
        translation: 平行移動ベクトル [x, y, z]。Noneの場合はデフォルト値を使用
    
    Returns:
        transformed_points: 変換後の3D座標
    """
    if translation is None:
        translation = np.array([-35, 189, 0])
    
    angle_rad = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    original_shape = points_3d.shape
    
    # 2次元配列に変換 (num_points, 3)
    if points_3d.ndim == 3:
        num_frames, num_keypoints, _ = points_3d.shape
        points_reshaped = points_3d.reshape(-1, 3)
    else:
        points_reshaped = points_3d
    
    # NaNのマスクを作成
    valid_mask = ~np.isnan(points_reshaped).any(axis=1)
    
    # 変換後の配列を初期化
    transformed = np.full_like(points_reshaped, np.nan)
    
    # 有効な点のみ変換
    if np.any(valid_mask):
        valid_points = points_reshaped[valid_mask]
        rotated = np.dot(valid_points, rotation_matrix.T)
        transformed[valid_mask] = rotated + translation
    
    # 元の形状に戻す
    return transformed.reshape(original_shape)

# =============================================================================
# データの読み込みと準備
# =============================================================================

def load_camera_parameters(params_file):
    """カメラパラメータをJSONファイルから読み込む"""
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # OpenCap形式に変換
    camera_matrix = np.array(params['intrinsics'])
    dist_coeffs = np.array(params.get('distortion', [0, 0, 0, 0, 0]))
    
    R = np.array(params['extrinsics']['rotation_matrix'])
    t = np.array(params['extrinsics']['translation_vector']).reshape(3, 1)
    
    return {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'R': R,
        't': t
    }


def create_projection_matrix(params):
    """カメラパラメータから3x4のプロジェクション行列を作成する"""
    K = params['camera_matrix']
    R = params['R']
    t = params['t']
    return K @ np.hstack([R, t])


def load_csv_2d_data(openpose_csv_path1, openpose_csv_path2):
    """2台のカメラのOpenPose CSVを読み込み、共通フレームのデータを返す"""
    op_df1 = pd.read_csv(openpose_csv_path1, index_col=0)
    op_df2 = pd.read_csv(openpose_csv_path2, index_col=0)
    common_frames = sorted(set(op_df1.index) & set(op_df2.index))
    
    all_kps1 = op_df1.to_numpy().reshape(-1, 25, 3)
    all_kps2 = op_df2.to_numpy().reshape(-1, 25, 3)
    
    return all_kps1, all_kps2, common_frames

# =============================================================================
# 3D座標の計算 (OpenCap風の高度な三角測量)
# =============================================================================

def calculate_3d_with_opencap_methods(kps1_seq, kps2_seq, params1, params2, config):
    """OpenCapの手法を使った3D再構成"""
    num_frames = len(kps1_seq)
    raw_3d_points = np.full((num_frames, 25, 3), np.nan)
    confidences_3d = np.full((num_frames, 25), np.nan)
    reprojection_errors = np.full(num_frames, np.nan)
    
    P1 = create_projection_matrix(params1)
    P2 = create_projection_matrix(params2)
    
    K1 = params1['camera_matrix']
    K2 = params2['camera_matrix']
    dist1 = params1['dist_coeffs']
    dist2 = params2['dist_coeffs']
    
    print(f"\nOpenCap方式で3D座標を計算中 (全{num_frames}フレーム)...")
    print(f"  設定:")
    print(f"    - 最小信頼度: {config['min_confidence']}")
    print(f"    - 最小視差角: {config['min_parallax_angle']}°")
    print(f"    - 最大再投影誤差: {config['max_reprojection_error']} px")
    print(f"    - バンドル調整: {config['use_bundle_adjustment']}")
    
    for i in tqdm(range(num_frames)):
        kp1, cf1 = kps1_seq[i][:, :2], kps1_seq[i][:, 2]
        kp2, cf2 = kps2_seq[i][:, :2], kps2_seq[i][:, 2]
        
        # ステップ1: レンズ歪み補正
        kp1_undist = undistort_points(kp1, K1, dist1)
        kp2_undist = undistort_points(kp2, K2, dist2)
        
        # ステップ2: 視差角チェックとエピポーラ制約を考慮した三角測量
        points_3d, conf_3d = triangulate_with_checks(
            P1, P2, kp1_undist, kp2_undist, cf1, cf2,
            min_confidence=config['min_confidence'],
            min_parallax=config['min_parallax_angle']
        )
        
        # ステップ3: バンドル調整(オプション)
        if config['use_bundle_adjustment']:
            points_3d = bundle_adjustment_simple(
                points_3d, P1, P2, kp1_undist, kp2_undist, conf_3d
            )
        
        # ステップ4: 座標変換(回転+平行移動)を適用
        points_3d = rotate_coordinates_x_axis(points_3d, angle_degrees=180)
        
        # ステップ5: 再投影誤差の計算
        valid_points = ~np.isnan(points_3d).any(axis=1)
        if np.any(valid_points):
            errors = []
            for j in range(25):
                if valid_points[j]:
                    error = calculate_reprojection_error(
                        points_3d[j], P1, P2, kp1_undist[j], kp2_undist[j]
                    )
                    errors.append(error)
            if errors:
                reprojection_errors[i] = np.mean(errors)
        
        raw_3d_points[i] = points_3d
        confidences_3d[i] = conf_3d
    
    print(f"\n  平均再投影誤差: {np.nanmean(reprojection_errors):.2f} px")
    print(f"  最大再投影誤差: {np.nanmax(reprojection_errors):.2f} px")
    
    return raw_3d_points, confidences_3d, reprojection_errors

# =============================================================================
# OpenCapスタイルの後処理パイプライン
# =============================================================================

def opencap_postprocessing_pipeline(raw_3d_points, config):
    """OpenCapの後処理パイプライン"""
    print("\nOpenCap後処理パイプラインを実行中...")
    
    # ステップ1: 品質評価
    print("  [1/4] データ品質を評価中...")
    metrics_raw = calculate_3d_confidence_metrics(raw_3d_points)
    print(f"    - 欠損率: {metrics_raw['missing_rate']*100:.2f}%")
    print(f"    - 最悪キーポイント(#{metrics_raw['worst_keypoint_idx']}): "
          f"{metrics_raw['worst_keypoint_missing_rate']*100:.2f}%")
    
    # ステップ2: 短いギャップの補間
    print(f"  [2/4] 短いギャップ(≤{config['max_gap_frames']}フレーム)を補間中...")
    gap_filled = interpolate_markers_temporal(
        raw_3d_points, 
        max_gap=config['max_gap_frames']
    )
    
    # ステップ3: 高度な欠損値補間
    print(f"  [3/4] {config['interpolation_method']}補間で欠損値を埋めています...")
    interpolated = filter_nans_advanced(
        gap_filled, 
        method=config['interpolation_method']
    )
    
    # ステップ4: バターワースフィルタ
    print(f"  [4/4] {config['filter_order']}次バターワースフィルタ(カットオフ={BUTTERWORTH_CUTOFF}Hz)を適用中...")
    filtered = butterworth_filter_advanced(
        interpolated, 
        cutoff=BUTTERWORTH_CUTOFF, 
        fs=FRAME_RATE,
        order=config['filter_order'],
        bidirectional=True
    )
    
    # 最終品質評価
    metrics_final = calculate_3d_confidence_metrics(filtered)
    print(f"\n  最終結果:")
    print(f"    - 欠損率: {metrics_final['missing_rate']*100:.2f}%")
    print(f"    - 平均速度: {metrics_final['mean_velocity']:.2f} mm/frame")
    print(f"    - 平均加速度: {metrics_final['mean_acceleration']:.2f} mm/frame²")
    
    return gap_filled, interpolated, filtered, (metrics_raw, metrics_final)

# =============================================================================
# OpenPoseスケルトン定義と配色
# =============================================================================

def get_skeleton_connections():
    """OpenPose (BODY_25) のスケルトン接続定義と公式の色分けを返す"""
    connections = [
        (1, 8),   (1, 2),   (1, 5),   (2, 3),   (3, 4),   (5, 6),   (6, 7),
        (8, 9),   (8, 12),  (9, 10),  (10, 11), (12, 13), (13, 14), (1, 0),
        (0, 15),  (15, 17), (0, 16),  (16, 18), (11, 24), (11, 22), (22, 23),
        (14, 21), (14, 19), (19, 20)
    ]

    colors = [
        '#FF0000', '#FF5500', '#FFAA00', '#FFFF00', '#AAFF00', '#55FF00', '#00FF00',
        '#00FF55', '#00FFAA', '#00FFFF', '#00AAFF', '#0055FF', '#0000FF', '#5500FF',
        '#AA00FF', '#FF00FF', '#FF00AA', '#FF0055', '#00FFFF', '#00FFFF', '#00FFFF',
        '#5500FF', '#5500FF', '#5500FF'
    ]
    return connections, colors

# =============================================================================
# アニメーション作成
# =============================================================================

def update_animation_frame(frame_idx, data_3d, skeleton_plots, text_plot, ax):
    """アニメーションの各フレームを更新する"""
    title_text, frame_info_text = text_plot
    keypoint_scatter, skeleton_lines = skeleton_plots
    
    all_plot_elements = []
    
    keypoints_3d = data_3d[frame_idx]
    
    # 座標系を変換 (Z, X, Y)
    transformed_points = np.column_stack([
        keypoints_3d[:, 2], 
        keypoints_3d[:, 0], 
        keypoints_3d[:, 1]
    ])
    
    # キーポイントの描画
    valid_points = transformed_points[~np.isnan(transformed_points).any(axis=1)]
    if valid_points.shape[0] > 0:
        keypoint_scatter._offsets3d = (
            valid_points[:, 0], 
            valid_points[:, 1], 
            valid_points[:, 2]
        )
    else:
        keypoint_scatter._offsets3d = ([], [], [])
    all_plot_elements.append(keypoint_scatter)
    
    # スケルトンの描画
    connections, _ = get_skeleton_connections()
    for i, (start_idx, end_idx) in enumerate(connections):
        if not (np.isnan(keypoints_3d[start_idx]).any() or 
                np.isnan(keypoints_3d[end_idx]).any()):
            start = transformed_points[start_idx]
            end = transformed_points[end_idx]
            skeleton_lines[i].set_data_3d(
                [start[0], end[0]], 
                [start[1], end[1]], 
                [start[2], end[2]]
            )
        else:
            skeleton_lines[i].set_data_3d([], [], [])
    all_plot_elements.extend(skeleton_lines)
    
    frame_info = f"Frame: {frame_idx + 1}/{len(data_3d)}"
    frame_info_text.set_text(frame_info)
    all_plot_elements.extend([title_text, frame_info_text])
    
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 2000)
    ax.set_zlim(0, 2000)
    
    return all_plot_elements


def create_3d_animation(data_3d, output_path, title, all_data_for_limits):
    """3Dスティックフィギュアのアニメーションを作成し保存する"""
    print(f"  - アニメーションを作成中: {output_path.name}")
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 2000)
    ax.set_zlim(0, 2000)
    
    ax.set_xlabel('Z-axis (mm) - Forward', fontsize=14, labelpad=15)
    ax.set_ylabel('X-axis (mm) - Sideways', fontsize=14, labelpad=15)
    ax.set_zlabel('Y-axis (mm) - Up', fontsize=14, labelpad=15)
    
    ax.view_init(elev=10, azim=45)
    ax.set_autoscale_on(False)
    
    connections, colors = get_skeleton_connections()
    keypoint_scatter = ax.scatter([], [], [], c='red', s=40, depthshade=True, zorder=5)
    skeleton_lines = [ax.plot([], [], [], color=c, lw=2.5)[0] for c in colors]
    skeleton_plots = (keypoint_scatter, skeleton_lines)
    
    title_text = ax.text2D(0.5, 0.95, title, transform=ax.transAxes, 
                           fontsize=16, ha='center')
    frame_info_text = ax.text2D(0.02, 0.02, "", transform=ax.transAxes, 
                                fontsize=10, va='bottom', 
                                bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))
    text_plot = (title_text, frame_info_text)
    
    ani = animation.FuncAnimation(
        fig, update_animation_frame, frames=len(data_3d),
        fargs=(data_3d, skeleton_plots, text_plot, ax), 
        blit=False, interval=1000/FRAME_RATE
    )
    
    print(f"    > 保存中...")
    try:
        writer = animation.FFMpegWriter(
            fps=FRAME_RATE, 
            metadata=dict(artist='OpenCap-Style'), 
            bitrate=3600
        )
        ani.save(output_path, writer=writer)
        print(f"  ✓ 保存が完了しました: {output_path.name}")
    except Exception as e:
        print(f"  ✗ アニメーション保存中にエラーが発生しました: {e}")
    
    plt.close(fig)

# =============================================================================
# メイン実行部
# =============================================================================

def main():
    """メインの処理パイプライン"""
    directions = ["fl", "fr"]
    
    try:
        params1 = load_camera_parameters(
            STEREO_CALI_DIR / directions[0] / "camera_params_with_ext_OC.json"
        )
        params2 = load_camera_parameters(
            STEREO_CALI_DIR / directions[1] / "camera_params_with_ext_OC.json"
        )
    except FileNotFoundError as e:
        print(f"✗ エラー: カメラパラメータファイルが見つかりません。{e}")
        return
    
    subject_dirs = sorted([
        d for d in ROOT_DIR.iterdir()
        if d.is_dir() and d.name.startswith("sub")
    ])
    
    for subject_dir in subject_dirs:
        thera_dirs = sorted([
            d for d in subject_dir.iterdir()
            if d.is_dir() and d.name.startswith("thera")
        ])
        
        for thera_dir in thera_dirs:
            if subject_dir.name != "sub1" or thera_dir.name != "thera0-3_claude":
                continue
            
            print(f"\n{'='*80}\n処理開始: {thera_dir.relative_to(ROOT_DIR)}")
            
            openpose_csv_path1 = thera_dir / "fl" / "openpose_kalman.csv"
            openpose_csv_path2 = thera_dir / "fr" / "openpose_kalman.csv"
            
            kps1_seq, kps2_seq, frames = load_csv_2d_data(
                openpose_csv_path1, openpose_csv_path2
            )
            
            if not frames:
                print(f"  - スキップ: 共通フレームが見つかりません。")
                continue
            
            print(f"kps1_seq shape: {kps1_seq.shape}, kps2_seq shape: {kps2_seq.shape}")
            
            # OpenCap風の3D再構成 (座標変換を含む)
            raw_kp_3d, confidences_3d, reproj_errors = calculate_3d_with_opencap_methods(
                kps1_seq, kps2_seq, params1, params2, OPENCAP_CONFIG
            )
            
            # OpenCap風の後処理
            gap_filled, interpolated, filtered, metrics = opencap_postprocessing_pipeline(
                raw_kp_3d, OPENCAP_CONFIG
            )
            
            # 3次元座標とメトリクスを保存
            npz_path = thera_dir / f"3d_kp_data_opencap_{openpose_csv_path1.stem}.npz"
            np.savez(
                npz_path, 
                frame=frames, 
                raw=raw_kp_3d, 
                gap_filled=gap_filled,
                interpolated=interpolated, 
                filtered=filtered, 
                confidence=confidences_3d,
                reprojection_errors=reproj_errors,
                metrics_raw=metrics[0],
                metrics_final=metrics[1],
                config=OPENCAP_CONFIG
            )
            print(f"  - 3Dキーポイントデータを保存しました: {npz_path.name}")
            
            # 3Dアニメーションを作成
            output_anim_dir = thera_dir / "3d_gait_anim_opencap"
            output_anim_dir.mkdir(parents=True, exist_ok=True)
            print("\n3Dスティックフィギュアアニメーションの作成を開始します...")
            
            all_data = np.concatenate([raw_kp_3d, gap_filled, interpolated, filtered], axis=0)
            
            create_3d_animation(
                raw_kp_3d, 
                output_anim_dir / "1_raw_opencap.mp4",
                f"Raw 3D (OpenCap + Rotated) - {subject_dir.name} / {thera_dir.name}",
                all_data
            )
            
            # create_3d_animation(
            #     gap_filled, 
            #     output_anim_dir / "2_gap_filled_opencap.mp4",
            #     f"Gap Filled - {subject_dir.name} / {thera_dir.name}",
            #     all_data
            # )
            
            # create_3d_animation(
            #     interpolated, 
            #     output_anim_dir / "3_interpolated_opencap.mp4",
            #     f"Interpolated ({OPENCAP_CONFIG['interpolation_method']}) - {subject_dir.name} / {thera_dir.name}",
            #     all_data
            # )
            
            create_3d_animation(
                filtered, 
                output_anim_dir / "4_filtered_opencap.mp4",
                f"Filtered (Butterworth) - {subject_dir.name} / {thera_dir.name}",
                all_data
            )
            
            print(f"\n処理完了: {thera_dir.relative_to(ROOT_DIR)}")


if __name__ == '__main__':
    main()