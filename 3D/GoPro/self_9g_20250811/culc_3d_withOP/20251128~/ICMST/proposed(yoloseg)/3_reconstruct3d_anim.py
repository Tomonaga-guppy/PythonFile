import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import warnings

from m_triangulation import triangulate_and_rotate

# =============================================================================
# 設定と定数
# =============================================================================
ROOT_DIR = Path(r"G:\gait_pattern\BR9G_shuron")
STEREO_CALI_DIR = Path(r"G:\gait_pattern\stereo_cali\9g_20250811")
BUTTERWORTH_CUTOFF = 6.0
FRAME_RATE = 60

# =============================================================================
# データの読み込みと準備
# =============================================================================

def load_camera_parameters(params_file):
    """カメラパラメータをJSONファイルから読み込む"""
    with open(params_file, 'r') as f:
        return json.load(f)


def create_projection_matrix(params):
    """カメラパラメータから3x4のプロジェクション行列を作成する"""
    K = np.array(params['intrinsics'])
    R = np.array(params['extrinsics']['rotation_matrix'])
    t = np.array(params['extrinsics']['translation_vector']).reshape(3, 1)
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
# 3D座標の計算 (三角測量)
# =============================================================================

def calculate_raw_3d_coordinates(kps1_seq, kps2_seq, P1, P2):
    """全フレームの生の3D座標を一括で計算する"""
    num_frames = len(kps1_seq)
    raw_3d_points = np.full((num_frames, 25, 3), np.nan)
    confidences_3d = np.full((num_frames, 25), np.nan)
    
    print(f"生の3D座標を計算中 (全{num_frames}フレーム)...")
    for i in tqdm(range(num_frames)):
        kp1, cf1 = kps1_seq[i][:, :2], kps1_seq[i][:, 2]
        kp2, cf2 = kps2_seq[i][:, :2], kps2_seq[i][:, 2]
        raw_3d_points[i], confidences_3d[i] = triangulate_and_rotate(P1, P2, kp1, kp2, cf1, cf2)
    
    return raw_3d_points, confidences_3d

# =============================================================================
# 欠損値補間とバターワースフィルタ
# =============================================================================

def spline_interpolate(data):
    """Cubic Spline補間を使用して欠損値を補間する"""
    interpolated_data = np.copy(data)
    num_frames, num_keypoints, _ = interpolated_data.shape
    
    for kp_idx in range(num_keypoints):
        for coord_idx in range(3):
            coord_series = interpolated_data[:, kp_idx, coord_idx]
            valid_mask = ~np.isnan(coord_series)
            
            if np.sum(valid_mask) < 2:
                continue
            
            cs = CubicSpline(np.where(valid_mask)[0], coord_series[valid_mask])
            interp_values = cs(np.arange(num_frames))
            interpolated_data[:, kp_idx, coord_idx] = interp_values
    
    return interpolated_data


def butterworth_filter(data, cutoff, fs, order=4):
    """バターワースフィルタを適用する"""
    butter_data = data.copy()
    num_frames, num_keypoints, _ = butter_data.shape
    
    for kp_idx in range(num_keypoints):
        for coord_idx in range(3):
            coord_series = butter_data[:, kp_idx, coord_idx]
            
            if np.all(np.isnan(coord_series)):
                continue
            
            valid_mask = ~np.isnan(coord_series)
            b, a = butter(order, cutoff / (0.5 * fs), btype='low')
            filtered_values = filtfilt(b, a, coord_series[valid_mask])
            butter_data[valid_mask, kp_idx, coord_idx] = filtered_values
    
    return butter_data

# =============================================================================
# OpenPoseスケルトン定義と配色（try_2_anim.pyから移植）
# =============================================================================

def get_skeleton_connections():
    """OpenPose (BODY_25) のスケルトン接続定義と公式の色分けを返す"""
    connections = [
        (1, 8),   (1, 2),   (1, 5),   (2, 3),   (3, 4),   (5, 6),   (6, 7),
        (8, 9),   (8, 12),  (9, 10),  (10, 11), (12, 13), (13, 14), (1, 0),
        (0, 15),  (15, 17), (0, 16),  (16, 18), (11, 24), (11, 22), (22, 23),
        (14, 21), (14, 19), (19, 20)
    ]

    # OpenPose公式GUIの配色を再現
    colors = [
        '#FF0000', '#FF5500', '#FFAA00', '#FFFF00', '#AAFF00', '#55FF00', '#00FF00',
        '#00FF55', '#00FFAA', '#00FFFF', '#00AAFF', '#0055FF', '#0000FF', '#5500FF',
        '#AA00FF', '#FF00FF', '#FF00AA', '#FF0055', '#00FFFF', '#00FFFF', '#00FFFF',
        '#5500FF', '#5500FF', '#5500FF'
    ]
    return connections, colors


# =============================================================================
# 外れ値検出とフィルタリング
# =============================================================================

def detect_valid_frame_range(data_3d, x_min=-2500, x_max=2500):
    """
    MidHipのX座標が指定範囲内にあるフレームを有効フレームとして検出

    Parameters:
    -----------
    data_3d : ndarray (num_frames, 25, 3)
        3D座標データ
    x_min, x_max : float
        MidHipのX座標の有効範囲 [mm]

    Returns:
    --------
    start_frame, end_frame : int, int
        有効なフレーム範囲（両端を含む）
    """
    num_frames = len(data_3d)
    MIDHIP_IDX = 8  # OpenPose BODY_25でのMidHipのインデックス

    # MidHipのX座標を取得
    midhip_x = data_3d[:, MIDHIP_IDX, 0]

    # 有効なフレームのマスク（X座標が範囲内かつNaNでない）
    valid_mask = (~np.isnan(midhip_x)) & (midhip_x >= x_min) & (midhip_x <= x_max)

    if not np.any(valid_mask):
        print(f"  警告: 有効なフレームが見つかりません（MidHip X座標が{x_min}~{x_max}の範囲外）")
        return 0, num_frames - 1

    # 最初と最後の有効フレームを見つける
    valid_indices = np.where(valid_mask)[0]
    start_frame = valid_indices[0]
    end_frame = valid_indices[-1]

    # 範囲内の無効フレーム数をカウント
    invalid_frames_in_range = np.sum(~valid_mask[start_frame:end_frame+1])

    print(f"\n有効フレーム範囲の検出 (MidHip X座標: {x_min} ~ {x_max} mm):")
    print(f"  - 検出範囲: Frame {start_frame} ~ {end_frame} (全{end_frame - start_frame + 1}フレーム)")
    print(f"  - 範囲内の無効フレーム数: {invalid_frames_in_range}")
    print(f"  - 除外フレーム: 開始前={start_frame}, 終了後={num_frames - end_frame - 1}")

    return start_frame, end_frame


def confidence_filter_keypoints(data_3d, confidences, conf_threshold=0.5):
    """
    信頼値をベースに外れ値となるキーポイントをNaNに置き換えてフィルタリング

    Parameters:
    -----------
    data_3d : ndarray (num_frames, 25, 3)
        3D座標データ
    confidences : ndarray (num_frames, 25)
        各キーポイントのconfidence値
    conf_threshold : float
        Confidenceの閾値（これ以下を除外）

    Returns:
    --------
    filtered_data : ndarray
        フィルタリング後の3D座標データ
    """
    filtered_data = data_3d.copy()
    num_frames = len(data_3d)

    # Eye, Ear のキーポイントインデックス
    # 0: Nose, 15: REye, 16: LEye, 17: REar, 18: LEar
    keypoints = list(range(25))  # 全キーポイント
    # # 0: Nose, 15: REye, 16: LEye, 17: REar, 18: LEar
    # face_keypoints = [0, 15, 16, 17, 18]

    print("\n外れ値フィルタリングを実行中...")
    outlier_counts = {
        'confidence': 0,
    }

    for frame_idx in range(num_frames):
        # 1. Confidenceベースのフィルタリング
        for kp_idx in keypoints:
            if confidences[frame_idx, kp_idx] < conf_threshold:
                filtered_data[frame_idx, kp_idx] = np.nan
                outlier_counts['confidence'] += 1
    print(f"  - Confidence閾値以下: {outlier_counts['confidence']} 点")

    return filtered_data


# =============================================================================
# キーポイント別時系列プロット
# =============================================================================

# OpenPose BODY_25 キーポイント名
KEYPOINT_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe",
    "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
]


def plot_keypoint_timeseries(raw_data, filtered_data, spline_data, filt_data, confidences_3d,
                              output_dir, frame_range=None):
    """
    各キーポイントのXYZ座標の時系列データをプロットして保存

    Parameters:
    -----------
    raw_data : ndarray (num_frames, 25, 3)
        生の3D座標データ
    filtered_data : ndarray (num_frames, 25, 3)
        外れ値フィルタリング後のデータ
    spline_data : ndarray (num_frames, 25, 3)
        スプライン補間後のデータ
    filt_data : ndarray (num_frames, 25, 3)
        バターワースフィルタ後のデータ
    confidences_3d : ndarray (num_frames, 25)
        各キーポイントのconfidence値
    output_dir : Path
        出力ディレクトリ
    frame_range : tuple (start, end), optional
        プロットするフレーム範囲
    """
    num_frames, num_keypoints, _ = raw_data.shape

    if frame_range is None:
        start_frame, end_frame = 0, num_frames - 1
    else:
        start_frame, end_frame = frame_range

    # フレーム範囲を切り出し
    frame_indices = np.arange(start_frame, end_frame + 1)
    raw_slice = raw_data[start_frame:end_frame+1]
    filtered_slice = filtered_data[start_frame:end_frame+1]
    spline_slice = spline_data[start_frame:end_frame+1]
    filt_slice = filt_data[start_frame:end_frame+1]
    confidences_slice = confidences_3d[start_frame:end_frame+1]

    # 出力ディレクトリ作成
    timeseries_dir = output_dir / "keypoint_timeseries_yoloseg"
    timeseries_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nキーポイント別時系列プロットを作成中...")
    print(f"  - フレーム範囲: {start_frame} ~ {end_frame}")
    print(f"  - 出力先: {timeseries_dir}")

    coord_labels = ['X (mm)', 'Y (mm)', 'Z (mm)']

    for kp_idx in tqdm(range(num_keypoints)):
        kp_name = KEYPOINT_NAMES[kp_idx]

        # 4x1のサブプロット（X, Y, Z, Confidence）
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f'{kp_name} (Keypoint {kp_idx}) - 3D Coordinates Time Series',
                     fontsize=16, fontweight='bold')

        # X, Y, Z座標のプロット
        for coord_idx in range(3):
            ax = axes[coord_idx]
            
            # データ抽出
            raw_coord = raw_slice[:, kp_idx, coord_idx]
            filtered_coord = filtered_slice[:, kp_idx, coord_idx]
            spline_coord = spline_slice[:, kp_idx, coord_idx]
            filt_coord = filt_slice[:, kp_idx, coord_idx]

            # プロット
            ax.plot(frame_indices, raw_coord, 'o', color='lightgray',
                    markersize=3, alpha=0.5, label='Raw')
            ax.plot(frame_indices, filtered_coord, 'x', color='orange',
                    markersize=4, alpha=0.7, label='Conf Filtered')
            ax.plot(frame_indices, spline_coord, '-', color='blue',
                    linewidth=1.5, alpha=0.8, label='Spline')
            ax.plot(frame_indices, filt_coord, '-', color='red',
                    linewidth=2, alpha=0.9, label='Butterworth')

            ax.set_ylabel(coord_labels[coord_idx], fontsize=12)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)

            # Y軸の範囲を設定（外れ値を除いて）
            valid_data = np.concatenate([
                filtered_coord[~np.isnan(filtered_coord)],
                spline_coord[~np.isnan(spline_coord)],
                filt_coord[~np.isnan(filt_coord)]
            ])
            if len(valid_data) > 0:
                y_min, y_max = np.percentile(valid_data, [1, 99])
                y_margin = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # 4行目: Confidenceのプロット
        ax_conf = axes[3]
        conf_values = confidences_slice[:, kp_idx]
        
        # 信頼度をプロット
        ax_conf.plot(frame_indices, conf_values, '-', color='green',
                     linewidth=1.5, alpha=0.8, label='Confidence')
        ax_conf.fill_between(frame_indices, 0, conf_values, 
                             color='green', alpha=0.2)
        ax_conf.set_ylabel('Confidence', fontsize=12)
        ax_conf.set_xlabel('Frame', fontsize=12)
        ax_conf.set_ylim(0, 1.05)
        ax_conf.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存
        output_path = timeseries_dir / f"kp{kp_idx:02d}_{kp_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  ✓ {num_keypoints}個のキーポイントのプロットを保存しました")


# =============================================================================
# アニメーション作成（try_2_anim.pyベース）
# =============================================================================

def update_animation_frame(frame_idx, data_3d, skeleton_plots, text_plot, ax,
                           confidences=None, conf_threshold=0.3):
    """
    アニメーションの各フレームを更新する

    Parameters:
    -----------
    confidences : ndarray, optional
        各キーポイントのconfidence値 (描画時の追加フィルタリング用)
    conf_threshold : float
        描画時のconfidence閾値
    """
    title_text, frame_info_text = text_plot
    keypoint_scatter, skeleton_lines = skeleton_plots

    all_plot_elements = []

    # 現在フレームのキーポイント取得
    keypoints_3d = data_3d[frame_idx].copy()

    # Confidenceによる追加フィルタリング（描画時）
    if confidences is not None:
        low_conf_mask = confidences[frame_idx] < conf_threshold
        keypoints_3d[low_conf_mask] = np.nan

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

    # スケルトンの描画（両端のキーポイントが有効な場合のみ）
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
            # 無効なキーポイントを持つボーンは描画しない
            skeleton_lines[i].set_data_3d([], [], [])
    all_plot_elements.extend(skeleton_lines)

    # フレーム情報のテキストを更新
    frame_info = f"Frame: {frame_idx + 1}/{len(data_3d)}"
    frame_info_text.set_text(frame_info)
    all_plot_elements.extend([title_text, frame_info_text])

    # 毎フレームで軸の範囲を強制的に設定
    ax.set_xlim(-2000, 2000) #前後方向
    ax.set_ylim(-2000, 2000) #左右方向
    ax.set_zlim(0, 2000) #高さ方向

    return all_plot_elements


def create_3d_animation(data_3d, output_path, title, all_data_for_limits,
                        confidences=None, conf_threshold=0.3):
    """
    3Dスティックフィギュアのアニメーションを作成し保存する

    Parameters:
    -----------
    confidences : ndarray, optional
        各キーポイントのconfidence値
    conf_threshold : float
        描画時のconfidence閾値
    """
    print(f"  - アニメーションを作成中: {output_path.name}")

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 全データから座標変換
    all_points_transformed = np.column_stack([
        all_data_for_limits[:, :, 2].flatten(),
        all_data_for_limits[:, :, 0].flatten(),
        all_data_for_limits[:, :, 1].flatten()
    ]).reshape(-1, 3)

    # データの実際の範囲を確認（デバッグ用）
    if np.any(~np.isnan(all_points_transformed)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x_min, x_max = np.nanmin(all_points_transformed[:, 0]), np.nanmax(all_points_transformed[:, 0])
            y_min, y_max = np.nanmin(all_points_transformed[:, 1]), np.nanmax(all_points_transformed[:, 1])
            z_min, z_max = np.nanmin(all_points_transformed[:, 2]), np.nanmax(all_points_transformed[:, 2])

            print(f"    データ範囲 - X: [{x_min:.1f}, {x_max:.1f}], Y: [{y_min:.1f}, {y_max:.1f}], Z: [{z_min:.1f}, {z_max:.1f}]")

    # 固定範囲を設定
    ax.set_xlim(-2000, 2000) #前後方向
    ax.set_ylim(-2000, 2000) #左右方向
    ax.set_zlim(0, 2000)  #高さ方向

    ax.set_xlabel('Z-axis (mm) - Forward', fontsize=14, labelpad=15)
    ax.set_ylabel('X-axis (mm) - Sideways', fontsize=14, labelpad=15)
    ax.set_zlabel('Y-axis (mm) - Up', fontsize=14, labelpad=15)

    ax.view_init(elev=10, azim=45)  # Oblique view
    # ax.view_init(elev=0, azim=-90)  #Sagittal view

    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)

    # 自動スケーリングを無効化
    ax.set_autoscale_on(False)

    # スケルトン描画オブジェクトを作成
    connections, colors = get_skeleton_connections()
    keypoint_scatter = ax.scatter([], [], [], c='red', s=40, depthshade=True, zorder=5)
    skeleton_lines = [ax.plot([], [], [], color=c, lw=2.5)[0] for c in colors]
    skeleton_plots = (keypoint_scatter, skeleton_lines)

    # タイトルとフレーム情報
    title_text = ax.text2D(0.5, 0.95, title, transform=ax.transAxes,
                           fontsize=16, ha='center')
    frame_info_text = ax.text2D(0.02, 0.02, "", transform=ax.transAxes,
                                fontsize=10, va='bottom',
                                bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))
    text_plot = (title_text, frame_info_text)

    # アニメーション作成
    ani = animation.FuncAnimation(
        fig, update_animation_frame, frames=len(data_3d),
        fargs=(data_3d, skeleton_plots, text_plot, ax, confidences, conf_threshold),
        blit=False, interval=1000/FRAME_RATE
    )

    print(f"    > 保存中...")
    try:
        writer = animation.FFMpegWriter(
            fps=FRAME_RATE,
            metadata=dict(artist='GaitAnalysis'),
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
        params_cam1 = load_camera_parameters(
            STEREO_CALI_DIR / directions[0] / "camera_params_with_ext_OC.json"
        )
        params_cam2 = load_camera_parameters(
            STEREO_CALI_DIR / directions[1] / "camera_params_with_ext_OC.json"
        )
        P1 = create_projection_matrix(params_cam1)
        P2 = create_projection_matrix(params_cam2)
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
            if subject_dir.name != "sub1" or thera_dir.name != "thera1-0":
                continue
            
            print(f"\n{'='*80}\n処理開始: {thera_dir.relative_to(ROOT_DIR)}")
            
            thera_dir = thera_dir / "ICMST"
                        
            openpose_csv_path1 = thera_dir / "fl_yoloseg" / "openpose.csv"
            openpose_csv_path2 = thera_dir / "fr_yoloseg" / "openpose.csv"
            
            kps1_seq, kps2_seq, frames = load_csv_2d_data(
                openpose_csv_path1, openpose_csv_path2
            )
            
            if not frames:
                print(f"  - スキップ: 共通フレームが見つかりません。")
                continue
            
            print(f"kps1_seq shape: {kps1_seq.shape}, kps2_seq shape: {kps2_seq.shape}")
            
            # 3D座標を計算
            raw_kp_3d, confidences_3d = calculate_raw_3d_coordinates(
                kps1_seq, kps2_seq, P1, P2
            )
            print(f"raw_kp_3d shape: {raw_kp_3d.shape}")

            # 信頼度による外れ値フィルタリングを適用
            conf_filt_kp_3d = confidence_filter_keypoints(
                raw_kp_3d,
                confidences_3d,
                conf_threshold=0.4,      # Confidence閾値
            )

            # 有効フレーム範囲を自動検出（MidHipのX座標が-2000~2000の範囲）
            start_frame, end_frame = detect_valid_frame_range(
                conf_filt_kp_3d,
                x_min=-2000,
                x_max=2000
            )

            # 欠損値補間とバターワースフィルタ（フィルタリング後のデータに適用）
            spline_kp_3d = spline_interpolate(conf_filt_kp_3d)
            butter_filt_kp_3d = butterworth_filter(spline_kp_3d, BUTTERWORTH_CUTOFF, FRAME_RATE)

            # 3次元座標を保存（外れ値フィルタリング後も保存）
            npz_path = thera_dir / f"3d_kp_data_{openpose_csv_path1.stem}_yoloseg.npz"
            np.savez(npz_path,
                     frame=frames,
                     raw=raw_kp_3d,
                     conf_filt=conf_filt_kp_3d,  # 外れ値除去後
                     spline=spline_kp_3d,
                     butter_filt=butter_filt_kp_3d,
                     conf=confidences_3d,
                     valid_frame_range=np.array([start_frame, end_frame]))
            print(f"  - 3Dキーポイントデータを保存しました。")

            # キーポイント別時系列プロットを作成
            plot_keypoint_timeseries(
                raw_kp_3d,
                conf_filt_kp_3d,
                spline_kp_3d,
                butter_filt_kp_3d,
                confidences_3d,
                thera_dir,
                frame_range=(start_frame, end_frame)
            )

            # 3Dアニメーションを作成（有効フレーム範囲で）
            output_anim_dir = thera_dir / "3d_gait_anim_yoloseg"
            output_anim_dir.mkdir(parents=True, exist_ok=True)
            print("\n3Dスティックフィギュアアニメーションの作成を開始します...")

            # 有効フレーム範囲でデータを切り出し
            raw_kp_3d_valid = raw_kp_3d[start_frame:end_frame+1]
            conf_filt_kp_3d_valid = conf_filt_kp_3d[start_frame:end_frame+1]
            spline_kp_3d_valid = spline_kp_3d[start_frame:end_frame+1]
            butter_filt_kp_3d_valid = butter_filt_kp_3d[start_frame:end_frame+1]
            confidences_3d_valid = confidences_3d[start_frame:end_frame+1]

            # 共通の軸スケールを計算（全データセットを結合）
            all_data = np.concatenate([conf_filt_kp_3d_valid, spline_kp_3d_valid, butter_filt_kp_3d_valid], axis=0)

            # 各データセットでアニメーションを作成（有効フレーム範囲のみ）
            # create_3d_animation(
            #     raw_kp_3d_valid,
            #     output_anim_dir / "raw_3d.mp4",
            #     f"Raw 3D (Frames {start_frame}-{end_frame}) - {subject_dir.name} / {thera_dir.name}",
            #     all_data,
            #     confidences=confidences_3d_valid,
            #     conf_threshold=0.3
            # )

            # create_3d_animation(
            #     conf_filt_kp_3d_valid,
            #     output_anim_dir / "conf_filt_3d.mp4",
            #     f"Conf Filtered 3D (Frames {start_frame}-{end_frame}) - {subject_dir.name} / {thera_dir.name}",
            #     all_data,
            #     confidences=confidences_3d_valid,
            #     conf_threshold=0.3
            # )

            # create_3d_animation(
            #     spline_kp_3d_valid,
            #     output_anim_dir / "spline_3d.mp4",
            #     f"Spline Interpolated 3D (Frames {start_frame}-{end_frame}) - {subject_dir.name} / {thera_dir.name}",
            #     all_data,
            #     confidences=confidences_3d_valid,
            #     conf_threshold=0.3
            # )

            # create_3d_animation(
            #     butter_filt_kp_3d_valid,
            #     output_anim_dir / "butter_filt_3d.mp4",
            #     f"Butterworth Filtered 3D (Frames {start_frame}-{end_frame}) - {subject_dir.name} / {thera_dir.name}",
            #     all_data,
            #     confidences=confidences_3d_valid,
            #     conf_threshold=0.3
            # )
            
            print(f"\n処理完了: {thera_dir.relative_to(ROOT_DIR)}")


if __name__ == '__main__':
    main()