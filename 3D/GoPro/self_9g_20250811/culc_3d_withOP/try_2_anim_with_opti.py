import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os
import glob
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt, resample

"""
OptiTrackのデータも同時にアニメーション化する
信頼度が低いキーポイントは描画しないように修正
"""

# ★★★★★★★★★★★★★★★★ 変更点① ★★★★★★★★★★★★★★★★
# 信頼度の閾値。この値未満の信頼度を持つキーポイントは描画されません。
CONFIDENCE_THRESHOLD = 0.3

# --- データ読み込み & ユーティリティ関数 ---

def load_gait_analysis_json(json_file_path, data_key):
    """
    複数人に対応した解析結果JSONファイルを読み込む
    3D座標と信頼度を結合して (N, 4) の配列として返す
    """
    with open(json_file_path, 'r') as f:
        all_frames_data = json.load(f)

    animation_data = {}
    if not all_frames_data:
        return None, None

    person_keys = [key for key in all_frames_data[0] if key.startswith('person_')]

    for p_key in person_keys:
        if data_key in all_frames_data[0][p_key]:
            combined_person_data = []
            for frame in all_frames_data:
                if p_key not in frame:
                    continue

                person_data = frame[p_key]
                points_3d = np.array(person_data.get(data_key, []))
                confidences = np.array(person_data.get("confidence", []))

                if points_3d.size > 0 and confidences.size > 0 and points_3d.shape[0] == confidences.shape[0]:
                    confidences = confidences.reshape(-1, 1)
                    combined_frame_data = np.hstack([points_3d, confidences])
                    combined_person_data.append(combined_frame_data)
                elif points_3d.size > 0:
                    # 信頼度がない場合は、ダミーの信頼度(1.0)を追加
                    dummy_confs = np.ones((points_3d.shape[0], 1))
                    combined_frame_data = np.hstack([points_3d, dummy_confs])
                    combined_person_data.append(combined_frame_data)

            if combined_person_data:
                animation_data[p_key] = combined_person_data

    if not animation_data:
        return None, None

    therapist_name = json_file_path.stem.replace('_3d_results', '')
    subject_name = json_file_path.parent.parent.parent.name

    metadata = {
        'subject': subject_name,
        'therapist': therapist_name,
        'total_frames': len(next(iter(animation_data.values()))),
        'data_source': data_key,
        'person_keys': person_keys
    }

    return animation_data, metadata


def get_skeleton_connections():
    """
    OpenPose (BODY_25) のスケルトン接続定義と公式の色分けを返す
    """
    connections = [
        (1, 8),  (1, 2),  (1, 5),  (2, 3),  (3, 4),  (5, 6),  (6, 7),
        (8, 9),  (8, 12), (9, 10), (10, 11), (12, 13), (13, 14), (1, 0),
        (0, 15), (15, 17), (0, 16), (16, 18), (11, 24), (11, 22), (22, 23),
        (14, 21), (14, 19), (19, 20)
    ]
    colors = [
        '#FF0000', '#FF5500', '#FFAA00', '#FFFF00', '#AAFF00', '#55FF00', '#00FF00',
        '#00FF55', '#00FFAA', '#00FFFF', '#00AAFF', '#0055FF', '#0000FF', '#5500FF',
        '#AA00FF', '#FF00FF', '#FF00AA', '#FF0055', '#00FFFF', '#00FFFF', '#00FFFF',
        '#5500FF', '#5500FF', '#5500FF'
    ]
    return connections, colors

# --- Mocapデータ読み込み関数 ---

def read_3d_optitrack(csv_path, down_hz, start_frame_100hz, end_frame_100hz, geometry_path=None):
    """
    OptiTrackの3Dデータを読み込み、前処理を行う。
    """
    def geometric_interpolation(marker_df, marker_to_fix, geometry, original_missing_mask):
        if marker_to_fix not in geometry:
            print(f"警告: ジオメトリ情報に '{marker_to_fix}' の定義がありません。")
            return marker_df
        if not original_missing_mask.any():
            return marker_df

        marker_geometry = geometry[marker_to_fix]
        ref_marker_names = marker_geometry["reference_markers"]
        target_cols = [c for c in marker_df.columns if marker_to_fix in c[0]]
        ref_cols_map = {name: [c for c in marker_df.columns if name in c[0]] for name in ref_marker_names}

        if not target_cols or not all(ref_cols_map.values()):
            return marker_df

        source_vectors = [np.array(marker_geometry["reference_vectors"][name]) for name in ref_marker_names]
        target_offset_vector = np.array(marker_geometry["target_offset_vector"])

        for index in range(len(marker_df)):
            row = marker_df.loc[index]
            if all(not row[cols].isnull().any() for cols in ref_cols_map.values()):
                ref_positions = [row[ref_cols_map[name]].values for name in ref_marker_names]
                centroid_current = np.mean(ref_positions, axis=0)
                target_vectors = [p - centroid_current for p in ref_positions]
                rot, _ = R.align_vectors(target_vectors, source_vectors)
                estimated_offset = rot.apply(target_offset_vector)
                estimated_target = centroid_current + estimated_offset
                marker_df.loc[index, target_cols] = estimated_target
        return marker_df

    df = pd.read_csv(csv_path, skiprows=[0, 1, 2, 4], header=[0, 2])
    if start_frame_100hz >= len(df) or end_frame_100hz >= len(df) or start_frame_100hz < 0:
        print(f"Error: Requested range ({start_frame_100hz}-{end_frame_100hz}) is outside data range")
        return None

    df_100hz = df.loc[start_frame_100hz:end_frame_100hz].reset_index(drop=True)
    marker_set = ["RASI", "LASI", "RPSI", "LPSI", "RKNE", "LKNE", "RANK", "LANK", "RTOE", "LTOE", "RHEE", "LHEE", "RKNE2", "LKNE2", "RANK2", "LANK2",
                  "RSHO", "LSHO", "C7", "T10", "CLAV", "STRN"]
    # marker_set = ["RASI", "LASI", "RPSI", "LPSI", "RKNE", "LKNE", "RANK", "LANK", "RTOE", "LTOE", "RHEE", "LHEE", "RKNE2", "LKNE2", "RANK2", "LANK2"]
    marker_set_df = df_100hz[[col for col in df_100hz.columns if any(marker in col[0] for marker in marker_set)]].copy()

    if marker_set_df.empty:
        return None

    marker_set_df.interpolate(method='cubic', limit_direction='both', inplace=True)

    if geometry_path and os.path.exists(geometry_path):
        with open(geometry_path, 'r') as f: geometry = json.load(f)
        for marker in ["LPSI"]:
            cols = [c for c in marker_set_df.columns if marker in c[0]]
            if cols:
                original_missing_mask = marker_set_df[cols].isnull().any(axis=1)
                marker_set_df = geometric_interpolation(marker_set_df, marker, geometry, original_missing_mask)

    if marker_set_df.isnull().values.any():
        marker_set_df.interpolate(method='linear', limit_direction='both', inplace=True)
    if marker_set_df.isnull().values.any():
        print("エラー: 補間後も欠損値が残っています。")
        return None

    if down_hz:
        target_length = int(len(marker_set_df) * 60 / 100)
        resampled_df = pd.DataFrame({col: resample(marker_set_df[col], target_length) for col in marker_set_df.columns})
        final_df = resampled_df
    else:
        final_df = marker_set_df

    return final_df.values.reshape(-1, len(marker_set), 3)

# --- 描画 & アニメーション関数 (Mocap対応) ---

def update_animation_frame(frame_idx, animation_data, mocap_data, skeleton_plots, mocap_scatter, text_plot, ax):
    """アニメーションの各フレームを更新する（Mocap対応）"""
    title_text, frame_info_text = text_plot
    all_plot_elements = []

    # OpenPoseスケルトンの更新
    for person_key, plots in skeleton_plots.items():
        keypoint_scatter, skeleton_lines = plots
        if frame_idx < len(animation_data[person_key]):
            person_points_data = animation_data[person_key][frame_idx].copy()

            # ★★★★★★★★★★★★★★★★ 変更点② ★★★★★★★★★★★★★★★★
            # 信頼度が閾値未満のキーポイントをNaNに設定して非表示にする
            if person_points_data.shape[1] > 3:
                low_confidence_mask = person_points_data[:, 3] < CONFIDENCE_THRESHOLD
                person_points_data[low_confidence_mask, :3] = np.nan

            # 3D座標のみを抽出
            person_points_3d = person_points_data[:, :3]

            transformed_points = np.column_stack([person_points_3d[:, 2], person_points_3d[:, 0], person_points_3d[:, 1]])

            valid_points_mask = ~np.isnan(transformed_points).any(axis=1)
            valid_points = transformed_points[valid_points_mask]

            keypoint_scatter._offsets3d = (valid_points[:, 0], valid_points[:, 1], valid_points[:, 2])
            all_plot_elements.append(keypoint_scatter)

            connections, _ = get_skeleton_connections()
            for i, (start_idx, end_idx) in enumerate(connections):
                if not (np.isnan(person_points_3d[start_idx]).any() or np.isnan(person_points_3d[end_idx]).any()):
                    start, end = transformed_points[start_idx], transformed_points[end_idx]
                    skeleton_lines[i].set_data_3d([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])
                else:
                    skeleton_lines[i].set_data_3d([], [], [])
            all_plot_elements.extend(skeleton_lines)

    # Mocapキーポイントの更新
    if mocap_data is not None and frame_idx < len(mocap_data):
        mocap_points_3d = mocap_data[frame_idx]
        transformed_mocap_points = np.column_stack([mocap_points_3d[:, 2], mocap_points_3d[:, 0], mocap_points_3d[:, 1]])
        mocap_scatter._offsets3d = (transformed_mocap_points[:, 0], transformed_mocap_points[:, 1], transformed_mocap_points[:, 2])
        all_plot_elements.append(mocap_scatter)

    total_frames = len(next(iter(animation_data.values())))
    if mocap_data is not None:
        total_frames = min(total_frames, len(mocap_data))

    frame_info_text.set_text(f"Frame: {frame_idx + 1}/{total_frames}")
    all_plot_elements.extend([title_text, frame_info_text])

    return all_plot_elements

def create_3d_animation(animation_data, metadata, save_path, view_config, mocap_data=None):
    """Mocapデータも同時に描画する3Dアニメーションを作成し保存する"""
    if not animation_data or not next(iter(animation_data.values())):
        print("アニメーションデータが空、またはスライス後にデータがなくなりました。")
        return

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 2000)
    ax.set_zlim(0, 2500)

    ax.set_xlabel('Z-axis (mm) - Forward')
    ax.set_ylabel('X-axis (mm) - Sideways')
    ax.set_zlabel('Y-axis (mm) - Up')
    ax.view_init(elev=view_config['elev'], azim=view_config['azim'])
    ax.set_autoscale_on(False)

    skeleton_plots = {}
    connections, colors = get_skeleton_connections()
    point_colors = ['red', 'blue', 'green', 'purple']
    for i, p_key in enumerate(metadata['person_keys']):
        keypoint_scatter = ax.scatter([], [], [], c=point_colors[i % len(point_colors)], s=40, depthshade=True, zorder=5, label=f'{p_key} (OpenPose)')
        skeleton_lines = [ax.plot([], [], [], color=c, lw=2.5)[0] for c in colors]
        skeleton_plots[p_key] = (keypoint_scatter, skeleton_lines)

    mocap_scatter = None
    if mocap_data is not None:
        print("mocap_dataが存在します。")
        mocap_scatter = ax.scatter([], [], [], c='lime', marker='x', s=60, depthshade=True, zorder=10, label='Mocap')
    else:
        print("mocap_dataは存在しません。")

    ax.legend()

    title_str = (f"3D Gait Animation ({metadata['data_source']}) - View: {view_config['name']}\n"
                 f"{metadata['subject']} / {metadata['therapist']}")
    title_text = ax.text2D(0.5, 0.95, title_str, transform=ax.transAxes, fontsize=14, ha='center')
    frame_info_text = ax.text2D(0.02, 0.02, "", transform=ax.transAxes, fontsize=10, va='bottom', bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))
    text_plot = (title_text, frame_info_text)

    total_frames = metadata['total_frames']
    print(f"total_frames: {total_frames}")
    if mocap_data is not None:
        total_frames = min(total_frames, len(mocap_data))
        print(f"同期フレーム数: {total_frames} (OpenPose: {metadata['total_frames']}, Mocap: {len(mocap_data)})")


    ani = animation.FuncAnimation(
        fig, update_animation_frame, frames=total_frames,
        fargs=(animation_data, mocap_data, skeleton_plots, mocap_scatter, text_plot, ax), blit=False, interval=1000/60
    )

    print(f"  > アニメーションを {save_path} に保存しています...")
    try:
        writer = animation.FFMpegWriter(fps=60, metadata=dict(artist='GaitAnalysis'), bitrate=3600)
        ani.save(save_path, writer=writer)
        print(f"  ✓ 保存が完了しました。")
    except Exception as e:
        print(f"  ✗ アニメーション保存中にエラーが発生しました: {e}")
    plt.close(fig)

# --- メイン実行部 ---
def main():
    root_dir = Path(r"G:\gait_pattern\20250811_br")
    results_folder_name = "3d_gait_analysis_kalman_v3"
    output_animation_dir = root_dir / "3d_animations_with_mocap"
    output_animation_dir.mkdir(exist_ok=True)

    view_options = {
        "sagittal": {"name": "Sagittal", "elev": 0, "azim": 90},
    }
    data_keys_to_animate = ["raw_unprocessed_3d"]

    print(f"\n{'='*40}\nモーションキャプチャデータを読み込みます\n{'='*40}")
    csv_path = r"G:\gait_pattern\20250811_br\sub1\thera0-2\mocap\1_0_2.csv"
    start_frame_100hz = 1000
    end_frame_100hz = 1440
    geometry_json_path = r"G:\gait_pattern\20250811_br\sub0\thera0-14\mocap\geometry.json"

    base_path = os.path.dirname(os.path.dirname(csv_path))
    mocap_json_path = os.path.join(base_path, "mocap", "gait_data_1_0_2.json")
    with open(mocap_json_path, 'r') as f:
        mocap_data = json.load(f)

    mocap_start_frame = mocap_data["start_frame_60Hz"]
    mocap_end_frame = mocap_data["end_frame_60Hz"]

    gopro_trim_json_path = os.path.join(base_path, "gopro_trimming_info.json")
    with open(gopro_trim_json_path, 'r') as f:
        gopro_trim_data = json.load(f)

    gopro_start_frame = gopro_trim_data['trimming_settings']['start_frame_relative']

    gopro_sync_start = mocap_start_frame - gopro_start_frame
    gopro_sync_end = mocap_end_frame - gopro_start_frame

    keypoints_mocap = None
    if os.path.exists(csv_path):
        print(f"Processing Mocap file: {csv_path}")
        keypoints_mocap = read_3d_optitrack(
            csv_path,
            down_hz=True,
            start_frame_100hz=start_frame_100hz,
            end_frame_100hz=end_frame_100hz,
            geometry_path=geometry_json_path
        )
        keypoints_mocap = keypoints_mocap*1000
        if keypoints_mocap is None:
            print(f"✗ Mocapデータの読み込みに失敗しました。")
        else:
            print(f"✓ Mocapデータ読み込み完了 ({len(keypoints_mocap)} フレーム)")
    else:
        print(f"✗ Mocapファイルが見つかりません: {csv_path}")


    json_files = list(root_dir.glob(f"**/{results_folder_name}/*.json"))
    if not json_files:
        print(f"✗ エラー: 解析結果のJSONファイルが見つかりませんでした。")
        return

    print(f"\n{'='*40}\nアニメーション作成処理を開始します\n{'='*40}")

    for json_file in tqdm(json_files, desc="ファイル処理"):
        if "thera0-2" not in str(json_file):
            continue

        print(f"\n--- ファイル: {json_file.name} ---")

        for data_key in data_keys_to_animate:
            tqdm.write(f"  - データソース '{data_key}' のアニメーションを作成中...")
            animation_data, metadata = load_gait_analysis_json(json_file, data_key=data_key)

            animation_data_to_use = None
            if animation_data and metadata:
                sliced_animation_data = {}
                for person_key, person_frames in animation_data.items():
                    start = max(0, gopro_sync_start)
                    end = min(len(person_frames), gopro_sync_end)
                    sliced_animation_data[person_key] = person_frames[start:end]

                if sliced_animation_data and next(iter(sliced_animation_data.values())):
                    metadata['total_frames'] = len(next(iter(sliced_animation_data.values())))
                    tqdm.write(f"    - OpenPoseデータを {gopro_sync_start} から {gopro_sync_end} までスライスしました。 (新しいフレーム数: {metadata['total_frames']})")
                    animation_data_to_use = sliced_animation_data
                else:
                    tqdm.write("    - スキップ: スライス後のOpenPoseデータが空になりました。")
                    continue

            if animation_data_to_use and metadata:
                for view_name, view_config in view_options.items():
                    tqdm.write(f"    - {view_config['name']}視点")
                    output_filename = f"{metadata['subject']}_{metadata['therapist']}_{metadata['data_source']}_{view_name}_with_mocap.mp4"
                    save_path = output_animation_dir / output_filename

                    create_3d_animation(animation_data_to_use, metadata, save_path, view_config, mocap_data=keypoints_mocap)
            else:
                tqdm.write(f"  - スキップ: '{data_key}' のデータが見つかりませんでした。")

    print("\n全ての処理が完了しました。")

if __name__ == '__main__':
    main()
