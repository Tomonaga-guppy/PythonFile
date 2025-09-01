import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm

# --- データ読み込み & ユーティリティ関数 (複数人対応に修正) ---

def load_gait_analysis_json(json_file_path, data_key):
    """
    複数人に対応した解析結果JSONファイルを読み込む
    """
    with open(json_file_path, 'r') as f:
        all_frames_data = json.load(f)

    animation_data = {}
    if not all_frames_data:
        return None, None

    # JSONの最初のフレームから存在する 'person_X' キーを全て検出
    person_keys = [key for key in all_frames_data[0] if key.startswith('person_')]

    # 各人物のデータを読み込む
    for p_key in person_keys:
        # 指定されたデータキーが存在するか確認
        if data_key in all_frames_data[0][p_key]:
            # 全フレームのデータをNumPy配列としてリストに格納
            person_anim_data = [np.array(frame[p_key][data_key]) for frame in all_frames_data]
            animation_data[p_key] = person_anim_data

    if not animation_data:
        return None, None

    therapist_name = json_file_path.stem.replace('_3d_results', '')
    subject_name = json_file_path.parent.parent.parent.name

    metadata = {
        'subject': subject_name,
        'therapist': therapist_name,
        'total_frames': len(next(iter(animation_data.values()))), # 最初の人物のフレーム数
        'data_source': data_key,
        'person_keys': person_keys # 検出された人物のキーリスト
    }

    return animation_data, metadata

# ★★★ 変更点: ご提示の画像に合わせて配色を更新 ★★★
def get_skeleton_connections():
    """
    OpenPose (BODY_25) のスケルトン接続定義と公式の色分けを返す
    """
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

# --- 描画 & アニメーション関数 (複数人対応に修正) ---

def update_animation_frame(frame_idx, animation_data, skeleton_plots, text_plot, ax):
    """アニメーションの各フレームを更新する（複数人対応）"""
    title_text, frame_info_text = text_plot

    all_plot_elements = []

    # 登録されている全人物のスケルトンを更新
    for person_key, plots in skeleton_plots.items():
        keypoint_scatter, skeleton_lines = plots
        person_points_3d = animation_data[person_key][frame_idx]

        # 座標系を変換 (Z, X, Y)
        transformed_points = np.column_stack([person_points_3d[:, 2], person_points_3d[:, 0], person_points_3d[:, 1]])

        # キーポイントの描画
        valid_points = transformed_points[~np.isnan(transformed_points).any(axis=1)]
        if valid_points.shape[0] > 0:
            keypoint_scatter._offsets3d = (valid_points[:, 0], valid_points[:, 1], valid_points[:, 2])
        else:
            keypoint_scatter._offsets3d = ([], [], [])
        all_plot_elements.append(keypoint_scatter)

        # スケルトンの描画
        connections, _ = get_skeleton_connections()
        for i, (start_idx, end_idx) in enumerate(connections):
            if not (np.isnan(person_points_3d[start_idx]).any() or np.isnan(person_points_3d[end_idx]).any()):
                start, end = transformed_points[start_idx], transformed_points[end_idx]
                skeleton_lines[i].set_data_3d([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])
            else:
                skeleton_lines[i].set_data_3d([], [], [])
        all_plot_elements.extend(skeleton_lines)

    # フレーム情報のテキストを更新
    frame_info = f"Frame: {frame_idx + 1}/{len(next(iter(animation_data.values())))}"
    frame_info_text.set_text(frame_info)
    all_plot_elements.extend([title_text, frame_info_text])

    # ★★★ 修正点: 毎フレームで軸の範囲を強制的に設定 ★★★
    ax.set_xlim(0, 2000)
    ax.set_ylim(-2000, 2000)
    ax.set_zlim(0, 2000)

    return all_plot_elements

def create_3d_animation(animation_data, metadata, save_path, view_config):
    """複数人対応の3Dスティックフィギュアアニメーションを作成し保存する"""
    if not animation_data:
        print("アニメーションデータが空です。")
        return

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 全員の全フレームのデータから描画範囲を決定
    all_points_list = []
    for p_key in metadata['person_keys']:
        all_points_list.append(np.vstack(animation_data[p_key]))

    all_points = np.vstack(all_points_list)
    all_points_transformed = np.column_stack([all_points[:, 2], all_points[:, 0], all_points[:, 1]])

    # ★★★ 修正点: より確実な軸範囲設定 ★★★
    if np.any(~np.isnan(all_points_transformed)):
        x_min, x_max = np.nanmin(all_points_transformed[:, 0]), np.nanmax(all_points_transformed[:, 0])
        y_min, y_max = np.nanmin(all_points_transformed[:, 1]), np.nanmax(all_points_transformed[:, 1])
        z_min, z_max = np.nanmin(all_points_transformed[:, 2]), np.nanmax(all_points_transformed[:, 2])

        # データの実際の範囲を確認
        print(f"データ範囲 - X: [{x_min:.1f}, {x_max:.1f}], Y: [{y_min:.1f}, {y_max:.1f}], Z: [{z_min:.1f}, {z_max:.1f}]")

        # 固定範囲を設定（データに基づいて調整可能）
        ax.set_xlim(0, 2000)
        ax.set_ylim(-2000, 2000)
        ax.set_zlim(0, 2000)
    else:
        # デフォルト範囲
        ax.set_xlim(0, 2000)
        ax.set_ylim(-2000, 2000)
        ax.set_zlim(0, 2000)

    ax.set_xlabel('Z-axis (mm) - Forward', fontsize=35, labelpad=30)
    ax.set_ylabel('X-axis (mm) - Sideways', fontsize=35, labelpad=30)
    ax.set_zlabel('Y-axis (mm) - Up', fontsize=35, labelpad=30)
    ax.view_init(elev=view_config['elev'], azim=view_config['azim'])

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)

    # ★★★ 修正点: set_aspectを明示的に無効化 ★★★
    # ax.set_aspect('equal')を使わない、または'auto'に設定

    # ★★★ 修正点: 軸の自動スケーリングを無効化 ★★★
    ax.set_autoscale_on(False)

    # 人物ごとにスケルトン描画オブジェクトを作成
    skeleton_plots = {}
    connections, colors = get_skeleton_connections()
    point_colors = ['red', 'blue', 'green', 'purple'] # 人物ごとの点の色

    for i, p_key in enumerate(metadata['person_keys']):
        keypoint_scatter = ax.scatter([], [], [], c=point_colors[i % len(point_colors)], s=40, depthshade=True, zorder=5, label=p_key)
        skeleton_lines = [ax.plot([], [], [], color=c, lw=2.5)[0] for c in colors]
        skeleton_plots[p_key] = (keypoint_scatter, skeleton_lines)

    ax.legend()

    title_str = (f"3D Gait Animation ({metadata['data_source']}) - View: {view_config['name']}\n"
                 f"{metadata['subject']} / {metadata['therapist']}")
    title_text = ax.text2D(0.5, 0.95, title_str, transform=ax.transAxes, fontsize=20, ha='center')
    frame_info_text = ax.text2D(0.02, 0.02, "", transform=ax.transAxes, fontsize=10, va='bottom', bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))
    text_plot = (title_text, frame_info_text)

    # ★★★ 修正点: update_animation_frame関数にaxを渡す ★★★
    ani = animation.FuncAnimation(
        fig, update_animation_frame, frames=metadata['total_frames'],
        fargs=(animation_data, skeleton_plots, text_plot, ax), blit=False, interval=1000/60
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
    # results_folder_name = "3d_gait_analysis" # 読み込み元フォルダ名
    results_folder_name = "3d_gait_analysis_kalman_v3" # 読み込み元フォルダ名
    output_animation_dir = root_dir / "3d_animations"
    output_animation_dir.mkdir(exist_ok=True)

    view_options = {
        "sagittal": {"name": "Sagittal", "elev": 0, "azim": 90},
        "frontal":  {"name": "Frontal",  "elev": 0, "azim": 0},
        "oblique":  {"name": "Oblique",  "elev": 10, "azim": 45}
    }

    # data_keys_to_animate = ["points_3d_raw", "points_3d_corrected_kalman", "points_3d_final"]
    data_keys_to_animate = ["raw_unprocessed_3d"]

    # data_keys_to_animate = ["points_3d_raw", "points_3d_corrected_kalman", "points_3d_final"]

    json_files = list(root_dir.glob(f"**/{results_folder_name}/*.json"))
    if not json_files:
        print(f"✗ エラー: 解析結果のJSONファイルが見つかりませんでした。")
        print(f"  検索パス: {root_dir}/**/{results_folder_name}/*.json")
        return

    print(f"\n{'='*80}\nアニメーション作成処理を開始します\n{'='*80}")
    print(f"{len(json_files)} 件の解析結果ファイルを処理します。")

    for json_file in tqdm(json_files, desc="ファイル処理"):
        print(f"\n--- ファイル: {json_file.name} ---")
        if json_file.parent.parent.name != "thera0-2":
            print(f"作成省略 {json_file.name}")
            continue

        for data_key in data_keys_to_animate:
            if data_key == "points_3d_corrected_kalman" or data_key == "points_3d_final":
                print(f"作成省略 {data_key}")
                continue

            tqdm.write(f"  - データソース '{data_key}' のアニメーションを作成中...")

            animation_data, metadata = load_gait_analysis_json(json_file, data_key=data_key)

            if animation_data and metadata:
                for view_name, view_config in view_options.items():
                    if view_name != "oblique":
                        print(f"作成省略 {view_name}視点")
                        continue

                    tqdm.write(f"    - {view_config['name']}視点")
                    output_filename = f"{metadata['subject']}_{metadata['therapist']}_{metadata['data_source']}_{view_name}.mp4"
                    save_path = output_animation_dir / output_filename
                    create_3d_animation(animation_data, metadata, save_path, view_config)
            else:
                tqdm.write(f"    - スキップ: '{data_key}' のデータが見つかりませんでした。")

    print("\n全ての処理が完了しました。")

if __name__ == '__main__':
    main()