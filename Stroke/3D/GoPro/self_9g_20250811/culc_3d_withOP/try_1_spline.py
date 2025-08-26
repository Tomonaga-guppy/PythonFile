import cv2
import numpy as np
import json
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline # 3次スプライン補間のためにインポート
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt  # グラフ描画用に追加

# --- ユーティリティ関数 (一部変更) ---

def load_camera_parameters(params_file):
    """カメラパラメータ (internal/external) をJSONファイルから読み込む"""
    with open(params_file, 'r') as f:
        params = json.load(f)
    return params

def create_projection_matrix(camera_params):
    """カメラパラメータから3x4のプロジェクション行列を作成する"""
    K = np.array(camera_params['intrinsics'])
    R = np.array(camera_params['extrinsics']['rotation_matrix'])
    t = np.array(camera_params['extrinsics']['translation_vector']).reshape(3, 1)
    P = K @ np.hstack([R, t])
    return P

def load_openpose_json(json_file_path, debug=False):
    """単一のOpenPose JSONファイルからキーポイントと信頼度を読み込む（複数人対応）"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    people_data = []
    if not data.get('people'):
        # 人が検出されなかった場合
        return [np.full((25, 2), np.nan)], [np.full((25,), np.nan)]
    
    if debug:
        print(f"  デバッグ: {json_file_path.name} で {len(data['people'])} 人検出")
    
    # 検出された人数分のデータを処理
    for person_idx, person_data in enumerate(data['people']):
        keypoints_raw = np.array(person_data['pose_keypoints_2d']).reshape(-1, 3)
        keypoints_2d = keypoints_raw[:, :2]
        confidence = keypoints_raw[:, 2]
        
        # 信頼度が0の点をNaNに変換
        keypoints_2d[confidence == 0] = np.nan
        
        # 有効なキーポイント数をカウント
        valid_points = np.sum(~np.isnan(keypoints_2d).any(axis=1))
        if debug:
            print(f"    人物{person_idx + 1}: {valid_points}/25 キーポイントが有効")
        
        people_data.append((keypoints_2d, confidence))
    
    # keypoints_2dとconfidenceを分離
    keypoints_list = [data[0] for data in people_data]
    confidence_list = [data[1] for data in people_data]
    
    return keypoints_list, confidence_list

def triangulate_points(P1, P2, points1, points2):
    """2組の2D点群から3D点群を三角測量する (OpenCV)"""
    if points1.shape[0] == 0 or points2.shape[0] == 0:
        return np.array([])
    
    points1_t = points1.T
    points2_t = points2.T
    points_4d_hom = cv2.triangulatePoints(P1, P2, points1_t, points2_t)
    points_3d = points_4d_hom[:3] / points_4d_hom[3]
    return points_3d.T

def rotate_coordinates_x_axis(points_3d, angle_degrees=180):
    """3D座標をX軸周りに回転させた後、モーキャプ座標と合わせるための平行移動を適用する"""
    angle_rad = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])
    # 回転を適用
    rotated_points = np.dot(points_3d, rotation_matrix.T)
    
    # 平行移動を適用（X座標を-35, Y座標を189ずらす）
    translation = np.array([-35, 189, 0])
    translated_points = rotated_points + translation
    
    return translated_points

def calculate_acceleration(p_prev2, p_prev1, p_curr):
    """3点間の二階差分で加速度の大きさを計算"""
    if np.isnan(p_prev2).any() or np.isnan(p_prev1).any() or np.isnan(p_curr).any():
        return np.inf
    v1 = p_prev1 - p_prev2
    v2 = p_curr - p_prev1
    acceleration_vec = v2 - v1
    return np.linalg.norm(acceleration_vec)

def swap_left_right_keypoints(keypoints):
    """キーポイント配列の左右の部位を入れ替える"""
    swapped = keypoints.copy()
    l_indices = [5, 6, 7, 12, 13, 14, 16, 18, 19, 20, 21]
    r_indices = [2, 3, 4,  9, 10, 11, 15, 17, 22, 23, 24]
    swapped[l_indices + r_indices] = swapped[r_indices + l_indices]
    return swapped

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """時系列データにバターワースローパスフィルタ（ゼロ位相）を適用する"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    not_nan = ~np.isnan(data)
    filtered_data = data.copy()
    
    if np.any(not_nan) and len(data[not_nan]) > order * 3:
        filtered_data[not_nan] = filtfilt(b, a, data[not_nan])
        
    return filtered_data

# ★★★ 修正版関数: 3次スプライン補間と線形補間の自動切り替え ★★★
def interpolate_spline(data):
    """
    NaNを含む1次元の時系列データを補間する。
    有効なデータ点数に応じて、3次スプライン補間と線形補間を自動的に使い分ける。
    """
    valid_points_count = np.sum(~np.isnan(data))

    # 有効なデータ点数が2点未満の場合、補間は不可能なので元のデータを返す
    if valid_points_count < 2:
        return data

    # 有効なデータのインデックスと値を取得
    valid_indices = np.where(~np.isnan(data))[0]
    valid_values = data[valid_indices]
    all_indices = np.arange(len(data))

    # デバッグ用: 入力データの確認
    if len(valid_values) > 0 and (np.all(np.isnan(valid_values)) or np.all(valid_values == 0)):
        print(f"警告: 有効データが全てNaNまたは0です。valid_count={valid_points_count}, valid_values={valid_values[:5]}")
        return data

    try:
        # 有効なデータが4点以上あれば、滑らかな3次スプライン補間を実行
        if valid_points_count >= 4:
            # bc_type='natural'は、端点での曲率を0にする自然な境界条件
            # extrapolate=False に変更して、範囲外の外挿を無効化
            spline = CubicSpline(valid_indices, valid_values, bc_type='natural', extrapolate=False)
            interpolated_data = spline(all_indices)
            
            # NaN値のチェックと処理
            if np.any(np.isnan(interpolated_data)):
                print(f"警告: スプライン補間後にNaNが発生しました。線形補間にフォールバック")
                interpolated_data = np.interp(all_indices, valid_indices, valid_values)
        # 有効なデータが2点か3点の場合は、シンプルな線形補間を実行
        else:
            interpolated_data = np.interp(all_indices, valid_indices, valid_values)
    except Exception as e:
        print(f"警告: スプライン補間でエラーが発生しました: {e}。線形補間にフォールバック")
        interpolated_data = np.interp(all_indices, valid_indices, valid_values)

    return interpolated_data


# --- メイン処理 ---
def main():
    root_dir = Path(r"G:\gait_pattern\20250811_br")
    stereo_cali_dir = Path(r"G:\gait_pattern\stereo_cali\9g_20250811")
    
    ACCELERATION_THRESHOLD = 150.0
    BUTTERWORTH_CUTOFF = 12.0
    FRAME_RATE = 60
    
    directions = ["fl", "fr"]

    print("統合版マーカーレス歩行解析（スプライン補間版）を開始します。")
    try:
        params_cam1 = load_camera_parameters(stereo_cali_dir / directions[0] / "camera_params_with_ext_OC.json")
        params_cam2 = load_camera_parameters(stereo_cali_dir / directions[1] / "camera_params_with_ext_OC.json")
        P1 = create_projection_matrix(params_cam1)
        P2 = create_projection_matrix(params_cam2)
        print("✓ カメラパラメータを正常に読み込みました。")
    except FileNotFoundError as e:
        print(f"✗ エラー: カメラパラメータファイルが見つかりません。{e}")
        return

    subject_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("sub")])
    if not subject_dirs:
        print(f"✗ エラー: {root_dir} 内に 'sub' で始まる被験者ディレクトリが見つかりません。")
        return

    for subject_dir in subject_dirs:
        thera_dirs = sorted([d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("thera")])
        for thera_dir in thera_dirs:
            # if thera_dir.name != "thera1-0" and thera_dir.name != "thera1-1":
            #     print(f"{thera_dir.name} はスキップします。")
            #     continue

            # if thera_dir.name != "thera0-16":
            #     print(f"{thera_dir.name} はスキップします。")
            #     continue

            print(f"\n{'='*80}")
            print(f"処理開始: {thera_dir.relative_to(root_dir)}")
            
            openpose_dir1 = thera_dir / directions[0] / "openpose.json"
            openpose_dir2 = thera_dir / directions[1] / "openpose.json"
            if not (openpose_dir1.exists() and openpose_dir2.exists()): continue
            files1 = {f.name for f in openpose_dir1.glob("*_keypoints.json")}
            files2 = {f.name for f in openpose_dir2.glob("*_keypoints.json")}
            common_frames = sorted(list(files1 & files2))
            if not common_frames: continue
            
            print(f"  - {len(common_frames)} フレームを処理します。")
            output_dir = thera_dir / "3d_gait_analysis"
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"{thera_dir.name}_3d_results.json"

            print("  - ステップ1: 全フレームの観測値を収集中...")
            all_raw_points = []
            all_points_with_nan = []
            history_for_accel_list = []  # 人物別の履歴リスト

            # 人数を確認するため全フレームをチェック（より正確に）
            all_people_counts = []
            print(f"  - 全フレームの人数を調査中...")
            
            for frame_file in common_frames:
                kp2d_cam1_check, _ = load_openpose_json(openpose_dir1 / frame_file)
                kp2d_cam2_check, _ = load_openpose_json(openpose_dir2 / frame_file)
                frame_max_people = max(len(kp2d_cam1_check), len(kp2d_cam2_check))
                all_people_counts.append(frame_max_people)
            
            # 統計情報を表示
            max_people = max(all_people_counts)
            avg_people = sum(all_people_counts) / len(all_people_counts)
            frames_with_2_people = sum(1 for count in all_people_counts if count >= 2)
            
            print(f"  - 人数統計:")
            print(f"    最大人数: {max_people}人")
            print(f"    平均人数: {avg_people:.2f}人")
            print(f"    2人以上のフレーム: {frames_with_2_people}/{len(all_people_counts)} ({frames_with_2_people/len(all_people_counts)*100:.1f}%)")
            print(f"  - 処理対象人数: {max_people}人")
            
            # 人物別のデータを格納するリストを初期化
            all_raw_points = [[] for _ in range(max_people)]
            all_points_with_nan = [[] for _ in range(max_people)]

            for frame_idx, frame_name in enumerate(common_frames):
                kp2d_cam1_list, _ = load_openpose_json(openpose_dir1 / frame_name)
                kp2d_cam2_list, _ = load_openpose_json(openpose_dir2 / frame_name)

                # 複数人物の処理（デバッグ情報付き）
                debug_info = frame_idx if frame_idx < 5 else None
                results = process_multiple_people(
                    kp2d_cam1_list, kp2d_cam2_list, P1, P2, 
                    history_for_accel_list, ACCELERATION_THRESHOLD, debug_info
                )

                # 結果を人物別に格納（安全な方法で）
                for person_idx in range(max_people):
                    if person_idx < len(results):
                        # 検出された人物のデータを使用
                        all_raw_points[person_idx].append(results[person_idx]['pattern0_points_3d'])
                        all_points_with_nan[person_idx].append(results[person_idx]['best_points_3d'])
                    else:
                        # 検出されなかった人物にはNaNデータを格納
                        all_raw_points[person_idx].append(np.full((25, 3), np.nan))
                        all_points_with_nan[person_idx].append(np.full((25, 3), np.nan))

            print("  - ステップ2: スプライン/線形補間で欠損値を補間中...")
            
            # 人物別にデータを処理
            all_person_results = []
            
            for person_idx in range(max_people):
                print(f"    人物{person_idx + 1}を処理中...")
                
                points_with_nan_array = np.array(all_points_with_nan[person_idx])
                spline_points_array = np.full_like(points_with_nan_array, np.nan)
                
                # rawデータの処理
                raw_points_array = np.array(all_raw_points[person_idx])
                raw_spline_points_array = np.full_like(raw_points_array, np.nan)
                
                # rawデータの補間
                for kp_idx in range(raw_points_array.shape[1]):
                    for axis_idx in range(raw_points_array.shape[2]):
                        sequence = raw_points_array[:, kp_idx, axis_idx]
                        raw_spline_points_array[:, kp_idx, axis_idx] = interpolate_spline(sequence)
                
                # 最適化後データの補間
                for kp_idx in range(points_with_nan_array.shape[1]):
                    for axis_idx in range(points_with_nan_array.shape[2]):
                        sequence = points_with_nan_array[:, kp_idx, axis_idx]
                        spline_points_array[:, kp_idx, axis_idx] = interpolate_spline(sequence)

                print(f"  - ステップ3: 人物{person_idx + 1}のバターワースフィルタ適用中...")
                # 最適化後データのフィルタリング
                final_points_array = np.full_like(spline_points_array, np.nan)
                for kp_idx in range(spline_points_array.shape[1]):
                    for axis_idx in range(spline_points_array.shape[2]):
                        sequence = spline_points_array[:, kp_idx, axis_idx]
                        final_points_array[:, kp_idx, axis_idx] = butter_lowpass_filter(sequence, BUTTERWORTH_CUTOFF, FRAME_RATE)
                
                # rawデータのフィルタリング
                raw_filt_points_array = np.full_like(raw_spline_points_array, np.nan)
                for kp_idx in range(raw_spline_points_array.shape[1]):
                    for axis_idx in range(raw_spline_points_array.shape[2]):
                        sequence = raw_spline_points_array[:, kp_idx, axis_idx]
                        raw_filt_points_array[:, kp_idx, axis_idx] = butter_lowpass_filter(sequence, BUTTERWORTH_CUTOFF, FRAME_RATE)
                
                # この人物の結果を保存
                all_person_results.append({
                    'raw_points': all_raw_points[person_idx],
                    'raw_spline_points': raw_spline_points_array,
                    'raw_filt_points': raw_filt_points_array,
                    'optimized_points': all_points_with_nan[person_idx],
                    'spline_points': spline_points_array,
                    'final_points': final_points_array
                })

            print("  - ステップ4: 全ての結果を結合して保存...")
            analysis_results = []
            for t, frame_name in enumerate(common_frames):
                frame_result = {"frame_name": frame_name}
                
                # 人物別にデータを保存
                for person_idx in range(max_people):
                    person_data = all_person_results[person_idx]
                    frame_result[f"person_{person_idx + 1}"] = {
                        "points_3d_raw": person_data['raw_points'][t].tolist(),
                        "points_3d_raw_filt": person_data['raw_filt_points'][t].tolist(),
                        "points_3d_optimized": person_data['optimized_points'][t].tolist(),
                        "points_3d_final": person_data['final_points'][t].tolist()
                    }
                
                analysis_results.append(frame_result)
            
            try:
                with open(output_file, 'w') as f:
                    json.dump(analysis_results, f, indent=4)
                print(f"  ✓ 処理完了。結果を {output_file.relative_to(root_dir)} に保存しました。")
            except Exception as e:
                print(f"  ✗ JSON保存エラー: {e}")
                traceback.print_exc()

            # ★★★ 新機能: 人物別のRKneeのZ座標変化をグラフ化 ★★★
            print("  - ステップ5: 人物別のRKneeのZ座標変化をグラフ化...")
            try:
                # RKneeのインデックスは10
                rknee_idx = 10
                z_axis_idx = 2
                frames = np.arange(len(common_frames))
                
                # 人物数に応じてサブプロットを作成
                fig_height = 6 * max_people
                plt.figure(figsize=(18, fig_height))
                
                for person_idx in range(max_people):
                    person_data = all_person_results[person_idx]
                    
                    # 各データセットからRKneeのZ座標を抽出
                    raw_z = np.array([person_data['raw_points'][t][rknee_idx, z_axis_idx] for t in range(len(common_frames))])
                    raw_filt_z = person_data['raw_filt_points'][:, rknee_idx, z_axis_idx]
                    optimized_z = np.array([person_data['optimized_points'][t][rknee_idx, z_axis_idx] for t in range(len(common_frames))])
                    final_z = person_data['final_points'][:, rknee_idx, z_axis_idx]
                    
                    # 人物ごとに3つのサブプロットを作成
                    base_plot = person_idx * 3 + 1
                    
                    # rawデータの処理過程
                    plt.subplot(max_people, 3, base_plot)
                    valid_raw = ~np.isnan(raw_z)
                    plt.plot(frames[valid_raw], raw_z[valid_raw], 'ro', label='Raw (valid)', markersize=3)
                    plt.plot(frames, raw_filt_z, 'red', label='Raw + Spline + Filter', linewidth=2)
                    plt.xlabel('Frame Number')
                    plt.ylabel('Z Coordinate (mm)')
                    plt.title(f'Person {person_idx + 1}: Raw Data Processing')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # 最適化後データの処理過程
                    plt.subplot(max_people, 3, base_plot + 1)
                    valid_opt = ~np.isnan(optimized_z)
                    plt.plot(frames[valid_opt], optimized_z[valid_opt], 'bo', label='Optimized (valid)', markersize=3)
                    plt.plot(frames, final_z, 'blue', label='Optimized + Spline + Filter', linewidth=2)
                    plt.xlabel('Frame Number')
                    plt.ylabel('Z Coordinate (mm)')
                    plt.title(f'Person {person_idx + 1}: Optimized Data Processing')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # 全データの比較
                    plt.subplot(max_people, 3, base_plot + 2)
                    plt.plot(frames[valid_raw], raw_z[valid_raw], 'ro', label='Raw', alpha=0.7, markersize=2)
                    plt.plot(frames, raw_filt_z, 'red', label='Raw Final', linewidth=2)
                    plt.plot(frames[valid_opt], optimized_z[valid_opt], 'bo', label='Optimized', alpha=0.7, markersize=2)
                    plt.plot(frames, final_z, 'blue', label='Optimized Final', linewidth=2)
                    plt.xlabel('Frame Number')
                    plt.ylabel('Z Coordinate (mm)')
                    plt.title(f'Person {person_idx + 1}: Raw vs Optimized Final')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # 統計をコンソールに表示
                    print(f"  人物{person_idx + 1} RKnee Z座標統計:")
                    print(f"    Raw有効率: {np.sum(valid_raw)/len(frames)*100:.1f}% ({np.sum(valid_raw)}/{len(frames)})")
                    print(f"    最適化後有効率: {np.sum(valid_opt)/len(frames)*100:.1f}% ({np.sum(valid_opt)}/{len(frames)})")
                    if np.sum(~np.isnan(raw_filt_z)) > 0:
                        print(f"    Raw最終範囲: {np.nanmin(raw_filt_z):.1f} ~ {np.nanmax(raw_filt_z):.1f} mm")
                    if np.sum(~np.isnan(final_z)) > 0:
                        print(f"    最適化最終範囲: {np.nanmin(final_z):.1f} ~ {np.nanmax(final_z):.1f} mm")
                
                plt.tight_layout()
                
                # グラフを保存
                graph_path = output_dir / f"{thera_dir.name}_RKnee_Z_multi_person_analysis.png"
                plt.savefig(graph_path, dpi=300, bbox_inches='tight')
                print(f"  ✓ 人物別RKneeのZ座標グラフを保存: {graph_path.relative_to(root_dir)}")
                
                plt.close()
                
            except Exception as e:
                print(f"  ✗ グラフ作成エラー: {e}")
                traceback.print_exc()
                
            # ★★★ 新機能: 人物別のRAnkleのZ座標変化をグラフ化 ★★★
            print("  - ステップ5: 人物別のRAnkleのZ座標変化をグラフ化...")
            try:
                # RAnkleのインデックスは14 (COCO-25フォーマット)
                rankle_idx = 14
                
                # Z軸のインデックス（座標変換後のY軸）
                z_axis_idx = 1
                
                # データが存在する場合のみ処理
                if len(all_person_results) > 0:
                    # 人数に応じて動的にサブプロットのレイアウトを決定
                    num_people = len(all_person_results)
                    
                    if num_people == 1:
                        # 1人の場合: 1x3のサブプロット
                        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                        fig.suptitle(f'{thera_dir.name} - 人物別RAnkleのZ座標変化分析 (1人)', fontsize=16, fontweight='bold')
                        # axesを2次元配列の形式に統一
                        axes = axes.reshape(1, 3)
                    else:
                        # 2人以上の場合: 2x3のサブプロット
                        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                        fig.suptitle(f'{thera_dir.name} - 人物別RAnkleのZ座標変化分析 ({num_people}人)', fontsize=16, fontweight='bold')
                    
                    person_colors = ['blue', 'red', 'green', 'orange', 'purple']
                    frames = np.arange(len(common_frames))
                    
                    for person_idx, person_data in enumerate(all_person_results):
                        if person_idx >= len(person_colors):
                            break
                        
                        # 2人以上の場合で、2人目以降は下段に配置
                        if num_people == 1:
                            row_idx = 0
                        else:
                            row_idx = person_idx  # 0番目は上段、1番目は下段
                        
                        color = person_colors[person_idx]
                        
                        # 各データセットからRAnkleのZ座標を抽出
                        raw_z = np.array([person_data['raw_points'][t][rankle_idx, z_axis_idx] for t in range(len(common_frames))])
                        raw_filt_z = person_data['raw_filt_points'][:, rankle_idx, z_axis_idx]
                        optimized_z = np.array([person_data['optimized_points'][t][rankle_idx, z_axis_idx] for t in range(len(common_frames))])
                        final_z = person_data['final_points'][:, rankle_idx, z_axis_idx]
                        
                        # 有効データのマスクを計算
                        valid_raw = ~np.isnan(raw_z)
                        valid_opt = ~np.isnan(optimized_z)
                        
                        # rawデータの処理過程
                        axes[row_idx, 0].plot(frames[valid_raw], raw_z[valid_raw], 'ro', label=f'Person {person_idx + 1} Raw (valid)', markersize=3)
                        axes[row_idx, 0].plot(frames, raw_filt_z, color='red', label=f'Person {person_idx + 1} Raw + Spline + Filter', linewidth=2)
                        axes[row_idx, 0].set_xlabel('Frame Number')
                        axes[row_idx, 0].set_ylabel('Z Coordinate (mm)')
                        axes[row_idx, 0].set_title(f'Person {person_idx + 1}: Raw Data Processing')
                        axes[row_idx, 0].legend()
                        axes[row_idx, 0].grid(True, alpha=0.3)
                        
                        # 最適化後データの処理過程
                        axes[row_idx, 1].plot(frames[valid_opt], optimized_z[valid_opt], 'bo', label=f'Person {person_idx + 1} Optimized (valid)', markersize=3)
                        axes[row_idx, 1].plot(frames, final_z, color='blue', label=f'Person {person_idx + 1} Optimized + Spline + Filter', linewidth=2)
                        axes[row_idx, 1].set_xlabel('Frame Number')
                        axes[row_idx, 1].set_ylabel('Z Coordinate (mm)')
                        axes[row_idx, 1].set_title(f'Person {person_idx + 1}: Optimized Data Processing')
                        axes[row_idx, 1].legend()
                        axes[row_idx, 1].grid(True, alpha=0.3)
                        
                        # 全データの比較
                        axes[row_idx, 2].plot(frames[valid_raw], raw_z[valid_raw], 'ro', label=f'Person {person_idx + 1} Raw', alpha=0.7, markersize=2)
                        axes[row_idx, 2].plot(frames, raw_filt_z, color='red', label=f'Person {person_idx + 1} Raw Final', linewidth=2)
                        axes[row_idx, 2].plot(frames[valid_opt], optimized_z[valid_opt], 'bo', label=f'Person {person_idx + 1} Optimized', alpha=0.7, markersize=2)
                        axes[row_idx, 2].plot(frames, final_z, color='blue', label=f'Person {person_idx + 1} Optimized Final', linewidth=2)
                        axes[row_idx, 2].set_xlabel('Frame Number')
                        axes[row_idx, 2].set_ylabel('Z Coordinate (mm)')
                        axes[row_idx, 2].set_title(f'Person {person_idx + 1}: Raw vs Optimized Final')
                        axes[row_idx, 2].legend()
                        axes[row_idx, 2].grid(True, alpha=0.3)
                        
                        # 統計をコンソールに表示
                        print(f"  人物{person_idx + 1} RAnkle Z座標統計:")
                        print(f"    Raw有効率: {np.sum(valid_raw)/len(frames)*100:.1f}% ({np.sum(valid_raw)}/{len(frames)})")
                        print(f"    最適化後有効率: {np.sum(valid_opt)/len(frames)*100:.1f}% ({np.sum(valid_opt)}/{len(frames)})")
                        if np.sum(~np.isnan(raw_filt_z)) > 0:
                            print(f"    Raw最終範囲: {np.nanmin(raw_filt_z):.1f} ~ {np.nanmax(raw_filt_z):.1f} mm")
                        if np.sum(~np.isnan(final_z)) > 0:
                            print(f"    最適化最終範囲: {np.nanmin(final_z):.1f} ~ {np.nanmax(final_z):.1f} mm")
                        
                        # 2人目以降で2人の場合のみ処理（2x3レイアウトの場合）
                        if num_people >= 2 and person_idx >= 1:
                            break
                    
                    plt.tight_layout()
                    
                    # グラフを保存
                    graph_path = output_dir / f"{thera_dir.name}_RAnkle_Z_multi_person_analysis.png"
                    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
                    print(f"  ✓ 人物別RAnkleのZ座標グラフを保存: {graph_path.relative_to(root_dir)}")
                    
                    plt.close()
                
            except Exception as e:
                print(f"  ⚠️ RAnkle Z座標グラフ作成中にエラー: {e}")
                if 'fig' in locals():
                    plt.close()
                
    print(f"\n{'='*80}")
    print("全ての処理が完了しました。")

def process_multiple_people(kp2d_cam1_list, kp2d_cam2_list, P1, P2, history_for_accel_list, ACCELERATION_THRESHOLD, debug_frame_idx=None):
    """
    複数人物の3D姿勢推定を処理する関数
    
    Args:
        kp2d_cam1_list: カメラ1の人物別キーポイントリスト
        kp2d_cam2_list: カメラ2の人物別キーポイントリスト
        P1, P2: プロジェクション行列
        history_for_accel_list: 人物別の加速度計算用履歴
        ACCELERATION_THRESHOLD: 加速度閾値
        debug_frame_idx: デバッグ表示するフレーム番号（Noneの場合は表示しない）
    
    Returns:
        results: 人物別の処理結果
    """
    # 実際に処理する人数を決定（両カメラで検出された人数の最大値）
    num_people = max(len(kp2d_cam1_list), len(kp2d_cam2_list))
    results = []
    
    if debug_frame_idx is not None and debug_frame_idx < 5:
        print(f"    フレーム{debug_frame_idx}: process_multiple_people開始")
        print(f"      カメラ1: {len(kp2d_cam1_list)}人, カメラ2: {len(kp2d_cam2_list)}人")
        print(f"      処理予定人数: {num_people}人")
    
    # history_for_accel_listが短い場合は拡張
    while len(history_for_accel_list) < num_people:
        history_for_accel_list.append([])
        if debug_frame_idx is not None and debug_frame_idx < 5:
            print(f"      履歴リストを拡張: {len(history_for_accel_list)}人分")
    
    for person_idx in range(num_people):
        if debug_frame_idx is not None and debug_frame_idx < 5:
            print(f"      人物{person_idx + 1}を処理開始...")
            
        # 各人物のキーポイントを取得（存在しない場合はNaNで埋める）
        if person_idx < len(kp2d_cam1_list):
            kp2d_cam1 = kp2d_cam1_list[person_idx]
            if debug_frame_idx is not None and debug_frame_idx < 5:
                valid_cam1 = np.sum(~np.isnan(kp2d_cam1).any(axis=1))
                print(f"        カメラ1: {valid_cam1}/25キーポイント有効")
        else:
            kp2d_cam1 = np.full((25, 2), np.nan)
            if debug_frame_idx is not None and debug_frame_idx < 5:
                print(f"        カメラ1: データなし（NaNで埋める）")
            
        if person_idx < len(kp2d_cam2_list):
            kp2d_cam2 = kp2d_cam2_list[person_idx]
            if debug_frame_idx is not None and debug_frame_idx < 5:
                valid_cam2 = np.sum(~np.isnan(kp2d_cam2).any(axis=1))
                print(f"        カメラ2: {valid_cam2}/25キーポイント有効")
        else:
            kp2d_cam2 = np.full((25, 2), np.nan)
            if debug_frame_idx is not None and debug_frame_idx < 5:
                print(f"        カメラ2: データなし（NaNで埋める）")
        
        # この人物の履歴を取得
        history_for_accel = history_for_accel_list[person_idx]
        
        best_points_3d = None
        min_avg_acceleration = np.inf
        pattern0_points_3d = np.full((25, 3), np.nan)

        for pattern_id in range(4):
            kp1_trial = swap_left_right_keypoints(kp2d_cam1) if pattern_id in [1, 3] else kp2d_cam1
            kp2_trial = swap_left_right_keypoints(kp2d_cam2) if pattern_id in [2, 3] else kp2d_cam2

            valid_indices = np.where(~np.isnan(kp1_trial).any(axis=1) & ~np.isnan(kp2_trial).any(axis=1))[0]
            if len(valid_indices) == 0: 
                continue

            points_3d_trial_raw = triangulate_points(P1, P2, kp1_trial[valid_indices], kp2_trial[valid_indices])
            points_3d_trial = rotate_coordinates_x_axis(points_3d_trial_raw)
            full_points_3d = np.full((25, 3), np.nan)
            full_points_3d[valid_indices] = points_3d_trial
            
            if pattern_id == 0:
                pattern0_points_3d = full_points_3d

            current_avg_accel = 0
            count = 0
            if len(history_for_accel) >= 2:
                eval_indices = [10, 11, 13, 14]
                for idx in eval_indices:
                    accel = calculate_acceleration(history_for_accel[-2][idx], history_for_accel[-1][idx], full_points_3d[idx])
                    if accel != np.inf:
                        current_avg_accel += accel; count += 1
                current_avg_accel = current_avg_accel / count if count > 0 else np.inf
            
            if current_avg_accel < min_avg_acceleration:
                min_avg_acceleration = current_avg_accel
                best_points_3d = full_points_3d

        # この人物の結果を保存
        is_error_frame = best_points_3d is None or min_avg_acceleration >= ACCELERATION_THRESHOLD
        input_points = np.full((25, 3), np.nan) if is_error_frame else best_points_3d
        
        result = {
            'pattern0_points_3d': pattern0_points_3d,
            'best_points_3d': input_points,
            'min_avg_acceleration': min_avg_acceleration
        }
        results.append(result)
        
        if debug_frame_idx is not None and debug_frame_idx < 5:
            valid_3d = np.sum(~np.isnan(pattern0_points_3d).any(axis=1))
            print(f"        結果: {valid_3d}/25の3Dキーポイント生成, 加速度={min_avg_acceleration:.2f}")
        
        # 履歴を更新
        history_for_accel.append(best_points_3d if best_points_3d is not None else np.full((25, 3), np.nan))
    
    if debug_frame_idx is not None and debug_frame_idx < 5:
        print(f"    フレーム{debug_frame_idx}: {len(results)}人分の結果を生成")
    
    return results

if __name__ == '__main__':
    main()
