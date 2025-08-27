import cv2
import numpy as np
import json
from pathlib import Path
from scipy.signal import butter, filtfilt
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
# 新しいカルマンフィルタモジュールをインポート
from kalman_module import predict_next_point

# --- ユーティリティ関数 ---

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

def load_openpose_json(json_file_path):
    """単一のOpenPose JSONファイルからキーポイントと信頼度を読み込む（複数人対応）"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    if not data.get('people'):
        return [np.full((25, 2), np.nan)], [np.full((25,), np.nan)]

    keypoints_list = []
    confidence_list = []
    for person_data in data['people']:
        keypoints_raw = np.array(person_data['pose_keypoints_2d']).reshape(-1, 3)
        keypoints_2d = keypoints_raw[:, :2]
        confidence = keypoints_raw[:, 2]
        keypoints_2d[confidence == 0] = np.nan
        confidence[confidence == 0] = np.nan
        keypoints_list.append(keypoints_2d)
        confidence_list.append(confidence)

    return keypoints_list, confidence_list

def p2e(projective):
    """projective座標からeuclidean座標に変換"""
    return (projective / projective[-1, :])[0:-1, :]

def construct_D_block(P, uv, w=1):
    """三角測量用のD行列のブロックを構築"""
    return w * np.vstack((
        uv[0] * P[2, :] - P[0, :],
        uv[1] * P[2, :] - P[1, :]
    ))

def weighted_linear_triangulation(P1, P2, correspondences, weights=None):
    """重み付き線形三角測量を実行"""
    projection_matrices = [P1, P2]
    n_cameras = len(projection_matrices)

    if weights is None:
        w = np.ones(n_cameras)
    else:
        w = [np.nan_to_num(wi, nan=0.1) for wi in weights]

    D = np.zeros((n_cameras * 2, 4))
    for cam_idx in range(n_cameras):
        P = projection_matrices[cam_idx]
        uv = correspondences[:, cam_idx]
        D[cam_idx * 2:cam_idx * 2 + 2, :] = construct_D_block(P, uv, w=w[cam_idx])

    Q = D.T.dot(D)
    u, s, vh = np.linalg.svd(Q)
    point_3d = p2e(u[:, -1, np.newaxis])

    return point_3d.flatten()

def triangulate_points_weighted(P1, P2, points1, points2, confidences1, confidences2):
    """重み付き三角測量を使用して2D点から3D点を計算"""
    points_3d_list = []
    for i in range(points1.shape[0]):
        correspondences = np.column_stack([points1[i], points2[i]])
        weights = [confidences1[i], confidences2[i]]
        try:
            point_3d = weighted_linear_triangulation(P1, P2, correspondences, weights)
            points_3d_list.append(point_3d)
        except Exception:
            points_3d_list.append(np.full(3, np.nan))
    return np.array(points_3d_list)

def rotate_coordinates_x_axis(points_3d, angle_degrees=180):
    """3D座標をX軸周りに回転させた後、平行移動を適用する"""
    angle_rad = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])
    rotated_points = np.dot(points_3d, rotation_matrix.T)
    translation = np.array([-35, 189, 0])
    return rotated_points + translation

def triangulate_and_rotate(P1, P2, points1, points2, confidences1, confidences2):
    """三角測量と座標回転をまとめて行うヘルパー関数"""
    valid_indices = np.where(~np.isnan(points1).any(axis=1) & ~np.isnan(points2).any(axis=1))[0]
    if len(valid_indices) == 0:
        return np.full((25, 3), np.nan)

    points_3d_raw = triangulate_points_weighted(
        P1, P2,
        points1[valid_indices],
        points2[valid_indices],
        confidences1[valid_indices],
        confidences2[valid_indices]
    )
    points_3d_rotated = rotate_coordinates_x_axis(points_3d_raw)
    full_points_3d = np.full((25, 3), np.nan)
    full_points_3d[valid_indices] = points_3d_rotated
    return full_points_3d

def calculate_acceleration(p_prev2, p_prev1, p_curr):
    """3点間の二階差分で加速度の大きさを計算"""
    if np.isnan(p_prev2).any() or np.isnan(p_prev1).any() or np.isnan(p_curr).any():
        return np.inf
    v1 = p_prev1 - p_prev2
    v2 = p_curr - p_prev1
    acceleration_vec = v2 - v1
    return np.linalg.norm(acceleration_vec)

def calculate_average_acceleration(history, current_points, eval_indices=[10, 11, 13, 14]):
    """特定キーポイントの平均加速度を計算"""
    if len(history) < 2:
        return 0.0

    p_prev2, p_prev1 = history[-2], history[-1]
    total_accel, count = 0, 0
    for idx in eval_indices:
        accel = calculate_acceleration(p_prev2[idx], p_prev1[idx], current_points[idx])
        if accel != np.inf:
            total_accel += accel
            count += 1
    return total_accel / count if count > 0 else np.inf

def swap_left_right_keypoints(keypoints):
    """キーポイント配列の左右の部位を入れ替える"""
    swapped = keypoints.copy()
    l_indices = [12, 13, 14, 16, 18, 19, 20, 21]
    r_indices = [9, 10, 11, 15, 17, 22, 23, 24]
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

def process_single_person_4patterns(kp1, cf1, kp2, cf2, P1, P2, history, threshold, frame_idx_for_debug=None):
    """
    4パターンのスイッチングを試し、最適なものを採用し、加速度も返す。
    """
    patterns_3d = []
    accelerations = []

    kp_combinations = [
        (kp1, cf1, kp2, cf2),
        (swap_left_right_keypoints(kp1), swap_left_right_keypoints(cf1), kp2, cf2),
        (kp1, cf1, swap_left_right_keypoints(kp2), swap_left_right_keypoints(cf2)),
        (swap_left_right_keypoints(kp1), swap_left_right_keypoints(cf1),
         swap_left_right_keypoints(kp2), swap_left_right_keypoints(cf2))
    ]

    for kp1_trial, cf1_trial, kp2_trial, cf2_trial in kp_combinations:
        points_3d = triangulate_and_rotate(P1, P2, kp1_trial, kp2_trial, cf1_trial, cf2_trial)
        accel = calculate_average_acceleration(history, points_3d)
        patterns_3d.append(points_3d)
        accelerations.append(accel)

    min_accel = np.inf
    best_pattern_idx = -1
    valid_accels = np.array([a if a != np.inf else np.nan for a in accelerations])
    if not np.all(np.isnan(valid_accels)):
        best_pattern_idx = np.nanargmin(valid_accels)
        min_accel = accelerations[best_pattern_idx]

    if frame_idx_for_debug is not None:
        accel_str = ", ".join([f"{a:8.2f}" if a != np.inf else "   inf" for a in accelerations])
        print(f"Frame {frame_idx_for_debug:04d} | Accels(N,S1,S2,S12): [{accel_str}] | Best: Idx={best_pattern_idx}, Accel={min_accel:.2f}")

    raw_points_3d = patterns_3d[0]

    if best_pattern_idx != -1 and min_accel < threshold:
        corrected_points_3d = patterns_3d[best_pattern_idx]
    else:
        corrected_points_3d = np.full((25, 3), np.nan)

    return raw_points_3d, corrected_points_3d, accelerations, min_accel

# --- メイン処理 ---
def main():
    root_dir = Path(r"G:\gait_pattern\20250811_br")
    stereo_cali_dir = Path(r"G:\gait_pattern\stereo_cali\9g_20250811")

    ACCELERATION_THRESHOLD = 200.0 # 閾値を少し上げる
    BUTTERWORTH_CUTOFF = 12.0
    FRAME_RATE = 60
    DEBUG_ACCELERATION = True

    directions = ["fl", "fr"]

    print("二階差分カルマンフィルタによる3D歩行解析を開始します。")
    try:
        params_cam1 = load_camera_parameters(stereo_cali_dir / directions[0] / "camera_params_with_ext_OC.json")
        params_cam2 = load_camera_parameters(stereo_cali_dir / directions[1] / "camera_params_with_ext_OC.json")
        P1 = create_projection_matrix(params_cam1)
        P2 = create_projection_matrix(params_cam2)
    except FileNotFoundError as e:
        print(f"✗ エラー: カメラパラメータファイルが見つかりません。{e}")
        return

    subject_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("sub")])
    for subject_dir in subject_dirs:
        thera_dirs = sorted([d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("thera")])
        for thera_dir in thera_dirs:
            if subject_dir.name != "sub1" and thera_dir.name != "thera0-2":
                continue

            print(f"\n{'='*80}\n処理開始: {thera_dir.relative_to(root_dir)}")

            openpose_dir1 = thera_dir / directions[0] / "openpose.json"
            openpose_dir2 = thera_dir / directions[1] / "openpose.json"
            if not (openpose_dir1.exists() and openpose_dir2.exists()): continue

            common_frames = sorted(list({f.name for f in openpose_dir1.glob("*_keypoints.json")} &
                                         {f.name for f in openpose_dir2.glob("*_keypoints.json")}))
            if not common_frames: continue

            output_dir = thera_dir / "3d_gait_analysis_kalman_v4"
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"{thera_dir.name}_3d_results_kalman.json"

            max_people = 0
            for frame_file in common_frames:
                kp1, _ = load_openpose_json(openpose_dir1 / frame_file)
                kp2, _ = load_openpose_json(openpose_dir2 / frame_file)
                max_people = max(max_people, len(kp1), len(kp2))

            all_raw_points = [[] for _ in range(max_people)]
            all_corrected_points = [[] for _ in range(max_people)]
            all_accelerations_data = [[] for _ in range(max_people)]
            history_list = [np.empty((0, 25, 3)) for _ in range(max_people)]

            print("  - ステップ1: フレーム毎の補正とオンラインでのカルマンフィルタ補間...")
            for frame_idx, frame_name in enumerate(tqdm(common_frames, desc="  フレーム処理中")):
                kp2d_cam1_list, conf_cam1_list = load_openpose_json(openpose_dir1 / frame_name)
                kp2d_cam2_list, conf_cam2_list = load_openpose_json(openpose_dir2 / frame_name)
                num_detected = max(len(kp2d_cam1_list), len(kp2d_cam2_list))

                for p_idx in range(max_people):
                    if p_idx < num_detected:
                        kp1 = kp2d_cam1_list[p_idx] if p_idx < len(kp2d_cam1_list) else np.full((25, 2), np.nan)
                        cf1 = conf_cam1_list[p_idx] if p_idx < len(conf_cam1_list) else np.full((25,), np.nan)
                        kp2 = kp2d_cam2_list[p_idx] if p_idx < len(kp2d_cam2_list) else np.full((25, 2), np.nan)
                        cf2 = conf_cam2_list[p_idx] if p_idx < len(conf_cam2_list) else np.full((25,), np.nan)

                        debug_frame_idx = frame_idx if DEBUG_ACCELERATION else None
                        raw_3d, corrected_3d, accels, min_accel = process_single_person_4patterns(kp1, cf1, kp2, cf2, P1, P2, history_list[p_idx], ACCELERATION_THRESHOLD, debug_frame_idx)

                        all_accelerations_data[p_idx].append({'all': accels, 'min': min_accel})

                        if np.any(np.isnan(corrected_3d)):
                            # ★★★ 修正点: 補間が行われたことを示すprint文を追加 ★★★
                            if DEBUG_ACCELERATION:
                                print(f"Frame {frame_idx:04d}, Person {p_idx + 1}: 欠損を検出。カルマンフィルタで補間します。")

                            interpolated_3d = np.full((25, 3), np.nan)
                            for joint_idx in range(25):
                                for axis_idx in range(3):
                                    history_seq = history_list[p_idx][:, joint_idx, axis_idx]
                                    valid_history = history_seq[~np.isnan(history_seq)]
                                    if len(valid_history) >= 2:
                                        interpolated_3d[joint_idx, axis_idx] = predict_next_point(valid_history)
                            corrected_3d = interpolated_3d

                        all_raw_points[p_idx].append(raw_3d)
                        all_corrected_points[p_idx].append(corrected_3d)
                        history_list[p_idx] = np.vstack([history_list[p_idx], corrected_3d[np.newaxis, ...]])
                    else:
                        nan_points = np.full((25, 3), np.nan)
                        all_raw_points[p_idx].append(nan_points)
                        all_corrected_points[p_idx].append(nan_points)
                        history_list[p_idx] = np.vstack([history_list[p_idx], nan_points[np.newaxis, ...]])
                        all_accelerations_data[p_idx].append({'all': [np.inf]*4, 'min': np.inf})

            # ... (後処理、JSON保存、軌道グラフ描画は変更なし) ...

            print("  - ステップ5: 全フレームの加速度をグラフ化...")
            plot_acceleration_graph(all_accelerations_data, common_frames, ACCELERATION_THRESHOLD, output_dir, thera_dir.name)


def plot_acceleration_graph(all_accelerations_data, frames, threshold, output_dir, file_prefix):
    """全フレームの加速度をプロットし、グラフを保存する関数"""
    try:
        max_people = len(all_accelerations_data)
        if max_people == 0: return

        plt.figure(figsize=(20, 7 * max_people))
        time_axis = np.arange(len(frames))

        for p_idx, person_accel_data in enumerate(all_accelerations_data):
            accel_n = np.array([d['all'][0] for d in person_accel_data], dtype=float)
            accel_s1 = np.array([d['all'][1] for d in person_accel_data], dtype=float)
            accel_s2 = np.array([d['all'][2] for d in person_accel_data], dtype=float)
            accel_s12 = np.array([d['all'][3] for d in person_accel_data], dtype=float)
            min_accels = np.array([d['min'] for d in person_accel_data], dtype=float)

            plt.subplot(max_people, 1, p_idx + 1)
            plt.plot(time_axis, accel_n, label='Normal (N)', alpha=0.6, color='blue')
            plt.plot(time_axis, accel_s1, label='Swap Cam1 (S1)', alpha=0.4, linestyle=':')
            plt.plot(time_axis, accel_s2, label='Swap Cam2 (S2)', alpha=0.4, linestyle=':')
            plt.plot(time_axis, accel_s12, label='Swap Both (S12)', alpha=0.4, linestyle=':')

            plt.plot(time_axis, min_accels, 'o', color='red', markersize=3, label='Selected Min Acceleration')

            plt.axhline(y=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')

            plt.title(f'Person {p_idx + 1}: Acceleration Analysis')
            plt.xlabel('Frame Number')
            plt.ylabel('Average Acceleration')
            plt.legend(loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.ylim(0, max(threshold * 3, 300))

        plt.tight_layout()
        graph_path = output_dir / f"{file_prefix}_acceleration_analysis.png"
        plt.savefig(graph_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 加速度の分析グラフを保存しました: {graph_path}")

    except Exception as e:
        print(f"  ✗ 加速度グラフ作成エラー: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
