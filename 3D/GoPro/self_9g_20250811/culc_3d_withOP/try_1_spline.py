import cv2
import numpy as np
import json
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline # 3次スプライン補間のためにインポート
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def calculate_average_acceleration(history, current_points, eval_indices=[14]):
    """履歴不足対策を追加した加速度計算"""
    # ★★★ 修正: 履歴不足の場合は0を返す（infではなく） ★★★
    if len(history) < 2:
        return 0.0  # infの代わりに0を返す

    # 有効な履歴を探す（最大5フレーム遡る）
    valid_history = []
    for i in range(len(history) - 1, max(-1, len(history) - 6), -1):
        frame_valid = True
        for idx in eval_indices:
            if np.isnan(history[i][idx]).any():
                frame_valid = False
                break
        if frame_valid:
            valid_history.append(history[i])
            if len(valid_history) == 2:
                break

    # ★★★ 修正: 履歴不足時の対処 ★★★
    if len(valid_history) < 2:
        return 0.0  # 履歴不足時は0を返す

    # 現在のフレームの有効性チェック
    current_valid = True
    for idx in eval_indices:
        if np.isnan(current_points[idx]).any():
            current_valid = False
            break

    if not current_valid:
        return 0.0  # 現在フレーム無効時は0を返す

    # 加速度計算
    p_prev2, p_prev1 = valid_history[1], valid_history[0]
    total_accel, count = 0, 0
    for idx in eval_indices:
        accel = calculate_acceleration(p_prev2[idx], p_prev1[idx], current_points[idx])
        if accel != np.inf:
            total_accel += accel
            count += 1

    return total_accel / count if count > 0 else 0.0

def swap_left_right_keypoints(keypoints):
    """キーポイント配列の左右の部位を入れ替える"""
    swapped = keypoints.copy()
    l_indices = [5, 6, 7, 12, 13, 14, 16, 18, 19, 20, 21]
    r_indices = [2, 3, 4,  9, 10, 11, 15, 17, 22, 23, 24]
    swapped[l_indices + r_indices] = swapped[r_indices + l_indices]
    return swapped

def calculate_left_right_acceleration(history, current_points, left_indices=[14], right_indices=[11]):
    """左右キーポイントの平均加速度を個別に計算"""
    left_accel = calculate_average_acceleration(history, current_points, left_indices)
    right_accel = calculate_average_acceleration(history, current_points, right_indices)
    return left_accel, right_accel

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
    左右キーポイントの加速度チェックに基づく4パターンスイッチング。
    両方が閾値を超えた場合→スイッチング、片方が閾値を超える場合→欠損値として扱う
    """
    patterns_3d = []
    accelerations = []
    left_right_accels = []

    kp_combinations = [
        (kp1, cf1, kp2, cf2),  # Normal (N)
        (swap_left_right_keypoints(kp1), swap_left_right_keypoints(cf1), kp2, cf2),  # Swap Cam1 (S1)
        (kp1, cf1, swap_left_right_keypoints(kp2), swap_left_right_keypoints(cf2)),  # Swap Cam2 (S2)
        (swap_left_right_keypoints(kp1), swap_left_right_keypoints(cf1),
         swap_left_right_keypoints(kp2), swap_left_right_keypoints(cf2))  # Swap Both (S12)
    ]

    for kp1_trial, cf1_trial, kp2_trial, cf2_trial in kp_combinations:
        points_3d = triangulate_and_rotate(P1, P2, kp1_trial, kp2_trial, cf1_trial, cf2_trial)
        # 左右の加速度を個別に計算
        left_accel, right_accel = calculate_left_right_acceleration(history, points_3d)
        total_accel = calculate_average_acceleration(history, points_3d)

        patterns_3d.append(points_3d)
        accelerations.append(total_accel)
        left_right_accels.append((left_accel, right_accel))

    # ★★★ 新しいロジック: 左右キーポイントの加速度チェック ★★★
    normal_left_accel, normal_right_accel = left_right_accels[0]

    # 両方とも閾値以下 → そのまま使用
    if normal_left_accel < threshold and normal_right_accel < threshold:
        best_pattern_idx = 0
        min_accel = accelerations[0]
        corrected_points_3d = patterns_3d[0]
        status_reason = "Both sides OK"

    # 片方のみ閾値を超える → 欠損値として扱う
    elif (normal_left_accel >= threshold) != (normal_right_accel >= threshold):
        best_pattern_idx = -1
        min_accel = np.inf
        corrected_points_3d = np.full((25, 3), np.nan)
        status_reason = f"One side error (L:{normal_left_accel:.1f}, R:{normal_right_accel:.1f})"

    # 両方とも閾値を超える → スイッチング試行
    else:
        best_pattern_idx = -1
        min_accel = np.inf
        best_left_accel, best_right_accel = np.inf, np.inf

        # 全パターンで左右両方が閾値以下のものを探す
        for i, (left_acc, right_acc) in enumerate(left_right_accels):
            if left_acc < threshold and right_acc < threshold:
                if accelerations[i] < min_accel:
                    best_pattern_idx = i
                    min_accel = accelerations[i]
                    best_left_accel, best_right_accel = left_acc, right_acc

        if best_pattern_idx != -1:
            corrected_points_3d = patterns_3d[best_pattern_idx]
            status_reason = f"Switched to pattern {best_pattern_idx} (L:{best_left_accel:.1f}, R:{best_right_accel:.1f})"
        else:
            corrected_points_3d = np.full((25, 3), np.nan)
            status_reason = "All patterns failed"

    if frame_idx_for_debug is not None:
        accel_str = ", ".join([f"{a:8.2f}" if a != np.inf else "   inf" for a in accelerations])
        lr_str = ", ".join([f"L:{l:.1f}/R:{r:.1f}" if l != np.inf and r != np.inf else "L:inf/R:inf"
                           for l, r in left_right_accels])
        history_info = f"History: {len(history)} frames"
        if len(history) > 0:
            last_valid = not np.isnan(history[-1]).all()
            history_info += f", last valid: {last_valid}"
        print(f"Frame {frame_idx_for_debug:04d} | Accels(N,S1,S2,S12): [{accel_str}] | LR: [{lr_str}] | {status_reason} | {history_info}")

    raw_points_3d = patterns_3d[0]  # 常に元の組み合わせ（Normal）をrawとして記録

    return raw_points_3d, corrected_points_3d, accelerations, min_accel

def apply_spline_interpolation(sequence):
    """★★★ 新設: NaNを含む1次元の時系列データに3次スプライン補間を適用する関数 ★★★"""
    frames = np.arange(len(sequence))
    is_valid = ~np.isnan(sequence)

    # 補間には最低4つの有効な点が必要
    if np.sum(is_valid) < 4:
        return sequence # 補間不可能なのでそのまま返す

    spline = CubicSpline(frames[is_valid], sequence[is_valid], extrapolate=False)
    interpolated_sequence = spline(frames)

    # スプライン補間が届かない始点と終点のNaNを最近傍の値で埋める（パディング）
    first_valid_idx = np.where(is_valid)[0][0]
    last_valid_idx = np.where(is_valid)[0][-1]

    # 先頭のNaNを埋める
    interpolated_sequence[:first_valid_idx] = sequence[first_valid_idx]
    # 末尾のNaNを埋める
    interpolated_sequence[last_valid_idx+1:] = sequence[last_valid_idx]

    # それでも内部にNaNが残っていた場合（非常に稀）、線形補間で最終的に埋める
    is_still_nan = np.isnan(interpolated_sequence)
    if np.any(is_still_nan):
        interpolated_sequence[is_still_nan] = np.interp(frames[is_still_nan], frames[~is_still_nan], interpolated_sequence[~is_still_nan])

    return interpolated_sequence

# --- メイン処理 ---
def main():
    root_dir = Path(r"G:\gait_pattern\20250811_br")
    stereo_cali_dir = Path(r"G:\gait_pattern\stereo_cali\9g_20250811")

    ACCELERATION_THRESHOLD = 100.0
    BUTTERWORTH_CUTOFF = 12.0
    FRAME_RATE = 60
    DEBUG_ACCELERATION = True

    directions = ["fl", "fr"]

    print("3次スプライン補間による3D歩行解析を開始します。")
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
            if subject_dir.name != "sub1" or thera_dir.name != "thera0-2":
                continue

            print(f"\n{'='*80}\n処理開始: {thera_dir.relative_to(root_dir)}")

            openpose_dir1 = thera_dir / directions[0] / "openpose.json"
            openpose_dir2 = thera_dir / directions[1] / "openpose.json"
            if not (openpose_dir1.exists() and openpose_dir2.exists()): continue

            common_frames = sorted(list({f.name for f in openpose_dir1.glob("*_keypoints.json")} &
                                         {f.name for f in openpose_dir2.glob("*_keypoints.json")}))
            if not common_frames: continue

            # デバッグ用コードを追加して確認
            print(f"カメラ1のフレーム数: {len(list(openpose_dir1.glob('*_keypoints.json')))}")
            print(f"カメラ2のフレーム数: {len(list(openpose_dir2.glob('*_keypoints.json')))}")
            print(f"共通フレーム数: {len(common_frames)}")
            print(f"最初の5フレーム: {common_frames[:5]}")
            print(f"最後の5フレーム: {common_frames[-5:]}")

            output_dir = thera_dir / "3d_gait_analysis_spline_v1"
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"{thera_dir.name}_3d_results_spline.json"

            max_people = 0
            for frame_file in common_frames:
                kp1, _ = load_openpose_json(openpose_dir1 / frame_file)
                kp2, _ = load_openpose_json(openpose_dir2 / frame_file)
                max_people = max(max_people, len(kp1), len(kp2))

            all_raw_points = [[] for _ in range(max_people)]
            all_corrected_points = [[] for _ in range(max_people)]
            all_accelerations_data = [[] for _ in range(max_people)]
            history_list = [np.empty((0, 25, 3)) for _ in range(max_people)]

            print("  - ステップ1: 全フレームのエラー判定...")
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

                        # ★★★ 変更点: オンライン補間をせず、エラー判定後のデータをそのまま保存 ★★★
                        all_raw_points[p_idx].append(raw_3d)
                        all_corrected_points[p_idx].append(corrected_3d)

                        # ★★★ 修正点: 履歴の更新方法を改善 ★★★
                        # 有効なデータのみ履歴に追加し、履歴サイズを制限
                        if not np.isnan(corrected_3d).all():
                            history_list[p_idx] = np.vstack([history_list[p_idx], corrected_3d[np.newaxis, ...]])
                            # 履歴サイズを制限（最大10フレーム）
                            if len(history_list[p_idx]) > 10:
                                history_list[p_idx] = history_list[p_idx][-10:]
                        # NaNの場合は履歴に追加しない
                    else:
                        nan_points = np.full((25, 3), np.nan)
                        all_raw_points[p_idx].append(nan_points)
                        all_corrected_points[p_idx].append(nan_points)
                        # ★★★ 修正点: NaNデータは履歴に追加しない ★★★
                        # history_list[p_idx] = np.vstack([history_list[p_idx], nan_points[np.newaxis, ...]])
                        all_accelerations_data[p_idx].append({'all': [np.inf]*4, 'min': np.inf})

            all_person_results = []
            for p_idx in range(max_people):
                print(f"  - 人物{p_idx + 1}のデータ後処理中...")
                raw_points_arr = np.array(all_raw_points[p_idx])
                corrected_points_arr = np.array(all_corrected_points[p_idx])

                # ★★★ 新規追加: Rawデータに直接スプライン補間とフィルタを適用 ★★★
                print("    - ステップ2a: Rawデータに3次スプライン補間を適用...")
                raw_spline_points_arr = np.full_like(raw_points_arr, np.nan)
                for kp_idx in range(raw_points_arr.shape[1]):
                    for axis_idx in range(raw_points_arr.shape[2]):
                        sequence = raw_points_arr[:, kp_idx, axis_idx]
                        raw_spline_points_arr[:, kp_idx, axis_idx] = apply_spline_interpolation(sequence)

                print("    - ステップ2b: Rawスプライン補間データにバターワースフィルタを適用...")
                raw_final_points_arr = np.full_like(raw_spline_points_arr, np.nan)
                for kp_idx in range(raw_spline_points_arr.shape[1]):
                    for axis_idx in range(raw_spline_points_arr.shape[2]):
                        sequence = raw_spline_points_arr[:, kp_idx, axis_idx]
                        raw_final_points_arr[:, kp_idx, axis_idx] = butter_lowpass_filter(sequence, BUTTERWORTH_CUTOFF, FRAME_RATE)

                # ★★★ 既存処理: スイッチング後データのスプライン補間 ★★★
                print("    - ステップ3: スイッチング後データに3次スプライン補間で欠損値を補間...")
                spline_points_arr = np.full_like(corrected_points_arr, np.nan)
                for kp_idx in range(corrected_points_arr.shape[1]):
                    for axis_idx in range(corrected_points_arr.shape[2]):
                        sequence = corrected_points_arr[:, kp_idx, axis_idx]
                        spline_points_arr[:, kp_idx, axis_idx] = apply_spline_interpolation(sequence)

                print("    - ステップ4: バターワースフィルタを適用...")
                final_points_arr = np.full_like(spline_points_arr, np.nan)
                for kp_idx in range(spline_points_arr.shape[1]):
                    for axis_idx in range(spline_points_arr.shape[2]):
                        sequence = spline_points_arr[:, kp_idx, axis_idx]
                        final_points_arr[:, kp_idx, axis_idx] = butter_lowpass_filter(sequence, BUTTERWORTH_CUTOFF, FRAME_RATE)

                all_person_results.append({
                    'raw': raw_points_arr,
                    'raw_processed': raw_final_points_arr,  # ★★★ 新規追加 ★★★
                    'corrected_with_nan': corrected_points_arr,
                    'spline': spline_points_arr,
                    'final': final_points_arr
                })

            print("  - ステップ5: 結果をJSONファイルに保存...")
            analysis_results = []
            for t, frame_name in enumerate(common_frames):
                frame_result = {"frame_name": frame_name}
                for p_idx in range(max_people):
                    person_data = all_person_results[p_idx]
                    frame_result[f"person_{p_idx + 1}"] = {
                        "points_3d_raw": person_data['raw'][t].tolist(),
                        "points_3d_raw_processed": person_data['raw_processed'][t].tolist(),  # ★★★ 新規追加 ★★★
                        "points_3d_spline": person_data['spline'][t].tolist(),
                        "points_3d_final": person_data['final'][t].tolist(),
                    }
                analysis_results.append(frame_result)

            with open(output_file, 'w') as f:
                json.dump(analysis_results, f, indent=4)
            print(f"  ✓ 処理完了。結果を {output_file.relative_to(root_dir)} に保存しました。")

            # ★★★ フォルダ分けでグラフを整理 ★★★
            graphs_dir = output_dir / "graphs"
            graphs_dir.mkdir(exist_ok=True)

            bilateral_dir = graphs_dir / "bilateral_comparison"
            bilateral_dir.mkdir(exist_ok=True)

            method_comparison_dir = graphs_dir / "method_comparison"
            method_comparison_dir.mkdir(exist_ok=True)

            acceleration_dir = graphs_dir / "acceleration_analysis"
            acceleration_dir.mkdir(exist_ok=True)

            print("  - ステップ6: 主要関節ペアの軌道と加速度をグラフ化...")
            plot_bilateral_joint_analysis(all_person_results, all_accelerations_data, common_frames, bilateral_dir, thera_dir.name)

            print("  - ステップ7: Raw処理 vs スイッチング処理の比較グラフ化...")
            plot_method_comparison_analysis(all_person_results, common_frames, method_comparison_dir, thera_dir.name)

            print("  - ステップ8: 全フレームの加速度をグラフ化...")
            plot_acceleration_graph(all_accelerations_data, common_frames, ACCELERATION_THRESHOLD, acceleration_dir, thera_dir.name)


def plot_bilateral_joint_analysis(all_person_results, all_accelerations_data, frames, output_dir, file_prefix):
    """左右対応関節ペアの軌道と加速度をまとめて分析・プロット"""

    # 関節ペアの定義 (右, 左, 名前)
    joint_pairs = [
        (11, 14, "Ankle"),    # RAnkle, LAnkle
        (9, 12, "Hip"),       # RHip, LHip
        (10, 13, "Knee"),     # RKnee, LKnee
        (23, 20, "SmallToe"), # RSmallToe, LSmallToe
        (22, 19, "BigToe"),   # RBigToe, LBigToe
        (24, 21, "Heel")      # RHeel, LHeel
    ]

    try:
        max_people = len(all_person_results)
        if max_people == 0:
            return

        # 各関節ペアについて分析とプロット
        for right_idx, left_idx, joint_name in joint_pairs:
            print(f"    - {joint_name}の分析...")

            # 1. 位置軌道のプロット (X, Y, Z軸別)
            plot_bilateral_trajectory(all_person_results, frames, right_idx, left_idx, joint_name, output_dir, file_prefix)

            # 2. 加速度比較のプロット
            plot_bilateral_acceleration(all_person_results, frames, right_idx, left_idx, joint_name, output_dir, file_prefix)

    except Exception as e:
        print(f"  ✗ 両側関節分析エラー: {e}")
        traceback.print_exc()

def plot_method_comparison_analysis(all_person_results, frames, output_dir, file_prefix):
    """Raw処理 vs スイッチング処理の比較分析"""

    # 関節ペアの定義 (右, 左, 名前)
    joint_pairs = [
        (11, 14, "Ankle"),    # RAnkle, LAnkle
        (9, 12, "Hip"),       # RHip, LHip
        (10, 13, "Knee"),     # RKnee, LKnee
        (23, 20, "SmallToe"), # RSmallToe, LSmallToe
        (22, 19, "BigToe"),   # RBigToe, LBigToe
        (24, 21, "Heel")      # RHeel, LHeel
    ]

    try:
        max_people = len(all_person_results)
        if max_people == 0:
            return

        # 各関節ペアについて比較分析
        for right_idx, left_idx, joint_name in joint_pairs:
            print(f"    - {joint_name}の手法比較分析...")

            # 1. 位置軌道の手法比較（Raw処理 vs スイッチング処理）
            plot_method_trajectory_comparison(all_person_results, frames, right_idx, left_idx, joint_name, output_dir, file_prefix)

            # 2. 加速度の手法比較
            plot_method_acceleration_comparison(all_person_results, frames, right_idx, left_idx, joint_name, output_dir, file_prefix)

    except Exception as e:
        print(f"  ✗ 手法比較分析エラー: {e}")
        traceback.print_exc()

def plot_method_trajectory_comparison(all_person_results, frames, right_idx, left_idx, joint_name, output_dir, file_prefix):
    """Raw処理 vs スイッチング処理の軌道比較"""
    try:
        max_people = len(all_person_results)
        if max_people == 0: return

        fig, axes = plt.subplots(3, max_people, figsize=(14 * max_people, 16))
        if max_people == 1:
            axes = axes.reshape(-1, 1)

        time_axis = np.arange(len(frames))
        axis_names = ['X', 'Y', 'Z']

        for p_idx, person_data in enumerate(all_person_results):
            for axis_idx in range(3):
                ax = axes[axis_idx, p_idx]

                # ★★★ Raw処理結果（Raw → スプライン → フィルタ）★★★
                right_raw_processed = person_data['raw_processed'][:, right_idx, axis_idx]
                left_raw_processed = person_data['raw_processed'][:, left_idx, axis_idx]

                # ★★★ スイッチング処理結果（Raw → スイッチング → スプライン → フィルタ）★★★
                right_switching_processed = person_data['final'][:, right_idx, axis_idx]
                left_switching_processed = person_data['final'][:, left_idx, axis_idx]

                # プロット
                ax.plot(time_axis, right_raw_processed, '--', color='orange', linewidth=2, alpha=0.8, label=f'R{joint_name} Raw→Processed')
                ax.plot(time_axis, left_raw_processed, '--', color='cyan', linewidth=2, alpha=0.8, label=f'L{joint_name} Raw→Processed')
                ax.plot(time_axis, right_switching_processed, color='darkred', linewidth=2, label=f'R{joint_name} Switching→Processed')
                ax.plot(time_axis, left_switching_processed, color='darkblue', linewidth=2, label=f'L{joint_name} Switching→Processed')

                ax.set_title(f'Person {p_idx + 1}: {joint_name} {axis_names[axis_idx]}-axis\n(Raw Processing vs Switching Processing)')
                ax.set_xlabel('Frame Number')
                ax.set_ylabel(f'{axis_names[axis_idx]} Coordinate (mm)')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        graph_path = output_dir / f"{file_prefix}_{joint_name}_method_trajectory_comparison.png"
        plt.savefig(str(graph_path), dpi=200, bbox_inches='tight')
        # plt.show()
        plt.close()
        print(f"      ✓ {joint_name}手法軌道比較グラフ保存: {graph_path.name}")

    except Exception as e:
        print(f"      ✗ {joint_name}手法軌道比較エラー: {e}")

def plot_method_acceleration_comparison(all_person_results, frames, right_idx, left_idx, joint_name, output_dir, file_prefix):
    """Raw処理 vs スイッチング処理の加速度比較"""
    try:
        max_people = len(all_person_results)
        if max_people == 0: return

        fig, axes = plt.subplots(max_people, 1, figsize=(18, 8 * max_people))
        if max_people == 1:
            axes = [axes]

        time_axis = np.arange(len(frames))

        for p_idx, person_data in enumerate(all_person_results):
            ax = axes[p_idx]

            # ★★★ Raw処理結果の加速度計算 ★★★
            right_raw_processed_accel = calculate_joint_acceleration_series(person_data['raw_processed'][:, right_idx, :])
            left_raw_processed_accel = calculate_joint_acceleration_series(person_data['raw_processed'][:, left_idx, :])

            # ★★★ スイッチング処理結果の加速度計算 ★★★
            right_switching_accel = calculate_joint_acceleration_series(person_data['final'][:, right_idx, :])
            left_switching_accel = calculate_joint_acceleration_series(person_data['final'][:, left_idx, :])

            # プロット
            ax.plot(time_axis[2:], right_raw_processed_accel, '--', color='orange', linewidth=2, alpha=0.8, label=f'R{joint_name} Raw→Processed Accel')
            ax.plot(time_axis[2:], left_raw_processed_accel, '--', color='cyan', linewidth=2, alpha=0.8, label=f'L{joint_name} Raw→Processed Accel')
            ax.plot(time_axis[2:], right_switching_accel, color='darkred', linewidth=2, label=f'R{joint_name} Switching→Processed Accel')
            ax.plot(time_axis[2:], left_switching_accel, color='darkblue', linewidth=2, label=f'L{joint_name} Switching→Processed Accel')

            ax.set_title(f'Person {p_idx + 1}: {joint_name} Acceleration Method Comparison\n(Raw Processing vs Switching Processing)')
            ax.set_xlabel('Frame Number')
            ax.set_ylabel('Acceleration (mm/frame²)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        graph_path = output_dir / f"{file_prefix}_{joint_name}_method_acceleration_comparison.png"
        plt.savefig(str(graph_path), dpi=200, bbox_inches='tight')
        # plt.show()
        plt.close()
        print(f"      ✓ {joint_name}手法加速度比較グラフ保存: {graph_path.name}")

    except Exception as e:
        print(f"      ✗ {joint_name}手法加速度比較エラー: {e}")

def plot_bilateral_trajectory(all_person_results, frames, right_idx, left_idx, joint_name, output_dir, file_prefix):
    """左右関節の3軸軌道比較プロット（生データ vs 処理後データ）"""
    try:
        max_people = len(all_person_results)
        if max_people == 0: return

        fig, axes = plt.subplots(3, max_people, figsize=(12 * max_people, 14))
        if max_people == 1:
            axes = axes.reshape(-1, 1)

        time_axis = np.arange(len(frames))
        axis_names = ['X', 'Y', 'Z']

        for p_idx, person_data in enumerate(all_person_results):
            for axis_idx in range(3):
                ax = axes[axis_idx, p_idx]

                # ★★★ 生データ（Raw）★★★
                right_raw = person_data['raw'][:, right_idx, axis_idx]
                left_raw = person_data['raw'][:, left_idx, axis_idx]

                # ★★★ スイッチング処理後（Corrected）★★★
                right_corrected = person_data['corrected_with_nan'][:, right_idx, axis_idx]
                left_corrected = person_data['corrected_with_nan'][:, left_idx, axis_idx]

                # ★★★ 最終処理後（Final: スプライン+フィルタ）★★★
                right_final = person_data['final'][:, right_idx, axis_idx]
                left_final = person_data['final'][:, left_idx, axis_idx]

                # 右側関節のプロット
                ax.plot(time_axis, right_raw, 'o', color='lightcoral', markersize=2, alpha=0.6, label=f'R{joint_name} Raw')
                ax.plot(time_axis, right_corrected, 'x', color='red', markersize=3, alpha=0.7, label=f'R{joint_name} Corrected')
                ax.plot(time_axis, right_final, color='darkred', linewidth=2, label=f'R{joint_name} Final')

                # 左側関節のプロット
                ax.plot(time_axis, left_raw, 'o', color='lightblue', markersize=2, alpha=0.6, label=f'L{joint_name} Raw')
                ax.plot(time_axis, left_corrected, 'x', color='blue', markersize=3, alpha=0.7, label=f'L{joint_name} Corrected')
                ax.plot(time_axis, left_final, color='darkblue', linewidth=2, label=f'L{joint_name} Final')

                ax.set_title(f'Person {p_idx + 1}: {joint_name} {axis_names[axis_idx]}-axis (Raw vs Processed)')
                ax.set_xlabel('Frame Number')
                ax.set_ylabel(f'{axis_names[axis_idx]} Coordinate (mm)')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        graph_path = output_dir / f"{file_prefix}_{joint_name}_bilateral_comparison.png"
        plt.savefig(str(graph_path), dpi=200, bbox_inches='tight')
        # plt.show()
        plt.close()
        print(f"      ✓ {joint_name}両側比較グラフ保存: {graph_path.name}")

    except Exception as e:
        print(f"      ✗ {joint_name}軌道グラフエラー: {e}")

def plot_bilateral_acceleration(all_person_results, frames, right_idx, left_idx, joint_name, output_dir, file_prefix):
    """左右関節の加速度比較プロット（生データ vs 処理後データ）"""
    try:
        max_people = len(all_person_results)
        if max_people == 0: return

        fig, axes = plt.subplots(max_people, 1, figsize=(18, 8 * max_people))
        if max_people == 1:
            axes = [axes]

        time_axis = np.arange(len(frames))

        for p_idx, person_data in enumerate(all_person_results):
            ax = axes[p_idx]

            # ★★★ 生データの加速度計算 ★★★
            right_raw_accel = calculate_joint_acceleration_series(person_data['raw'][:, right_idx, :])
            left_raw_accel = calculate_joint_acceleration_series(person_data['raw'][:, left_idx, :])

            # ★★★ スイッチング処理後の加速度計算 ★★★
            right_corrected_accel = calculate_joint_acceleration_series(person_data['corrected_with_nan'][:, right_idx, :])
            left_corrected_accel = calculate_joint_acceleration_series(person_data['corrected_with_nan'][:, left_idx, :])

            # ★★★ 最終処理後の加速度計算 ★★★
            right_final_accel = calculate_joint_acceleration_series(person_data['final'][:, right_idx, :])
            left_final_accel = calculate_joint_acceleration_series(person_data['final'][:, left_idx, :])

            # 右側関節
            ax.plot(time_axis[2:], right_raw_accel, 'o', color='lightcoral', markersize=3, alpha=0.6, label=f'R{joint_name} Raw Accel')
            ax.plot(time_axis[2:], right_corrected_accel, '--', color='red', linewidth=1.5, alpha=0.8, label=f'R{joint_name} Corrected Accel')
            ax.plot(time_axis[2:], right_final_accel, color='darkred', linewidth=2, label=f'R{joint_name} Final Accel')

            # 左側関節
            ax.plot(time_axis[2:], left_raw_accel, 'o', color='lightblue', markersize=3, alpha=0.6, label=f'L{joint_name} Raw Accel')
            ax.plot(time_axis[2:], left_corrected_accel, '--', color='blue', linewidth=1.5, alpha=0.8, label=f'L{joint_name} Corrected Accel')
            ax.plot(time_axis[2:], left_final_accel, color='darkblue', linewidth=2, label=f'L{joint_name} Final Accel')

            ax.set_title(f'Person {p_idx + 1}: {joint_name} Bilateral Acceleration Comparison (Raw vs Processed)')
            ax.set_xlabel('Frame Number')
            ax.set_ylabel('Acceleration (mm/frame²)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        graph_path = output_dir / f"{file_prefix}_{joint_name}_bilateral_acceleration_comparison.png"
        plt.savefig(str(graph_path), dpi=200, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"      ✓ {joint_name}両側加速度比較グラフ保存: {graph_path.name}")

    except Exception as e:
        print(f"      ✗ {joint_name}加速度グラフエラー: {e}")

def calculate_joint_acceleration_series(joint_trajectory):
    """関節軌道から加速度時系列を計算"""
    if len(joint_trajectory) < 3:
        return np.array([])

    accelerations = []
    for i in range(2, len(joint_trajectory)):
        p_prev2 = joint_trajectory[i-2]
        p_prev1 = joint_trajectory[i-1]
        p_current = joint_trajectory[i]

        # 有効なポイントかチェック
        if np.isnan(p_prev2).any() or np.isnan(p_prev1).any() or np.isnan(p_current).any():
            accelerations.append(np.nan)
        else:
            accel = calculate_acceleration(p_prev2, p_prev1, p_current)
            accelerations.append(accel if accel != np.inf else np.nan)

    return np.array(accelerations)

def plot_joint_trajectory_graph(all_person_results, frames, joint_idx, axis_idx, joint_name, output_dir, file_prefix):
    """指定された関節の軌道をプロットし、グラフを保存する"""
    try:
        max_people = len(all_person_results)
        if max_people == 0: return

        plt.figure(figsize=(18, 6 * max_people))
        time_axis = np.arange(len(frames))

        for p_idx, person_data in enumerate(all_person_results):
            raw_y = person_data['raw'][:, joint_idx, axis_idx]
            corrected_y = person_data['corrected_with_nan'][:, joint_idx, axis_idx]
            final_y = person_data['final'][:, joint_idx, axis_idx]

            plt.subplot(max_people, 1, p_idx + 1)
            plt.plot(time_axis, raw_y, 'o', color='gray', markersize=2, alpha=0.5, label='Raw (Normal Triangulation)')
            plt.plot(time_axis, corrected_y, 'x', color='red', markersize=3, alpha=0.7, label='Corrected (Before Spline)')
            plt.plot(time_axis, final_y, color='blue', linewidth=2, label='Final (Spline + Butterworth)')

            plt.title(f'Person {p_idx + 1}: {joint_name} Trajectory')
            plt.xlabel('Frame Number')
            plt.ylabel(f'Coordinate Axis {axis_idx} (mm)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        graph_path = output_dir / f"{file_prefix}_{joint_name}_trajectory.png"
        plt.savefig(graph_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {joint_name}の軌道グラフを保存しました: {graph_path}")

    except Exception as e:
        print(f"  ✗ グラフ作成エラー: {e}")
        traceback.print_exc()

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

            # ★★★ 無限大値をNaNに変換してプロット ★★★
            accel_n_clean = np.where(np.isinf(accel_n), np.nan, accel_n)
            accel_s1_clean = np.where(np.isinf(accel_s1), np.nan, accel_s1)
            accel_s2_clean = np.where(np.isinf(accel_s2), np.nan, accel_s2)
            accel_s12_clean = np.where(np.isinf(accel_s12), np.nan, accel_s12)
            min_accels_clean = np.where(np.isinf(min_accels), np.nan, min_accels)

            plt.subplot(max_people, 1, p_idx + 1)
            plt.plot(time_axis, accel_n_clean, label='Normal (N)', alpha=0.6, color='blue')
            plt.plot(time_axis, accel_s1_clean, label='Swap Cam1 (S1)', alpha=0.4, linestyle=':', color='orange')
            plt.plot(time_axis, accel_s2_clean, label='Swap Cam2 (S2)', alpha=0.4, linestyle=':', color='green')
            plt.plot(time_axis, accel_s12_clean, label='Swap Both (S12)', alpha=0.4, linestyle=':', color='purple')

            plt.plot(time_axis, min_accels_clean, 'o', color='red', markersize=3, label='Selected Min Acceleration')

            plt.axhline(y=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')

            plt.title(f'Person {p_idx + 1}: Acceleration Analysis')
            plt.xlabel('Frame Number')
            plt.ylabel('Average Acceleration')
            plt.legend(loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.ylim(0, max(threshold * 3, 300))

            plt.xlim(0, len(frames) - 1)

        plt.tight_layout()
        # ★★★ 修正: ファイル名をより安全な形式に変更 ★★★
        safe_file_prefix = file_prefix.replace('-', '_').replace(':', '_')  # 無効な文字を置換
        graph_filename = f"{safe_file_prefix}_acceleration_analysis.png"
        graph_path = output_dir / graph_filename

        # ★★★ 修正: ディレクトリの存在確認と作成 ★★★
        output_dir.mkdir(parents=True, exist_ok=True)

        # ★★★ 修正: 絶対パスでの保存を試行 ★★★
        plt.savefig(str(graph_path), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 加速度の分析グラフを保存しました: {graph_path}")

    except Exception as e:
        print(f"  ✗ 加速度グラフ作成エラー: {e}")
        traceback.print_exc()

        # ★★★ 追加: エラー時のデバッグ情報 ★★★
        print(f"  デバッグ情報:")
        print(f"    - output_dir: {output_dir}")
        print(f"    - file_prefix: {file_prefix}")
        if output_dir.exists():
            print(f"    - output_dir exists: True")
        else:
            print(f"    - output_dir exists: False")
        if output_dir.is_dir():
            print(f"    - output_dir is_dir: True")
        else:
            print(f"    - output_dir is_dir: False")

if __name__ == '__main__':
    main()
