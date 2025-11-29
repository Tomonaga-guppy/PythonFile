import cv2
import numpy as np
import json
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt

# MATLABカルマンフィルタの完全再現版をインポート
from m_kalman_filter import kalman2_matlab_exact
# 重み付き三角化モジュールをインポート
from m_triangulation import triangulate_and_rotate

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

from scipy.interpolate import CubicSpline

def cubic_spline_interpolate_nan(sequence):
    """
    欠損値(NaN)を3次スプライン補間で埋める
    """
    sequence = np.array(sequence, dtype=float)
    n = len(sequence)
    x = np.arange(n)
    mask = ~np.isnan(sequence)
    if np.sum(mask) < 2:
        # 有効値が2点未満なら補間不可
        return sequence.copy()
    # 3次スプライン補間
    cs = CubicSpline(x[mask], sequence[mask])
    filled = sequence.copy()
    filled[~mask] = cs(x[~mask])
    return filled

def calculate_acceleration_per_axis(position_series):
    """
    Matlabのdiff()関数スタイルで各軸の加速度を計算
    position_series: 位置の時系列データ (N x 3) または (N,)
    返り値: 加速度の時系列 (N-2, 3) または (N-2,)
    """
    if len(position_series) < 3:
        return np.array([])

    # 1階差分（速度）
    velocity = np.diff(position_series, axis=0)

    # 2階差分（加速度）
    acceleration = np.diff(velocity, axis=0)

    # 各軸の値をそのまま返す
    return acceleration

def calculate_average_acceleration(history, current_points, eval_indices=[14]):
    """
    履歴不足対策を追加したMatlabスタイル加速度計算
    """
    if len(history) < 2:
        return 0.0

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

    if len(valid_history) < 2:
        return 0.0

    current_valid = True
    for idx in eval_indices:
        if np.isnan(current_points[idx]).any():
            current_valid = False
            break
    if not current_valid:
        return 0.0

    p_prev2, p_prev1 = valid_history[1], valid_history[0]
    total_accel, count = 0, 0
    for idx in eval_indices:
        position_series = np.array([p_prev2[idx], p_prev1[idx], current_points[idx]])
        accel_series = calculate_acceleration_per_axis(position_series)
        if len(accel_series) > 0 and not np.isnan(accel_series[0]).any():
            # 加速度の大きさを計算して合計
            total_accel += np.linalg.norm(accel_series[0])
            count += 1

    return total_accel / count if count > 0 else 0.0

def swap_left_right_keypoints(keypoints):
    """キーポイント配列の左右の部位を入れ替える"""
    swapped = keypoints.copy()
    l_indices = [5, 6, 7, 12, 13, 14, 16, 18, 19, 20, 21]
    r_indices = [2, 3, 4,  9, 10, 11, 15, 17, 22, 23, 24]
    swapped[l_indices + r_indices] = swapped[r_indices + l_indices]
    return swapped

def calculate_keypoint_specific_acceleration(history, current_points, keypoint_idx):
    """
    指定されたキーポイントの加速度を個別に計算

    Args:
        history: 履歴データ
        current_points: 現在のフレームの3Dポイント
        keypoint_idx: 評価対象のキーポイントインデックス

    Returns:
        float: そのキーポイントの加速度
    """
    return calculate_average_acceleration(history, current_points, eval_indices=[keypoint_idx])

def calculate_all_keypoints_acceleration(history, points_3d):
    """
    全キーポイントの加速度を個別に計算

    Args:
        history: 履歴データ
        points_3d: 現在のフレームの3Dポイント

    Returns:
        dict: {keypoint_idx: acceleration} の辞書
    """
    keypoint_accelerations = {}

    # 主要なキーポイント（左右対応）
    important_keypoints = [
        11, 14,  # RAnkle, LAnkle
        9, 12,   # RHip, LHip
        10, 13,  # RKnee, LKnee
        23, 20,  # RSmallToe, LSmallToe
        22, 19,  # RBigToe, LBigToe
        24, 21   # RHeel, LHeel
    ]

    for kp_idx in important_keypoints:
        keypoint_accelerations[kp_idx] = calculate_keypoint_specific_acceleration(history, points_3d, kp_idx)

    return keypoint_accelerations


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

# ★★★ 変更点: キーポイント・軸別のパラメータを受け取るように関数シグネチャを更新 ★★★
def process_single_person_individual_keypoint_switching_kalman_matlab(kp1, cf1, kp2, cf2, P1, P2, history, kalman_params_per_keypoint, kalman_states, default_thresholds, default_initial_values, frame_idx_for_debug=None):
    """
    MATLABカルマンフィルタの完全再現版
    ローカルレベルモデル + 準ニュートン法による散漫対数尤度最大化 + 左右足入れ替わり検出

    Args:
        kp1, cf1: カメラ1のキーポイント座標と信頼度
        kp2, cf2: カメラ2のキーポイント座標と信頼度
        P1, P2: 投影行列
        history: 履歴データ
        kalman_params_per_keypoint: キーポイント・軸別のパラメータ辞書
        kalman_states: 未使用（MATLABでは状態保持しない）
        default_thresholds: デフォルトの軸別閾値
        default_initial_values: デフォルトの軸別初期値
        frame_idx_for_debug: デバッグ用フレームインデックス

    Returns:
        raw_3d: 生の3D座標
        corrected_3d: MATLABカルマンフィルタで補正済み3D座標
        accelerations: 加速度データ
        min_acceleration: 最小加速度
    """
    # 最適化: 通常パターンの三角測量のみを最初に実行
    raw_3d = triangulate_and_rotate(P1, P2, kp1, kp2, cf1, cf2)

    # MATLABのkalman2アルゴリズムを適用
    corrected_3d = np.full((25, 3), np.nan)
    all_accelerations = []

    # 主要な左右ペアのキーポイント
    keypoint_pairs = [
        (11, 14),  # RAnkle, LAnkle
        (10, 13),  # RKnee, LKnee
        (9, 12),   # RHip, LHip
        (22, 19),  # RBigToe, LBigToe
        (23, 20),  # RSmallToe, LSmallToe
        (24, 21),  # RHeel, LHeel
    ]

    # 履歴に現在のフレームを追加
    if len(history) == 0:
        history_with_current = raw_3d[np.newaxis, ...]
    else:
        history_with_current = np.vstack([history, raw_3d[np.newaxis, ...]])

    # 履歴が十分にある場合のみMATLABカルマンフィルタを適用
    if len(history_with_current) >= 5:
        for right_kp, left_kp in keypoint_pairs:
            # ★★★ 変更点: キーポイントに対応するパラメータセットを取得 ★★★
            # 左右ペアでは同じパラメータを共有すると仮定し、右足のキーポイントで代表させる
            keypoint_params = kalman_params_per_keypoint.get(right_kp, {
                'thresholds': default_thresholds,
                'initial_values': default_initial_values
            })

            for axis in range(3):  # X, Y, Z軸
                try:
                    # ★★★ 変更点: 軸別の閾値と初期値を取得 ★★★
                    axis_name = ['x', 'y', 'z'][axis]
                    current_threshold = keypoint_params['thresholds'][axis_name]
                    current_initial_value = keypoint_params['initial_values'][axis_name]

                    # 左右の座標データを取得
                    left_data = history_with_current[:, left_kp, axis]
                    right_data = history_with_current[:, right_kp, axis]

                    # NaNでない部分のみ抽出
                    valid_mask = ~(np.isnan(left_data) | np.isnan(right_data))
                    if np.sum(valid_mask) >= 3:
                        valid_left = left_data[valid_mask]
                        valid_right = right_data[valid_mask]

                        # ★★★ 変更点: 軸別のパラメータを渡す ★★★
                        # MATLABのkalman2関数を適用（完全再現版）
                        corrected_left, corrected_right, miss_point = kalman2_matlab_exact(
                            valid_left, valid_right, th=current_threshold, initial_value=current_initial_value)

                        # 結果を元の位置に戻す
                        corrected_3d[left_kp, axis] = corrected_left[-1]  # 最新の値
                        corrected_3d[right_kp, axis] = corrected_right[-1]  # 最新の値

                        # 加速度計算（MATLABと同じ二階差分）
                        if len(corrected_left) >= 3:
                            dt_sq = (1.0/60.0)**2  # フレームレート60fpsを想定
                            accel_left = (corrected_left[-1] - 2*corrected_left[-2] + corrected_left[-3]) / dt_sq
                            accel_right = (corrected_right[-1] - 2*corrected_right[-2] + corrected_right[-3]) / dt_sq
                            all_accelerations.extend([abs(accel_left), abs(accel_right)])

                except Exception as e:
                    # エラーの場合は生データを使用（デバッグメッセージは制限）
                    if frame_idx_for_debug is not None and frame_idx_for_debug < 5:  # 最初の5フレームのみ表示
                        print(f"MATLABカルマンフィルタエラー (フレーム{frame_idx_for_debug}, KP{left_kp}-{right_kp}, 軸{axis}): {e}")
                    corrected_3d[left_kp, axis] = raw_3d[left_kp, axis]
                    corrected_3d[right_kp, axis] = raw_3d[right_kp, axis]
    else:
        # 履歴が不十分な場合は生データを使用（デバッグメッセージは制限）
        corrected_3d = raw_3d.copy()
        if frame_idx_for_debug is not None and frame_idx_for_debug < 5:  # 最初の5フレームのみ表示
            print(f"フレーム{frame_idx_for_debug}: 履歴不足のため生データを使用 (履歴数: {len(history_with_current)})")

    # その他のキーポイントは生データを使用
    other_keypoints = [i for i in range(25) if i not in [kp for pair in keypoint_pairs for kp in pair]]
    for kp in other_keypoints:
        corrected_3d[kp] = raw_3d[kp]

    # 加速度の最小値
    min_acceleration = min(all_accelerations) if all_accelerations else np.inf

    return raw_3d, corrected_3d, all_accelerations, min_acceleration




def plot_bilateral_joint_analysis(all_person_results, all_accelerations_data, frames, output_dir, file_prefix):
    """左右対応関節ペアの軌道と加速度をまとめて分析・プロット"""
    joint_pairs = [
        (11, 14, "Ankle"), (9, 12, "Hip"), (10, 13, "Knee"),
        (23, 20, "SmallToe"), (22, 19, "BigToe"), (24, 21, "Heel")
    ]
    try:
        if not all_person_results: return
        for right_idx, left_idx, joint_name in joint_pairs:
            print(f"    - {joint_name}の分析...")
            plot_bilateral_trajectory(all_person_results, frames, right_idx, left_idx, joint_name, output_dir, file_prefix)
            plot_bilateral_acceleration_per_axis(all_person_results, frames, right_idx, left_idx, joint_name, output_dir, file_prefix)
    except Exception as e:
        print(f"  ✗ 両側関節分析エラー: {e}")
        traceback.print_exc()

def plot_method_comparison_analysis(all_person_results, frames, output_dir, file_prefix):
    """Raw処理 vs スイッチング処理の比較分析"""
    joint_pairs = [
        (11, 14, "Ankle"), (9, 12, "Hip"), (10, 13, "Knee"),
        (23, 20, "SmallToe"), (22, 19, "BigToe"), (24, 21, "Heel")
    ]
    try:
        if not all_person_results: return
        for right_idx, left_idx, joint_name in joint_pairs:
            print(f"    - {joint_name}の手法比較分析...")
            plot_method_trajectory_comparison(all_person_results, frames, right_idx, left_idx, joint_name, output_dir, file_prefix)
            plot_method_acceleration_comparison_per_axis(all_person_results, frames, right_idx, left_idx, joint_name, output_dir, file_prefix)
    except Exception as e:
        print(f"  ✗ 手法比較分析エラー: {e}")
        traceback.print_exc()

def plot_method_trajectory_comparison(all_person_results, frames, right_idx, left_idx, joint_name, output_dir, file_prefix):
    """Raw処理 vs スイッチング処理の軌道比較"""
    try:
        max_people = len(all_person_results)
        if max_people == 0: return

        fig, axes = plt.subplots(3, max_people, figsize=(14 * max_people, 16), sharex=True)
        if max_people == 1:
            axes = axes.reshape(-1, 1)

        time_axis = np.arange(len(frames))
        axis_names = ['X', 'Y', 'Z']

        for p_idx, person_data in enumerate(all_person_results):
            for axis_idx in range(3):
                ax = axes[axis_idx, p_idx]
                right_raw_processed = person_data['raw_processed'][:, right_idx, axis_idx]
                left_raw_processed = person_data['raw_processed'][:, left_idx, axis_idx]
                right_switching_processed = person_data['final'][:, right_idx, axis_idx]
                left_switching_processed = person_data['final'][:, left_idx, axis_idx]

                ax.plot(time_axis, right_raw_processed, '--', color='orange', linewidth=2, alpha=0.8, label=f'R{joint_name} Raw→Processed')
                ax.plot(time_axis, left_raw_processed, '--', color='cyan', linewidth=2, alpha=0.8, label=f'L{joint_name} Raw→Processed')
                ax.plot(time_axis, right_switching_processed, color='darkred', linewidth=2, label=f'R{joint_name} Switching→Processed')
                ax.plot(time_axis, left_switching_processed, color='darkblue', linewidth=2, label=f'L{joint_name} Switching→Processed')

                ax.set_title(f'Person {p_idx + 1}: {joint_name} {axis_names[axis_idx]}-axis\n(Raw Processing vs Switching Processing)')
                ax.set_ylabel(f'{axis_names[axis_idx]} Coordinate (mm)')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, linestyle='--', alpha=0.6)
            axes[-1, p_idx].set_xlabel('Frame Number')

        plt.tight_layout()
        graph_path = output_dir / f"{file_prefix}_{joint_name}_method_trajectory_comparison.png"
        plt.savefig(str(graph_path), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"      ✓ {joint_name}手法軌道比較グラフ保存: {graph_path.name}")
    except Exception as e:
        print(f"      ✗ {joint_name}手法軌道比較エラー: {e}")

def plot_method_acceleration_comparison_per_axis(all_person_results, frames, right_idx, left_idx, joint_name, output_dir, file_prefix):
    """Raw処理 vs スイッチング処理の加速度比較（軸別）"""
    try:
        max_people = len(all_person_results)
        if max_people == 0: return

        fig, axes = plt.subplots(3, max_people, figsize=(14 * max_people, 16), sharex=True)
        if max_people == 1:
            axes = axes.reshape(-1, 1)

        time_axis = np.arange(len(frames))
        accel_time_axis = time_axis[2:]
        axis_names = ['X', 'Y', 'Z']

        for p_idx, person_data in enumerate(all_person_results):
            right_raw_accel_3d = calculate_joint_acceleration_series_per_axis(person_data['raw_processed'][:, right_idx, :])
            left_raw_accel_3d = calculate_joint_acceleration_series_per_axis(person_data['raw_processed'][:, left_idx, :])
            right_switching_accel_3d = calculate_joint_acceleration_series_per_axis(person_data['final'][:, right_idx, :])
            left_switching_accel_3d = calculate_joint_acceleration_series_per_axis(person_data['final'][:, left_idx, :])

            for axis_idx in range(3):
                ax = axes[axis_idx, p_idx]
                if len(accel_time_axis) == 0: continue

                ax.plot(accel_time_axis, right_raw_accel_3d[2:, axis_idx], '--', color='orange', alpha=0.8, label=f'R{joint_name} Raw Accel')
                ax.plot(accel_time_axis, left_raw_accel_3d[2:, axis_idx], '--', color='cyan', alpha=0.8, label=f'L{joint_name} Raw Accel')
                ax.plot(accel_time_axis, right_switching_accel_3d[2:, axis_idx], color='darkred', label=f'R{joint_name} Switching Accel')
                ax.plot(accel_time_axis, left_switching_accel_3d[2:, axis_idx], color='darkblue', label=f'L{joint_name} Switching Accel')

                ax.set_title(f'Person {p_idx + 1}: {joint_name} {axis_names[axis_idx]}-Axis Acceleration Comparison')
                ax.set_ylabel('Acceleration (mm/frame²)')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, linestyle='--', alpha=0.6)
            axes[-1, p_idx].set_xlabel('Frame Number')

        plt.tight_layout()
        graph_path = output_dir / f"{file_prefix}_{joint_name}_method_acceleration_comparison.png"
        plt.savefig(str(graph_path), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"      ✓ {joint_name}手法加速度比較グラフ保存: {graph_path.name}")
    except Exception as e:
        print(f"      ✗ {joint_name}手法加速度比較エラー: {e}")

def plot_bilateral_trajectory(all_person_results, frames, right_idx, left_idx, joint_name, output_dir, file_prefix):
    """左右関節の3軸軌道比較プロット（生データ vs 処理後データ）"""
    try:
        max_people = len(all_person_results)
        if max_people == 0: return

        fig, axes = plt.subplots(3, max_people, figsize=(12 * max_people, 14), sharex=True)
        if max_people == 1:
            axes = axes.reshape(-1, 1)

        time_axis = np.arange(len(frames))
        axis_names = ['X', 'Y', 'Z']

        for p_idx, person_data in enumerate(all_person_results):
            for axis_idx in range(3):
                ax = axes[axis_idx, p_idx]
                right_raw = person_data['raw'][:, right_idx, axis_idx]
                left_raw = person_data['raw'][:, left_idx, axis_idx]
                right_final = person_data['final'][:, right_idx, axis_idx]
                left_final = person_data['final'][:, left_idx, axis_idx]

                ax.plot(time_axis, right_raw, 'o', color='lightcoral', markersize=2, alpha=0.6, label=f'R{joint_name} Raw')
                ax.plot(time_axis, right_final, color='darkred', linewidth=2, label=f'R{joint_name} Final')
                ax.plot(time_axis, left_raw, 'o', color='lightblue', markersize=2, alpha=0.6, label=f'L{joint_name} Raw')
                ax.plot(time_axis, left_final, color='darkblue', linewidth=2, label=f'L{joint_name} Final')

                ax.set_title(f'Person {p_idx + 1}: {joint_name} {axis_names[axis_idx]}-axis (Raw vs Processed)')
                ax.set_ylabel(f'{axis_names[axis_idx]} Coordinate (mm)')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, linestyle='--', alpha=0.6)
            axes[-1, p_idx].set_xlabel('Frame Number')

        plt.tight_layout()
        graph_path = output_dir / f"{file_prefix}_{joint_name}_bilateral_comparison.png"
        plt.savefig(str(graph_path), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"      ✓ {joint_name}両側比較グラフ保存: {graph_path.name}")
    except Exception as e:
        print(f"      ✗ {joint_name}軌道グラフエラー: {e}")

def plot_bilateral_acceleration_per_axis(all_person_results, frames, right_idx, left_idx, joint_name, output_dir, file_prefix):
    """左右関節の加速度比較プロット（軸別、生データ vs 処理後データ）"""
    try:
        max_people = len(all_person_results)
        if max_people == 0: return

        fig, axes = plt.subplots(3, max_people, figsize=(14 * max_people, 16), sharex=True)
        if max_people == 1:
            axes = axes.reshape(-1, 1)

        time_axis = np.arange(len(frames))
        accel_time_axis = time_axis[2:]
        axis_names = ['X', 'Y', 'Z']

        for p_idx, person_data in enumerate(all_person_results):
            right_raw_accel_3d = calculate_joint_acceleration_series_per_axis(person_data['raw'][:, right_idx, :])
            left_raw_accel_3d = calculate_joint_acceleration_series_per_axis(person_data['raw'][:, left_idx, :])
            right_final_accel_3d = calculate_joint_acceleration_series_per_axis(person_data['final'][:, right_idx, :])
            left_final_accel_3d = calculate_joint_acceleration_series_per_axis(person_data['final'][:, left_idx, :])

            for axis_idx in range(3):
                ax = axes[axis_idx, p_idx]
                if len(accel_time_axis) == 0: continue

                ax.plot(accel_time_axis, right_raw_accel_3d[2:, axis_idx], 'o', color='lightcoral', markersize=2, alpha=0.6, label=f'R{joint_name} Raw Accel')
                ax.plot(accel_time_axis, right_final_accel_3d[2:, axis_idx], color='darkred', linewidth=2, label=f'R{joint_name} Final Accel')
                ax.plot(accel_time_axis, left_raw_accel_3d[2:, axis_idx], 'o', color='lightblue', markersize=2, alpha=0.6, label=f'L{joint_name} Raw Accel')
                ax.plot(accel_time_axis, left_final_accel_3d[2:, axis_idx], color='darkblue', linewidth=2, label=f'L{joint_name} Final Accel')

                ax.set_title(f'Person {p_idx + 1}: {joint_name} {axis_names[axis_idx]}-Axis Acceleration')
                ax.set_ylabel('Acceleration (mm/frame²)')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, linestyle='--', alpha=0.6)
            axes[-1, p_idx].set_xlabel('Frame Number')

        plt.tight_layout()
        graph_path = output_dir / f"{file_prefix}_{joint_name}_bilateral_acceleration_comparison.png"
        plt.savefig(str(graph_path), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"      ✓ {joint_name}両側加速度比較グラフ保存: {graph_path.name}")
    except Exception as e:
        print(f"      ✗ {joint_name}加速度グラフエラー: {e}")

def calculate_joint_acceleration_series_per_axis(joint_trajectory):
    """
    関節軌道からMatlabスタイルの各軸加速度時系列を計算
    joint_trajectory: (N, 3) の関節位置時系列
    返り値: (N, 3) の加速度時系列（先頭2フレームはNaN）
    """
    if len(joint_trajectory) < 3:
        return np.full_like(joint_trajectory, np.nan)

    # NaNを無視せずに計算すると、NaNが含まれるウィンドウの結果がNaNになる
    velocity = np.diff(joint_trajectory, axis=0)
    acceleration = np.diff(velocity, axis=0)

    # 元の配列と同じ長さにNaNでパディングして返す
    padded_acceleration = np.full_like(joint_trajectory, np.nan)
    padded_acceleration[2:] = acceleration
    return padded_acceleration


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
            min_accels = np.array([d['min'] if 'min' in d else np.nan for d in person_accel_data], dtype=float)
            min_accels_clean = np.where(np.isinf(min_accels), np.nan, min_accels)

            plt.subplot(max_people, 1, p_idx + 1)
            plt.plot(time_axis, min_accels_clean, 'o', color='red', markersize=3, label='Selected Min Acceleration Magnitude')
            plt.axhline(y=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')

            plt.title(f'Person {p_idx + 1}: Acceleration Magnitude Analysis')
            plt.xlabel('Frame Number')
            plt.ylabel('Average Acceleration Magnitude')
            plt.legend(loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.ylim(0, max(threshold * 3, 300))
            plt.xlim(0, len(frames) - 1)

        plt.tight_layout()
        safe_file_prefix = file_prefix.replace('-', '_').replace(':', '_')
        graph_filename = f"{safe_file_prefix}_acceleration_analysis.png"
        graph_path = output_dir / graph_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(graph_path), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 加速度の分析グラフを保存しました: {graph_path}")

    except Exception as e:
        print(f"  ✗ 加速度グラフ作成エラー: {e}")
        traceback.print_exc()


def main():
    root_dir = Path(r"G:\gait_pattern\20250811_br")
    stereo_cali_dir = Path(r"G:\gait_pattern\stereo_cali\9g_20250811")

    # ★★★ 変更点: キーポイントごと、かつ軸ごとにパラメータを設定 ★★★
    # デフォルト値
    DEFAULT_THRESHOLDS = {'x': 100.0, 'y': 100.0, 'z': 100.0}
    DEFAULT_INITIAL_VALUES = {'x': 0.0005, 'y': 0.0005, 'z': 0.005}

    # キーポイント別カスタムパラメータ (左右ペアで同じ値を設定)
    # MATLABの `kalman2` 呼び出しを参考に設定
    x_threshold = 100.0
    y_threshold = 100.0
    z_threshold = 100.0
    KALMAN_PARAMETERS_PER_KEYPOINT = {
        # Ankle (足首)
        11: {'thresholds': DEFAULT_THRESHOLDS, 'initial_values': DEFAULT_INITIAL_VALUES}, # RAnkle
        14: {'thresholds': DEFAULT_THRESHOLDS, 'initial_values': DEFAULT_INITIAL_VALUES}, # LAnkle
        # Knee (膝)
        10: {'thresholds': DEFAULT_THRESHOLDS, 'initial_values': DEFAULT_INITIAL_VALUES}, # RKnee
        13: {'thresholds': DEFAULT_THRESHOLDS, 'initial_values': DEFAULT_INITIAL_VALUES}, # LKnee
        # Hip (股関節)
        9:  {'thresholds': DEFAULT_THRESHOLDS, 'initial_values': DEFAULT_INITIAL_VALUES}, # RHip
        12: {'thresholds': DEFAULT_THRESHOLDS, 'initial_values': DEFAULT_INITIAL_VALUES}, # LHip
        # Toes (つま先)
        22: {'thresholds': DEFAULT_THRESHOLDS, 'initial_values': DEFAULT_INITIAL_VALUES}, # RBigToe
        19: {'thresholds': DEFAULT_THRESHOLDS, 'initial_values': DEFAULT_INITIAL_VALUES}, # LBigToe
        23: {'thresholds': DEFAULT_THRESHOLDS, 'initial_values': DEFAULT_INITIAL_VALUES}, # RSmallToe
        20: {'thresholds': DEFAULT_THRESHOLDS, 'initial_values': DEFAULT_INITIAL_VALUES}, # LSmallToe
        # Heel (かかと)
        24: {'thresholds': DEFAULT_THRESHOLDS, 'initial_values': DEFAULT_INITIAL_VALUES}, # RHeel
        21: {'thresholds': DEFAULT_THRESHOLDS, 'initial_values': DEFAULT_INITIAL_VALUES}, # LHeel
    }
    # KALMAN_PARAMETERS_PER_KEYPOINT = {
    #     # Ankle (足首)
    #     11: {'thresholds': {'x': 100.0, 'y': 500.0, 'z': 100.0}, 'initial_values': {'x': 0.0005, 'y': 0.003, 'z': 0.0005}}, # RAnkle
    #     14: {'thresholds': {'x': 100.0, 'y': 500.0, 'z': 100.0}, 'initial_values': {'x': 0.0005, 'y': 0.003, 'z': 0.0005}}, # LAnkle
    #     # Knee (膝)
    #     10: {'thresholds': {'x': 100.0, 'y': 500.0, 'z': 100.0}, 'initial_values': {'x': 0.008, 'y': 0.002, 'z': 0.008}}, # RKnee
    #     13: {'thresholds': {'x': 100.0, 'y': 500.0, 'z': 100.0}, 'initial_values': {'x': 0.008, 'y': 0.002, 'z': 0.008}}, # LKnee
    #     # Hip (股関節)
    #     9:  {'thresholds': {'x': 100.0, 'y': 500.0, 'z': 100.0}, 'initial_values': {'x': 0.005, 'y': 0.003, 'z': 0.005}}, # RHip
    #     12: {'thresholds': {'x': 100.0, 'y': 500.0, 'z': 100.0}, 'initial_values': {'x': 0.005, 'y': 0.003, 'z': 0.005}}, # LHip
    #     # Toes (つま先)
    #     22: {'thresholds': {'x': 100.0, 'y': 500.0, 'z': 100.0}, 'initial_values': {'x': 0.008, 'y': 0.003, 'z': 0.008}}, # RBigToe
    #     19: {'thresholds': {'x': 100.0, 'y': 500.0, 'z': 100.0}, 'initial_values': {'x': 0.008, 'y': 0.003, 'z': 0.008}}, # LBigToe
    #     23: {'thresholds': {'x': 100.0, 'y': 500.0, 'z': 100.0}, 'initial_values': {'x': 0.008, 'y': 0.003, 'z': 0.008}}, # RSmallToe
    #     20: {'thresholds': {'x': 100.0, 'y': 500.0, 'z': 100.0}, 'initial_values': {'x': 0.008, 'y': 0.003, 'z': 0.008}}, # LSmallToe
    #     # Heel (かかと)
    #     24: {'thresholds': {'x': 100.0, 'y': 500.0, 'z': 100.0}, 'initial_values': {'x': 0.08, 'y': 0.003, 'z': 0.08}}, # RHeel
    #     21: {'thresholds': {'x': 100.0, 'y': 500.0, 'z': 100.0}, 'initial_values': {'x': 0.08, 'y': 0.003, 'z': 0.08}}, # LHeel
    # }

    BUTTERWORTH_CUTOFF = 12.0
    FRAME_RATE = 60
    DEBUG_ACCELERATION = True

    print(f"★★★ MATLABカルマンフィルタの完全再現版 (キーポイント・軸別パラメータ対応) ★★★")
    print("    ローカルレベルモデル + 準ニュートン法による散漫対数尤度最大化")
    print("    左右足入れ替わり検出機能付き二階差分カルマンフィルタ")
    print("    MATLABのSecond_Order_Difference_Kalman_Filter.mの正確な実装")
    print(f"    デフォルト閾値: {DEFAULT_THRESHOLDS}")
    print(f"    デフォルト初期値: {DEFAULT_INITIAL_VALUES}")


    directions = ["fl", "fr"]

    print("カルマンフィルタベースの3D歩行解析を開始します。")
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

            print(f"共通フレーム数: {len(common_frames)}")

            output_dir = thera_dir / "3d_gait_analysis_kalman_v1"
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
            kalman_states_list = [{} for _ in range(max_people)]

            frame_start = 300
            frame_end = 350
            selected_frames = common_frames[frame_start:frame_end]
            # selected_frames = common_frames

            print("  - ステップ1: 全フレームのエラー判定...")
            for frame_idx, frame_name in enumerate(tqdm(selected_frames, desc="  フレーム処理中")):
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
                        # キーポイント・軸別のパラメータ辞書を渡す
                        raw_3d, corrected_3d, accels, min_accel = process_single_person_individual_keypoint_switching_kalman_matlab(
                            kp1, cf1, kp2, cf2, P1, P2, history_list[p_idx],
                            KALMAN_PARAMETERS_PER_KEYPOINT,
                            kalman_states_list[p_idx],
                            default_thresholds=DEFAULT_THRESHOLDS,
                            default_initial_values=DEFAULT_INITIAL_VALUES,
                            frame_idx_for_debug=debug_frame_idx)

                        all_accelerations_data[p_idx].append({'all': accels, 'min': min_accel})
                        all_raw_points[p_idx].append(raw_3d)
                        all_corrected_points[p_idx].append(corrected_3d)

                        if not np.isnan(corrected_3d).all():
                            history_list[p_idx] = np.vstack([history_list[p_idx], corrected_3d[np.newaxis, ...]])
                            if len(history_list[p_idx]) > 10:
                                history_list[p_idx] = history_list[p_idx][-10:]

            all_person_results = []
            for p_idx in range(max_people):
                print(f"  - 人物{p_idx + 1}のデータ後処理中...")
                raw_points_arr = np.array(all_raw_points[p_idx])
                corrected_points_arr = np.array(all_corrected_points[p_idx])

                print("    - ステップ2a: Rawデータに3次スプライン補間を適用...")
                raw_spline_points_arr = np.full_like(raw_points_arr, np.nan)
                for kp_idx in range(raw_points_arr.shape[1]):
                    for axis_idx in range(raw_points_arr.shape[2]):
                        sequence = raw_points_arr[:, kp_idx, axis_idx]
                        raw_spline_points_arr[:, kp_idx, axis_idx] = cubic_spline_interpolate_nan(sequence)

                print("    - ステップ2b: Rawスプライン補間データにバターワースフィルタを適用...")
                raw_final_points_arr = np.full_like(raw_spline_points_arr, np.nan)
                for kp_idx in range(raw_spline_points_arr.shape[1]):
                    for axis_idx in range(raw_spline_points_arr.shape[2]):
                        sequence = raw_spline_points_arr[:, kp_idx, axis_idx]
                        raw_final_points_arr[:, kp_idx, axis_idx] = butter_lowpass_filter(sequence, BUTTERWORTH_CUTOFF, FRAME_RATE)

                print("    - ステップ3: バターワースフィルタを適用...")
                final_points_arr = np.full_like(corrected_points_arr, np.nan)
                for kp_idx in range(corrected_points_arr.shape[1]):
                    for axis_idx in range(corrected_points_arr.shape[2]):
                        sequence = corrected_points_arr[:, kp_idx, axis_idx]
                        final_points_arr[:, kp_idx, axis_idx] = butter_lowpass_filter(sequence, BUTTERWORTH_CUTOFF, FRAME_RATE)

                all_person_results.append({
                    'raw': raw_points_arr,
                    'raw_processed': raw_final_points_arr,
                    'corrected_with_nan': corrected_points_arr,
                    'final': final_points_arr
                })

            print("  - ステップ5: 結果をJSONファイルに保存...")
            analysis_results = []
            for t, frame_name in enumerate(selected_frames):
                frame_result = {"frame_name": frame_name}
                for p_idx in range(max_people):
                    person_data = all_person_results[p_idx]
                    frame_result[f"person_{p_idx + 1}"] = {
                        "points_3d_raw": person_data['raw'][t].tolist(),
                        "points_3d_raw_processed": person_data['raw_processed'][t].tolist(),
                        "points_3d_final": person_data['final'][t].tolist(),
                    }
                analysis_results.append(frame_result)

            with open(output_file, 'w') as f:
                json.dump(analysis_results, f, indent=4)
            print(f"  ✓ 処理完了。結果を {output_file.relative_to(root_dir)} に保存しました。")

            graphs_dir = output_dir / "graphs"
            graphs_dir.mkdir(exist_ok=True)
            bilateral_dir = graphs_dir / "bilateral_comparison"
            bilateral_dir.mkdir(exist_ok=True)
            method_comparison_dir = graphs_dir / "method_comparison"
            method_comparison_dir.mkdir(exist_ok=True)
            acceleration_dir = graphs_dir / "acceleration_analysis"
            acceleration_dir.mkdir(exist_ok=True)

            print("  - ステップ6: 主要関節ペアの軌道と加速度をグラフ化...")
            plot_bilateral_joint_analysis(all_person_results, all_accelerations_data, selected_frames, bilateral_dir, thera_dir.name)

            print("  - ステップ7: Raw処理 vs スイッチング処理の比較グラフ化...")
            plot_method_comparison_analysis(all_person_results, selected_frames, method_comparison_dir, thera_dir.name)

            print("  - ステップ8: 全フレームの加速度をグラフ化...")
            # プロットには代表的な閾値（Y軸のデフォルト）を渡す
            plot_acceleration_graph(all_accelerations_data, selected_frames, DEFAULT_THRESHOLDS['y'], acceleration_dir, thera_dir.name)


if __name__ == '__main__':
    main()
