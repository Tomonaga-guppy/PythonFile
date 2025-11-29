import cv2
import numpy as np
import json
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# 数値計算の警告を抑制（オーバーフロー、無効値など）
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# MATLABカルマンフィルタの完全再現版をインポート
from try_1_kalman_matlab_exact import (
    kalman2_matlab_exact,
    double_difference_kalman_filter_matlab,
    apply_kalman_filter_individual_keypoints
)


"""
★★★ カルマンフィルタベースの異常値検出・補正システム ★★★
MATLABのSecond_Order_Difference_Kalman_Filter.mを参考に実装

特徴:
- ローカルトレンドモデル（速度ベース）
- 最尤推定によるノイズパラメータ推定
- 加速度閾値による異常値検出と補正
"""


def estimate_noise_parameters_ml(positions, indices, dt, max_iter=50, tol=1e-3):
    """
    ニュートン・ラフソン法を用いて拡散対数尤度を最大化し、
    σ²_u (プロセスノイズ分散) と σ²_w (観測ノイズ分散) を推定

    Args:
        positions: 有効な位置データ
        indices: 有効な位置データのインデックス
        dt: 時間間隔
        max_iter: 最大反復回数
        tol: 収束判定閾値

    Returns:
        sigma_u_sq: プロセスノイズ分散
        sigma_w_sq: 観測ノイズ分散
    """
    n = len(positions)
    if n < 3:
        # デフォルト値を返す
        return 1.0, 1.0

    # 時間ベクトル
    t = np.array(indices) * dt
    y = np.array(positions)

    # 初期推定値（データ分散ベース）
    data_var = np.var(y)
    sigma_u_sq = data_var * 0.01  # プロセスノイズ（小さめ）
    sigma_w_sq = data_var * 0.1   # 観測ノイズ

    # ニュートン・ラフソン法による最適化
    for iteration in range(max_iter):
        # 状態空間モデルの構築
        F = np.array([[1.0, dt], [0.0, 1.0]])  # 状態遷移行列
        H = np.array([[1.0, 0.0]])             # 観測行列

        # ノイズ共分散行列
        Q = np.array([[sigma_u_sq * dt**3 / 3, sigma_u_sq * dt**2 / 2],
                      [sigma_u_sq * dt**2 / 2, sigma_u_sq * dt]])
        R = sigma_w_sq

        # カルマンフィルタによる対数尤度計算
        try:
            log_likelihood, grad_u, grad_w = compute_log_likelihood_and_gradients(
                y, F, H, Q, R, sigma_u_sq, sigma_w_sq, dt)
        except:
            # 数値的問題が発生した場合は現在の値を返す
            break

        # ヘッシアン行列の近似（対角近似）
        hess_uu = -abs(grad_u) / max(sigma_u_sq, 1e-6)  # 安定化
        hess_ww = -abs(grad_w) / max(sigma_w_sq, 1e-6)  # 安定化

        # ニュートン・ラフソン更新
        # 学習率を導入して安定化
        learning_rate = 0.1
        delta_u = learning_rate * grad_u / max(abs(hess_uu), 1e-6)
        delta_w = learning_rate * grad_w / max(abs(hess_ww), 1e-6)

        # パラメータ更新（正値制約）
        sigma_u_sq_new = max(sigma_u_sq + delta_u, 1e-6)
        sigma_w_sq_new = max(sigma_w_sq + delta_w, 1e-6)

        # 収束判定
        if abs(sigma_u_sq_new - sigma_u_sq) < tol and abs(sigma_w_sq_new - sigma_w_sq) < tol:
            sigma_u_sq, sigma_w_sq = sigma_u_sq_new, sigma_w_sq_new
            break

        sigma_u_sq, sigma_w_sq = sigma_u_sq_new, sigma_w_sq_new

        # 発散防止
        if sigma_u_sq > 1e6 or sigma_w_sq > 1e6:
            break

    # 最終的な範囲制限
    sigma_u_sq = np.clip(sigma_u_sq, 0.1, 1000.0)
    sigma_w_sq = np.clip(sigma_w_sq, 0.1, 1000.0)

    return sigma_u_sq, sigma_w_sq


def compute_log_likelihood_and_gradients(y, F, H, Q, R, sigma_u_sq, sigma_w_sq, dt):
    """
    対数尤度とその勾配を計算
    """
    n = len(y)

    # 初期状態（最小二乗推定）
    if n >= 2:
        x0 = y[0]
        v0 = (y[1] - y[0]) / dt if n > 1 else 0.0
    else:
        x0, v0 = y[0], 0.0

    x = np.array([x0, v0])
    P = np.eye(2) * 100.0

    log_likelihood = 0.0
    grad_u = 0.0
    grad_w = 0.0

    for k in range(n):
        # 予測ステップ
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # 観測更新
        innovation = y[k] - H @ x_pred
        S = H @ P_pred @ H.T + R

        # 数値安定性チェック
        if S <= 1e-10:
            S = 1e-6

        # 対数尤度への寄与
        log_likelihood += -0.5 * (np.log(2 * np.pi * S) + innovation**2 / S)

        # 勾配計算（簡易近似）
        grad_u += -0.5 * (1.0/S - innovation**2 / (S**2)) * dt
        grad_w += -0.5 * (1.0/S - innovation**2 / (S**2))

        # カルマンゲインと状態更新
        if S > 1e-10:
            K = P_pred @ H.T / S
            x = x_pred + K * innovation
            P = P_pred - K @ H @ P_pred
        else:
            x = x_pred
            P = P_pred

    return log_likelihood, grad_u, grad_w


def double_difference_kalman_filter(position_data, threshold=100.0, plot_name="", frame_rate=30.0):
    """
    MATLABのSecond_Order_Difference_Kalman_Filter.mベースの実装（安定性改善版）

    ローカルトレンドモデル（速度ベース）カルマンフィルタによる異常値検出・補正

    Args:
        position_data: 1次元位置データ (NaN含む可能性あり)
        threshold: 加速度閾値 [mm/s²]
        plot_name: プロット名（デバッグ用）
        frame_rate: フレームレート [fps]

    Returns:
        corrected_data: 補正済み位置データ
        acceleration: 算出された加速度
        outlier_flags: 異常値フラグ
    """

    n = len(position_data)
    dt = 1.0 / frame_rate

    # 状態ベクトル [position, velocity]
    # 状態遷移行列 (Local trend model)
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])

    # 観測行列 (位置のみ観測)
    H = np.array([[1.0, 0.0]])

    # 初期値設定の改善
    first_valid_idx = None
    valid_positions = []
    valid_indices = []

    # 有効なデータを収集
    for i in range(n):
        if not np.isnan(position_data[i]):
            valid_positions.append(position_data[i])
            valid_indices.append(i)
            if first_valid_idx is None:
                first_valid_idx = i

    if len(valid_positions) < 2:
        # 有効なデータが少なすぎる場合
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, False)

    # 初期状態推定の改善（複数点から最小二乗法で推定）
    if len(valid_positions) >= 3:
        # 最初の3つの有効点から初期速度を推定
        t_vals = np.array(valid_indices[:3]) * dt
        pos_vals = np.array(valid_positions[:3])

        # 線形回帰で初期速度を推定
        A = np.vstack([t_vals, np.ones(len(t_vals))]).T
        slope, intercept = np.linalg.lstsq(A, pos_vals, rcond=None)[0]
        initial_velocity = slope
        initial_position = valid_positions[0]
    else:
        # 2点のみの場合
        initial_velocity = (valid_positions[1] - valid_positions[0]) / \
                         ((valid_indices[1] - valid_indices[0]) * dt)
        initial_position = valid_positions[0]

    x = np.array([initial_position, initial_velocity])

    # ニュートン・ラフソン法による最尤推定でノイズパラメータを決定
    sigma_u_sq, sigma_w_sq = estimate_noise_parameters_ml(valid_positions, valid_indices, dt)

    # プロセスノイズ共分散行列（最尤推定結果を使用）
    Q = np.array([[sigma_u_sq * dt**3 / 3, sigma_u_sq * dt**2 / 2],
                  [sigma_u_sq * dt**2 / 2, sigma_u_sq * dt]])

    # 観測ノイズ共分散
    R = np.array([[sigma_w_sq]])

    # 初期共分散行列（より小さく設定）
    P = np.eye(2) * 100.0

    # 結果格納用
    corrected_data = np.full(n, np.nan)
    velocities = np.full(n, np.nan)
    accelerations = np.full(n, np.nan)
    outlier_flags = np.full(n, False)

    # カルマンフィルタ処理（安定性改善）
    consecutive_nan_count = 0
    max_consecutive_nan = 10  # 連続NaN数制限

    for k in range(n):
        # 予測ステップ
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # 共分散行列の発散防止
        P_pred = np.clip(P_pred, -1e6, 1e6)

        # 観測がある場合の更新
        if not np.isnan(position_data[k]):
            consecutive_nan_count = 0  # NaNカウントリセット

            # 観測残差
            y = position_data[k] - H @ x_pred

            # 残差共分散
            S = H @ P_pred @ H.T + R

            # 加速度による異常値検出（より安定した方法）
            is_outlier = False
            if k >= 2:
                # 予測値ベースでの加速度計算（より安定）
                if not np.isnan(corrected_data[k-1]) and not np.isnan(corrected_data[k-2]):
                    predicted_acceleration = (x_pred[0] - 2*corrected_data[k-1] + corrected_data[k-2]) / (dt**2)

                    # 閾値判定（絶対値で判定）
                    if abs(predicted_acceleration) > threshold:
                        is_outlier = True

            if is_outlier:
                outlier_flags[k] = True
                # 異常値の場合は観測を使わず予測値を使用
                x = x_pred.flatten()  # 形状を確保
                P = P_pred
            else:
                # 正常値として更新
                if S[0, 0] > 1e-10:  # 数値安定性チェック
                    K = P_pred @ H.T / S[0, 0]
                    K = K.flatten()  # 形状を確保
                    x = x_pred.flatten() + K * float(y)  # スカラー乗算を確保
                    I_KH = np.eye(2) - np.outer(K, H.flatten())
                    P = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * float(R[0, 0])  # Joseph form for stability
                else:
                    x = x_pred.flatten()
                    P = P_pred
        else:
            # 観測がない場合
            consecutive_nan_count += 1

            # 連続NaNが続く場合は速度を減衰させる
            if consecutive_nan_count > max_consecutive_nan:
                x[1] *= 0.95  # 速度を減衰
                P *= 1.1  # 不確実性を増加

            x = x_pred
            P = P_pred

        # 共分散行列の安定性確保
        P = np.clip(P, -1e6, 1e6)
        eigenvals = np.linalg.eigvals(P)
        if np.any(eigenvals <= 0):
            P = np.eye(2) * 100.0  # リセット

        # 状態ベクトルの形状を確保して安全に処理
        x_array = np.asarray(x).flatten()
        if len(x_array) == 0:
            x_array = np.array([0.0, 0.0])
        elif len(x_array) == 1:
            x_array = np.array([float(x_array[0]), 0.0])
        elif len(x_array) >= 2:
            x_array = np.array([float(x_array[0]), float(x_array[1])])

        # 結果を保存（スカラー値を確保）
        corrected_data[k] = x_array[0]
        velocities[k] = x_array[1]

        # 加速度計算（安定化）
        if k >= 2 and not np.isnan(corrected_data[k-1]) and not np.isnan(corrected_data[k-2]):
            accelerations[k] = (corrected_data[k] - 2*corrected_data[k-1] + corrected_data[k-2]) / (dt**2)
            # 加速度の異常値をクリップ
            accelerations[k] = np.clip(accelerations[k], -threshold, threshold)

    return corrected_data, accelerations, outlier_flags
def estimate_noise_parameters_mle(position_data, F, H, dt, threshold):
    """
    最尤推定によるノイズパラメータの推定

    Args:
        position_data: 位置データ
        F: 状態遷移行列
        H: 観測行列
        dt: 時間刻み
        threshold: 加速度閾値

    Returns:
        process_noise_var: プロセスノイズ分散
        observation_noise_var: 観測ノイズ分散
    """

    # 有効なデータのみ抽出
    valid_data = position_data[~np.isnan(position_data)]

    if len(valid_data) < 10:
        # データが少ない場合はデフォルト値
        return 1000.0, 100.0

    def negative_log_likelihood(params):
        """負の対数尤度関数"""
        if len(params) == 1:
            process_var = params[0]
            obs_var = np.var(valid_data) * 0.1  # 観測ノイズは小さめに設定
        else:
            process_var, obs_var = params

        if process_var <= 0 or obs_var <= 0:
            return 1e10

        # 簡易的な尤度計算
        # 実際のMATLABコードではより詳細な実装が必要

        # 差分の分散を計算
        diffs = np.diff(valid_data)
        if len(diffs) > 1:
            diff_var = np.var(diffs)
            # プロセスノイズとの適合度
            process_fit = abs(diff_var - process_var * dt**2)

            # 観測ノイズとの適合度
            obs_fit = abs(np.var(valid_data) - obs_var)

            return process_fit + obs_fit
        else:
            return 1e10

    # 最適化
    try:
        # プロセスノイズのみ最適化（簡易版）
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(lambda x: negative_log_likelihood([x]),
                               bounds=(1.0, 10000.0), method='bounded')

        process_noise_var = result.x
        observation_noise_var = np.var(valid_data) * 0.1

    except:
        # 最適化に失敗した場合はデフォルト値
        process_noise_var = 1000.0
        observation_noise_var = 100.0

    return process_noise_var, observation_noise_var


def calculate_joint_acceleration_kalman(sequence, threshold=100.0, plot_name="", frame_rate=30.0):
    """
    カルマンフィルタベースの加速度計算

    Args:
        sequence: 3D座標シーケンス [frames, 3]
        threshold: 加速度閾値
        plot_name: プロット名
        frame_rate: フレームレート

    Returns:
        corrected_sequence: 補正済み座標
        accelerations: 加速度 [frames, 3]
        outlier_flags: 異常値フラグ [frames, 3]
    """

    frames, dims = sequence.shape
    corrected_sequence = np.full_like(sequence, np.nan)
    accelerations = np.full_like(sequence, np.nan)
    outlier_flags = np.full((frames, dims), False)

    # 各次元に対してカルマンフィルタを適用
    for dim in range(dims):
        corrected_sequence[:, dim], accelerations[:, dim], outlier_flags[:, dim] = \
            double_difference_kalman_filter(
                sequence[:, dim],
                threshold=threshold,
                plot_name=f"{plot_name}_dim{dim}",
                frame_rate=frame_rate
            )

    return corrected_sequence, accelerations, outlier_flags



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

def calculate_acceleration_matlab_style(position_series):
    """
    Matlabのdiff()関数スタイルで加速度を計算
    position_series: 位置の時系列データ (N x 3) または (N,)
    返り値: 加速度の大きさの時系列 (N-2,)
    """
    if len(position_series) < 3:
        return np.array([])

    # 1階差分（速度）
    velocity = np.diff(position_series, axis=0)

    # 2階差分（加速度）
    acceleration = np.diff(velocity, axis=0)

    # 各フレームでの加速度の大きさを計算
    if acceleration.ndim == 2:  # 3D座標の場合
        acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
    else:  # 1D座標の場合
        acceleration_magnitude = np.abs(acceleration)

    return acceleration_magnitude

def calculate_acceleration(p_prev2, p_prev1, p_curr):
    """3点間の二階差分で加速度の大きさを計算（後方互換性のため残す）"""
    if np.isnan(p_prev2).any() or np.isnan(p_prev1).any() or np.isnan(p_curr).any():
        return np.inf
    v1 = p_prev1 - p_prev2
    v2 = p_curr - p_prev1
    acceleration_vec = v2 - v1
    return np.linalg.norm(acceleration_vec)

def calculate_average_acceleration(history, current_points, eval_indices=[14]):
    """
    履歴不足対策を追加したMatlabスタイル加速度計算
    """
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

    # ★★★ Matlabスタイル加速度計算 ★★★
    # 3点の位置データを準備
    p_prev2, p_prev1 = valid_history[1], valid_history[0]

    total_accel, count = 0, 0
    for idx in eval_indices:
        # 3点の位置データを時系列として準備
        position_series = np.array([p_prev2[idx], p_prev1[idx], current_points[idx]])

        # Matlabスタイルで加速度計算
        accel_series = calculate_acceleration_matlab_style(position_series)

        if len(accel_series) > 0 and not np.isnan(accel_series[0]):
            total_accel += accel_series[0]  # 最初（唯一）の要素
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

def check_keypoint_switching_needed(keypoint_accelerations, threshold):
    """
    各キーポイントの加速度に基づいてスイッチングが必要かチェック

    Args:
        keypoint_accelerations: {keypoint_idx: acceleration} の辞書
        threshold: 閾値

    Returns:
        tuple: (switching_needed, failed_keypoints, summary)
    """
    failed_keypoints = []
    for kp_idx, accel in keypoint_accelerations.items():
        if accel >= threshold:
            failed_keypoints.append(kp_idx)

    switching_needed = len(failed_keypoints) > 0

    # キーポイント名のマッピング
    keypoint_names = {
        11: "RAnkle", 14: "LAnkle",
        9: "RHip", 12: "LHip",
        10: "RKnee", 13: "LKnee",
        23: "RSmallToe", 20: "LSmallToe",
        22: "RBigToe", 19: "LBigToe",
        24: "RHeel", 21: "LHeel"
    }

    failed_names = [keypoint_names.get(kp, f"KP{kp}") for kp in failed_keypoints]
    summary = f"Failed: {failed_names}" if failed_names else "All OK"

    return switching_needed, failed_keypoints, summary

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

def process_single_person_individual_keypoint_switching_kalman_matlab(kp1, cf1, kp2, cf2, P1, P2, history, threshold, kalman_states, frame_idx_for_debug=None):
    """
    MATLABカルマンフィルタの完全再現版
    ローカルレベルモデル + 準ニュートン法による散漫対数尤度最大化 + 左右足入れ替わり検出

    Args:
        kp1, cf1: カメラ1のキーポイント座標と信頼度
        kp2, cf2: カメラ2のキーポイント座標と信頼度
        P1, P2: 投影行列
        history: 履歴データ
        threshold: 加速度閾値
        kalman_states: 未使用（MATLABでは状態保持しない）
        frame_idx_for_debug: デバッグ用フレームインデックス

    Returns:
        raw_3d: 生の3D座標
        corrected_3d: MATLABカルマンフィルタで補正済み3D座標
        accelerations: 加速度データ
        min_acceleration: 最小加速度
    """

    # 4つのパターンで重み付き三角測量を実行
    patterns_3d = []
    keypoint_accelerations = []

    kp_combinations = [
        (kp1, cf1, kp2, cf2),  # Normal (N)
        (swap_left_right_keypoints(kp1), swap_left_right_keypoints(cf1), kp2, cf2),  # Swap Cam1 (S1)
        (kp1, cf1, swap_left_right_keypoints(kp2), swap_left_right_keypoints(cf2)),  # Swap Cam2 (S2)
        (swap_left_right_keypoints(kp1), swap_left_right_keypoints(cf1),
         swap_left_right_keypoints(kp2), swap_left_right_keypoints(cf2))  # Swap Both (S12)
    ]

    # 全パターンで重み付き三角測量を実行
    for kp1_trial, cf1_trial, kp2_trial, cf2_trial in kp_combinations:
        # 重み付き三角測量を使用
        points_3d = triangulate_and_rotate(P1, P2, kp1_trial, kp2_trial, cf1_trial, cf2_trial)
        keypoint_accels = calculate_all_keypoints_acceleration(history, points_3d)

        patterns_3d.append(points_3d)
        keypoint_accelerations.append(keypoint_accels)

    # 生の3D座標として最初のパターンを使用
    raw_3d = patterns_3d[0].copy()

    # MATLABのkalman2アルゴリズムを適用
    corrected_3d = np.full((25, 3), np.nan)
    all_accelerations = []

    # 主要な左右ペアのキーポイント
    keypoint_pairs = [
        (11, 14),  # RAnkle, LAnkle
        (10, 13),  # RKnee, LKnee
        (9, 12),   # RHip, LHip
        (22, 19),  # RBigToe, LBigToe
        (24, 21),  # RHeel, LHeel
        (23, 20)   # RSmallToe, LSmallToe
    ]

    # 履歴に現在のフレームを追加
    if len(history) == 0:
        history_with_current = raw_3d[np.newaxis, ...]
    else:
        history_with_current = np.vstack([history, raw_3d[np.newaxis, ...]])

    # 履歴が十分にある場合のみMATLABカルマンフィルタを適用
    if len(history_with_current) >= 5:
        for right_kp, left_kp in keypoint_pairs:
            for axis in range(3):  # X, Y, Z軸
                try:
                    # 左右の座標データを取得
                    left_data = history_with_current[:, left_kp, axis]
                    right_data = history_with_current[:, right_kp, axis]

                    # NaNでない部分のみ抽出
                    valid_mask = ~(np.isnan(left_data) | np.isnan(right_data))
                    if np.sum(valid_mask) >= 3:
                        valid_left = left_data[valid_mask]
                        valid_right = right_data[valid_mask]

                        # MATLABのkalman2関数を適用（完全再現版）
                        corrected_left, corrected_right, miss_point = kalman2_matlab_exact(
                            valid_left, valid_right, th=threshold, initial_value=0.0005)

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


# def process_single_person_individual_keypoint_switching_kalman(kp1, cf1, kp2, cf2, P1, P2, history, threshold, kalman_states, frame_idx_for_debug=None):
#     """
#     ★★★ カルマンフィルタ統合版: 各キーポイントが独立して最適なパターンを選択し、必要時にカルマンフィルタ補間 ★★★
#     各キーポイント（膝、足首、つま先など）がそれぞれの加速度で個別に判定・スイッチング
#     すべてのパターンで閾値を超える場合はカルマンフィルタによる補間を適用
#     """
#     # 4つのパターンをすべて計算
#     patterns_3d = []
#     keypoint_accelerations = []

#     kp_combinations = [
#         (kp1, cf1, kp2, cf2),  # Normal (N)
#         (swap_left_right_keypoints(kp1), swap_left_right_keypoints(cf1), kp2, cf2),  # Swap Cam1 (S1)
#         (kp1, cf1, swap_left_right_keypoints(kp2), swap_left_right_keypoints(cf2)),  # Swap Cam2 (S2)
#         (swap_left_right_keypoints(kp1), swap_left_right_keypoints(cf1),
#          swap_left_right_keypoints(kp2), swap_left_right_keypoints(cf2))  # Swap Both (S12)
#     ]

#     # 全パターンの3D座標と各キーポイント加速度を計算
#     for kp1_trial, cf1_trial, kp2_trial, cf2_trial in kp_combinations:
#         points_3d = triangulate_and_rotate(P1, P2, kp1_trial, kp2_trial, cf1_trial, cf2_trial)
#         keypoint_accels = calculate_all_keypoints_acceleration(history, points_3d)

#         patterns_3d.append(points_3d)
#         keypoint_accelerations.append(keypoint_accels)

#     # ★★★ 各キーポイントごとに最適パターンを選択またはカルマンフィルタ適用 ★★★
#     normal_points_3d = patterns_3d[0].copy()  # ベースはNormalパターン
#     selective_points_3d = normal_points_3d.copy()
#     switching_log = {}
#     kalman_applied = {}

#     # 主要キーポイントリスト
#     important_keypoints = [11, 14, 9, 12, 10, 13, 23, 20, 22, 19, 24, 21]

#     for kp_idx in important_keypoints:
#         # Normalパターンでの加速度をチェック
#         normal_accel = keypoint_accelerations[0].get(kp_idx, np.inf)

#         if normal_accel >= threshold:
#             # 閾値を超えている場合、他のパターンから閾値以下のものを探す
#             best_pattern_idx = -1
#             best_accel = np.inf

#             for pattern_idx in range(1, 4):  # S1, S2, S12パターンを確認
#                 pattern_accel = keypoint_accelerations[pattern_idx].get(kp_idx, np.inf)
#                 # ★★★ 重要: 閾値以下かつより良い加速度の場合のみ選択 ★★★
#                 if pattern_accel < threshold and pattern_accel < best_accel:
#                     best_pattern_idx = pattern_idx
#                     best_accel = pattern_accel

#             # 閾値以下のパターンが見つかった場合のみ置き換え
#             if best_pattern_idx != -1:
#                 selective_points_3d[kp_idx] = patterns_3d[best_pattern_idx][kp_idx]
#                 switching_log[kp_idx] = {
#                     'from_pattern': 0,
#                     'to_pattern': best_pattern_idx,
#                     'from_accel': normal_accel,
#                     'to_accel': best_accel,
#                     'kalman_applied': False
#                 }
#             else:
#                 # ★★★ すべてのパターンで閾値を超える場合はカルマンフィルタ適用（安定性改善版） ★★★
#                 kalman_applied[kp_idx] = True

#                 # カルマンフィルタ状態の初期化（初回のみ）
#                 if kp_idx not in kalman_states:
#                     kalman_states[kp_idx] = {
#                         'x': np.zeros(2),  # [position, velocity]
#                         'P': np.eye(2) * 100.0,  # 共分散行列（小さく初期化）
#                         'history': [],  # 位置履歴
#                         'initialized': False
#                     }

#                 # 現在の観測値を履歴に追加（安定性のため有効な値のみ）
#                 current_pos = normal_points_3d[kp_idx]
#                 if not np.any(np.isnan(current_pos)):
#                     kalman_states[kp_idx]['history'].append(current_pos)

#                     # 履歴サイズを制限（最大20フレーム）
#                     if len(kalman_states[kp_idx]['history']) > 20:
#                         kalman_states[kp_idx]['history'] = kalman_states[kp_idx]['history'][-20:]

#                 # カルマンフィルタで補正された位置を計算（安定性改善）
#                 if len(kalman_states[kp_idx]['history']) >= 5:  # より多くの履歴を要求
#                     # 各軸に対してカルマンフィルタを適用
#                     corrected_pos = np.full(3, np.nan)
#                     history_array = np.array(kalman_states[kp_idx]['history'])

#                     for axis in range(3):
#                         axis_data = history_array[:, axis]
#                         if not np.all(np.isnan(axis_data)) and len(axis_data) >= 5:
#                             try:
#                                 # より保守的な閾値でカルマンフィルタを適用
#                                 conservative_threshold = threshold * 2  # 閾値を2倍に緩和
#                                 corrected_axis, _, _ = double_difference_kalman_filter(
#                                     axis_data, threshold=conservative_threshold,
#                                     plot_name=f"KP{kp_idx}_axis{axis}"
#                                 )
#                                 if not np.isnan(corrected_axis[-1]):
#                                     corrected_pos[axis] = corrected_axis[-1]  # 最新の補正値
#                                 else:
#                                     # カルマンフィルタが失敗した場合は履歴の平均値を使用
#                                     valid_history = axis_data[~np.isnan(axis_data)]
#                                     if len(valid_history) > 0:
#                                         corrected_pos[axis] = np.mean(valid_history[-3:])  # 最新3点の平均
#                             except Exception as e:
#                                 # カルマンフィルタでエラーが発生した場合の fallback
#                                 valid_history = axis_data[~np.isnan(axis_data)]
#                                 if len(valid_history) > 0:
#                                     corrected_pos[axis] = valid_history[-1]  # 最新の有効値

#                     # 補正結果が有効かチェック
#                     if not np.all(np.isnan(corrected_pos)):
#                         selective_points_3d[kp_idx] = corrected_pos
#                     else:
#                         # カルマンフィルタが完全に失敗した場合は履歴の最新値を使用
#                         if len(kalman_states[kp_idx]['history']) > 0:
#                             selective_points_3d[kp_idx] = kalman_states[kp_idx]['history'][-1]
#                         else:
#                             selective_points_3d[kp_idx] = current_pos
#                 else:
#                     # 履歴が不足している場合は元の値を使用
#                     selective_points_3d[kp_idx] = current_pos

#                 switching_log[kp_idx] = {
#                     'from_pattern': 0,
#                     'to_pattern': -2,  # -2はカルマンフィルタ適用を意味
#                     'from_accel': normal_accel,
#                     'to_accel': np.nan,
#                     'kalman_applied': True,
#                     'history_length': len(kalman_states[kp_idx]['history'])
#                 }    # 全パターンの平均加速度計算（デバッグ用）
#     pattern_avg_accels = []
#     for kp_accels in keypoint_accelerations:
#         avg_accel = np.mean([acc for acc in kp_accels.values() if acc != np.inf])
#         pattern_avg_accels.append(avg_accel if not np.isnan(avg_accel) else np.inf)

#     # 最終的な各キーポイント加速度を計算
#     final_keypoint_accels = calculate_all_keypoints_acceleration(history, selective_points_3d)

#     # ステータス作成
#     num_switched = len([log for log in switching_log.values() if log['to_pattern'] not in [-1, -2]])
#     num_kalman = len([log for log in switching_log.values() if log['to_pattern'] == -2])
#     num_nan_set = len([log for log in switching_log.values() if log['to_pattern'] == -1])

#     if len(switching_log) == 0:
#         status_reason = "All keypoints OK (Normal)"
#     else:
#         switched_kps = [kp for kp, log in switching_log.items() if log['to_pattern'] not in [-1, -2]]
#         kalman_kps = [kp for kp, log in switching_log.items() if log['to_pattern'] == -2]
#         nan_set_kps = [kp for kp, log in switching_log.items() if log['to_pattern'] == -1]

#         keypoint_names = {11: "RAnkle", 14: "LAnkle", 9: "RHip", 12: "LHip",
#                          10: "RKnee", 13: "LKnee", 23: "RSmallToe", 20: "LSmallToe",
#                          22: "RBigToe", 19: "LBigToe", 24: "RHeel", 21: "LHeel"}

#         status_parts = []
#         if switched_kps:
#             switched_names = [keypoint_names.get(kp, f"KP{kp}") for kp in switched_kps]
#             status_parts.append(f"Switched: {switched_names}")
#         if kalman_kps:
#             kalman_names = [keypoint_names.get(kp, f"KP{kp}") for kp in kalman_kps]
#             status_parts.append(f"Kalman: {kalman_names}")
#         if nan_set_kps:
#             nan_names = [keypoint_names.get(kp, f"KP{kp}") for kp in nan_set_kps]
#             status_parts.append(f"NaN_set: {nan_names}")

#         status_reason = " | ".join(status_parts)

#     # デバッグ出力
#     if frame_idx_for_debug is not None:
#         accel_str = ", ".join([f"{a:8.2f}" if a != np.inf else "   inf" for a in pattern_avg_accels])

#         # Normalパターンの加速度表示
#         normal_kp_accels = keypoint_accelerations[0]
#         kp_items = [f"{kp}:{acc:.1f}" if acc != np.inf else f"{kp}:inf"
#                    for kp, acc in normal_kp_accels.items()]
#         kp_str = f"Normal_KP_Accels: [{', '.join(kp_items)}]"

#         # 最終的な加速度表示
#         final_kp_items = [f"{kp}:{acc:.1f}" if acc != np.inf else f"{kp}:inf"
#                          for kp, acc in final_keypoint_accels.items()]
#         final_kp_str = f"Final_KP_Accels: [{', '.join(final_kp_items)}]"

#         # スイッチング詳細
#         switch_str = ""
#         if switching_log:
#             switch_details = []
#             for kp, info in switching_log.items():
#                 if info['to_pattern'] == -1:
#                     switch_details.append(f"KP{kp}:P{info['from_pattern']}→NaN({info['from_accel']:.1f})")
#                 elif info['to_pattern'] == -2:
#                     hist_len = info.get('history_length', 0)
#                     switch_details.append(f"KP{kp}:P{info['from_pattern']}→Kalman({info['from_accel']:.1f},H:{hist_len})")
#                 else:
#                     switch_details.append(f"KP{kp}:P{info['from_pattern']}→P{info['to_pattern']}({info['from_accel']:.1f}→{info['to_accel']:.1f})")
#             switch_str = f"Switches: [{', '.join(switch_details)}]"

#         history_info = f"History: {len(history)} frames"
#         if len(history) > 0:
#             last_valid = not np.isnan(history[-1]).all()
#             history_info += f", last valid: {last_valid}"

#         # ★★★ デバッグフレーム範囲を制限（200-400フレームのみ表示） ★★★
#         if 200 <= frame_idx_for_debug <= 400:
#             print(f"Frame {frame_idx_for_debug:04d} | AvgAccels(N,S1,S2,S12): [{accel_str}]")
#             print(f"                      | {kp_str}")
#             print(f"                      | {final_kp_str}")
#             if switch_str:
#                 print(f"                      | {switch_str}")
#             print(f"                      | {status_reason} | {history_info}")

#     return normal_points_3d, selective_points_3d, pattern_avg_accels, pattern_avg_accels[0]


def main():
    root_dir = Path(r"G:\gait_pattern\20250811_br")
    stereo_cali_dir = Path(r"G:\gait_pattern\stereo_cali\9g_20250811")

    # ★★★ 個別キーポイント加速度判定のための閾値設定 ★★★
    joint_acceleration_thresholds = {
        "RAnkle": 100.0, "LAnkle": 100.0, "RHip": 100.0, "LHip": 100.0,
        "RKnee": 100.0, "LKnee": 100.0, "RSmallToe": 100.0, "LSmallToe": 100.0,
        "RBigToe": 100.0, "LBigToe": 100.0, "RHeel": 100.0, "LHeel": 100.0
    }
    ACCELERATION_THRESHOLD = 100.0  # デフォルト閾値
    BUTTERWORTH_CUTOFF = 12.0
    FRAME_RATE = 60
    DEBUG_ACCELERATION = True

    print(f"★★★ MATLABカルマンフィルタの完全再現版 ★★★")
    print("    ローカルレベルモデル + 準ニュートン法による散漫対数尤度最大化")
    print("    左右足入れ替わり検出機能付き二階差分カルマンフィルタ")
    print("    MATLABのSecond_Order_Difference_Kalman_Filter.mの正確な実装")

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

            # デバッグ用コードを追加して確認
            print(f"カメラ1のフレーム数: {len(list(openpose_dir1.glob('*_keypoints.json')))}")
            print(f"カメラ2のフレーム数: {len(list(openpose_dir2.glob('*_keypoints.json')))}")
            print(f"共通フレーム数: {len(common_frames)}")
            print(f"最初の5フレーム: {common_frames[:5]}")
            print(f"最後の5フレーム: {common_frames[-5:]}")

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

            # ★★★ カルマンフィルタ状態を各人物について初期化 ★★★
            kalman_states_list = [{} for _ in range(max_people)]

            # ★★★ フレーム範囲を250～350に限定 ★★★
            frame_start = 250
            frame_end = 350  # 350まで処理
            # selected_frames = common_frames
            selected_frames = common_frames[frame_start:frame_end]

            print("  - ステップ1: 全フレームのエラー判定（カルマンフィルタ統合版）...")

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
                        raw_3d, corrected_3d, accels, min_accel = process_single_person_individual_keypoint_switching_kalman_matlab(
                            kp1, cf1, kp2, cf2, P1, P2, history_list[p_idx], ACCELERATION_THRESHOLD,
                            kalman_states_list[p_idx], debug_frame_idx)

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
            plot_bilateral_joint_analysis(all_person_results, all_accelerations_data, selected_frames, bilateral_dir, thera_dir.name)

            print("  - ステップ7: Raw処理 vs スイッチング処理の比較グラフ化...")
            plot_method_comparison_analysis(all_person_results, selected_frames, method_comparison_dir, thera_dir.name)

            print("  - ステップ8: 全フレームの加速度をグラフ化...")
            plot_acceleration_graph(all_accelerations_data, selected_frames, ACCELERATION_THRESHOLD, acceleration_dir, thera_dir.name)


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

            # プロット（データが存在する場合のみ）
            if len(right_raw_processed_accel) > 0:
                accel_time_axis = time_axis[2:2+len(right_raw_processed_accel)]
                ax.plot(accel_time_axis, right_raw_processed_accel, '--', color='orange', linewidth=2, alpha=0.8, label=f'R{joint_name} Raw→Processed Accel')

            if len(left_raw_processed_accel) > 0:
                accel_time_axis = time_axis[2:2+len(left_raw_processed_accel)]
                ax.plot(accel_time_axis, left_raw_processed_accel, '--', color='cyan', linewidth=2, alpha=0.8, label=f'L{joint_name} Raw→Processed Accel')

            if len(right_switching_accel) > 0:
                accel_time_axis = time_axis[2:2+len(right_switching_accel)]
                ax.plot(accel_time_axis, right_switching_accel, color='darkred', linewidth=2, label=f'R{joint_name} Switching→Processed Accel')

            if len(left_switching_accel) > 0:
                accel_time_axis = time_axis[2:2+len(left_switching_accel)]
                ax.plot(accel_time_axis, left_switching_accel, color='darkblue', linewidth=2, label=f'L{joint_name} Switching→Processed Accel')

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
                ax.plot(time_axis, right_raw, 'o', color='lightcoral', markersize=2, alpha=0.6, label='R{joint_name} Raw')
                ax.plot(time_axis, right_corrected, 'x', color='red', markersize=3, alpha=0.7, label='R{joint_name} Corrected')
                ax.plot(time_axis, right_final, color='darkred', linewidth=2, label='R{joint_name} Final')

                # 左側関節のプロット
                ax.plot(time_axis, left_raw, 'o', color='lightblue', markersize=2, alpha=0.6, label='L{joint_name} Raw')
                ax.plot(time_axis, left_corrected, 'x', color='blue', markersize=3, alpha=0.7, label='L{joint_name} Corrected')
                ax.plot(time_axis, left_final, color='darkblue', linewidth=2, label='L{joint_name} Final')

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

            # 有効な加速度データのみプロット
            accel_time_axis = time_axis[2:]  # 加速度は位置データより2フレーム短い

            # 右側関節（有効なデータのみプロット）
            if len(right_raw_accel) > 0:
                valid_accel_time = accel_time_axis[:len(right_raw_accel)]
                ax.plot(valid_accel_time, right_raw_accel, 'o', color='lightcoral', markersize=3, alpha=0.6, label=f'R{joint_name} Raw Accel')
            if len(right_corrected_accel) > 0:
                valid_accel_time = accel_time_axis[:len(right_corrected_accel)]
                ax.plot(valid_accel_time, right_corrected_accel, '--', color='red', linewidth=1.5, alpha=0.8, label=f'R{joint_name} Corrected Accel')
            if len(right_final_accel) > 0:
                valid_accel_time = accel_time_axis[:len(right_final_accel)]
                ax.plot(valid_accel_time, right_final_accel, color='darkred', linewidth=2, label=f'R{joint_name} Final Accel')

            # 左側関節（有効なデータのみプロット）
            if len(left_raw_accel) > 0:
                valid_accel_time = accel_time_axis[:len(left_raw_accel)]
                ax.plot(valid_accel_time, left_raw_accel, 'o', color='lightblue', markersize=3, alpha=0.6, label=f'L{joint_name} Raw Accel')
            if len(left_corrected_accel) > 0:
                valid_accel_time = accel_time_axis[:len(left_corrected_accel)]
                ax.plot(valid_accel_time, left_corrected_accel, '--', color='blue', linewidth=1.5, alpha=0.8, label=f'L{joint_name} Corrected Accel')
            if len(left_final_accel) > 0:
                valid_accel_time = accel_time_axis[:len(left_final_accel)]
                ax.plot(valid_accel_time, left_final_accel, color='darkblue', linewidth=2, label=f'L{joint_name} Final Accel')

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
    """
    関節軌道からMatlabスタイルの加速度時系列を計算
    joint_trajectory: (N, 3) の関節位置時系列
    返り値: (N-2,) の加速度大きさ時系列
    """
    if len(joint_trajectory) < 3:
        return np.array([])

    # NaNを含む行をチェック
    valid_mask = ~np.isnan(joint_trajectory).any(axis=1)

    if np.sum(valid_mask) < 3:
        # 有効な点が3点未満の場合は空配列を返す
        return np.array([])

    # 有効な点のみでMatlabスタイル加速度計算
    valid_trajectory = joint_trajectory[valid_mask]
    acceleration_series = calculate_acceleration_matlab_style(valid_trajectory)

    # 元の配列サイズに合わせて結果を作成（NaN埋め）
    full_acceleration = np.full(len(joint_trajectory) - 2, np.nan)

    # 有効な加速度値を対応する位置に配置
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) >= 3:
        # 加速度は位置インデックス+2から始まる
        for i, accel_val in enumerate(acceleration_series):
            if i + 2 < len(valid_indices):
                target_idx = valid_indices[i + 2] - 2  # 全体配列での加速度インデックス
                if 0 <= target_idx < len(full_acceleration):
                    full_acceleration[target_idx] = accel_val

    return full_acceleration

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
            accel_n = np.array([d['all'][0] if len(d['all']) > 0 else np.nan for d in person_accel_data], dtype=float)
            accel_s1 = np.array([d['all'][1] if len(d['all']) > 1 else np.nan for d in person_accel_data], dtype=float)
            accel_s2 = np.array([d['all'][2] if len(d['all']) > 2 else np.nan for d in person_accel_data], dtype=float)
            accel_s12 = np.array([d['all'][3] if len(d['all']) > 3 else np.nan for d in person_accel_data], dtype=float)
            min_accels = np.array([d['min'] if 'min' in d else np.nan for d in person_accel_data], dtype=float)


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

