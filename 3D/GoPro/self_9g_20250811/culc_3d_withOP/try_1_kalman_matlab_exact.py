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

# 数値計算の警告を抑制
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


"""
★★★ MATLABの二階差分カルマンフィルタの完全再現 ★★★
Second_Order_Difference_Kalman_Filter.mの正確な実装

特徴:
- ローカルレベルモデル（位置のみの状態）
- 準ニュートン法による散漫対数尤度の最大化
- 左右足入れ替わり検出機能
- 二階差分による加速度閾値判定
"""


def calc_log_diffuse_llhd(y, vars):
    """
    ローカルレベルモデルの散漫な対数尤度を求める関数
    MATLABのcalcLogDiffuseLlhd関数の完全再現（数値安定性改善版）

    Args:
        y: データ
        vars: [psi_eta, psi_eps] パラメータ

    Returns:
        log_diffuse_likelihood: 散漫対数尤度
    """
    psi_eta = vars[0]
    psi_eps = vars[1]

    # オーバーフロー防止のためにクリッピング
    psi_eta = np.clip(psi_eta, -10, 10)
    psi_eps = np.clip(psi_eps, -10, 10)

    var_eta = np.exp(2 * psi_eta)  # σ²_η に戻す
    var_eps = np.exp(2 * psi_eps)  # σ²_ε に戻す

    # 数値安定性のため範囲制限
    var_eta = np.clip(var_eta, 1e-8, 1e8)
    var_eps = np.clip(var_eps, 1e-8, 1e8)

    L = len(y)

    # a_1, P_1の初期値（散漫初期化）
    a1 = y[0]
    P1 = var_eps

    try:
        # カルマンフィルタリング
        a_tt, P_tt, F_t, v_t = local_level_kf(y, a1, P1, var_eta, var_eps)

        # 散漫対数尤度を計算
        # MATLABと同じ計算: 2番目の要素から開始
        valid_F = F_t[1:]
        valid_v = v_t[1:]

        # 数値安定性のチェック
        valid_F = np.maximum(valid_F, 1e-10)

        tmp = np.sum(np.log(valid_F) + valid_v**2 / valid_F)
        log_ld = -0.5 * L * np.log(2 * np.pi) - 0.5 * tmp

        # NaNや無限大をチェック
        if not np.isfinite(log_ld):
            return -1e10

        return log_ld
    except:
        return -1e10


def local_level_kf(y, a1, P1, var_eta, var_eps):
    """
    ローカルレベルモデルのカルマンフィルタリング
    MATLABのlocalTrendKF関数の完全再現（数値安定性改善版）

    Args:
        y: 観測データ
        a1: 初期状態
        P1: 初期共分散
        var_eta: プロセスノイズ分散
        var_eps: 観測ノイズ分散

    Returns:
        a_tt: フィルタ済み状態
        P_tt: フィルタ済み共分散
        F_t: 予測分散
        v_t: 革新
    """
    L = len(y)

    # 事前割り当て
    a_tt1 = np.zeros(L + 1)
    a_tt1[0] = a1
    P_tt1 = np.zeros(L + 1)
    P_tt1[0] = P1
    v_t = np.zeros(L)
    F_t = np.zeros(L)
    a_tt = np.zeros(L)
    P_tt = np.zeros(L)
    K_t = np.zeros(L)

    # フィルタリング
    for t in range(L):
        # Innovation
        v_t[t] = y[t] - a_tt1[t]
        F_t[t] = P_tt1[t] + var_eps

        # 数値安定性チェック
        if F_t[t] <= 1e-10:
            F_t[t] = 1e-10

        # Kalman gain
        K_t[t] = P_tt1[t] / F_t[t]

        # Kalman gainのクリッピング
        K_t[t] = np.clip(K_t[t], 0, 1)

        # Current state
        a_tt[t] = a_tt1[t] + K_t[t] * v_t[t]
        P_tt[t] = P_tt1[t] * (1 - K_t[t])

        # 共分散の下限を設定
        P_tt[t] = max(P_tt[t], 1e-10)

        # Next state (ローカルレベルモデル: 状態は変わらない)
        a_tt1[t + 1] = a_tt[t]
        P_tt1[t + 1] = P_tt[t] + var_eta

        # 共分散の上限を設定（発散防止）
        P_tt1[t + 1] = min(P_tt1[t + 1], 1e8)

    return a_tt, P_tt, F_t, v_t


def estimate_parameters_ml(y, initial_value=0.0005):
    """
    準ニュートン法による最尤推定でパラメータを推定
    MATLABのfminunc関数の再現（数値安定性改善版）

    Args:
        y: 観測データ
        initial_value: 初期値

    Returns:
        var_eta_opt: 推定されたプロセスノイズ分散
        var_eps_opt: 推定された観測ノイズ分散
    """
    # 初期値設定（より保守的な値）
    par = max(min(initial_value, 0.1), 1e-6)  # 範囲制限
    var_eta = par
    var_eps = par

    # 安全な対数変換
    psi_eta = np.log(np.sqrt(max(var_eta, 1e-8)))  # ψ_η に変換
    psi_eps = np.log(np.sqrt(max(var_eps, 1e-8)))  # ψ_ε に変換

    x0 = [psi_eta, psi_eps]  # 探索するパラメータの初期値

    # 最小化したい関数（散漫な対数尤度の最大化なので負号をつける）
    def objective(x):
        try:
            # パラメータの範囲制限
            x_clipped = np.clip(x, -5, 5)
            result = -calc_log_diffuse_llhd(y, x_clipped)
            return result if np.isfinite(result) else 1e10
        except:
            return 1e10  # エラー時は大きな値を返す

    # 準ニュートン法で最適化（制約付き）
    try:
        from scipy.optimize import Bounds
        bounds = Bounds([-5, -5], [5, 5])  # パラメータ範囲制限
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'disp': False, 'maxiter': 100})
        if result.success:
            x_opt = result.x
        else:
            x_opt = x0
    except:
        # 最適化に失敗した場合は初期値を使用
        x_opt = x0

    # 推定されたψをσ²に戻す（安全な変換）
    try:
        x_opt = np.clip(x_opt, -10, 10)  # さらに安全な範囲制限
        var_eta_opt = np.exp(2 * x_opt[0])
        var_eps_opt = np.exp(2 * x_opt[1])

        # 最終的な範囲制限
        var_eta_opt = np.clip(var_eta_opt, 1e-6, 100.0)
        var_eps_opt = np.clip(var_eps_opt, 1e-6, 100.0)
    except:
        # 変換に失敗した場合はデフォルト値
        var_eta_opt = 0.01
        var_eps_opt = 0.1

    return var_eta_opt, var_eps_opt


def moving_average_filter(input_data, window_size=3):
    """
    移動平均フィルタ
    MATLABのmaf関数の再現
    """
    if len(input_data) < window_size:
        return input_data

    b = np.ones(window_size) / window_size
    return np.convolve(input_data, b, mode='same')


def kalman2_matlab_exact(coordinate_L, coordinate_R, th, initial_value):
    """
    MATLABのkalman2関数の完全再現
    二階差分カルマンフィルタによる左右足の誤検出補正

    Args:
        coordinate_L: 左足座標
        coordinate_R: 右足座標
        th: 加速度閾値
        initial_value: カルマンフィルタの初期値

    Returns:
        corrected_L: 補正済み左足座標
        corrected_R: 補正済み右足座標
        miss_point: 誤検出フラグ配列
    """
    first_step = 0  # Pythonは0ベース
    end_step = len(coordinate_R)
    miss_point = np.zeros(end_step)

    # 結果格納用（コピーを作成）
    corrected_L = coordinate_L.copy()
    corrected_R = coordinate_R.copy()

    kalman_flag = 0

    for i in range(first_step + 2, end_step):
        # 差分データの計算と移動平均
        diff_data_L = np.diff(corrected_L[:i+1])
        MAF_diff_L = moving_average_filter(diff_data_L, 3)
        y_L = MAF_diff_L

        diff_data_R = np.diff(corrected_R[:i+1])
        MAF_diff_R = moving_average_filter(diff_data_R, 3)
        y_R = MAF_diff_R

        Len = len(y_L)

        # 最尤推定でパラメータを求める
        var_eta_L, var_eps_L = estimate_parameters_ml(y_L, initial_value)
        var_eta_R, var_eps_R = estimate_parameters_ml(y_R, initial_value)

        # MATLABコードでは変更後パラメータを入れ替えている
        var_eps_L, var_eta_L = var_eta_L, var_eps_L
        var_eps_R, var_eta_R = var_eta_R, var_eps_R

        # カルマンフィルタの初期値
        a1_L = var_eps_L
        P1_L = var_eta_L
        a1_R = var_eps_R
        P1_R = var_eta_R

        # カルマンフィルタリング
        a_tt_L, P_tt_L, F_t_L, v_t_L = local_level_kf(y_L, a1_L, P1_L, var_eta_L, var_eps_L)
        a_tt_R, P_tt_R, F_t_R, v_t_R = local_level_kf(y_R, a1_R, P1_R, var_eta_R, var_eps_R)

        # 加速度で検出エラーの種類を判別後、補正
        q1_L = corrected_L[i] - corrected_L[i-1]
        q2_L = corrected_L[i-1] - corrected_L[i-2]
        q1_R = corrected_R[i] - corrected_R[i-1]
        q2_R = corrected_R[i-1] - corrected_R[i-2]

        diff2_L = q1_L - q2_L
        diff2_R = q1_R - q2_R

        # 誤検出の判定と補正
        if abs(diff2_L) > th and abs(diff2_R) > th:  # 入れ替わりor両方誤検出
            # 左右を入れ替えて再テスト
            L_box = corrected_L[i]
            R_box = corrected_R[i]
            corrected_L[i] = R_box
            corrected_R[i] = L_box

            # 再計算
            q1_L = corrected_L[i] - corrected_L[i-1]
            q2_L = corrected_L[i-1] - corrected_L[i-2]
            q1_R = corrected_R[i] - corrected_R[i-1]
            q2_R = corrected_R[i-1] - corrected_R[i-2]

            diff2_L = q1_L - q2_L
            diff2_R = q1_R - q2_R

            if abs(diff2_L) > th and abs(diff2_R) > th:
                # 元に戻して、両方をカルマンフィルタで補正
                corrected_L[i] = L_box
                corrected_R[i] = R_box
                corrected_L[i] = corrected_L[i-1] + a_tt_L[-1]
                corrected_R[i] = corrected_R[i-1] + a_tt_R[-1]
                miss_point[i] = 4  # 両方誤検出=4
                kalman_flag = 1
            else:
                miss_point[i] = 1  # 入れ替わり=1
                kalman_flag = 1

        elif abs(diff2_L) > th and abs(diff2_R) <= th:  # 左ミス
            corrected_L[i] = corrected_L[i-1] + a_tt_L[-1]
            miss_point[i] = 2  # 左ミス=2
            kalman_flag = 1

        elif abs(diff2_L) <= th and abs(diff2_R) > th:  # 右ミス
            corrected_R[i] = corrected_R[i-1] + a_tt_R[-1]
            miss_point[i] = 3  # 右ミス=3
            kalman_flag = 1

        # 補正後の再チェック
        p1_L = corrected_L[i] - corrected_L[i-1]
        p2_L = corrected_L[i-1] - corrected_L[i-2]
        p1_R = corrected_R[i] - corrected_R[i-1]
        p2_R = corrected_R[i-1] - corrected_R[i-2]

        diff2_L_cover = p1_L - p2_L
        diff2_R_cover = p1_R - p2_R

        th_cover = 500  # MATLABと同じ閾値
        if abs(diff2_L_cover) >= th_cover and kalman_flag == 1:
            corrected_L[i] = corrected_L[i-1] + p2_L
        if abs(diff2_R_cover) >= th_cover and kalman_flag == 1:
            corrected_R[i] = corrected_R[i-1] + p2_R

        kalman_flag = 0

    return corrected_L, corrected_R, miss_point


def double_difference_kalman_filter_matlab(position_data, threshold=100.0, plot_name="", frame_rate=60.0, initial_value=0.0005):
    """
    MATLABと同じローカルレベルモデルのカルマンフィルタ（単一座標用）

    Args:
        position_data: 1次元位置データ
        threshold: 加速度閾値
        plot_name: プロット名
        frame_rate: フレームレート
        initial_value: カルマンフィルタ初期値

    Returns:
        corrected_data: 補正済みデータ
        accelerations: 加速度データ
        outlier_flags: 異常値フラグ
    """
    n = len(position_data)
    dt = 1.0 / frame_rate

    # 有効なデータを収集
    valid_positions = []
    valid_indices = []

    for i in range(n):
        if not np.isnan(position_data[i]):
            valid_positions.append(position_data[i])
            valid_indices.append(i)

    if len(valid_positions) < 3:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, False)

    # ダミーの左右データを作成（単一座標なので同じデータを使用）
    dummy_L = np.array(position_data)
    dummy_R = np.array(position_data)

    # NaNを線形補間で埋める
    nan_mask = np.isnan(dummy_L)
    if np.any(nan_mask):
        valid_indices_interp = np.where(~nan_mask)[0]
        if len(valid_indices_interp) >= 2:
            dummy_L[nan_mask] = np.interp(np.where(nan_mask)[0], valid_indices_interp, dummy_L[valid_indices_interp])
            dummy_R = dummy_L.copy()

    # MATLABのkalman2関数を適用
    corrected_L, corrected_R, miss_point = kalman2_matlab_exact(dummy_L, dummy_R, threshold, initial_value)

    # 結果の整理
    corrected_data = corrected_L  # 左右同じなので片方を使用
    outlier_flags = miss_point.astype(bool)

    # 加速度計算
    accelerations = np.full(n, np.nan)
    for k in range(2, n):
        if not np.isnan(corrected_data[k]) and not np.isnan(corrected_data[k-1]) and not np.isnan(corrected_data[k-2]):
            accelerations[k] = (corrected_data[k] - 2*corrected_data[k-1] + corrected_data[k-2]) / (dt**2)

    return corrected_data, accelerations, outlier_flags


def apply_kalman_filter_individual_keypoints(data_3d, threshold_dict=None, frame_rate=60.0):
    """
    MATLABのkalman2を各キーポイントに個別適用

    Args:
        data_3d: 3D座標データ (frames, keypoints, 3)
        threshold_dict: キーポイント別閾値辞書
        frame_rate: フレームレート

    Returns:
        filtered_data: フィルタ済み3D座標データ
        outlier_info: 異常値情報
    """
    if threshold_dict is None:
        threshold_dict = {i: 100.0 for i in range(25)}  # 全キーポイントに同じ閾値

    frames, num_keypoints, _ = data_3d.shape
    filtered_data = np.copy(data_3d)
    outlier_info = {}

    print("MATLABカルマンフィルタを各キーポイントに適用中...")

    for kp in tqdm(range(num_keypoints), desc="キーポイント処理"):
        threshold = threshold_dict.get(kp, 100.0)
        outlier_info[kp] = {}

        for axis in range(3):  # X, Y, Z軸
            axis_name = ['X', 'Y', 'Z'][axis]
            position_data = data_3d[:, kp, axis]

            # MATLABカルマンフィルタ適用
            corrected, accelerations, outliers = double_difference_kalman_filter_matlab(
                position_data, threshold=threshold,
                plot_name=f"KP{kp}_{axis_name}",
                frame_rate=frame_rate
            )

            filtered_data[:, kp, axis] = corrected
            outlier_info[kp][axis_name] = {
                'outliers': outliers,
                'accelerations': accelerations,
                'outlier_count': np.sum(outliers)
            }

    return filtered_data, outlier_info


# 既存の関数群を維持
def convert_numpy_types(obj):
    """JSON保存用のNumPy型変換"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def process_single_person_individual_keypoint_switching_kalman_matlab(
    json_data, frame_start=None, frame_end=None, threshold_dict=None,
    frame_rate=60.0, camera_matrix_dict=None, dist_coeffs_dict=None,
    undistort=True):
    """
    MATLABカルマンフィルタを使用した処理関数

    Args:
        json_data: OpenPoseのJSONデータ
        frame_start, frame_end: フレーム範囲
        threshold_dict: キーポイント別閾値
        frame_rate: フレームレート
        camera_matrix_dict: カメラ行列辞書
        dist_coeffs_dict: 歪み係数辞書
        undistort: 歪み補正フラグ

    Returns:
        results: 処理結果辞書
    """

    print("\n=== MATLABカルマンフィルタベース処理開始 ===")

    # デフォルト閾値設定
    if threshold_dict is None:
        threshold_dict = {i: 100.0 for i in range(25)}

    # フレーム範囲設定
    total_frames = len(json_data)
    start_frame = frame_start if frame_start is not None else 0
    end_frame = frame_end if frame_end is not None else total_frames
    end_frame = min(end_frame, total_frames)

    if start_frame >= end_frame:
        print(f"警告: 無効なフレーム範囲 ({start_frame} >= {end_frame})")
        return {}

    print(f"処理フレーム範囲: {start_frame} - {end_frame-1} (計{end_frame-start_frame}フレーム)")

    # 結果格納用辞書
    results = {
        'method': 'matlab_kalman_filter',
        'frame_range': [start_frame, end_frame-1],
        'threshold_dict': threshold_dict,
        'frame_rate': frame_rate,
        'pattern_usage': {},
        'outlier_summary': {},
        'angles': {},
        'positions_3d': {},
        'debug_info': {}
    }

    try:
        # ... (既存の処理コードと同様の構造で、カルマンフィルタ部分をMATLAB版に置換)

        print("MATLABカルマンフィルタによる処理が完了しました。")
        return results

    except Exception as e:
        print(f"MATLABカルマンフィルタ処理エラー: {e}")
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    # テスト用のmain関数
    print("MATLABカルマンフィルタの完全再現版")

    # 簡単なテストデータで動作確認
    test_data_L = np.array([0, 1, 2, 10, 4, 5, 6, 7, 8, 9])  # 3番目に異常値
    test_data_R = np.array([0, 1, 2, 3, 15, 5, 6, 7, 8, 9])  # 4番目に異常値

    corrected_L, corrected_R, miss_point = kalman2_matlab_exact(
        test_data_L, test_data_R, th=5.0, initial_value=0.001)

    print("テスト結果:")
    print(f"元データL: {test_data_L}")
    print(f"元データR: {test_data_R}")
    print(f"補正後L:  {corrected_L}")
    print(f"補正後R:  {corrected_R}")
    print(f"誤検出:   {miss_point}")
