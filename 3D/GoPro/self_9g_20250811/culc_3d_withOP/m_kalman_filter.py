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
        if F_t[t] <= 1e-10 or not np.isfinite(F_t[t]):
            F_t[t] = 1e-10
            P_tt1[t] = 1e-8 # Pが不安定ならリセット

        # Kalman gain
        K_t[t] = P_tt1[t] / F_t[t]
        K_t[t] = np.clip(K_t[t], 0, 1)

        # Current state
        a_tt[t] = a_tt1[t] + K_t[t] * v_t[t]
        P_tt[t] = P_tt1[t] * (1 - K_t[t])

        # ★★★ 追加の安定化処理 ★★★
        # 状態（速度）が非現実的な値になったらリセット
        if not np.isfinite(a_tt[t]) or abs(a_tt[t]) > 1e4: # 1e4は十分に大きな速度
            a_tt[t] = y[t] if np.isfinite(y[t]) else 0 # 観測値 or 0でリセット

        # 共分散が負または発散したらリセット
        if not np.isfinite(P_tt[t]) or P_tt[t] < 0:
            P_tt[t] = var_eps # 観測ノイズ分散でリセット

        # Next state (ローカルレベルモデル: 状態は変わらない)
        a_tt1[t + 1] = a_tt[t]
        P_tt1[t + 1] = P_tt[t] + var_eta

        # 共分散の上限を設定（上限値を少し厳しくする）
        P_tt1[t + 1] = np.clip(P_tt1[t + 1], 1e-10, 1e6)

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
    par = max(min(initial_value, 0.1), 1e-6)
    var_eta = par
    var_eps = par

    psi_eta = np.log(np.sqrt(max(var_eta, 1e-8)))
    psi_eps = np.log(np.sqrt(max(var_eps, 1e-8)))
    x0 = [psi_eta, psi_eps]

    def objective(x):
        try:
            x_clipped = np.clip(x, -5, 5)
            result = -calc_log_diffuse_llhd(y, x_clipped)
            return result if np.isfinite(result) else 1e10
        except:
            return 1e10

    try:
        from scipy.optimize import Bounds
        bounds = Bounds([-5, -5], [5, 5])
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'disp': False, 'maxiter': 100})
        x_opt = result.x if result.success else x0
    except:
        x_opt = x0

    try:
        x_opt = np.clip(x_opt, -10, 10)
        var_eta_opt = np.exp(2 * x_opt[0])
        var_eps_opt = np.exp(2 * x_opt[1])
        var_eta_opt = np.clip(var_eta_opt, 1e-6, 100.0)
        var_eps_opt = np.clip(var_eps_opt, 1e-6, 100.0)
    except:
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


def kalman2_matlab_exact(coordinate_L, coordinate_R, th, initial_value, debug_context=not None):
    """
    MATLABのkalman2関数の完全再現（高速化・安定化版）
    二階差分カルマンフィルタによる左右足の誤検出補正

    Args:
        coordinate_L: 左足座標
        coordinate_R: 右足座標
        th: 加速度閾値
        initial_value: カルマンフィルタの初期値
        debug_context (str, optional): デバッグ出力用のコンテキスト情報. Defaults to not None.

    Returns:
        corrected_L: 補正済み左足座標
        corrected_R: 補正済み右足座標
        miss_point: 誤検出フラグ配列
    """
    first_step = 0
    end_step = len(coordinate_R)
    miss_point = np.zeros(end_step)
    corrected_L = coordinate_L.copy()
    corrected_R = coordinate_R.copy()

    # パラメータ推定をループの外で1回だけ実行
    diff_data_L = np.diff(corrected_L)
    diff_data_R = np.diff(corrected_R)
    y_L = moving_average_filter(diff_data_L, 3)
    y_R = moving_average_filter(diff_data_R, 3)

    var_eta_L, var_eps_L = estimate_parameters_ml(y_L, initial_value)
    var_eta_R, var_eps_R = estimate_parameters_ml(y_R, initial_value)
    var_eps_L, var_eta_L = var_eta_L, var_eps_L
    var_eps_R, var_eta_R = var_eta_R, var_eps_R
    a1_L, P1_L = var_eps_L, var_eta_L
    a1_R, P1_R = var_eps_R, var_eta_R

    # 全データに対して一度にフィルタリングを実行
    a_tt_L, _, _, _ = local_level_kf(y_L, a1_L, P1_L, var_eta_L, var_eps_L)
    a_tt_R, _, _, _ = local_level_kf(y_R, a1_R, P1_R, var_eta_R, var_eps_R)

    kalman_flag = 0
    for i in range(first_step + 2, end_step):
        q1_L = corrected_L[i] - corrected_L[i-1]
        q2_L = corrected_L[i-1] - corrected_L[i-2]
        q1_R = corrected_R[i] - corrected_R[i-1]
        q2_R = corrected_R[i-1] - corrected_R[i-2]
        diff2_L = q1_L - q2_L
        diff2_R = q1_R - q2_R

        # ★★★ 修正点: カルマンフィルタの推定速度を使用する ★★★
        # a_tt_L/R は速度(1階差分)の推定値。インデックスは i-1 に対応。
        # 配列の範囲外アクセスを防ぐためのフォールバックも追加
        kalman_velocity_L = a_tt_L[i-1] if i-1 < len(a_tt_L) else (corrected_L[i-1] - corrected_L[i-2])
        kalman_velocity_R = a_tt_R[i-1] if i-1 < len(a_tt_R) else (corrected_R[i-1] - corrected_R[i-2])

        # ★★★ 追加点: デバッグ出力用のフラグ ★★★
        is_debug = debug_context is not None

        if abs(diff2_L) > th and abs(diff2_R) > th:
            L_box, R_box = corrected_L[i], corrected_R[i]
            corrected_L[i], corrected_R[i] = R_box, L_box

            q1_L_sw = corrected_L[i] - corrected_L[i-1]
            q2_L_sw = corrected_L[i-1] - corrected_L[i-2]
            q1_R_sw = corrected_R[i] - corrected_R[i-1]
            q2_R_sw = corrected_R[i-1] - corrected_R[i-2]
            diff2_L_sw = q1_L_sw - q2_L_sw
            diff2_R_sw = q1_R_sw - q2_R_sw

            if abs(diff2_L_sw) > th and abs(diff2_R_sw) > th:
                if is_debug:
                    print(f"デバッグ {debug_context} [ローカルフレーム={i}]: 補正タイプ4 (両足とも高加速度、交換なし). 左_加速度={diff2_L:.2f}, 右_加速度={diff2_R:.2f} > 閾値={th:.2f}")
                    print(f"  -> 左: 座標={L_box:.2f} -> 補正後_座標={corrected_L[i-1] + kalman_velocity_L:.2f} (使用速度={kalman_velocity_L:.2f})")
                    print(f"  -> 右: 座標={R_box:.2f} -> 補正後_座標={corrected_R[i-1] + kalman_velocity_R:.2f} (使用速度={kalman_velocity_R:.2f})")

                corrected_L[i], corrected_R[i] = L_box, R_box
                # ★★★ 修正点: カルマンフィルタの推定速度で補正 ★★★
                corrected_L[i] = corrected_L[i-1] + kalman_velocity_L
                corrected_R[i] = corrected_R[i-1] + kalman_velocity_R
                miss_point[i] = 4
                kalman_flag = 1
            else:
                if is_debug:
                    print(f"デバッグ {debug_context} [ローカルフレーム={i}]: 補正タイプ1 (左右交換). 左_加速度={diff2_L:.2f}, 右_加速度={diff2_R:.2f} > 閾値={th:.2f}")
                miss_point[i] = 1
                kalman_flag = 1

        elif abs(diff2_L) > th and abs(diff2_R) <= th:
            if is_debug:
                print(f"デバッグ {debug_context} [ローカルフレーム={i}]: 補正タイプ2 (左足が高加速度). 左_加速度={diff2_L:.2f} > 閾値={th:.2f}")
                print(f"  -> 左: 座標={corrected_L[i]:.2f} -> 補正後_座標={corrected_L[i-1] + kalman_velocity_L:.2f} (使用速度={kalman_velocity_L:.2f})")

            # ★★★ 修正点: カルマンフィルタの推定速度で補正 ★★★
            corrected_L[i] = corrected_L[i-1] + kalman_velocity_L
            miss_point[i] = 2
            kalman_flag = 1

        elif abs(diff2_L) <= th and abs(diff2_R) > th:
            if is_debug:
                print(f"デバッグ {debug_context} [ローカルフレーム={i}]: 補正タイプ3 (右足が高加速度). 右_加速度={diff2_R:.2f} > 閾値={th:.2f}")
                print(f"  -> 右: 座標={corrected_R[i]:.2f} -> 補正後_座標={corrected_R[i-1] + kalman_velocity_R:.2f} (使用速度={kalman_velocity_R:.2f})")

            # ★★★ 修正点: カルマンフィルタの推定速度で補正 ★★★
            corrected_R[i] = corrected_R[i-1] + kalman_velocity_R
            miss_point[i] = 3
            kalman_flag = 1

        kalman_flag = 0

    return corrected_L, corrected_R, miss_point