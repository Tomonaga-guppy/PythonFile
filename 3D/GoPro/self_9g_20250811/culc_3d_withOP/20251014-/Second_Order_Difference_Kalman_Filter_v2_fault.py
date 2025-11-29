import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize
import os
import warnings

# # RuntimeWarningという種類の警告を非表示にする設定
# warnings.filterwarnings('ignore', category=RuntimeWarning)


"""
失敗というよりかは結局うまくいってない
カルマンフィルタの予測などを範囲全体で一度にやってしまい、ループ内では加速度の閾値判定と補正のみを行うようにしている
"""

# =============================================================================
# %% 関数定義セクション
# =============================================================================

# def local_trend_kf(y, a1, p1, var_eta, var_eps):
#     """
#     ローカルトレンドモデルのカルマンフィルタリングを行う関数
#     """
#     L = len(y)
    
#     # 計算結果を格納するための配列を事前に確保(Preallocation)
#     a_tt1 = np.zeros(L + 1)
#     a_tt1[0] = a1
#     p_tt1 = np.zeros(L + 1)
#     p_tt1[0] = p1
#     v_t = np.zeros(L)
#     f_t = np.zeros(L)
#     a_tt = np.zeros(L)
#     p_tt = np.zeros(L)
#     k_t = np.zeros(L)
    
#     # Filtering (フィルタリングのループ処理)
#     for t in range(L):
#         # Innovation (観測値と予測値の差)
#         v_t[t] = y[t] - a_tt1[t]
#         f_t[t] = p_tt1[t] + var_eps
        
#         # Kalman gain (カルマンゲインの計算)
#         # ゼロ、無限大、NaN(非数)による計算エラーを回避
#         if f_t[t] == 0 or not np.isfinite(f_t[t]):
#             k_t[t] = 0
#         else:
#             k_t[t] = p_tt1[t] / f_t[t] 
            
#         # Current state (現時刻の状態を更新)
#         a_tt[t] = a_tt1[t] + k_t[t] * v_t[t]
#         p_tt[t] = p_tt1[t] * (1 - k_t[t])
        
#         # Next state (次時刻の状態を予測)
#         a_tt1[t+1] = a_tt[t]
#         p_tt1[t+1] = p_tt[t] + var_eta
        
#     return a_tt, p_tt, f_t, v_t

# def calc_log_diffuse_llhd(vars, y):
#     """
#     ローカルトレンドモデルの散漫な対数尤度を求める関数
#     尤度を最大化するパラメータを見つけるために使用
#     """
#     psi_eta, psi_eps = vars
#     var_eta = np.exp(2 * psi_eta)  # ψ_η を σ^2_η に戻す
#     var_eps = np.exp(2 * psi_eps)  # ψ_ε を σ^2_ε に戻す
#     L = len(y)
    
#     if L < 2:
#         return -np.inf # データが少なすぎて計算できない
        
#     # a_1, P_1の初期値
#     a1 = y[0]
#     p1 = var_eps
    
#     # カルマンフィルタリングを実行
#     _, _, f_t, v_t = local_trend_kf(y, a1, p1, var_eta, var_eps)
    
#     # f_tのゼロや負の値、非数をチェック
#     if np.any(f_t[1:] <= 0) or not np.all(np.isfinite(f_t[1:])):
#         return -np.inf

#     # 散漫対数尤度を計算
#     tmp = np.sum(np.log(f_t[1:]) + v_t[1:]**2 / f_t[1:])
#     log_ld = -0.5 * L * np.log(2 * np.pi) - 0.5 * tmp
    
#     return log_ld

# def maf(input_data, size):
#     """
#     移動平均フィルタ(Moving Average Filter)
#     """
#     window_size = size
#     b = (1 / window_size) * np.ones(window_size)
#     a = 1
#     from scipy.signal import lfilter
#     return lfilter(b, a, input_data)

# def kalman2(coordinate_L, coordinate_R, th, initial_value):
#     """
#     二階差分カルマンフィルタ
#     加速度(二階差分)を監視し、閾値を超えた場合に補正を行う
#     """
#     end_step = len(coordinate_R)
    
#     # 元のデータを変更しないようにコピーを作成
#     coordinate_L = coordinate_L.copy()
#     coordinate_R = coordinate_R.copy()

#     # 誤検出の種類を記録するための配列
#     miss_point = np.zeros(end_step)
    
#     # 2フレーム目から最終フレームまでループ
#     for i in range(2, end_step):
#         kalman_flag = 0 # カルマンフィルタによる補正が行われたかを判定するフラグ
        
#         # 現在のフレームまでのデータスライスを取得
#         current_cooredinate_L = coordinate_L[:i]
#         current_coordinate_R = coordinate_R[:i]

#         # 座標の差分(速度)を計算し、移動平均を適用
#         diff_data_Lx = np.diff(current_cooredinate_L)
#         yL = maf(diff_data_Lx, 3)
#         diff_data_Rx = np.diff(current_coordinate_R)
#         yR = maf(diff_data_Rx, 3)
        
#         if len(yL) < 2 or len(yR) < 2:
#             continue

#         # 最尤推定よりパラメータを求める
#         parL = initial_value
#         parR = initial_value
        
#         # パラメータを対数変換 (ψ_η, ψ_ε に変換)
#         psi_eta_L = np.log(np.sqrt(parL))
#         psi_eps_L = np.log(np.sqrt(parL))
#         psi_eta_R = np.log(np.sqrt(parR))
#         psi_eps_R = np.log(np.sqrt(parR))
        
#         # 探索するパラメータの初期値
#         x0L = [psi_eta_L, psi_eps_L]
#         x0R = [psi_eta_R, psi_eps_R]
        
#         # 最小化したい関数（散漫な対数尤度の最大化なので負号をつける）
#         fL = lambda xL: -calc_log_diffuse_llhd(xL, yL)
#         fR = lambda xR: -calc_log_diffuse_llhd(xR, yR)
        
        
#         # パラメータの探索範囲を制約する
#         bounds = ((-20, 20), (-20, 20)) 
        
#         # 最適化を実行 (準ニュートン法の一種であるL-BFGS-Bを使用)
#         resL = minimize(fL, x0L, method='L-BFGS-B', bounds=bounds)
#         resR = minimize(fR, x0R, method='L-BFGS-B', bounds=bounds)
        
#         # 最適化されたパラメータを取得
#         xoptL = resL.x
#         xoptR = resR.x
        
#         # 推定されたψをσ^2に戻す
#         var_eta_opt_L = np.exp(2 * xoptL[0])
#         var_eps_opt_L = np.exp(2 * xoptL[1])
#         var_eta_opt_R = np.exp(2 * xoptR[0])
#         var_eps_opt_R = np.exp(2 * xoptR[1])
        
#         # パラメータを更新(元々のやつ：epsとeta入れ替わってる？)
#         var_eps_L = var_eta_opt_L
#         var_eta_L = var_eps_opt_L
#         var_eps_R = var_eta_opt_R
#         var_eta_R = var_eps_opt_R
        
#         # もともとのやつ
#         a1L = var_eps_L
#         p1L = var_eta_L
#         a1R = var_eps_R
#         p1R = var_eta_R
        
#         # # パラメータを更新
#         # var_eps_L = var_eps_opt_L
#         # var_eta_L = var_eta_opt_L
#         # var_eps_R = var_eps_opt_R
#         # var_eta_R = var_eta_opt_R
        
#         # # L180-181の代わりに以下を使用
#         # a1L = yL[0] if len(yL) > 0 else 0  # yL（速度データ）の最初の値を初期状態とする
#         # p1L = var_eps_L                   # 初期状態の不確かさは観測ノイズ分散で代用
#         # a1R = yR[0] if len(yR) > 0 else 0
#         # p1R = var_eps_R

#         # カルマンフィルタを実行し、状態変数を取得
#         a_tt_L, _, _, _ = local_trend_kf(yL, a1L, p1L, var_eta_L, var_eps_L)
#         a_tt_R, _, _, _ = local_trend_kf(yR, a1R, p1R, var_eta_R, var_eps_R)
        
#         # 次の状態の予測値を取得
#         a_tt1L_end = a_tt_L[-2] if len(a_tt_L) > 1 else 0
#         a_tt1R_end = a_tt_R[-2] if len(a_tt_R) > 1 else 0
#         # a_tt1L_end = a_tt_L[-1] if len(a_tt_L) > 0 else 0
#         # a_tt1R_end = a_tt_R[-1] if len(a_tt_R) > 0 else 0
#         # a_tt1L_end = a_tt_L[-1] + var_eta_L if len(a_tt_L) > 0 else 0
#         # a_tt1R_end = a_tt_R[-1] + var_eta_R if len(a_tt_R) > 0 else 0

#         # 加速度で検出エラーの種類を判別後、補正
#         q1L = coordinate_L[i] - coordinate_L[i-1]
#         q2L = coordinate_L[i-1] - coordinate_L[i-2]
#         diff2_cankle_Lx_update = q1L - q2L
        
#         q1R = coordinate_R[i] - coordinate_R[i-1]
#         q2R = coordinate_R[i-1] - coordinate_R[i-2]
#         diff2_cankle_Rx_update = q1R - q2R

#         # パターン1: 左右両方の加速度が閾値を超えた場合 (入れ替わり or 両方誤検出)
#         if abs(diff2_cankle_Lx_update) > th and abs(diff2_cankle_Rx_update) > th:
#             # いったん左右の座標を入れ替えてみる
#             Lbox, Rbox = coordinate_L[i], coordinate_R[i]
#             coordinate_L[i], coordinate_R[i] = Rbox, Lbox
            
#             # 入れ替えた後、再度加速度を計算
#             q1L = coordinate_L[i] - coordinate_L[i-1]
#             q2L = coordinate_L[i-1] - coordinate_L[i-2]
#             diff2_cankle_Lx_update = q1L - q2L
            
#             q1R = coordinate_R[i] - coordinate_R[i-1]
#             q2R = coordinate_R[i-1] - coordinate_R[i-2]
#             diff2_cankle_Rx_update = q1R - q2R
            
#             # それでも両方の加速度が閾値を超えている場合 -> 両方誤検出と判断
#             if abs(diff2_cankle_Lx_update) > th and abs(diff2_cankle_Rx_update) > th:
#                 coordinate_L[i], coordinate_R[i] = Lbox, Rbox # 入れ替えを元に戻す
#                 coordinate_L[i] = coordinate_L[i-1] + a_tt1L_end # カルマンフィルタで予測した値で補正
#                 coordinate_R[i] = coordinate_R[i-1] + a_tt1R_end # カルマンフィルタで予測した値で補正
#                 miss_point[i] = 4 # 両方誤検出=4
#                 kalman_flag = 1
#             else: # 入れ替えたら加速度が閾値内に収まった -> 入れ替わりと判断
#                 miss_point[i] = 1 # 入れ替わり=1
#                 kalman_flag = 1
        
#         # パターン2: 左足のみ加速度が閾値を超えた場合
#         if abs(diff2_cankle_Lx_update) > th and abs(diff2_cankle_Rx_update) <= th:
#             coordinate_L[i] = coordinate_L[i-1] + a_tt1L_end # カルマンフィルタで予測した値で補正
#             miss_point[i] = 2 # 左足の誤検出=2
#             kalman_flag = 1
            
#         # パターン3: 右足のみ加速度が閾値を超えた場合
#         if abs(diff2_cankle_Lx_update) <= th and abs(diff2_cankle_Rx_update) > th:
#             coordinate_R[i] = coordinate_R[i-1] + a_tt1R_end # カルマンフィルタで予測した値で補正
#             miss_point[i] = 3 # 右足の誤検出=3
#             kalman_flag = 1
            
#         # 補正後の値が極端に飛びすぎないようにする追加の補正
#         p1L = coordinate_L[i] - coordinate_L[i-1]
#         p2L = coordinate_L[i-1] - coordinate_L[i-2]
#         diff2_cankle_Lx_update_cover = p1L - p2L
        
#         p1R = coordinate_R[i] - coordinate_R[i-1]
#         p2R = coordinate_R[i-1] - coordinate_R[i-2]
#         diff2_cankle_Rx_update_cover = p1R - p2R
        
#         th_cover = 500
#         if abs(diff2_cankle_Lx_update_cover) >= th_cover and kalman_flag == 1:
#             coordinate_L[i] = coordinate_L[i-1] + p2L
            
#         if abs(diff2_cankle_Rx_update_cover) >= th_cover and kalman_flag == 1:
#             coordinate_R[i] = coordinate_R[i-1] + p2R

#     return coordinate_L, coordinate_R


# [変更点 1] local_trend_kf が a_tt1 を返すように修正

def local_trend_kf_mod(y, a1, p1, var_eta, var_eps):
    """
    ローカルトレンドモデルのカルマンフィルタリングを行う関数
    (修正：a_tt1 を返すように変更)
    """
    L = len(y)
    
    # 計算結果を格納するための配列を事前に確保(Preallocation)
    a_tt1 = np.zeros(L + 1)
    a_tt1[0] = a1
    p_tt1 = np.zeros(L + 1)
    p_tt1[0] = p1
    v_t = np.zeros(L)
    f_t = np.zeros(L)
    a_tt = np.zeros(L)
    p_tt = np.zeros(L)
    k_t = np.zeros(L)
    
    # Filtering (フィルタリングのループ処理)
    for t in range(L):
        # Innovation (観測値と予測値の差)
        v_t[t] = y[t] - a_tt1[t]
        f_t[t] = p_tt1[t] + var_eps
        
        # Kalman gain (カルマンゲインの計算)
        if f_t[t] == 0 or not np.isfinite(f_t[t]):
            k_t[t] = 0
        else:
            k_t[t] = p_tt1[t] / f_t[t] 
            
        # Current state (現時刻の状態を更新)
        a_tt[t] = a_tt1[t] + k_t[t] * v_t[t]
        p_tt[t] = p_tt1[t] * (1 - k_t[t])
        
        # Next state (次時刻の状態を予測)
        a_tt1[t+1] = a_tt[t]
        p_tt1[t+1] = p_tt[t] + var_eta
        
    # ★ 変更点：a_tt1 (L+1次元) を返すように変更
    return a_tt, p_tt, f_t, v_t, a_tt1

# [変更点 2] 尤度関数が正しい初期化を使い、修正版KFを呼ぶように変更

def calc_log_diffuse_llhd_mod(vars, y):
    """
    ローカルトレンドモデルの散漫な対数尤度を求める関数
    (修正：正しい初期化を使用し、local_trend_kf_mod を呼ぶ)
    """
    psi_eta, psi_eps = vars
    var_eta = np.exp(2 * psi_eta)  # ψ_η を σ^2_η に戻す
    var_eps = np.exp(2 * psi_eps)  # ψ_ε を σ^2_ε に戻す
    L = len(y)
    
    if L < 2:
        return -np.inf # データが少なすぎて計算できない
        
    # ★ 変更点：正しい初期化
    a1 = y[0]
    p1 = var_eps
    
    # ★ 変更点：修正版のkfを呼び、a_tt1 は _ で受け取る
    _, _, f_t, v_t, _ = local_trend_kf_mod(y, a1, p1, var_eta, var_eps)
    
    # f_tのゼロや負の値、非数をチェック
    if np.any(f_t[1:] <= 0) or not np.all(np.isfinite(f_t[1:])):
        return -np.inf

    # 散漫対数尤度を計算
    tmp = np.sum(np.log(f_t[1:]) + v_t[1:]**2 / f_t[1:])
    log_ld = -0.5 * L * np.log(2 * np.pi) - 0.5 * tmp
    
    return log_ld

# [変更点 3] kalman2 関数を「推定部」と「補正部」に分離

def kalman2(coordinate_L, coordinate_R, th, initial_value):
    """
    二階差分カルマンフィルタ (設計見直し版)
    1. 最初に全データでパラメータを推定
    2. 最初に全データでフィルタリング(予測値)を実行
    3. ループ内では加速度判定と補正のみ行う
    """
    end_step = len(coordinate_R)
    
    # 元のコードのtypo(cooredinate_L)を修正
    coordinate_L_copy = coordinate_L.copy() 
    coordinate_R_copy = coordinate_R.copy()
    miss_point = np.zeros(end_step)
    
    # --- 1. ループの外でパラメータ推定とフィルタリングを実行 ---
    
    # 座標の差分(速度)を計算 (★ 変更点：mafを除去)
    yL = np.diff(coordinate_L)
    yR = np.diff(coordinate_R)
    
    if len(yL) < 2 or len(yR) < 2:
        warnings.warn("データが短すぎるため、kalman2処理をスキップします。")
        return coordinate_L, coordinate_R

    # パラメータ推定 (元のコードの推定部分を流用)
    parL = initial_value
    parR = initial_value
    psi_eta_L = np.log(np.sqrt(parL))
    psi_eps_L = np.log(np.sqrt(parL))
    psi_eta_R = np.log(np.sqrt(parR))
    psi_eps_R = np.log(np.sqrt(parR))
    
    x0L = [psi_eta_L, psi_eps_L]
    x0R = [psi_eta_R, psi_eps_R]
    
    # ★ 変更点：修正版の尤度関数を使用
    fL = lambda xL: -calc_log_diffuse_llhd_mod(xL, yL) 
    fR = lambda xR: -calc_log_diffuse_llhd_mod(xR, yR)
    
    bounds = ((-20, 20), (-20, 20)) 
    
    # ★ 変更点：minimizeをループの外で1回だけ実行
    resL = minimize(fL, x0L, method='L-BFGS-B', bounds=bounds)
    resR = minimize(fR, x0R, method='L-BFGS-B', bounds=bounds)
    
    xoptL = resL.x
    xoptR = resR.x
    
    # ★ 変更点：正しいパラメータの割り当て
    var_eta_L = np.exp(2 * xoptL[0])
    var_eps_L = np.exp(2 * xoptL[1])
    var_eta_R = np.exp(2 * xoptR[0])
    var_eps_R = np.exp(2 * xoptR[1])
    
    # ★ 変更点：正しい初期化
    a1L = yL[0]
    p1L = var_eps_L
    a1R = yR[0]
    p1R = var_eps_R

    # ★ 変更点：修正版のkfをループの外で1回だけ実行
    # a_tt1 (予測値の時系列) を受け取る
    _, _, _, _, a_tt1_L = local_trend_kf_mod(yL, a1L, p1L, var_eta_L, var_eps_L)
    _, _, _, _, a_tt1_R = local_trend_kf_mod(yR, a1R, p1R, var_eta_R, var_eps_R)
    
    # a_tt1_L/R は (速度データの長さ L) + 1 の長さを持つ。
    
    # --- 2. ループ内では加速度判定と補正のみ ---
    
    for i in range(2, end_step):
        kalman_flag = 0 
        
        # 加速度で検出エラーの種類を判別後、補正
        q1L = coordinate_L_copy[i] - coordinate_L_copy[i-1]
        q2L = coordinate_L_copy[i-1] - coordinate_L_copy[i-2]
        diff2_cankle_Lx_update = q1L - q2L
        
        q1R = coordinate_R_copy[i] - coordinate_R_copy[i-1]
        q2R = coordinate_R_copy[i-1] - coordinate_R_copy[i-2]
        diff2_cankle_Rx_update = q1R - q2R

        # 補正に使う予測速度を取得
        # フレームiの座標 = フレームi-1の座標 + 速度[i-1]
        # 速度[i-1] (yL[i-1]) の予測値は a_tt1_L[i-1] に格納されている
        pred_vel_L = a_tt1_L[i-1] if (i-1) < len(a_tt1_L) else yL[-1] # 配列外参照を防ぐ
        pred_vel_R = a_tt1_R[i-1] if (i-1) < len(a_tt1_R) else yR[-1]

        # パターン1: 左右両方の加速度が閾値を超えた場合
        if abs(diff2_cankle_Lx_update) > th and abs(diff2_cankle_Rx_update) > th:
            Lbox, Rbox = coordinate_L_copy[i], coordinate_R_copy[i]
            coordinate_L_copy[i], coordinate_R_copy[i] = Rbox, Lbox
            
            q1L = coordinate_L_copy[i] - coordinate_L_copy[i-1]
            q2L = coordinate_L_copy[i-1] - coordinate_L_copy[i-2]
            diff2_cankle_Lx_update = q1L - q2L
            
            q1R = coordinate_R_copy[i] - coordinate_R_copy[i-1]
            q2R = coordinate_R_copy[i-1] - coordinate_R_copy[i-2]
            diff2_cankle_Rx_update = q1R - q2R
            
            if abs(diff2_cankle_Lx_update) > th and abs(diff2_cankle_Rx_update) > th:
                coordinate_L_copy[i], coordinate_R_copy[i] = Lbox, Rbox
                # ★ 変更点：事前に計算した予測値で補正
                coordinate_L_copy[i] = coordinate_L_copy[i-1] + pred_vel_L
                coordinate_R_copy[i] = coordinate_R_copy[i-1] + pred_vel_R
                miss_point[i] = 4 
                kalman_flag = 1
            else: 
                miss_point[i] = 1 
                kalman_flag = 1
        
        # パターン2: 左足のみ
        if abs(diff2_cankle_Lx_update) > th and abs(diff2_cankle_Rx_update) <= th:
            # ★ 変更点：事前に計算した予測値で補正
            coordinate_L_copy[i] = coordinate_L_copy[i-1] + pred_vel_L
            miss_point[i] = 2 
            kalman_flag = 1
            
        # パターン3: 右足のみ
        if abs(diff2_cankle_Lx_update) <= th and abs(diff2_cankle_Rx_update) > th:
            # ★ 変更点：事前に計算した予測値で補正
            coordinate_R_copy[i] = coordinate_R_copy[i-1] + pred_vel_R
            miss_point[i] = 3 
            kalman_flag = 1
            
        # (以下、補正後の追加補正ロジックは変更なし)
        p1L = coordinate_L_copy[i] - coordinate_L_copy[i-1]
        p2L = coordinate_L_copy[i-1] - coordinate_L_copy[i-2]
        diff2_cankle_Lx_update_cover = p1L - p2L
        
        p1R = coordinate_R_copy[i] - coordinate_R_copy[i-1]
        p2R = coordinate_R_copy[i-1] - coordinate_R_copy[i-2]
        diff2_cankle_Rx_update_cover = p1R - p2R
        
        th_cover = 500
        if abs(diff2_cankle_Lx_update_cover) >= th_cover and kalman_flag == 1:
            coordinate_L_copy[i] = coordinate_L_copy[i-1] + p2L
            
        if abs(diff2_cankle_Rx_update_cover) >= th_cover and kalman_flag == 1:
            coordinate_R_copy[i] = coordinate_R_copy[i-1] + p2R

    return coordinate_L_copy, coordinate_R_copy


def bufilter(coordinate_R, coordinate_L):
    """
    バターワースフィルタ
    """
    order = 4      # 4次
    fs = 100.0     # サンプリング周波数 (Hz)
    fc = 12.0       # カットオフ周波数 (Hz) もともと6Hz
    b, a = butter(order, fc / (fs / 2), btype='low')
    # ゼロ位相フィルタリング（順方向と逆方向にフィルタをかける）
    fcoordinate_Rx = filtfilt(b, a, coordinate_R)
    fcoordinate_Lx = filtfilt(b, a, coordinate_L)
    return fcoordinate_Rx, fcoordinate_Lx

def kangle(Xk, Xhi, Xa, Yk, Yhi, Ya):
    """
    膝関節角度算出
    """
    vA = (Xk - Xhi) * (Xk - Xa)
    vB = (Yk - Yhi) * (Yk - Ya)
    vAvB = vA + vB
    Asize = np.sqrt((Xk - Xhi)**2 + (Yk - Yhi)**2)
    Bsize = np.sqrt((Xk - Xa)**2 + (Yk - Ya)**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = vAvB / (Asize * Bsize)
    cos_theta = np.nan_to_num(cos_theta)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = 180 - (180 / np.pi * np.arccos(cos_theta))
    return angle

def aangle(Xk, Xa, Xb, Xhe, Yk, Ya, Yb, Yhe):
    """
    足関節角度算出
    """
    vC = (Xk - Xa) * (Xb - Xhe)
    vD = (Yk - Ya) * (Yb - Yhe)
    vCvD = vC + vD
    Csize = np.sqrt((Xk - Xa)**2 + (Yk - Ya)**2)
    Dsize = np.sqrt((Xb - Xhe)**2 + (Yb - Yhe)**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = vCvD / (Csize * Dsize)
    cos_theta = np.nan_to_num(cos_theta)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = 90 - (180 / np.pi * np.arccos(cos_theta))
    return angle

def hangle(Xhi, Xk, Yhi, Yk):
    """
    股関節角度算出
    """
    Esize = np.sqrt((Xk - Xhi)**2 + (Yk - Yhi)**2)
    Ey = Yk - Yhi
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = -Ey / Esize
    cos_theta = np.nan_to_num(cos_theta)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    hip_angle = 180 - (180 / np.pi * np.arccos(cos_theta))
    
    angle_m = np.where(Xhi >= Xk, hip_angle, -1 * hip_angle)
    angle = -angle_m
    return angle
    
# =============================================================================
# %% メインスクリプト
# =============================================================================

# --- 1. openposeから得られた座標をエクセルから取得 ---
# ★ ユーザーはこれらのパスを自分の環境に合わせて変更する必要があります。
path_op = r'G:\gait_pattern\20250811_br\sub1\thera0-3\fl\json_excel' # OpenPoseの座標データ(Excel)があるパス
name_op_excel = 'openpose.xlsx'  # 処理対象のファイル名
full_path_op = os.path.join(path_op, name_op_excel)
name = os.path.splitext(name_op_excel)[0] # 拡張子を除いたファイル名を取得

# --- 結果保存用ディレクトリの作成 ---
output_dir = os.path.join(path_op, f"{name}_results")
os.makedirs(output_dir, exist_ok=True)
print(f"グラフは '{output_dir}' に保存されます。")

# 座標データをexcelから取得
# 右足
df_R = pd.read_excel(full_path_op, sheet_name='Sheet1', header=None)
ankle_R_x, ankle_R_y = df_R[4].values, df_R[5].values
knee_R_x, knee_R_y = df_R[2].values, df_R[3].values
hip_R_x, hip_R_y = df_R[0].values, df_R[1].values
bigtoe_R_x, bigtoe_R_y = df_R[6].values, df_R[7].values
heel_R_x, heel_R_y = df_R[8].values, df_R[9].values

# 左足
df_L = pd.read_excel(full_path_op, sheet_name='Sheet2', header=None)
ankle_L_x, ankle_L_y = df_L[4].values, df_L[5].values
knee_L_x, knee_L_y = df_L[2].values, df_L[3].values
hip_L_x, hip_L_y = df_L[0].values, df_L[1].values
bigtoe_L_x, bigtoe_L_y = df_L[6].values, df_L[7].values
heel_L_x, heel_L_y = df_L[8].values, df_L[9].values

# データをまとめる [右x, 左x, 右y, 左y] の順で格納
ankle = np.column_stack([ankle_R_x, ankle_L_x, ankle_R_y, ankle_L_y])
knee = np.column_stack([knee_R_x, knee_L_x, knee_R_y, knee_L_y])
hip = np.column_stack([hip_R_x, hip_L_x, hip_R_y, hip_L_y])
bigtoe = np.column_stack([bigtoe_R_x, bigtoe_L_x, bigtoe_R_y, bigtoe_L_y])
heel = np.column_stack([heel_R_x, heel_L_x, heel_R_y, heel_L_y])

# --- 3. 前後時間削除 ---
# timestep = 60.0  # OpenPose動画のフレームレート ★
# rtimestep = 1 / timestep

start_frame = 170 #FL約-2m地点
end_frame = 459 #FLの最大検出フレーム
# start_frame = 340
# end_frame = 440

# 一人歩行 1_0-3
# start_frame = int(943*0.6)
# end_frame = int(1400*0.6)

# # 2人歩行 1_1-1
# start_frame = int(1090*0.6)
# end_frame = int(1252*0.6)

print(f"データはフレーム {start_frame} から {end_frame} まで使用されます。")

# 座標データをカット(しない)
cankle = ankle[start_frame:end_frame]
cknee = knee[start_frame:end_frame]
chip = hip[start_frame:end_frame]
cbigtoe = bigtoe[start_frame:end_frame]
cheel = heel[start_frame:end_frame]

# 時間軸を作成 (MATLABの挙動と合わせるため、切り取った開始時間を加算)
# start_time = 0 * rtimestep
# ctime = np.arange(len(cankle)) * rtimestep + start_time
cframe = np.arange(len(cankle)) + start_frame

# --- 4. 補正前の加速度算出 & グラフ描画 ---
display_pre_correction_plots = True  # True:表示, False:非表示
if display_pre_correction_plots:
    print("補正前の座標と加速度グラフを作成中...")
    pre_correction_data = {
        'hip': chip, 'knee': cknee, 'ankle': cankle, 
        'bigtoe': cbigtoe, 'heel': cheel
    }
    
    for joint_name, data in pre_correction_data.items():
        # 速度を計算(データ数が1つ減る)
        vel_Rx = np.diff(data[:, 0])
        vel_Lx = np.diff(data[:, 1])
        vel_Ry = np.diff(data[:, 2])
        vel_Ly = np.diff(data[:, 3])
        cframe_v = cframe[:-1]  # 速度の長さに合わせる

        # 加速度を計算 (データ数が2つ減る)
        accel_Rx = np.diff(data[:, 0], 2)
        accel_Lx = np.diff(data[:, 1], 2)
        accel_Ry = np.diff(data[:, 2], 2)
        accel_Ly = np.diff(data[:, 3], 2)
        cframe_a = cframe[:-2] # 加速度の長さに合わせる

        # --- 座標のプロット ---
        plt.figure(figsize=(12, 7))
        plt.subplot(2, 1, 1)
        plt.plot(cframe, data[:, 0], label='Right X', color='red', alpha=0.8)
        plt.plot(cframe, data[:, 1], label='Left X', color='blue', alpha=0.8)
        plt.title(f'Pre-correction {joint_name.capitalize()} X Coordinates', fontsize=18)
        plt.ylabel('Coordinate [px]', fontsize=16)
        plt.xlabel('Frame [-]', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(cframe, data[:, 2], label='Right Y', color='red', alpha=0.8)
        plt.plot(cframe, data[:, 3], label='Left Y', color='blue', alpha=0.8)
        plt.title(f'Pre-correction {joint_name.capitalize()} Y Coordinates', fontsize=18)
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('Coordinate [px]', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pre_{joint_name}_coords.png'))
        plt.close()
        
        # --- 速度のプロット ---
        plt.figure(figsize=(12, 7))
        plt.subplot(2, 1, 1)
        plt.plot(cframe_v, vel_Rx, label='Right X Vel', color='red', alpha=0.8)
        plt.plot(cframe_v, vel_Lx, label='Left X Vel', color='blue', alpha=0.8)
        plt.title(f'Pre-correction {joint_name.capitalize()} X Velocity', fontsize=18)
        plt.ylabel('Velocity [px/s]', fontsize=16)  
        plt.xlabel('Frame [-]', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(cframe_v, vel_Ry, label='Right Y Vel', color='red', alpha=0.8)
        plt.plot(cframe_v, vel_Ly, label='Left Y Vel', color='blue', alpha=0.8)
        plt.title(f'Pre-correction {joint_name.capitalize()} Y Velocity', fontsize=18)
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('Velocity [px/s]', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pre_{joint_name}_vel.png'))
        plt.close()

        # --- 加速度のプロット ---
        plt.figure(figsize=(12, 7))
        plt.subplot(2, 1, 1)
        plt.plot(cframe_a, accel_Rx, label='Right X Accel', color='red', alpha=0.8)
        plt.plot(cframe_a, accel_Lx, label='Left X Accel', color='blue', alpha=0.8)
        plt.ylim(-1000,1000)
        plt.title(f'Pre-correction {joint_name.capitalize()} X Acceleration', fontsize=18)
        plt.ylabel('Acceleration [px/s²]', fontsize=16)
        plt.xlabel('Frame [-]', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(cframe_a, accel_Ry, label='Right Y Accel', color='red', alpha=0.8)
        plt.plot(cframe_a, accel_Ly, label='Left Y Accel', color='blue', alpha=0.8)
        plt.ylim(-100,100)
        plt.title(f'Pre-correction {joint_name.capitalize()} Y Acceleration', fontsize=18)
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('Acceleration [px/s²]', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pre_{joint_name}_accel.png'))
        plt.close()
    print("補正前の座標と加速度グラフの作成が完了しました。")

# --- 5. カルマンフィルタで補正 ---
# (左座標, 右座標, 加速度の閾値, カルマンの初期値) ★データによって閾値を変更する必要あり
print("カルマンフィルタを適用中...")
# FR #
# kankle_Lx, kankle_Rx = cankle[:, 1], cankle[:, 0]
# kankle_Ly, kankle_Ry = cankle[:, 3], cankle[:, 2]
# kknee_Lx, kknee_Rx = cknee[:, 1], cknee[:, 0]
# kknee_Ly, kknee_Ry = cknee[:, 3], cknee[:, 2]
# khip_Lx, khip_Rx = chip[:, 1], chip[:, 0]
# khip_Ly, khip_Ry = chip[:, 3], chip[:, 2]
# kbigtoe_Lx, kbigtoe_Rx = cbigtoe[:, 1], cbigtoe[:, 0]
# kbigtoe_Ly, kbigtoe_Ry = kalman2(cbigtoe[:, 3], cbigtoe[:, 2], 25, 0.003)  ##################
# kheel_Lx, kheel_Rx = cheel[:, 1], cheel[:, 0]
# kheel_Ly, kheel_Ry = cheel[:, 3], cheel[:, 2]

# kankle_Lx, kankle_Rx = kalman2(cankle[:, 1], cankle[:, 0], 200, 0.0005)
# kankle_Ly, kankle_Ry = kalman2(cankle[:, 3], cankle[:, 2], 50, 0.003)
# kknee_Lx, kknee_Rx = kalman2(cknee[:, 1], cknee[:, 0], 200, 0.008)
# kknee_Ly, kknee_Ry = kalman2(cknee[:, 3], cknee[:, 2], 50, 0.002)
# khip_Lx, khip_Rx = kalman2(chip[:, 1], chip[:, 0], 50, 0.005)
# khip_Ly, khip_Ry = kalman2(chip[:, 3], chip[:, 2], 50, 0.003)
# kbigtoe_Lx, kbigtoe_Rx = kalman2(cbigtoe[:, 1], cbigtoe[:, 0], 200, 0.008)
# kbigtoe_Ly, kbigtoe_Ry = kalman2(cbigtoe[:, 3], cbigtoe[:, 2], 50, 0.003)
# kheel_Lx, kheel_Rx = kalman2(cheel[:, 1], cheel[:, 0], 200, 0.08)
# kheel_Ly, kheel_Ry = kalman2(cheel[:, 3], cheel[:, 2], 50, 0.003)

"""
値調整中 bigtoe_y以外はうまくいく
"""
# kankle_Lx, kankle_Rx = cankle[:, 1], cankle[:, 0]
# kankle_Ly, kankle_Ry = cankle[:, 3], cankle[:, 2]
# kknee_Lx, kknee_Rx = cknee[:, 1], cknee[:, 0]
# kknee_Ly, kknee_Ry = cknee[:, 3], cknee[:, 2]
# khip_Lx, khip_Rx = chip[:, 1], chip[:, 0]
# khip_Ly, khip_Ry = chip[:, 3], chip[:, 2]
# kbigtoe_Lx, kbigtoe_Rx = cbigtoe[:, 1], cbigtoe[:, 0]
# kbigtoe_Ly, kbigtoe_Ry = kalman2(cbigtoe[:, 3], cbigtoe[:, 2], 40, 0.001)
# print("カルマンフィルタ: 右母趾Y座標補正完了")
# kheel_Lx, kheel_Rx = cheel[:, 1], cheel[:, 0]
# kheel_Ly, kheel_Ry = cheel[:, 3], cheel[:, 2]

# kankle_Lx, kankle_Rx = kalman2(cankle[:, 1], cankle[:, 0], 200, 0.1)
# print("カルマンフィルタ: 右足首X座標補正完了")
# kankle_Ly, kankle_Ry = kalman2(cankle[:, 3], cankle[:, 2], 50, 0.1)
# print("カルマンフィルタ: 右足首Y座標補正完了")
# kknee_Lx, kknee_Rx = kalman2(cknee[:, 1], cknee[:, 0], 200, 0.1)
# print("カルマンフィルタ: 右膝X座標補正完了")
# kknee_Ly, kknee_Ry = kalman2(cknee[:, 3], cknee[:, 2], 50, 0.1)
# print("カルマンフィルタ: 右膝Y座標補正完了")
# khip_Lx, khip_Rx = kalman2(chip[:, 1], chip[:, 0], 50, 0.1)
# print("カルマンフィルタ: 右股関節X座標補正完了")
# khip_Ly, khip_Ry = kalman2(chip[:, 3], chip[:, 2], 50, 0.1)
# print("カルマンフィルタ: 右股関節Y座標補正完了")
# kbigtoe_Lx, kbigtoe_Rx = kalman2(cbigtoe[:, 1], cbigtoe[:, 0], 200, 0.1)
# print("カルマンフィルタ: 右母趾X座標補正完了")
# kbigtoe_Ly, kbigtoe_Ry = kalman2(cbigtoe[:, 3], cbigtoe[:, 2], 40, 0.1)
# print("カルマンフィルタ: 右母趾Y座標補正完了")
# kheel_Lx, kheel_Rx = kalman2(cheel[:, 1], cheel[:, 0], 200, 0.1)
# print("カルマンフィルタ: 右踵X座標補正完了")
# kheel_Ly, kheel_Ry = kalman2(cheel[:, 3], cheel[:, 2], 50, 0.1)
# print("カルマンフィルタ: 右踵Y座標補正完了")






# # FL#
# kankle_Lx, kankle_Rx = cankle[:, 1], cankle[:, 0]
# # kankle_Lx, kankle_Rx = kalman2(cankle[:, 1], cankle[:, 0], 200, 0.0005)
# kankle_Ly, kankle_Ry = cankle[:, 3], cankle[:, 2]
# kknee_Lx, kknee_Rx = cknee[:, 1], cknee[:, 0]
# kknee_Ly, kknee_Ry = cknee[:, 3], cknee[:, 2]
# khip_Lx, khip_Rx = chip[:, 1], chip[:, 0]
# khip_Ly, khip_Ry = chip[:, 3], chip[:, 2]
# kbigtoe_Lx, kbigtoe_Rx = cbigtoe[:, 1], cbigtoe[:, 0]
# # kbigtoe_Ly, kbigtoe_Ry = cbigtoe[:, 3], cbigtoe[:, 2]
# kbigtoe_Ly, kbigtoe_Ry = kalman2(cbigtoe[:, 3], cbigtoe[:, 2], 40, 0.003)  ##################
# kheel_Lx, kheel_Rx = cheel[:, 1], cheel[:, 0]
# kheel_Ly, kheel_Ry = kalman2(cheel[:, 3], cheel[:, 2], 50, 0.003)
# # kheel_Ly, kheel_Ry = cheel[:, 3], cheel[:, 2]


# kankle_Lx, kankle_Rx = kalman2(cankle[:, 1], cankle[:, 0], 200, 0.0005)
# print("カルマンフィルタ: 右足首X座標補正完了")
# kankle_Ly, kankle_Ry = kalman2(cankle[:, 3], cankle[:, 2], 50, 0.003)
# print("カルマンフィルタ: 右足首Y座標補正完了")
# kknee_Lx, kknee_Rx = kalman2(cknee[:, 1], cknee[:, 0], 200, 0.008)
# print("カルマンフィルタ: 右膝X座標補正完了")
# kknee_Ly, kknee_Ry = kalman2(cknee[:, 3], cknee[:, 2], 50, 0.002)
# print("カルマンフィルタ: 右膝Y座標補正完了")
# khip_Lx, khip_Rx = kalman2(chip[:, 1], chip[:, 0], 50, 0.005)
# print("カルマンフィルタ: 右股関節X座標補正完了")
# khip_Ly, khip_Ry = kalman2(chip[:, 3], chip[:, 2], 50, 0.003)
# print("カルマンフィルタ: 右股関節Y座標補正完了")
# kbigtoe_Lx, kbigtoe_Rx = kalman2(cbigtoe[:, 1], cbigtoe[:, 0], 200, 0.008)
# print("カルマンフィルタ: 右母趾X座標補正完了")
# kbigtoe_Ly, kbigtoe_Ry = kalman2(cbigtoe[:, 3], cbigtoe[:, 2], 40, 0.003)
# print("カルマンフィルタ: 右母趾Y座標補正完了")
# kheel_Lx, kheel_Rx = kalman2(cheel[:, 1], cheel[:, 0], 200, 0.08)
# print("カルマンフィルタ: 右踵X座標補正完了")
# kheel_Ly, kheel_Ry = kalman2(cheel[:, 3], cheel[:, 2], 50, 0.003)
# print("カルマンフィルタ: 右踵Y座標補正完了")

"""
この値でいけた
"""
kankle_Lx, kankle_Rx = kalman2(cankle[:, 1], cankle[:, 0], 200, 0.1)
print("カルマンフィルタ: 右足首X座標補正完了")
kankle_Ly, kankle_Ry = kalman2(cankle[:, 3], cankle[:, 2], 50, 0.1)
print("カルマンフィルタ: 右足首Y座標補正完了")
kknee_Lx, kknee_Rx = kalman2(cknee[:, 1], cknee[:, 0], 200, 0.1)
print("カルマンフィルタ: 右膝X座標補正完了")
kknee_Ly, kknee_Ry = kalman2(cknee[:, 3], cknee[:, 2], 50, 0.1)
print("カルマンフィルタ: 右膝Y座標補正完了")
khip_Lx, khip_Rx = kalman2(chip[:, 1], chip[:, 0], 50, 0.1)
print("カルマンフィルタ: 右股関節X座標補正完了")
khip_Ly, khip_Ry = kalman2(chip[:, 3], chip[:, 2], 50, 0.1)
print("カルマンフィルタ: 右股関節Y座標補正完了")
kbigtoe_Lx, kbigtoe_Rx = kalman2(cbigtoe[:, 1], cbigtoe[:, 0], 200, 0.1)
print("カルマンフィルタ: 右母趾X座標補正完了")
kbigtoe_Ly, kbigtoe_Ry = kalman2(cbigtoe[:, 3], cbigtoe[:, 2], 40, 0.1)
print("カルマンフィルタ: 右母趾Y座標補正完了")
kheel_Lx, kheel_Rx = kalman2(cheel[:, 1], cheel[:, 0], 200, 0.1)
print("カルマンフィルタ: 右踵X座標補正完了")
kheel_Ly, kheel_Ry = kalman2(cheel[:, 3], cheel[:, 2], 50, 0.1)
print("カルマンフィルタ: 右踵Y座標補正完了")


# --- 6. バターワースフィルタ ---
# 4次バターワースフィルタ (カットオフ周波数 6Hz)
kankle_Rx_filter, kankle_Lx_filter = bufilter(kankle_Rx, kankle_Lx)
kankle_Ry_filter, kankle_Ly_filter = bufilter(kankle_Ry, kankle_Ly)
kknee_Rx_filter, kknee_Lx_filter = bufilter(kknee_Rx, kknee_Lx)
kknee_Ry_filter, kknee_Ly_filter = bufilter(kknee_Ry, kknee_Ly)
khip_Rx_filter, khip_Lx_filter = bufilter(khip_Rx, khip_Lx)
khip_Ry_filter, khip_Ly_filter = bufilter(khip_Ry, khip_Ly)
kbigtoe_Rx_filter, kbigtoe_Lx_filter = bufilter(kbigtoe_Rx, kbigtoe_Lx)
kbigtoe_Ry_filter, kbigtoe_Ly_filter = bufilter(kbigtoe_Ry, kbigtoe_Ly)
kheel_Rx_filter, kheel_Lx_filter = bufilter(kheel_Rx, kheel_Lx)
kheel_Ry_filter, kheel_Ly_filter = bufilter(kheel_Ry, kheel_Ly)

# --- 7. 関節角度計算 ---
bfilter_knee_angle_R = kangle(kknee_Rx_filter, khip_Rx_filter, kankle_Rx_filter, kknee_Ry_filter, khip_Ry_filter, kankle_Ry_filter)
bfilter_knee_angle_L = kangle(kknee_Lx_filter, khip_Lx_filter, kankle_Lx_filter, kknee_Ly_filter, khip_Ly_filter, kankle_Ly_filter)
bfilter_ankle_angle_R = aangle(kknee_Rx_filter, kankle_Rx_filter, kbigtoe_Rx_filter, kheel_Rx_filter, kknee_Ry_filter, kankle_Ry_filter, kbigtoe_Ry_filter, kheel_Ry_filter)
bfilter_ankle_angle_L = aangle(kknee_Lx_filter, kankle_Lx_filter, kbigtoe_Lx_filter, kheel_Lx_filter, kknee_Ly_filter, kankle_Ly_filter, kbigtoe_Ly_filter, kheel_Ly_filter)
bfilter_hip_angle_R = hangle(khip_Rx_filter, kknee_Rx_filter, khip_Ry_filter, kknee_Ry_filter)
bfilter_hip_angle_L = hangle(khip_Lx_filter, kknee_Lx_filter, khip_Ly_filter, kknee_Ly_filter)

# --- 8. 結果表示 ---
# バターワースフィルタの端点効果を除くため、最初と最後の30フレームをカットして表示
start_index = 0
end_index = len(cframe)
# start_index = 60
# end_index = len(cframe) - 60

# --- 8.1 関節角度のグラフ描画 & 保存 ---
display_angle_plots = False
if display_angle_plots:
    print("最終的な関節角度グラフを作成中...")
    # --- 右膝関節角 ---
    plt.figure(figsize=(10, 6))
    plt.plot(cframe[start_index:end_index], bfilter_knee_angle_R[start_index:end_index], linewidth=1.5, color='#fc9d03', label='Proposed Method')
    plt.fill_between(cframe[start_index:end_index], bfilter_knee_angle_R[start_index:end_index] - 5, bfilter_knee_angle_R[start_index:end_index] + 5, color='#fc9d03', alpha=0.3)
    plt.xlabel('Frame [-]', fontsize=16)
    plt.ylabel('Knee Joint Angle_R [°]', fontsize=16)
    plt.title('Right Knee Joint Angle', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim([-10, 80])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'angle_knee_right.png'))
    plt.close()

    # --- 左膝関節角の描画 ---
    plt.figure(figsize=(10, 6))
    plt.plot(cframe[start_index:end_index], bfilter_knee_angle_L[start_index:end_index], linewidth=1.5, color='#fc9d03', label='Proposed Method')
    plt.fill_between(cframe[start_index:end_index], bfilter_knee_angle_L[start_index:end_index] - 5, bfilter_knee_angle_L[start_index:end_index] + 5, color='#fc9d03', alpha=0.3)
    plt.xlabel('Frame [-]', fontsize=16)
    plt.ylabel('Knee Joint Angle_L [°]', fontsize=16)
    plt.title('Left Knee Joint Angle', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim([-10, 80])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'angle_knee_left.png'))
    plt.close()

    # --- 右足関節角度 ---
    plt.figure(figsize=(10, 6))
    plt.plot(cframe[start_index:end_index], bfilter_ankle_angle_R[start_index:end_index], linewidth=1.5, color='#fc9d03', label='Proposed Method')
    plt.fill_between(cframe[start_index:end_index], bfilter_ankle_angle_R[start_index:end_index] - 5, bfilter_ankle_angle_R[start_index:end_index] + 5, color='#fc9d03', alpha=0.3)
    plt.xlabel('Frame [-]', fontsize=16)
    plt.ylabel('Ankle Joint Angle_R [°]', fontsize=16)
    plt.title('Right Ankle Joint Angle', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim([-30, 30])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'angle_ankle_right.png'))
    plt.close()

    # --- 左足関節角度 ---
    plt.figure(figsize=(10, 6))
    plt.plot(cframe[start_index:end_index], bfilter_ankle_angle_L[start_index:end_index], linewidth=1.5, color='#fc9d03', label='Proposed Method')
    plt.fill_between(cframe[start_index:end_index], bfilter_ankle_angle_L[start_index:end_index] - 5, bfilter_ankle_angle_L[start_index:end_index] + 5, color='#fc9d03', alpha=0.3)
    plt.xlabel('Frame [-]', fontsize=16)
    plt.ylabel('Ankle Joint Angle_L [°]', fontsize=16)
    plt.title('Left Ankle Joint Angle', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim([-30, 30])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'angle_ankle_left.png'))
    plt.close()
    
    # --- 右股関節角度 ---
    plt.figure(figsize=(10, 6))
    plt.plot(cframe[start_index:end_index], bfilter_hip_angle_R[start_index:end_index], linewidth=1.5, color='#fc9d03', label='Proposed Method')
    plt.fill_between(cframe[start_index:end_index], bfilter_hip_angle_R[start_index:end_index] - 5, bfilter_hip_angle_R[start_index:end_index] + 5, color='#fc9d03', alpha=0.3)
    plt.xlabel('Frame [-]', fontsize=16)
    plt.ylabel('Hip Joint Angle_R [°]', fontsize=16)
    plt.title('Right Hip Joint Angle', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim([-30, 40])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'angle_hip_right.png'))
    plt.close()

    # --- 左股関節角度の描画 ---
    plt.figure(figsize=(10, 6))
    plt.plot(cframe[start_index:end_index], bfilter_hip_angle_L[start_index:end_index], linewidth=1.5, color='#fc9d03', label='Proposed Method')
    plt.fill_between(cframe[start_index:end_index], bfilter_hip_angle_L[start_index:end_index] - 5, bfilter_hip_angle_L[start_index:end_index] + 5, color='#fc9d03', alpha=0.3)
    plt.xlabel('Frame [-]', fontsize=16)
    plt.ylabel('Hip Joint Angle_L [°]', fontsize=16)
    plt.title('Left Hip Joint Angle', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim([-30, 40])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'angle_hip_left.png'))
    plt.close()

# --- 8.2 最終的な座標データの描画 & 保存 ---
display_coordinates = True
if display_coordinates:
    # 描画対象のデータを辞書にまとめる
    plot_data = {
        'hip': {'raw': chip, 'kalman_Rx':khip_Rx,'kalman_Lx':khip_Lx,'kalman_Ry':khip_Ry,'kalman_Ly':khip_Ly,'filtered_Rx': khip_Rx_filter, 'filtered_Lx': khip_Lx_filter, 'filtered_Ry': khip_Ry_filter, 'filtered_Ly': khip_Ly_filter},
        'knee': {'raw': cknee, 'kalman_Rx':kknee_Rx,'kalman_Lx':kknee_Lx,'kalman_Ry':kknee_Ry,'kalman_Ly':kknee_Ly,'filtered_Rx': kknee_Rx_filter, 'filtered_Lx': kknee_Lx_filter, 'filtered_Ry': kknee_Ry_filter, 'filtered_Ly': kknee_Ly_filter},
        'ankle': {'raw': cankle, 'kalman_Rx':kankle_Rx,'kalman_Lx':kankle_Lx,'kalman_Ry':kankle_Ry,'kalman_Ly':kankle_Ly,'filtered_Rx': kankle_Rx_filter, 'filtered_Lx': kankle_Lx_filter, 'filtered_Ry': kankle_Ry_filter, 'filtered_Ly': kankle_Ly_filter},
        'bigtoe': {'raw': cbigtoe, 'kalman_Rx':kbigtoe_Rx,'kalman_Lx':kbigtoe_Lx,'kalman_Ry':kbigtoe_Ry,'kalman_Ly':kbigtoe_Ly,'filtered_Rx': kbigtoe_Rx_filter, 'filtered_Lx': kbigtoe_Lx_filter, 'filtered_Ry': kbigtoe_Ry_filter, 'filtered_Ly': kbigtoe_Ly_filter},
        'heel': {'raw': cheel, 'kalman_Rx':kheel_Rx,'kalman_Lx':kheel_Lx,'kalman_Ry':kheel_Ry,'kalman_Ly':kheel_Ly,'filtered_Rx': kheel_Rx_filter, 'filtered_Lx': kheel_Lx_filter, 'filtered_Ry': kheel_Ry_filter, 'filtered_Ly': kheel_Ly_filter},
    }

    for joint_name, data in plot_data.items():
        # --- X座標のプロット ---
        plt.figure(figsize=(10, 6))
        plt.plot(cframe[start_index:end_index], data['raw'][start_index:end_index, 0], color='r', label='Raw Right', alpha=0.3)
        plt.plot(cframe[start_index:end_index], data['raw'][start_index:end_index, 1], color='b', label='Raw Left', alpha=0.3)
        plt.plot(cframe[start_index:end_index], data['kalman_Rx'][start_index:end_index], color='r', label='Kalman Right')
        plt.plot(cframe[start_index:end_index], data['kalman_Lx'][start_index:end_index], color='b', label='Kalman Left')
        # plt.plot(cframe[start_index:end_index], data['filtered_Rx'][start_index:end_index], color='r', label='Filtered Right')
        # plt.plot(cframe[start_index:end_index], data['filtered_Lx'][start_index:end_index], color='b', label='Filtered Left')
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('X Coordinate [px]', fontsize=16)
        plt.title(f'{joint_name.capitalize()} X Coordinate', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'coord_{joint_name}_x.png'))
        plt.close()

        # --- Y座標のプロット ---
        plt.figure(figsize=(10, 6))
        plt.plot(cframe[start_index:end_index], data['raw'][start_index:end_index, 2], color='r', label='Raw Right', alpha=0.3)
        plt.plot(cframe[start_index:end_index], data['raw'][start_index:end_index, 3], color='b', label='Raw Left', alpha=0.3)
        plt.plot(cframe[start_index:end_index], data['kalman_Ry'][start_index:end_index], color='r', label='Kalman Right')
        plt.plot(cframe[start_index:end_index], data['kalman_Ly'][start_index:end_index], color='b', label='Kalman Left')
        # plt.plot(cframe[start_index:end_index], data['filtered_Ry'][start_index:end_index], color='r', label='Filtered Right')
        # plt.plot(cframe[start_index:end_index], data['filtered_Ly'][start_index:end_index], color='b', label='Filtered Left')
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('Y Coordinate [px]', fontsize=16)
        plt.title(f'{joint_name.capitalize()} Y Coordinate', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'coord_{joint_name}_y.png'))
        plt.close()

print("\n処理が完了し、すべてのグラフが保存されました。")