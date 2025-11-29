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
うまくいかなかった
運動が加速度的であることを考慮して、ローカルリニアトレンドモデル(LLT)を用いたカルマンフィルタを実装してみた場合
LLTなどの理解を深める必要あり
"""


# =============================================================================
# %% 関数定義セクション
# =============================================================================

def local_linear_trend_kf(y, a1, p1, var_eta, var_zeta, var_eps):
    """
    ローカルリニアトレンドモデルのカルマンフィルタリング
    状態ベクトル: [速度, 加速度]
    
    モデル:
    y_t = [1, 0] * [vel_t, acc_t]^T + eps_t   (観測方程式)
    
    [vel_t+1, acc_t+1]^T = [[1, 1], [0, 1]] * [vel_t, acc_t]^T + [eta_t, zeta_t]^T (状態方程式)
    """
    L = len(y)
    m = len(a1) # 状態の次元 (m=2)
    
    # モデルの行列 (固定)
    Z = np.array([1.0, 0.0])              # 観測行列 (速度のみ観測)
    T = np.array([[1.0, 1.0], [0.0, 1.0]]) # 遷移行列
    Q = np.diag([var_eta, var_zeta])    # プロセスノイズ (速度, 加速度)
    H = var_eps                         # 観測ノイズ
    
    # 結果格納用
    a_tt1 = np.zeros((L + 1, m))      # 1期先予測状態 a_t|t-1
    a_tt1[0, :] = a1
    p_tt1 = np.zeros((L + 1, m, m))   # 1期先予測誤差分散 P_t|t-1
    p_tt1[0, :, :] = p1
    v_t = np.zeros(L)                 # 予測誤差
    f_t = np.zeros(L)                 # 予測誤差分散
    a_tt = np.zeros((L, m))           # フィルタリング状態 a_t|t
    p_tt = np.zeros((L, m, m))        # フィルタリング誤差分散 P_t|t
    
    for t in range(L):
        # 予測ステップで使う値
        a_pred = a_tt1[t, :]
        p_pred = p_tt1[t, :, :]
        
        # 観測誤差(Innovation)の計算
        v_t[t] = y[t] - Z.dot(a_pred)
        f_t[t] = Z.dot(p_pred).dot(Z.T) + H
        
        if f_t[t] <= 0 or not np.isfinite(f_t[t]):
            # 予測誤差分散が非正定値など -> ゲイン0で更新スキップ
            a_tt[t, :] = a_pred
            p_tt[t, :, :] = p_pred
        else:
            # カルマンゲインの計算 (K_t = P_t|t-1 * Z^T * F_t^-1)
            k_t_vec = p_pred.dot(Z.T) / f_t[t] # (m x 1)
            
            # 更新(フィルタリング)ステップ
            a_tt[t, :] = a_pred + k_t_vec * v_t[t]
            p_tt[t, :, :] = p_pred - np.outer(k_t_vec, k_t_vec) * f_t[t]
            
        # 次状態の予測ステップ
        a_tt1[t+1, :] = T.dot(a_tt[t, :])
        p_tt1[t+1, :, :] = T.dot(p_tt[t, :, :]).dot(T.T) + Q
        
    return a_tt, p_tt, f_t, v_t, a_tt1

def calc_log_linear_trend_llhd(vars, y):
    """
    ローカルリニアトレンドモデルの対数尤度
    vars = [psi_eta, psi_zeta, psi_eps]
    """
    if len(vars) != 3:
        raise ValueError("vars must have 3 elements: [psi_eta, psi_zeta, psi_eps]")
        
    psi_eta, psi_zeta, psi_eps = vars
    var_eta = np.exp(2 * psi_eta)   # 速度のプロセスノイズ
    var_zeta = np.exp(2 * psi_zeta) # 加速度のプロセスノイズ
    var_eps = np.exp(2 * psi_eps)   # 観測ノイズ
    
    L = len(y)
    
    if L < 3: # 少なくとも3点(速度2点)ないと加速度が計算できない
        return -np.inf 
        
    # 初期値の設定 (簡易的な方法)
    # a1 = [初期速度, 初期加速度]
    a1 = np.array([y[1], y[1] - y[0]]) 
    # p1 = 初期誤差分散 (対角行列で仮定)
    p1 = np.diag([var_eps, var_eps]) 
    
    # カルマンフィルタリングを実行
    _, _, f_t, v_t, _ = local_linear_trend_kf(y, a1, p1, var_eta, var_zeta, var_eps)
    
    # f_tのチェック (インデックス[2:]から尤度計算)
    if np.any(f_t[2:] <= 0) or not np.all(np.isfinite(f_t[2:])):
        return -np.inf

    # 対数尤度を計算 (最初の2点は初期化に使ったので飛ばす)
    tmp = np.sum(np.log(f_t[2:]) + v_t[2:]**2 / f_t[2:])
    log_ld = -0.5 * (L - 2) * np.log(2 * np.pi) - 0.5 * tmp
    
    # 尤度が計算できない場合は-infを返す
    if not np.isfinite(log_ld):
        return -np.inf
        
    return log_ld

def kalman2(coordinate_L, coordinate_R, th, initial_value):
    """
    二階差分カルマンフィルタ (ローカルリニアトレンド版)
    1. 最初に全データでパラメータ(3つ)を推定
    2. 最初に全データでLLTフィルタリング(予測値)を実行
    3. ループ内では加速度判定と補正のみ行う
    """
    end_step = len(coordinate_R)
    
    coordinate_L_copy = coordinate_L.copy() 
    coordinate_R_copy = coordinate_R.copy()
    miss_point = np.zeros(end_step)
    
    # --- 1. ループの外でパラメータ推定とフィルタリングを実行 ---
    
    # 速度データ (mafなし)
    yL = np.diff(coordinate_L)
    yR = np.diff(coordinate_R)
    
    if len(yL) < 3 or len(yR) < 3: # LLTは最低3点(速度2点)必要
        warnings.warn("データが短すぎるため、kalman2_llt処理をスキップします。")
        return coordinate_L, coordinate_R

    # パラメータ推定 (3次元に変更)
    par = initial_value
    
    # 3つのパラメータの初期値を設定
    psi_eta_L = np.log(np.sqrt(par))
    psi_zeta_L = np.log(np.sqrt(par)) # ★ 追加
    psi_eps_L = np.log(np.sqrt(par))
    psi_eta_R = np.log(np.sqrt(par))
    psi_zeta_R = np.log(np.sqrt(par)) # ★ 追加
    psi_eps_R = np.log(np.sqrt(par))
    
    x0L = [psi_eta_L, psi_zeta_L, psi_eps_L] # ★ 3次元
    x0R = [psi_eta_R, psi_zeta_R, psi_eps_R] # ★ 3次元
    
    # ★ 変更点：LLT版の尤度関数を使用
    fL = lambda xL: -calc_log_linear_trend_llhd(xL, yL) 
    fR = lambda xR: -calc_log_linear_trend_llhd(xR, yR)
    
    bounds = ((-20, 20), (-20, 20), (-20, 20)) # ★ 3次元
    
    resL = minimize(fL, x0L, method='L-BFGS-B', bounds=bounds)
    resR = minimize(fR, x0R, method='L-BFGS-B', bounds=bounds)
    
    if not resL.success:
        warnings.warn(f"L側 kalman2_llt のパラメータ推定に失敗: {resL.message}")
    if not resR.success:
        warnings.warn(f"R側 kalman2_llt のパラメータ推定に失敗: {resR.message}")

    xoptL = resL.x
    xoptR = resR.x
    
    # ★ 変更点：3つのパラメータを取得
    var_eta_L = np.exp(2 * xoptL[0])
    var_zeta_L = np.exp(2 * xoptL[1])
    var_eps_L = np.exp(2 * xoptL[2])
    var_eta_R = np.exp(2 * xoptR[0])
    var_zeta_R = np.exp(2 * xoptR[1])
    var_eps_R = np.exp(2 * xoptR[2])
    
    # ★ 変更点：LLT版の初期化
    a1L = np.array([yL[1], yL[1] - yL[0]])
    p1L = np.diag([var_eps_L, var_eps_L]) # 簡略化
    a1R = np.array([yR[1], yR[1] - yR[0]])
    p1R = np.diag([var_eps_R, var_eps_R]) # 簡略化

    # ★ 変更点：LLT版のkfを実行
    _, _, _, _, a_tt1_L = local_linear_trend_kf(yL, a1L, p1L, var_eta_L, var_zeta_L, var_eps_L)
    _, _, _, _, a_tt1_R = local_linear_trend_kf(yR, a1R, p1R, var_eta_R, var_zeta_R, var_eps_R)
    
    # a_tt1_L/R は (速度データの長さ L) + 1 の長さで、2列(速度, 加速度)を持つ
    
    # --- 2. ループ内では加速度判定と補正のみ ---
    
    for i in range(2, end_step):
        kalman_flag = 0 
        
        # 観測された加速度
        q1L = coordinate_L_copy[i] - coordinate_L_copy[i-1]
        q2L = coordinate_L_copy[i-1] - coordinate_L_copy[i-2]
        diff2_cankle_Lx_update = q1L - q2L
        
        q1R = coordinate_R_copy[i] - coordinate_R_copy[i-1]
        q2R = coordinate_R_copy[i-1] - coordinate_R_copy[i-2]
        diff2_cankle_Rx_update = q1R - q2R

        # ★ 変更点：予測値の取得
        # 補正に使う予測速度を取得
        # フレームiの座標 = フレームi-1の座標 + 速度[i-1]
        # 速度[i-1] (yL[i-1]) の予測値は a_tt1_L[i-1, 0] に格納されている
        
        idx = i - 1 # yL/yR のインデックス (diffで1つ減っているため)
        if idx < len(a_tt1_L):
            pred_vel_L = a_tt1_L[idx, 0] # 予測された速度 [vel_t|t-1]
        else:
            pred_vel_L = yL[-1] # 配列外参照を防ぐ (フォールバック)
            
        if idx < len(a_tt1_R):
            pred_vel_R = a_tt1_R[idx, 0]
        else:
            pred_vel_R = yR[-1] # フォールバック

        # (補正ロジック自体は変更なし。予測値 pred_vel が改善されていることを期待)
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
                coordinate_L_copy[i] = coordinate_L_copy[i-1] + pred_vel_L
                coordinate_R_copy[i] = coordinate_R_copy[i-1] + pred_vel_R
                miss_point[i] = 4 
                kalman_flag = 1
            else: 
                miss_point[i] = 1 
                kalman_flag = 1
        
        # パターン2: 左足のみ
        if abs(diff2_cankle_Lx_update) > th and abs(diff2_cankle_Rx_update) <= th:
            coordinate_L_copy[i] = coordinate_L_copy[i-1] + pred_vel_L
            miss_point[i] = 2 
            kalman_flag = 1
            
        # パターン3: 右足のみ
        if abs(diff2_cankle_Lx_update) <= th and abs(diff2_cankle_Rx_update) > th:
            coordinate_R_copy[i] = coordinate_R_copy[i-1] + pred_vel_R
            miss_point[i] = 3 
            kalman_flag = 1
            
        # (追加補正ロジックも変更なし)
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