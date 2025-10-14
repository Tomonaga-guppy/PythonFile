import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize
import os

### 警告文を非表示にする (ゼロ除算やNaNに関する警告)
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# %% 関数定義セクション
# =============================================================================

def local_trend_kf(y, a1, p1, var_eta, var_eps):
    """
    ローカルトレンドモデルのカルマンフィルタリングを行う関数
    """
    L = len(y)
    
    # Preallocation
    a_tt1 = np.zeros(L + 1)
    a_tt1[0] = a1
    p_tt1 = np.zeros(L + 1)
    p_tt1[0] = p1
    v_t = np.zeros(L)
    f_t = np.zeros(L)
    a_tt = np.zeros(L)
    p_tt = np.zeros(L)
    k_t = np.zeros(L)
    
    # Filtering
    for t in range(L):
        # Innovation
        v_t[t] = y[t] - a_tt1[t]
        f_t[t] = p_tt1[t] + var_eps
        
        # Kalman gain
        
        # =============================================================================
        # ▼▼▼ 変更点: ゼロ、無限大、NaNのチェックを強化 ▼▼▼
        # =============================================================================
        if f_t[t] == 0 or not np.isfinite(f_t[t]):
            k_t[t] = 0
        else:
            k_t[t] = p_tt1[t] / f_t[t]
        # =============================================================================
        # ▲▲▲ 変更点ここまで ▲▲▲
        # =============================================================================
        
        # # 以前のコード
        # # F_t[t]が非常に小さい、またはゼロの場合のゼロ除算を回避
        # if f_t[t] == 0:
        #     k_t[t] = 0
        # else:
        #     k_t[t] = p_tt1[t] / f_t[t]
            
        # Current state
        a_tt[t] = a_tt1[t] + k_t[t] * v_t[t]
        p_tt[t] = p_tt1[t] * (1 - k_t[t])
        
        # Next state
        a_tt1[t+1] = a_tt[t]
        p_tt1[t+1] = p_tt[t] + var_eta
        
    return a_tt, p_tt, f_t, v_t

def calc_log_diffuse_llhd(vars, y):
    """
    ローカルトレンドモデルの散漫な対数尤度を求める関数
    """
    psi_eta, psi_eps = vars
    var_eta = np.exp(2 * psi_eta)  # σ^2_η に戻す
    var_eps = np.exp(2 * psi_eps)  # σ^2_ε に戻す
    L = len(y)
    
    if L < 2:
        return -np.inf # データが少なすぎて計算できない
        
    # a_1, P_1の初期値
    a1 = y[0]
    p1 = var_eps
    
    # カルマンフィルタリング
    _, _, f_t, v_t = local_trend_kf(y, a1, p1, var_eta, var_eps)
    
    # f_tのゼロや負の値をチェック
    if np.any(f_t[1:] <= 0):
        return -np.inf

    # 散漫対数尤度を計算
    tmp = np.sum(np.log(f_t[1:]) + v_t[1:]**2 / f_t[1:])
    log_ld = -0.5 * L * np.log(2 * np.pi) - 0.5 * tmp
    
    return log_ld

def maf(input_data, size):
    """
    移動平均フィルタ
    """
    window_size = size
    b = (1 / window_size) * np.ones(window_size)
    a = 1
    # Note: MATLABのfilterと挙動を合わせるためlfilterを使用
    from scipy.signal import lfilter
    return lfilter(b, a, input_data)

def kalman2(coordinate_L, coordinate_R, th, initial_value):
    """
    二階差分カルマンフィルタ
    """
    end_step = len(coordinate_R)
    
    # Pythonは参照渡しなので、コピーを作成して元のデータを変更しないようにする
    cooredinate_L = coordinate_L.copy()
    coordinate_R = coordinate_R.copy()

    miss_point = np.zeros(end_step)
    
    for i in range(2, end_step):
        kalman_flag = 0
        
        # 処理に必要な最小限のデータスライスを取得
        current_cooredinate_L = cooredinate_L[:i+1]
        current_coordinate_R = coordinate_R[:i+1]

        diff_data_Lx = np.diff(current_cooredinate_L)
        yL = maf(diff_data_Lx, 3)
        
        diff_data_Rx = np.diff(current_coordinate_R)
        yR = maf(diff_data_Rx, 3)
        
        if len(yL) < 2 or len(yR) < 2:
            continue

        # 最尤推定よりパラメータを求める
        parL = initial_value  
        parR = initial_value
        
        psi_eta_L = np.log(np.sqrt(parL))
        psi_eps_L = np.log(np.sqrt(parL))
        psi_eta_R = np.log(np.sqrt(parR))
        psi_eps_R = np.log(np.sqrt(parR))

        x0L = [psi_eta_L, psi_eps_L]  # 探索するパラメータの初期値
        x0R = [psi_eta_R, psi_eps_R]
        
        # 最小化したい関数（対数尤度の最大化なので負号をつける）
        fL = lambda xL: -calc_log_diffuse_llhd(xL, yL)
        fR = lambda xR: -calc_log_diffuse_llhd(xR, yR)

        resL = minimize(fL, x0L, method='BFGS')  # 最適化を実行
        resR = minimize(fR, x0R, method='BFGS')
        
        xoptL = resL.x
        xoptR = resR.x

        var_eta_opt_L = np.exp(2 * xoptL[0])  # 推定されたψ_ηをσ^2_ηに戻す
        var_eps_opt_L = np.exp(2 * xoptL[1])  # 推定されたψ_εをσ^2_εに戻す
        var_eta_opt_R = np.exp(2 * xoptR[0])  # 推定されたψ_ηをσ^2_ηに戻す
        var_eps_opt_R = np.exp(2 * xoptR[1])  # 推定されたψ_εをσ^2_εに戻す

        # 変更後パラメータの代入
        var_eps_L = var_eta_opt_L
        var_eta_L = var_eps_opt_L
        var_eps_R = var_eta_opt_R
        var_eta_R = var_eps_opt_R
        
        a1L = var_eps_L
        p1L = var_eta_L
        a1R = var_eps_R
        p1R = var_eta_R

        # カルマンフィルタ実行
        _, _, _, v_tL = local_trend_kf(yL, a1L, p1L, var_eta_L, var_eps_L)
        _, _, _, v_tR = local_trend_kf(yR, a1R, p1R, var_eta_R, var_eps_R)
        
        # Next stateの予測値 (a_tt1) を取得
        a_tt_L, _, _, _ = local_trend_kf(yL, a1L, p1L, var_eta_L, var_eps_L)
        a_tt_R, _, _, _ = local_trend_kf(yR, a1R, p1R, var_eta_R, var_eps_R)
        
        # # a_tt1の最終要素が次の状態の予測値：変なことしてる
        # a_tt1L_end = a_tt_L[-1] + var_eta_L if len(a_tt_L) > 0 else 0
        # a_tt1R_end = a_tt_R[-1] + var_eta_R if len(a_tt_R) > 0 else 0

        # 次の状態の予測値を取得 (ノイズ分散 var_eta を加算しないように修正)
        a_tt1L_end = a_tt_L[-1] if len(a_tt_L) > 0 else 0
        a_tt1R_end = a_tt_R[-1] if len(a_tt_R) > 0 else 0

        # 加速度で検出エラーの種類を判別後、補正
        q1L = cooredinate_L[i] - cooredinate_L[i-1]
        q2L = cooredinate_L[i-1] - cooredinate_L[i-2]
        diff2_cankle_Lx_update = q1L - q2L
        
        q1R = coordinate_R[i] - coordinate_R[i-1]
        q2R = coordinate_R[i-1] - coordinate_R[i-2]
        diff2_cankle_Rx_update = q1R - q2R

        if abs(diff2_cankle_Lx_update) > th and abs(diff2_cankle_Rx_update) > th:
            Lbox, Rbox = cooredinate_L[i], coordinate_R[i]
            cooredinate_L[i], coordinate_R[i] = Rbox, Lbox
            
            q1L = cooredinate_L[i] - cooredinate_L[i-1]
            q2L = cooredinate_L[i-1] - cooredinate_L[i-2]
            diff2_cankle_Lx_update = q1L - q2L
            
            q1R = coordinate_R[i] - coordinate_R[i-1]
            q2R = coordinate_R[i-1] - coordinate_R[i-2]
            diff2_cankle_Rx_update = q1R - q2R
            
            if abs(diff2_cankle_Lx_update) > th and abs(diff2_cankle_Rx_update) > th:
                cooredinate_L[i], coordinate_R[i] = Lbox, Rbox
                cooredinate_L[i] = cooredinate_L[i-1] + a_tt1L_end
                coordinate_R[i] = coordinate_R[i-1] + a_tt1R_end
                miss_point[i] = 4
                kalman_flag = 1
            else:
                miss_point[i] = 1
                kalman_flag = 1
        
        if abs(diff2_cankle_Lx_update) > th and abs(diff2_cankle_Rx_update) <= th:
            cooredinate_L[i] = cooredinate_L[i-1] + a_tt1L_end
            miss_point[i] = 2
            kalman_flag = 1
            
        if abs(diff2_cankle_Lx_update) <= th and abs(diff2_cankle_Rx_update) > th:
            coordinate_R[i] = coordinate_R[i-1] + a_tt1R_end
            miss_point[i] = 3
            kalman_flag = 1
            
        p1L = cooredinate_L[i] - cooredinate_L[i-1]
        p2L = cooredinate_L[i-1] - cooredinate_L[i-2]
        diff2_cankle_Lx_update_cover = p1L - p2L
        
        p1R = coordinate_R[i] - coordinate_R[i-1]
        p2R = coordinate_R[i-1] - coordinate_R[i-2]
        diff2_cankle_Rx_update_cover = p1R - p2R
        
        th_cover = 500
        if abs(diff2_cankle_Lx_update_cover) >= th_cover and kalman_flag == 1:
            cooredinate_L[i] = cooredinate_L[i-1] + p2L
            
        if abs(diff2_cankle_Rx_update_cover) >= th_cover and kalman_flag == 1:
            coordinate_R[i] = coordinate_R[i-1] + p2R

    return cooredinate_L, coordinate_R

def bufilter(coordinate_R, coordinate_L):
    """
    バターワースフィルタ
    """
    order = 4
    fs = 100.0 
    fc = 12.0
    # fc = 6.0 # 中崎さん元々(ただ滑らかになりすぎてしまう)
    b, a = butter(order, fc / (fs / 2), btype='low')
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
    # ゼロ除算を回避し、cosの中身を-1から1の範囲に収める
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = vAvB / (Asize * Bsize)
    cos_theta = np.nan_to_num(cos_theta) # nanを0に置換
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
path_op = r'G:\gait_pattern\20251014_nakazaki_pythontest\json_excel'
name_op_excel = 'sub4_com_nfpa_2.xlsx'
full_path_op = os.path.join(path_op, name_op_excel)
name = os.path.splitext(name_op_excel)[0]

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

# データをまとめる [右x, 左x, 右y, 左y]
ankle = np.column_stack([ankle_R_x, ankle_L_x, ankle_R_y, ankle_L_y])
knee = np.column_stack([knee_R_x, knee_L_x, knee_R_y, knee_L_y])
hip = np.column_stack([hip_R_x, hip_L_x, hip_R_y, hip_L_y])
bigtoe = np.column_stack([bigtoe_R_x, bigtoe_L_x, bigtoe_R_y, bigtoe_L_y])
heel = np.column_stack([heel_R_x, heel_L_x, heel_R_y, heel_L_y])

# --- 2. 正解角度(motion capture)を取得 ---
path_mc = r'G:\gait_pattern\20251014_nakazaki_pythontest\qualisys' # ★
name_mc_excel = 'angle_120Hz_sub4_com_nfpa0002.csv' # ★
full_path_mc = os.path.join(path_mc, name_mc_excel)
df_mc = pd.read_csv(full_path_mc, header=0) 

frame_angle = df_mc.iloc[:, 0].values
hip_angle_R = df_mc.iloc[:, 1].values
knee_angle_R = df_mc.iloc[:, 2].values
ankle_angle_R = df_mc.iloc[:, 3].values
hip_angle_L = df_mc.iloc[:, 4].values
knee_angle_L = df_mc.iloc[:, 5].values
ankle_angle_L = df_mc.iloc[:, 6].values

# --- 3. 前後時間削除 ---
timestep = 30.0
rtimestep = 1 / timestep

beginend_excel_path = r'G:\gait_pattern\20251014_nakazaki_pythontest\開始・終了時間.xlsx' # ★
df_time = pd.read_excel(beginend_excel_path, sheet_name="切り取り時間", header=None)
timedata_name = df_time.iloc[1:, 0].values
Excel_TimeData = df_time.iloc[1:, 1:3].values.astype(float)

filenumber = np.where(timedata_name == name)[0]
if filenumber.size > 0:
    filenumber = filenumber[0]
else:
    raise ValueError(f"File '{name}' not found in time data Excel.")

front = int(np.floor(Excel_TimeData[filenumber, 0] * timestep + 0.5))
back = int(np.floor(Excel_TimeData[filenumber, 1] * timestep + 0.5))

cankle = ankle[front:back, :]
cknee = knee[front:back, :]
chip = hip[front:back, :]
cbigtoe = bigtoe[front:back, :]
cheel = heel[front:back, :]

start_time = front * rtimestep
ctime = np.arange(len(cankle)) * rtimestep + start_time  #matlabと同じ時間軸にするためにstart_timeを加算

print(f"切り取りフレーム: {front} から {back} まで")
print(f"front: {front}, back: {back}, データ長さ: {len(cankle)}")
      
correct_angle_frame_front = front * 4
correct_angle_frame_back = back * 4

index_front_candidates = np.where(frame_angle == correct_angle_frame_front)[0]
index_back_candidates = np.where(frame_angle == correct_angle_frame_back)[0]

if index_front_candidates.size == 0 or index_back_candidates.size == 0:
    raise ValueError("Could not find start or end frame in motion capture data.")
index_front = index_front_candidates[0]
index_back = index_back_candidates[0]

adjustment_frame = 1
p120 = 4 * adjustment_frame

cut_hip_cangle_R_120 = hip_angle_R[index_front + p120 : index_back + p120]
cut_knee_cangle_R_120 = knee_angle_R[index_front + p120 : index_back + p120]
cut_ankle_cangle_R_120 = ankle_angle_R[index_front + p120 : index_back + p120]
cut_hip_cangle_L_120 = hip_angle_L[index_front + p120 : index_back + p120]
cut_knee_cangle_L_120 = knee_angle_L[index_front + p120 : index_back + p120]
cut_ankle_cangle_L_120 = ankle_angle_L[index_front + p120 : index_back + p120]

hip_cangle_R = cut_hip_cangle_R_120[::4]
knee_cangle_R = cut_knee_cangle_R_120[::4]
ankle_cangle_R = cut_ankle_cangle_R_120[::4]
hip_cangle_L = cut_hip_cangle_L_120[::4]
knee_cangle_L = cut_knee_cangle_L_120[::4]
ankle_cangle_L = cut_ankle_cangle_L_120[::4]

# --- 4. 補正前の加速度算出 & グラフ描画 ---
# =============================================================================
# %% ▼▼▼ 追加機能: 補正前の座標と加速度の描画 ▼▼▼
# =============================================================================
display_pre_correction_plots = True  # Trueにすると描画
if display_pre_correction_plots:
    print("補正前の座標と加速度グラフを作成中...")
    pre_correction_data = {
        'hip': chip, 'knee': cknee, 'ankle': cankle, 
        'bigtoe': cbigtoe, 'heel': cheel
    }
    
    for joint_name, data in pre_correction_data.items():
        # 加速度を計算 (データ数が2つ減る)
        accel_Rx = np.diff(data[:, 0], 2)
        accel_Lx = np.diff(data[:, 1], 2)
        accel_Ry = np.diff(data[:, 2], 2)
        accel_Ly = np.diff(data[:, 3], 2)
        ctime_a = ctime[:-2] # 加速度の長さに合わせる

        # --- 座標のプロット ---
        plt.figure(figsize=(12, 7))
        plt.subplot(2, 1, 1)
        plt.plot(ctime, data[:, 0], label='Right X', color='red', alpha=0.8)
        plt.plot(ctime, data[:, 1], label='Left X', color='blue', alpha=0.8)
        plt.title(f'Pre-correction {joint_name.capitalize()} X Coordinates', fontsize=14)
        plt.ylabel('Coordinate')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(ctime, data[:, 2], label='Right Y', color='orange', alpha=0.8)
        plt.plot(ctime, data[:, 3], label='Left Y', color='green', alpha=0.8)
        plt.title(f'Pre-correction {joint_name.capitalize()} Y Coordinates', fontsize=14)
        plt.xlabel('Time [s]')
        plt.ylabel('Coordinate')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pre_correction_{joint_name}_coords.png'))
        plt.close()

        # --- 加速度のプロット ---
        plt.figure(figsize=(12, 7))
        plt.subplot(2, 1, 1)
        plt.plot(ctime_a, accel_Rx, label='Right X Accel', color='red', alpha=0.8)
        plt.plot(ctime_a, accel_Lx, label='Left X Accel', color='blue', alpha=0.8)
        plt.title(f'Pre-correction {joint_name.capitalize()} X Acceleration', fontsize=14)
        plt.ylabel('Acceleration')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(ctime_a, accel_Ry, label='Right Y Accel', color='orange', alpha=0.8)
        plt.plot(ctime_a, accel_Ly, label='Left Y Accel', color='green', alpha=0.8)
        plt.title(f'Pre-correction {joint_name.capitalize()} Y Acceleration', fontsize=14)
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pre_correction_{joint_name}_accel.png'))
        plt.close()
# =============================================================================
# %% ▲▲▲ 追加機能ここまで ▲▲▲
# =============================================================================

# --- 5. カルマンフィルタで補正 ---
# ★ データによって閾値を変更  (左座標,　右座標, 加速度の閾値, カルマンの初期値)
print("カルマンフィルタを適用中...")
# kankle_Lx, kankle_Rx = kalman2(cankle[:, 1], cankle[:, 0], 100, 0.0005)
# kankle_Ly, kankle_Ry = kalman2(cankle[:, 3], cankle[:, 2], 500, 0.003)
# kknee_Lx, kknee_Rx = kalman2(cknee[:, 1], cknee[:, 0], 100, 0.008)
# kknee_Ly, kknee_Ry = kalman2(cknee[:, 3], cknee[:, 2], 500, 0.002)
# khip_Lx, khip_Rx = kalman2(chip[:, 1], chip[:, 0], 100, 0.005)
# khip_Ly, khip_Ry = kalman2(chip[:, 3], chip[:, 2], 500, 0.003)
# kbigtoe_Lx, kbigtoe_Rx = kalman2(cbigtoe[:, 1], cbigtoe[:, 0], 40, 0.008)
# kbigtoe_Ly, kbigtoe_Ry = kalman2(cbigtoe[:, 3], cbigtoe[:, 2], 40, 0.003)
# kheel_Lx, kheel_Rx = kalman2(cheel[:, 1], cheel[:, 0], 100, 0.08)
# kheel_Ly, kheel_Ry = kalman2(cheel[:, 3], cheel[:, 2], 500, 0.003)

kankle_Lx, kankle_Rx = kalman2(cankle[:, 1], cankle[:, 0], 100, 0.0005)
kankle_Ly, kankle_Ry = kalman2(cankle[:, 3], cankle[:, 2], 500, 0.003)
kknee_Lx, kknee_Rx = kalman2(cknee[:, 1], cknee[:, 0], 100, 0.008)
kknee_Ly, kknee_Ry = kalman2(cknee[:, 3], cknee[:, 2], 500, 0.002)
khip_Lx, khip_Rx = kalman2(chip[:, 1], chip[:, 0], 100, 0.005)
khip_Ly, khip_Ry = kalman2(chip[:, 3], chip[:, 2], 500, 0.003)
kbigtoe_Lx, kbigtoe_Rx = kalman2(cbigtoe[:, 1], cbigtoe[:, 0], 100, 0.008)
kbigtoe_Ly, kbigtoe_Ry = kalman2(cbigtoe[:, 3], cbigtoe[:, 2], 500, 0.003)
kheel_Lx, kheel_Rx = kalman2(cheel[:, 1], cheel[:, 0], 100, 0.08)
kheel_Ly, kheel_Ry = kalman2(cheel[:, 3], cheel[:, 2], 500, 0.003)

# =============================================================================
# %% ▼▼▼ 追加機能: バターワースフィルタ適用前のデータ描画 ▼▼▼
# =============================================================================
display_before_butterworth_plot = True # Trueにすると描画
if display_before_butterworth_plot:
    print("カルマンフィルタ適用後のグラフを作成中...")
    plt.figure(figsize=(10, 6))
    plt.plot(ctime, cankle[:,0], color='gray', linestyle='--', label='Raw Right X')
    plt.plot(ctime, cankle[:,1], color='lightgray', linestyle=':', label='Raw Left X')
    plt.plot(ctime, kankle_Rx, color='red', label='Kalman Filtered Right X')
    plt.plot(ctime, kankle_Lx, color='blue', label='Kalman Filtered Left X')
    plt.title('Ankle X Coordinate (After Kalman Filter)', fontsize=18)
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Coordinate', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'kalman_only_ankle_x.png'))
    plt.close()
# =============================================================================
# %% ▲▲▲ 追加機能ここまで ▲▲▲
# =============================================================================

# --- 6. バターワースフィルタ ---
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
start_index = 30
end_index = len(ctime) - 30

print(f"ctime[start_index]: {ctime[start_index]}")
print(f"ctime[end_index-1]: {ctime[end_index-1]}")

# --- 8.1 関節角度のグラフ描画 & 保存 ---
display_angle_plots = True
if display_angle_plots:
    print("最終的な関節角度グラフを作成中...")
    # --- 右膝関節角 ---
    plt.figure(figsize=(10, 6))
    plt.plot(ctime[start_index:end_index], bfilter_knee_angle_R[start_index:end_index], linewidth=1.5, color='#fc9d03', label='Proposed Method')
    plt.plot(ctime[start_index:end_index], knee_cangle_R[start_index:end_index], linewidth=1.5, color='#42f58d', label='3DMC')
    plt.fill_between(ctime[start_index:end_index], bfilter_knee_angle_R[start_index:end_index] - 5, bfilter_knee_angle_R[start_index:end_index] + 5, color='#fc9d03', alpha=0.3)
    plt.fill_between(ctime[start_index:end_index], knee_cangle_R[start_index:end_index] - 5, knee_cangle_R[start_index:end_index] + 5, color='g', alpha=0.2)
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Knee Joint Angle_R [°]', fontsize=16)
    plt.title('Right Knee Joint Angle', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim([-10, 80])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'angle_knee_right.png'))
    plt.close()

    # =============================================================================
    # %% ▼▼▼ 追加機能: 左膝関節角の描画 ▼▼▼
    # =============================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(ctime[start_index:end_index], bfilter_knee_angle_L[start_index:end_index], linewidth=1.5, color='#fc9d03', label='Proposed Method')
    plt.plot(ctime[start_index:end_index], knee_cangle_L[start_index:end_index], linewidth=1.5, color='#42f58d', label='3DMC')
    plt.fill_between(ctime[start_index:end_index], bfilter_knee_angle_L[start_index:end_index] - 5, bfilter_knee_angle_L[start_index:end_index] + 5, color='#fc9d03', alpha=0.3)
    plt.fill_between(ctime[start_index:end_index], knee_cangle_L[start_index:end_index] - 5, knee_cangle_L[start_index:end_index] + 5, color='g', alpha=0.2)
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Knee Joint Angle_L [°]', fontsize=16)
    plt.title('Left Knee Joint Angle', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim([-10, 80])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'angle_knee_left.png'))
    plt.close()
    # =============================================================================
    # %% ▲▲▲ 追加機能ここまで ▲▲▲
    # =============================================================================

    # --- 右足関節角度 ---
    plt.figure(figsize=(10, 6))
    plt.plot(ctime[start_index:end_index], bfilter_ankle_angle_R[start_index:end_index], linewidth=1.5, color='#fc9d03', label='Proposed Method')
    plt.plot(ctime[start_index:end_index], ankle_cangle_R[start_index:end_index], linewidth=1.5, color='#42f58d', label='3DMC')
    plt.fill_between(ctime[start_index:end_index], bfilter_ankle_angle_R[start_index:end_index] - 5, bfilter_ankle_angle_R[start_index:end_index] + 5, color='#fc9d03', alpha=0.3)
    plt.fill_between(ctime[start_index:end_index], ankle_cangle_R[start_index:end_index] - 5, ankle_cangle_R[start_index:end_index] + 5, color='g', alpha=0.2)
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Ankle Joint Angle_R [°]', fontsize=16)
    plt.title('Right Ankle Joint Angle', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim([-30, 30])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'angle_ankle_right.png'))
    plt.close()

    # =============================================================================
    # %% ▼▼▼ 追加機能: 左足関節角度の描画 ▼▼▼
    # =============================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(ctime[start_index:end_index], bfilter_ankle_angle_L[start_index:end_index], linewidth=1.5, color='#fc9d03', label='Proposed Method')
    plt.plot(ctime[start_index:end_index], ankle_cangle_L[start_index:end_index], linewidth=1.5, color='#42f58d', label='3DMC')
    plt.fill_between(ctime[start_index:end_index], bfilter_ankle_angle_L[start_index:end_index] - 5, bfilter_ankle_angle_L[start_index:end_index] + 5, color='#fc9d03', alpha=0.3)
    plt.fill_between(ctime[start_index:end_index], ankle_cangle_L[start_index:end_index] - 5, ankle_cangle_L[start_index:end_index] + 5, color='g', alpha=0.2)
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Ankle Joint Angle_L [°]', fontsize=16)
    plt.title('Left Ankle Joint Angle', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim([-30, 30])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'angle_ankle_left.png'))
    plt.close()
    # =============================================================================
    # %% ▲▲▲ 追加機能ここまで ▲▲▲
    # =============================================================================
    
    # --- 右股関節角度 ---
    plt.figure(figsize=(10, 6))
    plt.plot(ctime[start_index:end_index], bfilter_hip_angle_R[start_index:end_index], linewidth=1.5, color='#fc9d03', label='Proposed Method')
    plt.plot(ctime[start_index:end_index], hip_cangle_R[start_index:end_index], linewidth=1.5, color='#42f58d', label='3DMC')
    plt.fill_between(ctime[start_index:end_index], bfilter_hip_angle_R[start_index:end_index] - 5, bfilter_hip_angle_R[start_index:end_index] + 5, color='#fc9d03', alpha=0.3)
    plt.fill_between(ctime[start_index:end_index], hip_cangle_R[start_index:end_index] - 5, hip_cangle_R[start_index:end_index] + 5, color='g', alpha=0.2)
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Hip Joint Angle_R [°]', fontsize=16)
    plt.title('Right Hip Joint Angle', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim([-30, 40])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'angle_hip_right.png'))
    plt.close()

    # =============================================================================
    # %% ▼▼▼ 追加機能: 左股関節角度の描画 ▼▼▼
    # =============================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(ctime[start_index:end_index], bfilter_hip_angle_L[start_index:end_index], linewidth=1.5, color='#fc9d03', label='Proposed Method')
    plt.plot(ctime[start_index:end_index], hip_cangle_L[start_index:end_index], linewidth=1.5, color='#42f58d', label='3DMC')
    plt.fill_between(ctime[start_index:end_index], bfilter_hip_angle_L[start_index:end_index] - 5, bfilter_hip_angle_L[start_index:end_index] + 5, color='#fc9d03', alpha=0.3)
    plt.fill_between(ctime[start_index:end_index], hip_cangle_L[start_index:end_index] - 5, hip_cangle_L[start_index:end_index] + 5, color='g', alpha=0.2)
    plt.xlabel('Time [s]', fontsize=16)
    plt.ylabel('Hip Joint Angle_L [°]', fontsize=16)
    plt.title('Left Hip Joint Angle', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.ylim([-30, 40])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'angle_hip_left.png'))
    plt.close()
    # =============================================================================
    # %% ▲▲▲ 追加機能ここまで ▲▲▲
    # =============================================================================

# --- 8.2 最終的な座標データの描画 & 保存 ---
display_coordinates = True
if display_coordinates:
    # 描画対象のデータを辞書にまとめる
    plot_data = {
        'hip': {'raw': chip, 'filtered_R': khip_Rx_filter, 'filtered_L': khip_Lx_filter, 'filtered_Ry': khip_Ry_filter, 'filtered_Ly': khip_Ly_filter},
        'knee': {'raw': cknee, 'filtered_R': kknee_Rx_filter, 'filtered_L': kknee_Lx_filter, 'filtered_Ry': kknee_Ry_filter, 'filtered_Ly': kknee_Ly_filter},
        'ankle': {'raw': cankle, 'filtered_R': kankle_Rx_filter, 'filtered_L': kankle_Lx_filter, 'filtered_Ry': kankle_Ry_filter, 'filtered_Ly': kankle_Ly_filter},
        'bigtoe': {'raw': cbigtoe, 'filtered_R': kbigtoe_Rx_filter, 'filtered_L': kbigtoe_Lx_filter, 'filtered_Ry': kbigtoe_Ry_filter, 'filtered_Ly': kbigtoe_Ly_filter},
        'heel': {'raw': cheel, 'filtered_R': kheel_Rx_filter, 'filtered_L': kheel_Lx_filter, 'filtered_Ry': kheel_Ry_filter, 'filtered_Ly': kheel_Ly_filter},
    }

    for joint_name, data in plot_data.items():
        # --- X座標のプロット ---
        plt.figure(figsize=(10, 6))
        plt.plot(ctime[start_index:end_index], data['raw'][start_index:end_index, 0], color='k', label='Raw Right')
        plt.plot(ctime[start_index:end_index], data['raw'][start_index:end_index, 1], color='gray', linestyle='--', label='Raw Left')
        plt.plot(ctime[start_index:end_index], data['filtered_R'][start_index:end_index], color='r', label='Filtered Right')
        plt.plot(ctime[start_index:end_index], data['filtered_L'][start_index:end_index], color='b', label='Filtered Left')
        plt.xlabel('Time [s]', fontsize=16)
        plt.ylabel('X Coordinate', fontsize=16)
        plt.title(f'{joint_name.capitalize()} X Coordinate (Final)', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'coord_{joint_name}_x.png'))
        plt.close()

        # --- Y座標のプロット ---
        plt.figure(figsize=(10, 6))
        plt.plot(ctime[start_index:end_index], data['raw'][start_index:end_index, 2], color='k', label='Raw Right')
        plt.plot(ctime[start_index:end_index], data['raw'][start_index:end_index, 3], color='gray', linestyle='--', label='Raw Left')
        plt.plot(ctime[start_index:end_index], data['filtered_Ry'][start_index:end_index], color='r', label='Filtered Right')
        plt.plot(ctime[start_index:end_index], data['filtered_Ly'][start_index:end_index], color='b', label='Filtered Left')
        plt.xlabel('Time [s]', fontsize=16)
        plt.ylabel('Y Coordinate', fontsize=16)
        plt.title(f'{joint_name.capitalize()} Y Coordinate (Final)', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'coord_{joint_name}_y.png'))
        plt.close()

# --- 9. MAE表示 ---
def mae(error):
    return np.mean(np.abs(error))

print("\n--- MAE (Mean Absolute Error) ---")
print(f"右股関節角度のMAE: {mae(bfilter_hip_angle_R[start_index:end_index] - hip_cangle_R[start_index:end_index])}")
print(f"右膝関節角度のMAE: {mae(bfilter_knee_angle_R[start_index:end_index] - knee_cangle_R[start_index:end_index])}")
print(f"右足関節角度のMAE: {mae(bfilter_ankle_angle_R[start_index:end_index] - ankle_cangle_R[start_index:end_index])}")

print(f"左股関節角度のMAE: {mae(bfilter_hip_angle_L[start_index:end_index] - hip_cangle_L[start_index:end_index])}")
print(f"左膝関節角度のMAE: {mae(bfilter_knee_angle_L[start_index:end_index] - knee_cangle_L[start_index:end_index])}")
print(f"左足関節角度のMAE: {mae(bfilter_ankle_angle_L[start_index:end_index] - ankle_cangle_L[start_index:end_index])}")

# 全ての処理が完了したことを通知
print("\n処理が完了し、すべてのグラフが保存されました。")