import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize
import os
import warnings

# # RuntimeWarningという種類の警告を非表示にする設定
# warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# 関数定義セクション
# =============================================================================

def local_trend_kf(y, a1, p1, var_eta, var_eps):
    """
    ローカルトレンドモデルのカルマンフィルタリングを行う関数
    
    引数:
    y       : np.array (L,) - 観測値の時系列データ (このスクリプトでは「速度」が該当)
    a1      : float         - 状態 (速度) の初期予測値
    p1      : float         - 状態 (速度) の初期予測誤差の分散 (予測の不確かさ)
    var_eta : float         - システムノイズ (状態遷移ノイズ) の分散 (σ^2_η)
                            (予測が次のステップでどれだけ変動しうるか)
    var_eps : float         - 観測ノイズの分散 (σ^2_ε)
                            (観測値yがどれだけ信頼できないか)
    """
    
    L = len(y)
    
    # 計算結果を格納するための配列を事前に確保(Preallocation)
    a_tt1 = np.zeros(L + 1)  #状態(速度)の予測値を格納
    a_tt1[0] = a1
    p_tt1 = np.zeros(L + 1)  #状態(速度)の予測誤差分散を格納
    p_tt1[0] = p1
    v_t = np.zeros(L)  #イノベーション(観測値と予測値の差)を格納
    f_t = np.zeros(L)  #イノベーション分散を格納
    a_tt = np.zeros(L)  #状態(速度)の推定値(フィルタ値)を格納
    p_tt = np.zeros(L)  #状態(速度)の推定誤差分散を格納
    k_t = np.zeros(L)  #カルマンゲインを格納
    
    # Filtering (フィルタリングのループ処理)
    for t in range(L):
        # ----------------------------------------------------
        # 1. 更新 (Update) ステップ: 新しい観測 y[t] を使って予測を補正
        # --------------------------------------------------------
        v_t[t] = y[t] - a_tt1[t]  # イノベーション(観測値と予測値の差)
        f_t[t] = p_tt1[t] + var_eps  # イノベーションの分散: 予測誤差分散 + 観測ノイズ分散
        
        # Kalman gain (カルマンゲインの計算): (予測の不確かさ) / (イノベーション全体の不確かさ)
        # 1に近い: 観測値を信頼する (予測が不確か or 観測が正確)    0に近い: 予測値を信頼する (予測が正確 or 観測がノイズだらけ)
        # f_tが無限大、NaN(非数)にの場合は，k_tを1に設定して観測値をそのまま使う
        if not np.isfinite(f_t[t]):
            k_t[t] = 1.0
            print(f"Warning: f_t[{t}] is not finite (value: {f_t[t]}). Setting k_t[{t}] to 1.0.")
        elif f_t[t] == 0:
            k_t[t] = 0.0
            print(f"Warning: f_t[{t}] is zero. Setting k_t[{t}] to 0.0.")
        else:
            k_t[t] = p_tt1[t] / f_t[t] 
            k_t[t] = np.clip(k_t[t], 0, 1)  # カルマンゲインを[0,1]に制限
            
        # Current state (現時刻の状態推定値 a_tt[t] の計算)
        a_tt[t] = a_tt1[t] + k_t[t] * v_t[t]  #: (予測値) + (カルマンゲイン) * (観測残差)
        # Current state variance (現時刻の推定誤差の分散 p_tt[t] の計算)
        p_tt[t] = p_tt1[t] * (1 - k_t[t])  # 観測値を使った分、予測誤差の分散(p_tt1[t])が(1-k_t)倍だけ減少する
        
        # ----------------------------------------------------
        # 2. 予測 (Prediction) ステップ: 次の時刻(t+1)の状態を予測
        # ----------------------------------------------------
        # Next state prediction (次時刻の状態予測値 a_tt1[t+1])
        a_tt1[t+1] = a_tt[t]  # ローカルトレンドモデルでは、次の速度は現在の速度と同じと予測
        # Next state variance (次時刻の予測誤差の分散 p_tt1[t+1])
        p_tt1[t+1] = p_tt[t] + var_eta  # (現在の推定誤差) + (システムノイズ)   時間が経過した分、システムノイズ(var_eta)だけ不確かさが増加する
        
    return a_tt, p_tt, f_t, v_t

def calc_log_diffuse_llhd(vars, y):
    """
    ローカルトレンドモデルの散漫な対数尤度を求める関数
    尤度を最大化するパラメータを見つけるために使用
    """
    psi_eta, psi_eps = vars
    var_eta = np.exp(2 * psi_eta)  # ψ_η を σ^2_η に戻す
    var_eps = np.exp(2 * psi_eps)  # ψ_ε を σ^2_ε に戻す
    L = len(y)
    
    if L < 2:
        return -np.inf # データが少なすぎて計算できない
        
    # a_1, P_1の初期値
    a1 = y[0]
    p1 = var_eps
    
    # カルマンフィルタリングを実行
    _, _, f_t, v_t = local_trend_kf(y, a1, p1, var_eta, var_eps)
    
    # f_tのゼロや負の値、非数をチェック
    if np.any(f_t[1:] <= 0) or not np.all(np.isfinite(f_t[1:])):
        return -np.inf

    # 散漫対数尤度を計算
    tmp = np.sum(np.log(f_t[1:]) + v_t[1:]**2 / f_t[1:])
    log_ld = -0.5 * L * np.log(2 * np.pi) - 0.5 * tmp
    
    return log_ld

# def maf(input_data, size):
#     """
#     移動平均フィルタ(Moving Average Filter)
#     """
#     window_size = size
#     b = (1 / window_size) * np.ones(window_size)
#     a = 1
#     from scipy.signal import lfilter
#     return lfilter(b, a, input_data)

def maf(input_data, size):
    '''
    中心化された移動平均フィルタ（位相遅延なし）
    '''
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(input_data, size=size, mode='nearest')

def kalman2(coordinate_L, coordinate_R, th, initial_value):
    """
    二階差分カルマンフィルタ
    加速度(二階差分)を監視し、閾値を超えた場合に補正を行う
    """
    end_step = len(coordinate_R)
    
    # 元のデータを変更しないようにコピーを作成
    coordinate_L = coordinate_L.copy()
    coordinate_R = coordinate_R.copy()

    # 誤検出の種類を記録するための配列
    miss_point = np.zeros(end_step)
    
    # 2フレーム目から最終フレームまでループ
    for i in range(2, end_step):
        kalman_flag = 0 # カルマンフィルタによる補正が行われたかを判定するフラグ
        
        # 現在のフレームまでのデータスライスを取得
        current_cooredinate_L = coordinate_L[:i]
        current_coordinate_R = coordinate_R[:i]

        # 座標の差分(速度)を計算し、移動平均を適用
        diff_data_Lx = np.diff(current_cooredinate_L)
        yL = maf(diff_data_Lx, 3)
        diff_data_Rx = np.diff(current_coordinate_R)
        yR = maf(diff_data_Rx, 3)
        
        # # 移動平均使いたくなかったのでテスト
        # yL = diff_data_Lx
        # yR = diff_data_Rx
        
        if len(yL) < 2 or len(yR) < 2:
            continue

        # 最尤推定よりパラメータを求める
        parL = initial_value
        parR = initial_value
        
        # パラメータを対数変換 (ψ_η, ψ_ε に変換)
        psi_eta_L = np.log(np.sqrt(parL))
        psi_eps_L = np.log(np.sqrt(parL))
        psi_eta_R = np.log(np.sqrt(parR))
        psi_eps_R = np.log(np.sqrt(parR))
        
        # 探索するパラメータの初期値
        x0L = [psi_eta_L, psi_eps_L]
        x0R = [psi_eta_R, psi_eps_R]
        
        # 最小化したい関数（散漫な対数尤度の最大化なので負号をつける）
        fL = lambda xL: -calc_log_diffuse_llhd(xL, yL)
        fR = lambda xR: -calc_log_diffuse_llhd(xR, yR)
        
        
        # パラメータの探索範囲を制約する
        bounds = ((-20, 20), (-20, 20)) 
        
        # 最適化を実行 (準ニュートン法の一種であるL-BFGS-Bを使用)
        resL = minimize(fL, x0L, method='L-BFGS-B', bounds=bounds)
        resR = minimize(fR, x0R, method='L-BFGS-B', bounds=bounds)
        
        # 最適化されたパラメータを取得
        xoptL = resL.x
        xoptR = resR.x
        
        # 推定されたψをσ^2に戻す
        var_eta_opt_L = np.exp(2 * xoptL[0])
        var_eps_opt_L = np.exp(2 * xoptL[1])
        var_eta_opt_R = np.exp(2 * xoptR[0])
        var_eps_opt_R = np.exp(2 * xoptR[1])
        
        # # パラメータを更新(元々のやつ：epsとeta入れ替わってる？)
        # var_eps_L = var_eta_opt_L
        # var_eta_L = var_eps_opt_L
        # var_eps_R = var_eta_opt_R
        # var_eta_R = var_eps_opt_R
        
        # # もともとのやつ
        # a1L = var_eps_L
        # p1L = var_eta_L
        # a1R = var_eps_R
        # p1R = var_eta_R
        
        # パラメータを更新
        var_eta_L = var_eta_opt_L  # システムノイズ（状態遷移の不確かさ）
        var_eps_L = var_eps_opt_L  # 観測ノイズ（観測値の不確かさ）
        var_eta_R = var_eta_opt_R
        var_eps_R = var_eps_opt_R
        
        # a1: 状態（速度）の初期推定値 → 直前の速度データを使用
        # p1: 予測誤差分散の初期値 → 観測ノイズとシステムノイズの和
        a1L = yL[-1] if len(yL) > 0 else 0.0  # 直前の速度を初期状態に
        p1L = var_eps_L + var_eta_L           # 適切な予測誤差分散
        a1R = yR[-1] if len(yR) > 0 else 0.0
        p1R = var_eps_R + var_eta_R

        # カルマンフィルタを実行し、状態変数を取得
        a_tt_L, _, _, _ = local_trend_kf(yL, a1L, p1L, var_eta_L, var_eps_L)
        a_tt_R, _, _, _ = local_trend_kf(yR, a1R, p1R, var_eta_R, var_eps_R)
        
        # 次の状態の予測値を取得
        a_tt1L_end = a_tt_L[-2] if len(a_tt_L) > 1 else 0  
        a_tt1R_end = a_tt_R[-2] if len(a_tt_R) > 1 else 0
        # a_tt1L_end = a_tt_L[-1] if len(a_tt_L) > 0 else 0
        # a_tt1R_end = a_tt_R[-1] if len(a_tt_R) > 0 else 0

        # 加速度で検出エラーの種類を判別後、補正
        q1L = coordinate_L[i] - coordinate_L[i-1]
        q2L = coordinate_L[i-1] - coordinate_L[i-2]
        diff2_cankle_Lx_update = q1L - q2L
        
        q1R = coordinate_R[i] - coordinate_R[i-1]
        q2R = coordinate_R[i-1] - coordinate_R[i-2]
        diff2_cankle_Rx_update = q1R - q2R

        # パターン1: 左右両方の加速度が閾値を超えた場合 (入れ替わり or 両方誤検出)
        if abs(diff2_cankle_Lx_update) > th and abs(diff2_cankle_Rx_update) > th:
            # いったん左右の座標を入れ替えてみる
            Lbox, Rbox = coordinate_L[i], coordinate_R[i]
            coordinate_L[i], coordinate_R[i] = Rbox, Lbox
            
            # 入れ替えた後、再度加速度を計算
            q1L = coordinate_L[i] - coordinate_L[i-1]
            q2L = coordinate_L[i-1] - coordinate_L[i-2]
            diff2_cankle_Lx_update = q1L - q2L
            
            q1R = coordinate_R[i] - coordinate_R[i-1]
            q2R = coordinate_R[i-1] - coordinate_R[i-2]
            diff2_cankle_Rx_update = q1R - q2R
            
            # それでも両方の加速度が閾値を超えている場合 -> 両方誤検出と判断
            if abs(diff2_cankle_Lx_update) > th and abs(diff2_cankle_Rx_update) > th:
                coordinate_L[i], coordinate_R[i] = Lbox, Rbox # 入れ替えを元に戻す
                coordinate_L[i] = coordinate_L[i-1] + a_tt1L_end # カルマンフィルタで予測した値で補正
                coordinate_R[i] = coordinate_R[i-1] + a_tt1R_end # カルマンフィルタで予測した値で補正
                miss_point[i] = 4 # 両方誤検出=4
                kalman_flag = 1
            else: # 入れ替えたら加速度が閾値内に収まった -> 入れ替わりと判断
                miss_point[i] = 1 # 入れ替わり=1
                kalman_flag = 1
        
        # パターン2: 左足のみ加速度が閾値を超えた場合
        if abs(diff2_cankle_Lx_update) > th and abs(diff2_cankle_Rx_update) <= th:
            coordinate_L[i] = coordinate_L[i-1] + a_tt1L_end # カルマンフィルタで予測した値で補正
            miss_point[i] = 2 # 左足の誤検出=2
            kalman_flag = 1
            
        # パターン3: 右足のみ加速度が閾値を超えた場合
        if abs(diff2_cankle_Lx_update) <= th and abs(diff2_cankle_Rx_update) > th:
            coordinate_R[i] = coordinate_R[i-1] + a_tt1R_end # カルマンフィルタで予測した値で補正
            miss_point[i] = 3 # 右足の誤検出=3
            kalman_flag = 1
            
        # 補正後の値が極端に飛びすぎないようにする追加の補正
        p1L = coordinate_L[i] - coordinate_L[i-1]
        p2L = coordinate_L[i-1] - coordinate_L[i-2]
        diff2_cankle_Lx_update_cover = p1L - p2L
        
        p1R = coordinate_R[i] - coordinate_R[i-1]
        p2R = coordinate_R[i-1] - coordinate_R[i-2]
        diff2_cankle_Rx_update_cover = p1R - p2R
        
        th_cover = 500
        if abs(diff2_cankle_Lx_update_cover) >= th_cover and kalman_flag == 1:
            coordinate_L[i] = coordinate_L[i-1] + p2L
            
        if abs(diff2_cankle_Rx_update_cover) >= th_cover and kalman_flag == 1:
            coordinate_R[i] = coordinate_R[i-1] + p2R

    return coordinate_L, coordinate_R

# =============================================================================
# メインスクリプト
# =============================================================================

# --- 1. openposeから得られた座標をエクセルから取得 ---
# ★ ユーザーはこれらのパスを自分の環境に合わせて変更する必要があります。
path_op = r'G:\gait_pattern\20250811_br\sub1\thera1-0\fr_yoloseg' # OpenPoseの座標データ(csv)があるパス
name_op_excel = 'openpose.csv'  # 処理対象のファイル名
full_path_op = os.path.join(path_op, name_op_excel)
name = os.path.splitext(name_op_excel)[0] # 拡張子を除いたファイル名を取得

# --- 結果保存用ディレクトリの作成 ---
output_dir = os.path.join(path_op, f"kalman_results_savgol")
os.makedirs(output_dir, exist_ok=True)
print(f"グラフは '{output_dir}' に保存されます。")

# 座標データをcsvから取得
""" openpose keypoint index
//     {0,  "Nose"},
//     {1,  "Neck"},
//     {2,  "RShoulder"},
//     {3,  "RElbow"},
//     {4,  "RWrist"},
//     {5,  "LShoulder"},
//     {6,  "LElbow"},
//     {7,  "LWrist"},
//     {8,  "MidHip"},
//     {9,  "RHip"},
//     {10, "RKnee"},
//     {11, "RAnkle"},
//     {12, "LHip"},
//     {13, "LKnee"},
//     {14, "LAnkle"},
//     {15, "REye"},
//     {16, "LEye"},
//     {17, "REar"},
//     {18, "LEar"},
//     {19, "LBigToe"},
//     {20, "LSmallToe"},
//     {21, "LHeel"},
//     {22, "RBigToe"},
//     {23, "RSmallToe"},
//     {24, "RHeel"},
"""

df = pd.read_csv(full_path_op)
# 各キーポイントの座標，信頼度を取得
# x, y, pの順で格納 (左右存在するものは右x, 右y, 右p, 左x, 左y, 左p の順)
midhip = df.iloc[:, [25,26,27]].values  # 股関節中心の座標データ
hip = df.iloc[:, [28,29,30, 37,38,39]].values # 左右股関節の座標データ
knee = df.iloc[:, [31,32,33, 40,41,42]].values # 膝の座標データ
ankle = df.iloc[:, [34,35,36, 43,44,45]].values # 足首の座標データ
bigtoe = df.iloc[:, [67,68,69, 58,59,60]].values # 親指の座標データ
smalltoe = df.iloc[:, [70,71,72, 61,62,63]].values # 小指の座標データ
heel = df.iloc[:, [73,74,75, 64,65,66]].values # かかとの座標データ

# おまけ
nose = df.iloc[:, [1,2,3]].values  # 鼻の座標データ
neck = df.iloc[:, [4,5,6]].values  # 首の座標データ
shoulder = df.iloc[:, [7,8,9, 16,17,18]].values # 肩の座標データ
elbow = df.iloc[:, [10,11,12, 19,20,21]].values # 肘の座標データ
wrist = df.iloc[:, [13,14,15, 22,23,24]].values # 手首の座標データ
eye = df.iloc[:, [46,47,48, 49,50,51]].values # 目の座標データ
ear = df.iloc[:, [52,53,54, 55,56,57]].values # 耳の座標データ


# --- 3. 前後フレーム設定 ---
# start_frame = 170 #FL約-2m地点 0-0-16
# end_frame = 350 #FLの最大検出フレーム
# start_frame = 170 #FL約-2m地点 1-0-3
# end_frame = 459 #FLの最大検出フレーム
start_frame = 170 #FL約-2m地点 1--0
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
csmalltoe = smalltoe[start_frame:end_frame]
cheel = heel[start_frame:end_frame]

cframe = np.arange(len(cankle)) + start_frame

# --- 4. 補正前の加速度算出 & グラフ描画 ---
display_pre_correction_plots = True  # True:表示, False:非表示
if display_pre_correction_plots:
    print("補正前の座標と加速度グラフを作成中...")
    pre_correction_data = {
        'hip': chip, 'knee': cknee, 'ankle': cankle, 
        'bigtoe': cbigtoe, 'smalltoe': csmalltoe, 'heel': cheel
    }
    
    for joint_name, data in pre_correction_data.items():
        # 速度を計算(データ数が1つ減る)
        vel_Rx = np.diff(data[:, 0])
        vel_Lx = np.diff(data[:, 3])
        vel_Ry = np.diff(data[:, 1])
        vel_Ly = np.diff(data[:, 4])
        cframe_v = cframe[:-1]  # 速度の長さに合わせる

        # 加速度を計算 (データ数が2つ減る)
        accel_Rx = np.diff(data[:, 0], 2)
        accel_Lx = np.diff(data[:, 3], 2)
        accel_Ry = np.diff(data[:, 1], 2)
        accel_Ly = np.diff(data[:, 4], 2)
        cframe_a = cframe[:-2] # 加速度の長さに合わせる

        # --- 座標のプロット ---
        plt.figure(figsize=(12, 12))
        plt.subplot(3, 1, 1)
        plt.plot(cframe, data[:, 0], label='Right X', color='red', alpha=0.8)
        plt.plot(cframe, data[:, 3], label='Left X', color='blue', alpha=0.8)
        plt.title(f'Pre {joint_name.capitalize()} X Coordinates', fontsize=18)
        plt.ylabel('Coordinate [px]', fontsize=16)
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylim(0, min(3840, np.max([data[:, 0], data[:, 3]])))
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(cframe, data[:, 1], label='Right Y', color='red', alpha=0.8)
        plt.plot(cframe, data[:, 4], label='Left Y', color='blue', alpha=0.8)
        plt.title(f'Pre {joint_name.capitalize()} Y Coordinates', fontsize=18)
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('Coordinate [px]', fontsize=16)
        plt.ylim(0, min(2160, np.max([data[:, 1], data[:, 4]])))
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(cframe, data[:, 2], label='Right Confidence', color='red', alpha=0.8)
        plt.plot(cframe, data[:, 5], label='Left Confidence', color='blue', alpha=0.8)
        plt.title(f'Pre {joint_name.capitalize()} Confidence', fontsize=18)
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('Confidence [-]', fontsize=16)
        plt.ylim(0, 1)
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
        plt.ylim(-200,200)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(cframe_v, vel_Ry, label='Right Y Vel', color='red', alpha=0.8)
        plt.plot(cframe_v, vel_Ly, label='Left Y Vel', color='blue', alpha=0.8)
        plt.title(f'Pre-correction {joint_name.capitalize()} Y Velocity', fontsize=18)
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('Velocity [px/s]', fontsize=16)
        plt.ylim(-50,50)
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
"""
値調整中 bigtoe_y以外はうまくいく
"""
# kankle_Lx, kankle_Rx = cankle[:, 3], cankle[:, 0]
# kankle_Ly, kankle_Ry = cankle[:, 4], cankle[:, 1]
# kknee_Lx, kknee_Rx = cknee[:, 3], cknee[:, 0]
# kknee_Ly, kknee_Ry = cknee[:, 4], cknee[:, 1]
# khip_Lx, khip_Rx = chip[:, 3], chip[:, 0]
# khip_Ly, khip_Ry = chip[:, 4], chip[:, 1]
# kbigtoe_Lx, kbigtoe_Rx = cbigtoe[:, 3], cbigtoe[:, 0]
# kbigtoe_Ly, kbigtoe_Ry = cbigtoe[:, 4], cbigtoe[:, 1]
# # kbigtoe_Ly, kbigtoe_Ry = kalman2(cbigtoe[:, 4], cbigtoe[:, 1], 40, 0.001)
# print("カルマンフィルタ: 母趾Y座標補正完了")
# # ksmalltoe_Lx, ksmalltoe_Rx = csmalltoe[:, 3], csmalltoe[:, 0]
# # ksmalltoe_Ly, ksmalltoe_Ry = csmalltoe[:, 4], csmalltoe[:, 1]
# ksmalltoe_Lx, ksmalltoe_Rx = kalman2(csmalltoe[:, 3], csmalltoe[:, 0], 200, 0.1)
# print("カルマンフィルタ: 小趾X座標補正完了")
# ksmalltoe_Ly, ksmalltoe_Ry = kalman2(csmalltoe[:, 4], csmalltoe[:, 1], 50, 0.1)
# print("カルマンフィルタ: 小趾Y座標補正完了")
# kheel_Lx, kheel_Rx = cheel[:, 3], cheel[:, 0]
# kheel_Ly, kheel_Ry = cheel[:, 4], cheel[:, 1]

# kankle_Lx, kankle_Rx = kalman2(cankle[:, 3], cankle[:, 0], 200, 0.1)
# print("カルマンフィルタ: 足首X座標補正完了")
# kankle_Ly, kankle_Ry = kalman2(cankle[:, 4], cankle[:, 1], 50, 0.1)
# print("カルマンフィルタ: 足首Y座標補正完了")
# kknee_Lx, kknee_Rx = kalman2(cknee[:, 3], cknee[:, 0], 200, 0.1)
# print("カルマンフィルタ: 膝X座標補正完了")
# kknee_Ly, kknee_Ry = kalman2(cknee[:, 4], cknee[:, 1], 50, 0.1)
# print("カルマンフィルタ: 膝Y座標補正完了")
# khip_Lx, khip_Rx = kalman2(chip[:, 3], chip[:, 0], 50, 0.1)
# print("カルマンフィルタ: 股関節X座標補正完了")
# khip_Ly, khip_Ry = kalman2(chip[:, 4], chip[:, 1], 50, 0.1)
# print("カルマンフィルタ: 股関節Y座標補正完了")
# kbigtoe_Lx, kbigtoe_Rx = kalman2(cbigtoe[:, 3], cbigtoe[:, 0], 200, 0.1)
# print("カルマンフィルタ: 母趾X座標補正完了")
# # kbigtoe_Ly, kbigtoe_Ry = kalman2(cbigtoe[:, 4], cbigtoe[:, 1], 100, 0.1)
# kbigtoe_Ly, kbigtoe_Ry = kalman2(cbigtoe[:, 4], cbigtoe[:, 1], 40, 0.1)
# print("カルマンフィルタ: 母趾Y座標補正完了")
# ksmalltoe_Lx, ksmalltoe_Rx = kalman2(csmalltoe[:, 3], csmalltoe[:, 0], 200, 0.1)
# print("カルマンフィルタ: 小趾X座標補正完了")
# ksmalltoe_Ly, ksmalltoe_Ry = kalman2(csmalltoe[:, 4], csmalltoe[:, 1], 50, 0.1)
# print("カルマンフィルタ: 小趾Y座標補正完了")
# kheel_Lx, kheel_Rx = kalman2(cheel[:, 3], cheel[:, 0], 200, 0.1)
# print("カルマンフィルタ: 踵X座標補正完了")
# kheel_Ly, kheel_Ry = kalman2(cheel[:, 4], cheel[:, 1], 50, 0.1)
# print("カルマンフィルタ: 踵Y座標補正完了")



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

# """
# FL この値でいけた
# """
# kankle_Lx, kankle_Rx = kalman2(cankle[:, 3], cankle[:, 0], 200, 0.1)
# print("カルマンフィルタ: 足首X座標補正完了")
# kankle_Ly, kankle_Ry = kalman2(cankle[:, 4], cankle[:, 1], 50, 0.1)
# print("カルマンフィルタ: 足首Y座標補正完了")
# kknee_Lx, kknee_Rx = kalman2(cknee[:, 3], cknee[:, 0], 200, 0.1)
# print("カルマンフィルタ: 膝X座標補正完了")
# kknee_Ly, kknee_Ry = kalman2(cknee[:, 4], cknee[:, 1], 50, 0.1)
# print("カルマンフィルタ: 膝Y座標補正完了")
# khip_Lx, khip_Rx = kalman2(chip[:, 3], chip[:, 0], 50, 0.1)
# print("カルマンフィルタ: 股関節X座標補正完了")
# khip_Ly, khip_Ry = kalman2(chip[:, 4], chip[:, 1], 50, 0.1)
# print("カルマンフィルタ: 股関節Y座標補正完了")
# kbigtoe_Lx, kbigtoe_Rx = kalman2(cbigtoe[:, 3], cbigtoe[:, 0], 200, 0.1)
# print("カルマンフィルタ: 母趾X座標補正完了")
# kbigtoe_Ly, kbigtoe_Ry = kalman2(cbigtoe[:, 4], cbigtoe[:, 1], 40, 0.1)
# print("カルマンフィルタ: 母趾Y座標補正完了")
# ksmalltoe_Lx, ksmalltoe_Rx = kalman2(csmalltoe[:, 3], csmalltoe[:, 0], 200, 0.1)
# print("カルマンフィルタ: 小趾X座標補正完了")
# ksmalltoe_Ly, ksmalltoe_Ry = kalman2(csmalltoe[:, 4], csmalltoe[:, 1], 50, 0.1)
# print("カルマンフィルタ: 小趾Y座標補正完了")
# kheel_Lx, kheel_Rx = kalman2(cheel[:, 3], cheel[:, 0], 200, 0.1)
# print("カルマンフィルタ: 踵X座標補正完了")
# kheel_Ly, kheel_Ry = kalman2(cheel[:, 4], cheel[:, 1], 50, 0.1)
# print("カルマンフィルタ: 踵Y座標補正完了")



from model_free_correction_full import model_free_correction

"""
window_lengthは奇数: 7, 9, 11, 13, 15, 17, 19, 21, 23, 25...
polyorderは3固定: 変更不要

windowサイズの意味

小さい (7-11): ギザギザ残るが細部保存、遅延小
標準 (13-17): バランス良好 👍
大きい (19-25): 非常に滑らか、細部消失、遅延大
"""

# ========== 股関節 ==========
khip_Lx, khip_Rx = model_free_correction(
    chip[:, 3], chip[:, 0], chip[:, 5], chip[:, 2],
    method='savgol', window_length=13, polyorder=3)
khip_Ly, khip_Ry = model_free_correction(
    chip[:, 4], chip[:, 1], chip[:, 5], chip[:, 2],
    method='savgol', window_length=9, polyorder=3)

# ========== 膝 ==========
kknee_Lx, kknee_Rx = model_free_correction(
    cknee[:, 3], cknee[:, 0], cknee[:, 5], cknee[:, 2],
    method='savgol', window_length=15, polyorder=3)
kknee_Ly, kknee_Ry = model_free_correction(
    cknee[:, 4], cknee[:, 1], cknee[:, 5], cknee[:, 2],
    method='savgol', window_length=11, polyorder=3)

# ========== 足首 ==========
kankle_Lx, kankle_Rx = model_free_correction(
    cankle[:, 3], cankle[:, 0], cankle[:, 5], cankle[:, 2],
    method='savgol', window_length=15, polyorder=3)
kankle_Ly, kankle_Ry = model_free_correction(
    cankle[:, 4], cankle[:, 1], cankle[:, 5], cankle[:, 2],
    method='savgol', window_length=11, polyorder=3)

# ========== 踵 ==========
kheel_Lx, kheel_Rx = model_free_correction(
    cheel[:, 3], cheel[:, 0], cheel[:, 5], cheel[:, 2],
    method='savgol', window_length=17, polyorder=3)
kheel_Ly, kheel_Ry = model_free_correction(
    cheel[:, 4], cheel[:, 1], cheel[:, 5], cheel[:, 2],
    method='savgol', window_length=13, polyorder=3)

# ========== 母趾 ==========
kbigtoe_Lx, kbigtoe_Rx = model_free_correction(
    cbigtoe[:, 3], cbigtoe[:, 0], cbigtoe[:, 5], cbigtoe[:, 2],
    method='savgol', window_length=19, polyorder=3)
kbigtoe_Ly, kbigtoe_Ry = model_free_correction(
    cbigtoe[:, 4], cbigtoe[:, 1], cbigtoe[:, 5], cbigtoe[:, 2],
    method='savgol', window_length=15, polyorder=3)

# ========== 小趾 ==========
ksmalltoe_Lx, ksmalltoe_Rx = model_free_correction(
    csmalltoe[:, 3], csmalltoe[:, 0], csmalltoe[:, 5], csmalltoe[:, 2],
    method='savgol', window_length=19, polyorder=3)
ksmalltoe_Ly, ksmalltoe_Ry = model_free_correction(
    csmalltoe[:, 4], csmalltoe[:, 1], csmalltoe[:, 5], csmalltoe[:, 2],
    method='savgol', window_length=15, polyorder=3)




# --- 6 最終的な座標データの描画 & 保存 ---
display_coordinates = True
if display_coordinates:
    # 描画対象のデータを辞書にまとめる
    plot_data = {
        'hip': {'raw': chip, 'kalman_Rx':khip_Rx,'kalman_Lx':khip_Lx,'kalman_Ry':khip_Ry,'kalman_Ly':khip_Ly},
        'knee': {'raw': cknee, 'kalman_Rx':kknee_Rx,'kalman_Lx':kknee_Lx,'kalman_Ry':kknee_Ry,'kalman_Ly':kknee_Ly},
        'ankle': {'raw': cankle, 'kalman_Rx':kankle_Rx,'kalman_Lx':kankle_Lx,'kalman_Ry':kankle_Ry,'kalman_Ly':kankle_Ly},
        'bigtoe': {'raw': cbigtoe, 'kalman_Rx':kbigtoe_Rx,'kalman_Lx':kbigtoe_Lx,'kalman_Ry':kbigtoe_Ry,'kalman_Ly':kbigtoe_Ly},
        'smalltoe': {'raw': csmalltoe, 'kalman_Rx':ksmalltoe_Rx,'kalman_Lx':ksmalltoe_Lx,'kalman_Ry':ksmalltoe_Ry,'kalman_Ly':ksmalltoe_Ly},
        'heel': {'raw': cheel, 'kalman_Rx':kheel_Rx,'kalman_Lx':kheel_Lx,'kalman_Ry':kheel_Ry,'kalman_Ly':kheel_Ly},
    }

    for joint_name, data in plot_data.items():
        # --- X座標のプロット ---
        plt.figure(figsize=(10, 6))
        plt.plot(cframe, data['raw'][:, 0], color='r', label='Raw Right', alpha=0.3)
        plt.plot(cframe, data['raw'][:, 3], color='b', label='Raw Left', alpha=0.3)
        plt.plot(cframe, data['kalman_Rx'], color='r', label='Kalman Right')
        plt.plot(cframe, data['kalman_Lx'], color='b', label='Kalman Left')
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('X Coordinate [px]', fontsize=16)
        plt.ylim(0, min(3840, np.max([data['raw'][:, 0], data['raw'][:, 3], data['kalman_Rx'], data['kalman_Lx']])))
        plt.title(f'{joint_name.capitalize()} X Coordinate', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'coord_{joint_name}_x.png'))
        plt.close()

        # --- Y座標のプロット ---
        plt.figure(figsize=(10, 6))
        plt.plot(cframe, data['raw'][:, 1], color='r', label='Raw Right', alpha=0.3)
        plt.plot(cframe, data['raw'][:, 4], color='b', label='Raw Left', alpha=0.3)
        plt.plot(cframe, data['kalman_Ry'], color='r', label='Kalman Right')
        plt.plot(cframe, data['kalman_Ly'], color='b', label='Kalman Left')
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('Y Coordinate [px]', fontsize=16)
        plt.ylim(0, min(2160, np.max([data['raw'][:, 1], data['raw'][:, 4], data['kalman_Ry'], data['kalman_Ly']])))
        plt.title(f'{joint_name.capitalize()} Y Coordinate', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'coord_{joint_name}_y.png'))
        plt.close()

print("\n処理が完了し、すべてのグラフが保存されました。")

# --- 7 最終的な座標データの保存 ---
df_final = df.copy()

corrected_data = {
    'Ankle': (kankle_Lx, kankle_Rx, kankle_Ly, kankle_Ry),
    'Knee': (kknee_Lx, kknee_Rx, kknee_Ly, kknee_Ry),
    'Hip': (khip_Lx, khip_Rx, khip_Ly, khip_Ry),
    'BigToe': (kbigtoe_Lx, kbigtoe_Rx, kbigtoe_Ly, kbigtoe_Ry),
    'SmallToe': (ksmalltoe_Lx, ksmalltoe_Rx, ksmalltoe_Ly, ksmalltoe_Ry),
    'Heel': (kheel_Lx, kheel_Rx, kheel_Ly, kheel_Ry),
}

for col_name in df_final.columns:
        for joint_name, data_tuple in corrected_data.items():
            
            # 列名に関節名が含まれているかチェック
            if joint_name not in col_name:
                continue
                
            Lx, Rx, Ly, Ry = data_tuple
            
            if 'x' in col_name:
                if 'L' in col_name:
                    df_final.loc[start_frame:end_frame-1, col_name] = Lx
                    break # この列の処理は完了
                elif 'R' in col_name:
                    df_final.loc[start_frame:end_frame-1, col_name] = Rx
                    break # この列の処理は完了
            
            elif 'y' in col_name:
                if 'L' in col_name:
                    df_final.loc[start_frame:end_frame-1, col_name] = Ly
                    break # この列の処理は完了
                elif 'R' in col_name:
                    df_final.loc[start_frame:end_frame-1, col_name] = Ry
                    break # この列の処理は完了
                
# 保存
output_csv_path = os.path.join(path_op, f"{name}_savgol.csv")
# output_csv_path = os.path.join(path_op, f"{name}_kalman.csv")
df_final.to_csv(output_csv_path, index=False)