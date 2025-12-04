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
def exponential_smoothing(data, alpha):
    """
    指数平滑法による予測値を計算（改善版）
    
    引数:
    data  : np.array - 時系列データ
    alpha : float    - 平滑化パラメータ (0 < alpha <= 1)
                      大きいほど最新のデータを重視
    
    戻り値:
    次のステップの予測値
    """
    if len(data) < 2:
        return data[-1] if len(data) > 0 else 0
    
    # 単純指数平滑法
    smoothed_values = np.zeros(len(data))
    smoothed_values[0] = data[0]
    
    for i in range(1, len(data)):
        smoothed_values[i] = alpha * data[i] + (1 - alpha) * smoothed_values[i-1]
    
    # 最新の平滑化値を返す（トレンドは加えない）
    return smoothed_values[-1]

def exponential_smoothing_with_trend(data, alpha, beta):
    """
    Holt's指数平滑法（トレンドを考慮）
    
    引数:
    data  : np.array - 時系列データ
    alpha : float    - レベルの平滑化パラメータ (0 < alpha <= 1)
    beta  : float    - トレンドの平滑化パラメータ (0 < beta <= 1)
    
    戻り値:
    次のステップの予測値
    """
    if len(data) < 2:
        return data[-1] if len(data) > 0 else 0
    
    # 初期値の設定
    level = data[0]
    trend = data[1] - data[0] if len(data) > 1 else 0
    
    # Holt's指数平滑法の計算
    for i in range(1, len(data)):
        prev_level = level
        level = alpha * data[i] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    
    # 次のステップの予測
    prediction = level + trend
    
    return prediction

def diff2_with_exponential_smoothing(coordinate_L, coordinate_R, th, alpha, beta, window_size=5):
    """
    二階差分による外れ値検出 + 指数平滑法による補正
    
    引数:
    coordinate_L : np.array - 左側の座標データ
    coordinate_R : np.array - 右側の座標データ
    th           : float    - 加速度の閾値
    alpha        : float    - 指数平滑法の平滑化パラメータ (0 < alpha <= 1)
    beta         : float    - トレンドの平滑化パラメータ (0 < beta <= 1)
    window_size  : int      - 予測に使用する過去データのウィンドウサイズ
    
    戻り値:
    補正された左右の座標データ
    """
    end_step = len(coordinate_R)
    
    # 元のデータを変更しないようにコピーを作成
    coordinate_L = coordinate_L.copy()
    coordinate_R = coordinate_R.copy()

    # 誤検出の種類を記録するための配列
    miss_point = np.zeros(end_step)
    
    # 2フレーム目から最終フレームまでループ
    for i in range(2, end_step):
        correction_flag = 0  # 補正が行われたかを判定するフラグ
        
        # 加速度を計算
        q1L = coordinate_L[i] - coordinate_L[i-1]
        q2L = coordinate_L[i-1] - coordinate_L[i-2]
        diff2_L = q1L - q2L
        
        q1R = coordinate_R[i] - coordinate_R[i-1]
        q2R = coordinate_R[i-1] - coordinate_R[i-2]
        diff2_R = q1R - q2R

        # パターン1: 左右両方の加速度が閾値を超えた場合
        if abs(diff2_L) > th and abs(diff2_R) > th:
            # 座標を入れ替えてみる
            Lbox, Rbox = coordinate_L[i], coordinate_R[i]
            coordinate_L[i], coordinate_R[i] = Rbox, Lbox
            
            # 入れ替え後の加速度を再計算
            q1L_swap = coordinate_L[i] - coordinate_L[i-1]
            q2L_swap = coordinate_L[i-1] - coordinate_L[i-2]
            diff2_L_swap = q1L_swap - q2L_swap
            
            q1R_swap = coordinate_R[i] - coordinate_R[i-1]
            q2R_swap = coordinate_R[i-1] - coordinate_R[i-2]
            diff2_R_swap = q1R_swap - q2R_swap
            
            # それでも閾値を超えている場合 -> 両方誤検出
            if abs(diff2_L_swap) > th and abs(diff2_R_swap) > th:
                coordinate_L[i], coordinate_R[i] = Lbox, Rbox  # 入れ替えを戻す
                
                # 指数平滑法で補正
                start_idx = max(0, i - window_size)
                pred_L = exponential_smoothing_with_trend(coordinate_L[start_idx:i], alpha, beta)
                pred_R = exponential_smoothing_with_trend(coordinate_R[start_idx:i], alpha, beta)
                
                coordinate_L[i] = pred_L
                coordinate_R[i] = pred_R
                miss_point[i] = 4  # 両方誤検出
                correction_flag = 1
            else:  # 入れ替えで改善した場合
                miss_point[i] = 1  # 入れ替わり
                correction_flag = 1
        
        # パターン2: 左足のみ閾値を超えた場合
        elif abs(diff2_L) > th and abs(diff2_R) <= th:
            start_idx = max(0, i - window_size)
            pred_L = exponential_smoothing_with_trend(coordinate_L[start_idx:i], alpha, beta)
            coordinate_L[i] = pred_L
            miss_point[i] = 2  # 左足誤検出
            correction_flag = 1
            
        # パターン3: 右足のみ閾値を超えた場合
        elif abs(diff2_L) <= th and abs(diff2_R) > th:
            start_idx = max(0, i - window_size)
            pred_R = exponential_smoothing_with_trend(coordinate_R[start_idx:i], alpha, beta)
            coordinate_R[i] = pred_R
            miss_point[i] = 3  # 右足誤検出
            correction_flag = 1
            
        # 補正後の追加チェック
        if correction_flag == 1:
            p1L = coordinate_L[i] - coordinate_L[i-1]
            p2L = coordinate_L[i-1] - coordinate_L[i-2]
            diff2_L_cover = p1L - p2L
            
            p1R = coordinate_R[i] - coordinate_R[i-1]
            p2R = coordinate_R[i-1] - coordinate_R[i-2]
            diff2_R_cover = p1R - p2R
            
            th_cover = 500
            if abs(diff2_L_cover) >= th_cover:
                coordinate_L[i] = coordinate_L[i-1] + p2L
                
            if abs(diff2_R_cover) >= th_cover:
                coordinate_R[i] = coordinate_R[i-1] + p2R

    return coordinate_L, coordinate_R

# =============================================================================
# メインスクリプト
# =============================================================================

# --- 1. openposeから得られた座標をエクセルから取得 ---
# ★ ユーザーはこれらのパスを自分の環境に合わせて変更する必要があります。
# path_op = r'G:\gait_pattern\20250811_br\sub0\thera0-16\fl' # OpenPoseの座標データ(csv)があるパス
path_op = r'G:\gait_pattern\20250811_br\sub1\thera0-3\fr' # OpenPoseの座標データ(csv)があるパス
name_op_excel = 'openpose.csv'  # 処理対象のファイル名
full_path_op = os.path.join(path_op, name_op_excel)
name = os.path.splitext(name_op_excel)[0] # 拡張子を除いたファイル名を取得

# --- 結果保存用ディレクトリの作成 ---
output_dir = os.path.join(path_op, f"newfilt_results")
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
start_frame = 170 #FL約-2m地点 1-0-3
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
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(cframe, data[:, 1], label='Right Y', color='red', alpha=0.8)
        plt.plot(cframe, data[:, 4], label='Left Y', color='blue', alpha=0.8)
        plt.title(f'Pre {joint_name.capitalize()} Y Coordinates', fontsize=18)
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('Coordinate [px]', fontsize=16)
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

# --- 5. フィルタで補正 ---
# (左座標, 右座標, 加速度の閾値, カルマンの初期値) ★データによって閾値を変更する必要あり
print("フィルタを適用中...")

# パラメータ設定
alpha = 0.2  # 平滑化パラメータ (0.1~0.9で調整)
beta = 0.3   # トレンド平滑化パラメータ (Holt's法用)
window_size = 10  # 予測に使う過去データ数（200フレーム中なら5~15推奨）

alpha_hip_x, beta_hip_x = 0.2, 0.2
alpha_hip_y, beta_hip_y = 0.2, 0.2
alpha_knee_x, beta_knee_x = 0.2, 0.3
alpha_knee_y, beta_knee_y = 0.2, 0.3
alpha_ankle_x, beta_ankle_x = 0.2, 0.3
alpha_ankle_y, beta_ankle_y = 0.2, 0.3
alpha_smalltoe_x, beta_smalltoe_x = 0.2, 0.3
alpha_smalltoe_y, beta_smalltoe_y = 0.2, 0.3
alpha_bigtoe_x, beta_bigtoe_x = 0.2, 0.3
alpha_bigtoe_y, beta_bigtoe_y = 0.2, 0.2
alpha_heel_x, beta_heel_x = 0.2, 0.3
alpha_heel_y, beta_heel_y = 0.2, 0.3

filter_params = {
    'hip_params': (alpha_hip_x, beta_hip_x, alpha_hip_y, beta_hip_y),
    'knee_params': (alpha_knee_x, beta_knee_x, alpha_knee_y, beta_knee_y),
    'ankle_params': (alpha_ankle_x, beta_ankle_x, alpha_ankle_y, beta_ankle_y),
    'bigtoe_params': (alpha_bigtoe_x, beta_bigtoe_x, alpha_bigtoe_y, beta_bigtoe_y),
    'smalltoe_params': (alpha_smalltoe_x, beta_smalltoe_x, alpha_smalltoe_y, beta_smalltoe_y),
    'heel_params': (alpha_heel_x, beta_heel_x, alpha_heel_y, beta_heel_y),
}

#filter_paramsを保存
params_df = pd.DataFrame.from_dict(filter_params, orient='index', columns=['alpha_x', 'beta_x', 'alpha_y', 'beta_y'])
params_df.to_csv(os.path.join(output_dir, 'filter_params.csv'))

# FR #
khip_Lx, khip_Rx = diff2_with_exponential_smoothing(chip[:, 3], chip[:, 0], 50, alpha_hip_x, beta_hip_x, window_size)
print("カルマンフィルタ: 股関節X座標補正完了")
khip_Ly, khip_Ry = diff2_with_exponential_smoothing(chip[:, 4], chip[:, 1], 50, alpha_hip_y, beta_hip_y, window_size)
print("カルマンフィルタ: 股関節Y座標補正完了")
kknee_Lx, kknee_Rx = diff2_with_exponential_smoothing(cknee[:, 3], cknee[:, 0], 200, alpha_knee_x, beta_knee_x, window_size)
print("カルマンフィルタ: 膝X座標補正完了")
kknee_Ly, kknee_Ry = diff2_with_exponential_smoothing(cknee[:, 4], cknee[:, 1], 50, alpha_knee_y, beta_knee_y, window_size)
print("カルマンフィルタ: 膝Y座標補正完了")
kankle_Lx, kankle_Rx = diff2_with_exponential_smoothing(cankle[:, 3], cankle[:, 0], 200, alpha_ankle_x, beta_ankle_x, window_size)
print("カルマンフィルタ: 足首X座標補正完了")
kankle_Ly, kankle_Ry = diff2_with_exponential_smoothing(cankle[:, 4], cankle[:, 1], 50, alpha_ankle_y, beta_ankle_y, window_size)
print("カルマンフィルタ: 足首Y座標補正完了")
kbigtoe_Lx, kbigtoe_Rx = diff2_with_exponential_smoothing(cbigtoe[:, 3], cbigtoe[:, 0], 200, alpha_bigtoe_x, beta_bigtoe_x, window_size)
print("カルマンフィルタ: 母趾X座標補正完了")
kbigtoe_Ly, kbigtoe_Ry = diff2_with_exponential_smoothing(cbigtoe[:, 4], cbigtoe[:, 1], 40, alpha_bigtoe_y, beta_bigtoe_y, window_size)
print("カルマンフィルタ: 母趾Y座標補正完了")
ksmalltoe_Lx, ksmalltoe_Rx = diff2_with_exponential_smoothing(csmalltoe[:, 3], csmalltoe[:, 0], 200, alpha_smalltoe_x, beta_smalltoe_x, window_size)
print("カルマンフィルタ: 小趾X座標補正完了")
ksmalltoe_Ly, ksmalltoe_Ry = diff2_with_exponential_smoothing(csmalltoe[:, 4], csmalltoe[:, 1], 50, alpha_smalltoe_y, beta_smalltoe_y, window_size)
print("カルマンフィルタ: 小趾Y座標補正完了")
kheel_Lx, kheel_Rx = diff2_with_exponential_smoothing(cheel[:, 3], cheel[:, 0], 200, alpha_heel_x, beta_heel_x, window_size)
print("カルマンフィルタ: 踵X座標補正完了")
kheel_Ly, kheel_Ry = diff2_with_exponential_smoothing(cheel[:, 4], cheel[:, 1], 50, alpha_heel_y, beta_heel_y, window_size)
print("カルマンフィルタ: 踵Y座標補正完了")

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
        plt.title(f'{joint_name.capitalize()} X Coordinate', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'coord_{joint_name}_x_a{alpha}_b{beta}.png'))
        plt.close()

        # --- Y座標のプロット ---
        plt.figure(figsize=(10, 6))
        plt.plot(cframe, data['raw'][:, 1], color='r', label='Raw Right', alpha=0.3)
        plt.plot(cframe, data['raw'][:, 4], color='b', label='Raw Left', alpha=0.3)
        plt.plot(cframe, data['kalman_Ry'], color='r', label='Kalman Right')
        plt.plot(cframe, data['kalman_Ly'], color='b', label='Kalman Left')
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('Y Coordinate [px]', fontsize=16)
        plt.title(f'{joint_name.capitalize()} Y Coordinate', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'coord_{joint_name}_y_a{alpha}_b{beta}.png'))
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
output_csv_path = os.path.join(path_op, f"{name}_exp.csv")
df_final.to_csv(output_csv_path, index=False)