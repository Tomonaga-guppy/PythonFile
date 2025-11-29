import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

# 警告を非表示
warnings.filterwarnings('ignore')

# =============================================================================
# 関数定義セクション (変更なし)
# =============================================================================

def run_ca_kalman_rts(coords, confidences, dt=1/60.0, process_noise_scale=50.0, base_measure_noise=1.0):
    """
    等加速度モデル + RTSスムーシング (参照軌道の生成用)
    ※ base_measure_noise を小さくし、信頼できるデータには強く追従するように設定
    """
    
    n_steps = len(coords)
    dim_state = 3 # [位置, 速度, 加速度]
    
    # --- 1. 行列の定義 ---
    F = np.array([
        [1, dt, 0.5 * dt**2],
        [0, 1, dt],
        [0, 0, 1]
    ])
    
    H = np.array([
        [1, 0, 0]
    ])
    
    q_var = process_noise_scale ** 2
    dt2 = dt**2; dt3 = dt**3; dt4 = dt**4; dt5 = dt**5
    
    Q = np.array([
        [dt5/20, dt4/8, dt3/6],
        [dt4/8,  dt3/3, dt2/2],
        [dt3/6,  dt2/2, dt]
    ]) * q_var
    
    P = np.eye(dim_state) * 100.0
    x = np.zeros(dim_state)
    
    # 初期値設定
    for i in range(n_steps):
        if not np.isnan(coords[i]) and coords[i] != 0:
            x = np.array([coords[i], 0, 0])
            break
            
    # --- 保存用バッファ ---
    xs_pred = np.zeros((n_steps, dim_state))
    Ps_pred = np.zeros((n_steps, dim_state, dim_state))
    xs_filt = np.zeros((n_steps, dim_state))
    Ps_filt = np.zeros((n_steps, dim_state, dim_state))
    
    # --- 2. Forward Filtering ---
    for t in range(n_steps):
        # 予測
        if t == 0:
            x_pred = x
            P_pred = P
        else:
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
        
        xs_pred[t] = x_pred
        Ps_pred[t] = P_pred
        
        # 更新
        z = coords[t]
        conf = confidences[t] if confidences is not None else 1.0
        is_valid_obs = (not np.isnan(z)) and (z != 0) and (conf > 0.1)
        
        if is_valid_obs:
            r_val = base_measure_noise / (conf + 1e-6)
            R = np.array([[r_val]])
            
            y_res = z - (H @ x_pred)[0]
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            x = x_pred + K.flatten() * y_res
            P = (np.eye(dim_state) - K @ H) @ P_pred
        else:
            x = x_pred
            P = P_pred
        
        xs_filt[t] = x
        Ps_filt[t] = P
        
    # --- 3. Backward Smoothing (RTS) ---
    xs_smooth = np.zeros_like(xs_filt)
    Ps_smooth = np.zeros_like(Ps_filt)
    xs_smooth[-1] = xs_filt[-1]
    Ps_smooth[-1] = Ps_filt[-1]
    
    for t in range(n_steps - 2, -1, -1):
        P_pred_next = Ps_pred[t+1]
        P_filt_curr = Ps_filt[t]
        
        try:
            C = P_filt_curr @ F.T @ np.linalg.inv(P_pred_next)
        except np.linalg.LinAlgError:
            C = np.zeros_like(P_filt_curr)
            
        xs_smooth[t] = xs_filt[t] + C @ (xs_smooth[t+1] - xs_pred[t+1])
        Ps_smooth[t] = P_filt_curr + C @ (Ps_smooth[t+1] - P_pred_next) @ C.T
        
    return xs_smooth[:, 0]

# =============================================================================
# メインスクリプト (ハイブリッド処理の実装)
# =============================================================================

path_op = r'G:\gait_pattern\20250811_br\sub1\thera1-0\fl_yoloseg' 
name_op_excel = 'openpose.csv'
full_path_op = os.path.join(path_op, name_op_excel)
name = os.path.splitext(name_op_excel)[0]

output_dir = os.path.join(path_op, f"kalman_hybrid_results")
os.makedirs(output_dir, exist_ok=True)
print(f"グラフは '{output_dir}' に保存されます。")

# --- データ読み込み ---
try:
    df = pd.read_csv(full_path_op)
except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません -> {full_path_op}")
    exit()

# 座標データ取得
hip = df.iloc[:, [28,29,30, 37,38,39]].values
knee = df.iloc[:, [31,32,33, 40,41,42]].values
ankle = df.iloc[:, [34,35,36, 43,44,45]].values
bigtoe = df.iloc[:, [67,68,69, 58,59,60]].values
smalltoe = df.iloc[:, [70,71,72, 61,62,63]].values
heel = df.iloc[:, [73,74,75, 64,65,66]].values

# --- 範囲設定 ---
start_frame = 250
end_frame = 450
cframe = np.arange(start_frame, end_frame)

cankle = ankle[start_frame:end_frame]
cknee = knee[start_frame:end_frame]
chip = hip[start_frame:end_frame]
cbigtoe = bigtoe[start_frame:end_frame]
csmalltoe = smalltoe[start_frame:end_frame]
cheel = heel[start_frame:end_frame]

# --- 前処理 (0をNaNへ) ---
def mask_zeros(data_array):
    data_masked = data_array.copy()
    data_masked[:, 0] = np.where(data_masked[:, 0] == 0, np.nan, data_masked[:, 0])
    data_masked[:, 3] = np.where(data_masked[:, 3] == 0, np.nan, data_masked[:, 3])
    data_masked[:, 1] = np.where(data_masked[:, 1] == 0, np.nan, data_masked[:, 1])
    data_masked[:, 4] = np.where(data_masked[:, 4] == 0, np.nan, data_masked[:, 4])
    return data_masked

cankle = mask_zeros(cankle)
cknee = mask_zeros(cknee)
chip = mask_zeros(chip)
cbigtoe = mask_zeros(cbigtoe)
csmalltoe = mask_zeros(csmalltoe)
cheel = mask_zeros(cheel)

# --- ハイブリッドフィルタリング実行 ---
print("異常値のみKF補間するハイブリッド処理を適用中...")

# パラメータ設定
PN_SCALE = 50.0   # 加速度の追従性
MN_BASE = 1.0     # 観測ノイズ (小さくして、KFがRawデータに張り付くようにする)
DT = 1/60.0

# ★ 閾値設定（重要）
CONF_TH = 0.4     # これより信頼度が低ければ異常値とみなす
DEV_TH = 50.0     # KF予測値とRawデータの差がこれ(px)以上なら異常値とみなす

def apply_hybrid_filter(joint_data):
    """
    KF予測軌道を作成し、異常値のみを置き換える関数
    """
    # 結果格納用
    final_Lx = joint_data[:, 3].copy()
    final_Rx = joint_data[:, 0].copy()
    final_Ly = joint_data[:, 4].copy()
    final_Ry = joint_data[:, 1].copy()

    # --- 右側 (Right) ---
    # 1. KFで参照軌道を生成
    kf_Rx = run_ca_kalman_rts(joint_data[:, 0], joint_data[:, 2], dt=DT, process_noise_scale=PN_SCALE, base_measure_noise=MN_BASE)
    kf_Ry = run_ca_kalman_rts(joint_data[:, 1], joint_data[:, 2], dt=DT, process_noise_scale=PN_SCALE, base_measure_noise=MN_BASE)
    
    # 2. 異常値判定マスクを作成
    # 条件A: 信頼度が低い
    # 条件B: データがNaN (欠損)
    # 条件C: KF予測値との偏差が大きすぎる (スパイクノイズ)
    diff_Rx = np.abs(joint_data[:, 0] - kf_Rx)
    diff_Ry = np.abs(joint_data[:, 1] - kf_Ry)
    
    mask_outlier_Rx = (joint_data[:, 2] < CONF_TH) | np.isnan(joint_data[:, 0]) | (diff_Rx > DEV_TH)
    mask_outlier_Ry = (joint_data[:, 2] < CONF_TH) | np.isnan(joint_data[:, 1]) | (diff_Ry > DEV_TH)
    
    # 3. 置き換え (異常な部分だけKFの値を入れる)
    final_Rx[mask_outlier_Rx] = kf_Rx[mask_outlier_Rx]
    final_Ry[mask_outlier_Ry] = kf_Ry[mask_outlier_Ry]
    
    # --- 左側 (Left) ---
    kf_Lx = run_ca_kalman_rts(joint_data[:, 3], joint_data[:, 5], dt=DT, process_noise_scale=PN_SCALE, base_measure_noise=MN_BASE)
    kf_Ly = run_ca_kalman_rts(joint_data[:, 4], joint_data[:, 5], dt=DT, process_noise_scale=PN_SCALE, base_measure_noise=MN_BASE)
    
    diff_Lx = np.abs(joint_data[:, 3] - kf_Lx)
    diff_Ly = np.abs(joint_data[:, 4] - kf_Ly)
    
    mask_outlier_Lx = (joint_data[:, 5] < CONF_TH) | np.isnan(joint_data[:, 3]) | (diff_Lx > DEV_TH)
    mask_outlier_Ly = (joint_data[:, 5] < CONF_TH) | np.isnan(joint_data[:, 4]) | (diff_Ly > DEV_TH)
    
    final_Lx[mask_outlier_Lx] = kf_Lx[mask_outlier_Lx]
    final_Ly[mask_outlier_Ly] = kf_Ly[mask_outlier_Ly]
    
    return final_Lx, final_Rx, final_Ly, final_Ry

# 適用
kankle_Lx, kankle_Rx, kankle_Ly, kankle_Ry = apply_hybrid_filter(cankle)
kknee_Lx, kknee_Rx, kknee_Ly, kknee_Ry = apply_hybrid_filter(cknee)
khip_Lx, khip_Rx, khip_Ly, khip_Ry = apply_hybrid_filter(chip)
kbigtoe_Lx, kbigtoe_Rx, kbigtoe_Ly, kbigtoe_Ry = apply_hybrid_filter(cbigtoe)
ksmalltoe_Lx, ksmalltoe_Rx, ksmalltoe_Ly, ksmalltoe_Ry = apply_hybrid_filter(csmalltoe)
kheel_Lx, kheel_Rx, kheel_Ly, kheel_Ry = apply_hybrid_filter(cheel)

print("全関節の補正完了")

# --- グラフ描画 ---
# Rawデータ(点)と補正後データ(線)を重ねて表示し、Rawデータが維持されているか確認
display_coordinates = True
if display_coordinates:
    plot_data = {
        'hip': {'raw': chip, 'kalman_Rx':khip_Rx, 'kalman_Lx':khip_Lx},
        'knee': {'raw': cknee, 'kalman_Rx':kknee_Rx, 'kalman_Lx':kknee_Lx},
        'ankle': {'raw': cankle, 'kalman_Rx':kankle_Rx, 'kalman_Lx':kankle_Lx},
        'bigtoe': {'raw': cbigtoe, 'kalman_Rx':kbigtoe_Rx, 'kalman_Lx':kbigtoe_Lx},
        'smalltoe': {'raw': csmalltoe, 'kalman_Rx':ksmalltoe_Rx, 'kalman_Lx':ksmalltoe_Lx},
        'heel': {'raw': cheel, 'kalman_Rx':kheel_Rx, 'kalman_Lx':kheel_Lx},
    }

    for joint_name, data in plot_data.items():
        plt.figure(figsize=(10, 6))
        # 生データを散布図（点）で描画して、データの残り具合を確認
        plt.scatter(cframe, data['raw'][:, 0], color='r', s=10, label='Raw Right', alpha=0.3)
        plt.scatter(cframe, data['raw'][:, 3], color='b', s=10, label='Raw Left', alpha=0.3)
        
        # 補正後を線で描画
        plt.plot(cframe, data['kalman_Rx'], color='r', label='Hybrid Right')
        plt.plot(cframe, data['kalman_Lx'], color='b', label='Hybrid Left')
        
        plt.title(f'{joint_name.capitalize()} X (Raw vs Hybrid)', fontsize=18)
        valid_data = data['raw'][:, 0][~np.isnan(data['raw'][:, 0])]
        if len(valid_data) > 0:
            plt.ylim(0, np.max(valid_data) + 100)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'coord_{joint_name}_x.png'))
        plt.close()

# --- CSV保存 (変更なし) ---
df_final = df.copy()
final_results = {
    'Ankle': (kankle_Lx, kankle_Rx, kankle_Ly, kankle_Ry),
    'Knee': (kknee_Lx, kknee_Rx, kknee_Ly, kknee_Ry),
    'Hip': (khip_Lx, khip_Rx, khip_Ly, khip_Ry),
    'BigToe': (kbigtoe_Lx, kbigtoe_Rx, kbigtoe_Ly, kbigtoe_Ry),
    'SmallToe': (ksmalltoe_Lx, ksmalltoe_Rx, ksmalltoe_Ly, ksmalltoe_Ry),
    'Heel': (kheel_Lx, kheel_Rx, kheel_Ly, kheel_Ry),
}

for col_name in df_final.columns:
    for joint_name, (Lx, Rx, Ly, Ry) in final_results.items():
        if joint_name not in col_name:
            continue
        target_slice = slice(start_frame, end_frame)
        if 'x' in col_name:
            if 'L' in col_name: df_final.loc[target_slice, col_name] = Lx
            elif 'R' in col_name: df_final.loc[target_slice, col_name] = Rx
        elif 'y' in col_name:
            if 'L' in col_name: df_final.loc[target_slice, col_name] = Ly
            elif 'R' in col_name: df_final.loc[target_slice, col_name] = Ry

output_csv_path = os.path.join(path_op, f"{name}_kalman_hybrid.csv")
df_final.to_csv(output_csv_path, index=False)
print(f"補正後のCSVファイル: {output_csv_path}")