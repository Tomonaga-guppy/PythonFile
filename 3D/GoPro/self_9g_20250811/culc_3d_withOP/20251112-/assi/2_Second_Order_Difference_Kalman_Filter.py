import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize
import os
import warnings

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
    var_eps : float         - 観測ノイズの分散 (σ^2_ε)
    """
    
    L = len(y)
    
    # 計算結果を格納するための配列を事前に確保
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
        v_t[t] = y[t] - a_tt1[t]
        f_t[t] = p_tt1[t] + var_eps
        
        if not np.isfinite(f_t[t]):
            k_t[t] = 1.0
        elif f_t[t] == 0:
            k_t[t] = 0.0
        else:
            k_t[t] = p_tt1[t] / f_t[t]
            
        a_tt[t] = a_tt1[t] + k_t[t] * v_t[t]
        p_tt[t] = p_tt1[t] * (1 - k_t[t])
        
        a_tt1[t+1] = a_tt[t]
        p_tt1[t+1] = p_tt[t] + var_eta
        
    return a_tt, p_tt, f_t, v_t

def calc_log_diffuse_llhd(vars, y):
    """
    ローカルトレンドモデルの散漫な対数尤度を求める関数
    """
    psi_eta, psi_eps = vars
    var_eta = np.exp(2 * psi_eta)
    var_eps = np.exp(2 * psi_eps)
    L = len(y)
    
    if L < 2:
        return -np.inf
        
    a1 = y[0]
    p1 = var_eps
    
    _, _, f_t, v_t = local_trend_kf(y, a1, p1, var_eta, var_eps)
    
    if np.any(f_t[1:] <= 0) or not np.all(np.isfinite(f_t[1:])):
        return -np.inf

    tmp = np.sum(np.log(f_t[1:]) + v_t[1:]**2 / f_t[1:])
    log_ld = -0.5 * L * np.log(2 * np.pi) - 0.5 * tmp
    
    return log_ld

def maf(input_data, size):
    """
    移動平均フィルタ(Moving Average Filter)
    """
    window_size = size
    b = (1 / window_size) * np.ones(window_size)
    a = 1
    from scipy.signal import lfilter
    return lfilter(b, a, input_data)

def is_valid_point(coordinate, index):
    """
    座標が有効（検出されている）かをチェック
    座標が0の場合は未検出とみなす
    """
    if index >= len(coordinate):
        return False
    return coordinate[index] != 0

def calculate_acceleration(coordinate, i):
    """
    指定されたインデックスでの加速度を計算
    """
    if i < 2:
        return 0
    # 二階差分 (加速度)
    accel = coordinate[i] - 2 * coordinate[i-1] + coordinate[i-2]
    return accel

def kalman2_multi_person(coordinate_L_p0, coordinate_R_p0, 
                         coordinate_L_p1, coordinate_R_p1,
                         th, initial_value):
    """
    複数人対応の二階差分カルマンフィルタ
    person0の座標を補正（person1は参照のみ）
    
    引数:
    coordinate_L_p0 : person0の左座標
    coordinate_R_p0 : person0の右座標
    coordinate_L_p1 : person1の左座標
    coordinate_R_p1 : person1の右座標
    th : 加速度の閾値
    initial_value : カルマンフィルタの初期値
    """
    end_step = len(coordinate_R_p0)
    
    # 元のデータを変更しないようにコピーを作成
    coordinate_L_p0 = coordinate_L_p0.copy()
    coordinate_R_p0 = coordinate_R_p0.copy()

    # 誤検出の種類を記録するための配列
    miss_point = np.zeros(end_step)
    
    # 統計情報
    swap_stats = {
        'original': 0,
        'swap_p0_LR': 0,
        'swap_p1_R': 0,
        'swap_p1_L': 0
    }
    
    # 2フレーム目から最終フレームまでループ
    for i in range(2, end_step):
        kalman_flag = 0
        
        # 現在のフレームまでのデータスライスを取得
        current_coordinate_L_p0 = coordinate_L_p0[:i]
        current_coordinate_R_p0 = coordinate_R_p0[:i]

        # 座標の差分(速度)を計算し、移動平均を適用
        diff_data_Lx = np.diff(current_coordinate_L_p0)
        yL = maf(diff_data_Lx, 3)
        diff_data_Rx = np.diff(current_coordinate_R_p0)
        yR = maf(diff_data_Rx, 3)
        
        if len(yL) < 2 or len(yR) < 2:
            continue

        # パラメータ設定
        parL = initial_value
        parR = initial_value
        
        psi_eta_L = np.log(np.sqrt(parL))
        psi_eps_L = np.log(np.sqrt(parL))
        psi_eta_R = np.log(np.sqrt(parR))
        psi_eps_R = np.log(np.sqrt(parR))
        
        x0L = [psi_eta_L, psi_eps_L]
        x0R = [psi_eta_R, psi_eps_R]
        
        fL = lambda xL: -calc_log_diffuse_llhd(xL, yL)
        fR = lambda xR: -calc_log_diffuse_llhd(xR, yR)
        
        bounds = ((-20, 20), (-20, 20))
        
        resL = minimize(fL, x0L, method='L-BFGS-B', bounds=bounds)
        resR = minimize(fR, x0R, method='L-BFGS-B', bounds=bounds)
        
        xoptL = resL.x
        xoptR = resR.x
        
        var_eta_opt_L = np.exp(2 * xoptL[0])
        var_eps_opt_L = np.exp(2 * xoptL[1])
        var_eta_opt_R = np.exp(2 * xoptR[0])
        var_eps_opt_R = np.exp(2 * xoptR[1])
        
        var_eps_L = var_eta_opt_L
        var_eta_L = var_eps_opt_L
        var_eps_R = var_eta_opt_R
        var_eta_R = var_eps_opt_R
        
        a1L = var_eps_L
        p1L = var_eta_L
        a1R = var_eps_R
        p1R = var_eta_R
        
        # カルマンフィルタ実行
        a_ttL, p_ttL, f_tL, v_tL = local_trend_kf(yL, a1L, p1L, var_eta_L, var_eps_L)
        a_ttR, p_ttR, f_tR, v_tR = local_trend_kf(yR, a1R, p1R, var_eta_R, var_eps_R)
        
        # 加速度計算（二階差分）
        accel_L = calculate_acceleration(coordinate_L_p0, i)
        accel_R = calculate_acceleration(coordinate_R_p0, i)
        
        # 右側の異常値チェック
        if abs(accel_R) > th:
            # 4パターンの候補を準備
            candidates = []
            
            # パターン0: オリジナル（入れ替えなし）
            candidates.append({
                'accel': abs(accel_R),
                'type': 'original',
                'value': None
            })
            
            # パターン1: person0の左右入れ替え
            if is_valid_point(coordinate_L_p0, i):
                accel_swap_p0_LR = calculate_acceleration(coordinate_L_p0, i)
                candidates.append({
                    'accel': abs(accel_swap_p0_LR),
                    'type': 'swap_p0_LR',
                    'value': coordinate_L_p0[i]
                })
            
            # パターン2: person1の右（同側）と入れ替え
            if is_valid_point(coordinate_R_p1, i):
                # 仮想的に入れ替えた場合の加速度を計算
                temp_coord_R = coordinate_R_p0.copy()
                temp_coord_R[i] = coordinate_R_p1[i]
                accel_swap_p1_R = calculate_acceleration(temp_coord_R, i)
                candidates.append({
                    'accel': abs(accel_swap_p1_R),
                    'type': 'swap_p1_R',
                    'value': coordinate_R_p1[i]
                })
            
            # パターン3: person1の左（対側）と入れ替え
            if is_valid_point(coordinate_L_p1, i):
                temp_coord_R = coordinate_R_p0.copy()
                temp_coord_R[i] = coordinate_L_p1[i]
                accel_swap_p1_L = calculate_acceleration(temp_coord_R, i)
                candidates.append({
                    'accel': abs(accel_swap_p1_L),
                    'type': 'swap_p1_L',
                    'value': coordinate_L_p1[i]
                })
            
            # 最小加速度のパターンを選択
            best_candidate = min(candidates, key=lambda x: x['accel'])
            
            # 統計情報を更新
            swap_stats[best_candidate['type']] += 1
            
            # 最適なパターンの加速度が閾値以下の場合のみ補正
            if best_candidate['accel'] < th:
                if best_candidate['type'] == 'swap_p0_LR':
                    # person0の左右入れ替え
                    coordinate_R_p0[i] = best_candidate['value']
                    kalman_flag = 1
                    miss_point[i] = 1
                elif best_candidate['type'] == 'swap_p1_R':
                    # person1の右と入れ替え
                    coordinate_R_p0[i] = best_candidate['value']
                    kalman_flag = 1
                    miss_point[i] = 2
                elif best_candidate['type'] == 'swap_p1_L':
                    # person1の左と入れ替え
                    coordinate_R_p0[i] = best_candidate['value']
                    kalman_flag = 1
                    miss_point[i] = 3
            else:
                # どのパターンでも閾値を超える場合、カルマンフィルタで補正
                coordinate_R_p0[i] = a_ttR[-1]
                kalman_flag = 1
                miss_point[i] = 4
        
        # 左側の異常値チェック
        if abs(accel_L) > th:
            candidates = []
            
            # パターン0: オリジナル
            candidates.append({
                'accel': abs(accel_L),
                'type': 'original',
                'value': None
            })
            
            # パターン1: person0の左右入れ替え
            if is_valid_point(coordinate_R_p0, i):
                accel_swap_p0_RL = calculate_acceleration(coordinate_R_p0, i)
                candidates.append({
                    'accel': abs(accel_swap_p0_RL),
                    'type': 'swap_p0_LR',
                    'value': coordinate_R_p0[i]
                })
            
            # パターン2: person1の左（同側）と入れ替え
            if is_valid_point(coordinate_L_p1, i):
                temp_coord_L = coordinate_L_p0.copy()
                temp_coord_L[i] = coordinate_L_p1[i]
                accel_swap_p1_L = calculate_acceleration(temp_coord_L, i)
                candidates.append({
                    'accel': abs(accel_swap_p1_L),
                    'type': 'swap_p1_L',
                    'value': coordinate_L_p1[i]
                })
            
            # パターン3: person1の右（対側）と入れ替え
            if is_valid_point(coordinate_R_p1, i):
                temp_coord_L = coordinate_L_p0.copy()
                temp_coord_L[i] = coordinate_R_p1[i]
                accel_swap_p1_R = calculate_acceleration(temp_coord_L, i)
                candidates.append({
                    'accel': abs(accel_swap_p1_R),
                    'type': 'swap_p1_R',
                    'value': coordinate_R_p1[i]
                })
            
            # 最小加速度のパターンを選択
            best_candidate = min(candidates, key=lambda x: x['accel'])
            
            # 統計情報を更新
            swap_stats[best_candidate['type']] += 1
            
            # 補正
            if best_candidate['accel'] < th:
                if best_candidate['type'] == 'swap_p0_LR':
                    coordinate_L_p0[i] = best_candidate['value']
                    kalman_flag = 1
                    miss_point[i] = 1
                elif best_candidate['type'] == 'swap_p1_L':
                    coordinate_L_p0[i] = best_candidate['value']
                    kalman_flag = 1
                    miss_point[i] = 2
                elif best_candidate['type'] == 'swap_p1_R':
                    coordinate_L_p0[i] = best_candidate['value']
                    kalman_flag = 1
                    miss_point[i] = 3
            else:
                coordinate_L_p0[i] = a_ttL[-1]
                kalman_flag = 1
                miss_point[i] = 4
    
    # 統計情報を出力
    total_corrections = sum(swap_stats.values()) - swap_stats['original']
    if total_corrections > 0:
        print(f"  補正統計:")
        print(f"    - Person0内で左右入れ替え: {swap_stats['swap_p0_LR']}回")
        print(f"    - Person1の同側と入れ替え: {swap_stats['swap_p1_R']}回")
        print(f"    - Person1の対側と入れ替え: {swap_stats['swap_p1_L']}回")
    
    return coordinate_L_p0, coordinate_R_p0

# =============================================================================
# メイン処理
# =============================================================================

# --- 1. CSVファイルの選択 ---
print("CSVファイルを選択してください...")
# ★ ここにパスを入力
path_op = r'G:\gait_pattern\20250811_br\sub1\thera1-0\fr'
name = 'openpose'  # ファイル名のベース（_person0, _person1の前の部分）

# Person0とPerson1のファイルパス
file_path_p0 = os.path.join(path_op, f"{name}_person0.csv")
file_path_p1 = os.path.join(path_op, f"{name}_person1.csv")

# ファイルの存在確認
if not os.path.exists(file_path_p0):
    raise FileNotFoundError(f"Person0のファイルが見つかりません: {file_path_p0}")
if not os.path.exists(file_path_p1):
    raise FileNotFoundError(f"Person1のファイルが見つかりません: {file_path_p1}")

# --- 2. データ読み込み ---
print("データを読み込み中...")
df_p0 = pd.read_csv(file_path_p0, index_col=0)
df_p1 = pd.read_csv(file_path_p1, index_col=0)

print(f"Person0: {len(df_p0)} フレーム")
print(f"Person1: {len(df_p1)} フレーム")

# フレーム数が一致するか確認
if len(df_p0) != len(df_p1):
    print(f"警告: Person0とPerson1のフレーム数が異なります")

# --- 3. 必要なデータを抽出 ---
start_frame = 0
end_frame = len(df_p0)

# フレーム番号
cframe = np.arange(start_frame, end_frame)

# Person0のデータ抽出
print("Person0の座標データを抽出中...")
chip_p0 = df_p0.loc[start_frame:end_frame-1, ['RHip_x', 'RHip_y', 'RHip_p', 'LHip_x', 'LHip_y', 'LHip_p']].values
cknee_p0 = df_p0.loc[start_frame:end_frame-1, ['RKnee_x', 'RKnee_y', 'RKnee_p', 'LKnee_x', 'LKnee_y', 'LKnee_p']].values
cankle_p0 = df_p0.loc[start_frame:end_frame-1, ['RAnkle_x', 'RAnkle_y', 'RAnkle_p', 'LAnkle_x', 'LAnkle_y', 'LAnkle_p']].values
cbigtoe_p0 = df_p0.loc[start_frame:end_frame-1, ['RBigToe_x', 'RBigToe_y', 'RBigToe_p', 'LBigToe_x', 'LBigToe_y', 'LBigToe_p']].values
csmalltoe_p0 = df_p0.loc[start_frame:end_frame-1, ['RSmallToe_x', 'RSmallToe_y', 'RSmallToe_p', 'LSmallToe_x', 'LSmallToe_y', 'LSmallToe_p']].values
cheel_p0 = df_p0.loc[start_frame:end_frame-1, ['RHeel_x', 'RHeel_y', 'RHeel_p', 'LHeel_x', 'LHeel_y', 'LHeel_p']].values

# Person1のデータ抽出
print("Person1の座標データを抽出中...")
chip_p1 = df_p1.loc[start_frame:end_frame-1, ['RHip_x', 'RHip_y', 'RHip_p', 'LHip_x', 'LHip_y', 'LHip_p']].values
cknee_p1 = df_p1.loc[start_frame:end_frame-1, ['RKnee_x', 'RKnee_y', 'RKnee_p', 'LKnee_x', 'LKnee_y', 'LKnee_p']].values
cankle_p1 = df_p1.loc[start_frame:end_frame-1, ['RAnkle_x', 'RAnkle_y', 'RAnkle_p', 'LAnkle_x', 'LAnkle_y', 'LAnkle_p']].values
cbigtoe_p1 = df_p1.loc[start_frame:end_frame-1, ['RBigToe_x', 'RBigToe_y', 'RBigToe_p', 'LBigToe_x', 'LBigToe_y', 'LBigToe_p']].values
csmalltoe_p1 = df_p1.loc[start_frame:end_frame-1, ['RSmallToe_x', 'RSmallToe_y', 'RSmallToe_p', 'LSmallToe_x', 'LSmallToe_y', 'LSmallToe_p']].values
cheel_p1 = df_p1.loc[start_frame:end_frame-1, ['RHeel_x', 'RHeel_y', 'RHeel_p', 'LHeel_x', 'LHeel_y', 'LHeel_p']].values

# --- 4. グラフ保存用ディレクトリ作成 ---
output_dir = os.path.join(path_op, 'kalman_plots')
os.makedirs(output_dir, exist_ok=True)
print(f"グラフ保存先: {output_dir}")

# --- 5. 補正前のグラフ作成 ---
display_pre_correction = True
if display_pre_correction:
    print("補正前の座標と加速度グラフを作成中...")
    
    # 座標データの辞書
    joints_data_p0 = {
        'hip': chip_p0,
        'knee': cknee_p0,
        'ankle': cankle_p0,
        'bigtoe': cbigtoe_p0,
        'smalltoe': csmalltoe_p0,
        'heel': cheel_p0,
    }
    
    for joint_name, joint_data in joints_data_p0.items():
        # 加速度計算用のフレーム
        cframe_a = cframe[2:]
        
        # X座標の加速度
        accel_Rx = np.diff(joint_data[:, 0], n=2)
        accel_Lx = np.diff(joint_data[:, 3], n=2)
        
        # Y座標の加速度
        accel_Ry = np.diff(joint_data[:, 1], n=2)
        accel_Ly = np.diff(joint_data[:, 4], n=2)
        
        # X座標のプロット
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(cframe, joint_data[:, 0], label='Right X', color='red', alpha=0.8)
        plt.plot(cframe, joint_data[:, 3], label='Left X', color='blue', alpha=0.8)
        plt.title(f'Pre-correction {joint_name.capitalize()} X Coordinate (Person0)', fontsize=18)
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('X Coordinate [px]', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(cframe_a, accel_Rx, label='Right X Accel', color='red', alpha=0.8)
        plt.plot(cframe_a, accel_Lx, label='Left X Accel', color='blue', alpha=0.8)
        plt.ylim(-100,100)
        plt.title(f'Pre-correction {joint_name.capitalize()} X Acceleration (Person0)', fontsize=18)
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('Acceleration [px/s²]', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pre_{joint_name}_x_accel.png'))
        plt.close()
        
        # Y座標のプロット
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(cframe, joint_data[:, 1], label='Right Y', color='red', alpha=0.8)
        plt.plot(cframe, joint_data[:, 4], label='Left Y', color='blue', alpha=0.8)
        plt.title(f'Pre-correction {joint_name.capitalize()} Y Coordinate (Person0)', fontsize=18)
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('Y Coordinate [px]', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(cframe_a, accel_Ry, label='Right Y Accel', color='red', alpha=0.8)
        plt.plot(cframe_a, accel_Ly, label='Left Y Accel', color='blue', alpha=0.8)
        plt.ylim(-100,100)
        plt.title(f'Pre-correction {joint_name.capitalize()} Y Acceleration (Person0)', fontsize=18)
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('Acceleration [px/s²]', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pre_{joint_name}_y_accel.png'))
        plt.close()
    
    print("補正前の座標と加速度グラフの作成が完了しました。")

# --- 6. カルマンフィルタで補正（複数人対応） ---
print("\n複数人対応カルマンフィルタを適用中...")

# 股関節
print("股関節X座標補正中...")
khip_Lx_p0, khip_Rx_p0 = kalman2_multi_person(
    chip_p0[:, 3], chip_p0[:, 0], chip_p1[:, 3], chip_p1[:, 0], 50, 0.1)
print("股関節Y座標補正中...")
khip_Ly_p0, khip_Ry_p0 = kalman2_multi_person(
    chip_p0[:, 4], chip_p0[:, 1], chip_p1[:, 4], chip_p1[:, 1], 50, 0.1)

# 膝
print("膝X座標補正中...")
kknee_Lx_p0, kknee_Rx_p0 = kalman2_multi_person(
    cknee_p0[:, 3], cknee_p0[:, 0], cknee_p1[:, 3], cknee_p1[:, 0], 200, 0.1)
print("膝Y座標補正中...")
kknee_Ly_p0, kknee_Ry_p0 = kalman2_multi_person(
    cknee_p0[:, 4], cknee_p0[:, 1], cknee_p1[:, 4], cknee_p1[:, 1], 50, 0.1)

# 足首
print("足首X座標補正中...")
kankle_Lx_p0, kankle_Rx_p0 = kalman2_multi_person(
    cankle_p0[:, 3], cankle_p0[:, 0], cankle_p1[:, 3], cankle_p1[:, 0], 200, 0.1)
print("足首Y座標補正中...")
kankle_Ly_p0, kankle_Ry_p0 = kalman2_multi_person(
    cankle_p0[:, 4], cankle_p0[:, 1], cankle_p1[:, 4], cankle_p1[:, 1], 50, 0.1)

# 母趾
print("母趾X座標補正中...")
kbigtoe_Lx_p0, kbigtoe_Rx_p0 = kalman2_multi_person(
    cbigtoe_p0[:, 3], cbigtoe_p0[:, 0], cbigtoe_p1[:, 3], cbigtoe_p1[:, 0], 200, 0.1)
print("母趾Y座標補正中...")
kbigtoe_Ly_p0, kbigtoe_Ry_p0 = kalman2_multi_person(
    cbigtoe_p0[:, 4], cbigtoe_p0[:, 1], cbigtoe_p1[:, 4], cbigtoe_p1[:, 1], 100, 0.1)

# 小趾
print("小趾X座標補正中...")
ksmalltoe_Lx_p0, ksmalltoe_Rx_p0 = kalman2_multi_person(
    csmalltoe_p0[:, 3], csmalltoe_p0[:, 0], csmalltoe_p1[:, 3], csmalltoe_p1[:, 0], 200, 0.1)
print("小趾Y座標補正中...")
ksmalltoe_Ly_p0, ksmalltoe_Ry_p0 = kalman2_multi_person(
    csmalltoe_p0[:, 4], csmalltoe_p0[:, 1], csmalltoe_p1[:, 4], csmalltoe_p1[:, 1], 50, 0.1)

# 踵
print("踵X座標補正中...")
kheel_Lx_p0, kheel_Rx_p0 = kalman2_multi_person(
    cheel_p0[:, 3], cheel_p0[:, 0], cheel_p1[:, 3], cheel_p1[:, 0], 200, 0.1)
print("踵Y座標補正中...")
kheel_Ly_p0, kheel_Ry_p0 = kalman2_multi_person(
    cheel_p0[:, 4], cheel_p0[:, 1], cheel_p1[:, 4], cheel_p1[:, 1], 50, 0.1)

print("\nすべての関節の補正が完了しました。")

# --- 7. 最終的な座標データの描画 ---
display_coordinates = True
if display_coordinates:
    print("\n補正後の座標グラフを作成中...")
    
    plot_data = {
        'hip': {'raw': chip_p0, 'kalman_Rx':khip_Rx_p0, 'kalman_Lx':khip_Lx_p0, 
                'kalman_Ry':khip_Ry_p0, 'kalman_Ly':khip_Ly_p0},
        'knee': {'raw': cknee_p0, 'kalman_Rx':kknee_Rx_p0, 'kalman_Lx':kknee_Lx_p0, 
                 'kalman_Ry':kknee_Ry_p0, 'kalman_Ly':kknee_Ly_p0},
        'ankle': {'raw': cankle_p0, 'kalman_Rx':kankle_Rx_p0, 'kalman_Lx':kankle_Lx_p0, 
                  'kalman_Ry':kankle_Ry_p0, 'kalman_Ly':kankle_Ly_p0},
        'bigtoe': {'raw': cbigtoe_p0, 'kalman_Rx':kbigtoe_Rx_p0, 'kalman_Lx':kbigtoe_Lx_p0, 
                   'kalman_Ry':kbigtoe_Ry_p0, 'kalman_Ly':kbigtoe_Ly_p0},
        'smalltoe': {'raw': csmalltoe_p0, 'kalman_Rx':ksmalltoe_Rx_p0, 'kalman_Lx':ksmalltoe_Lx_p0, 
                     'kalman_Ry':ksmalltoe_Ry_p0, 'kalman_Ly':ksmalltoe_Ly_p0},
        'heel': {'raw': cheel_p0, 'kalman_Rx':kheel_Rx_p0, 'kalman_Lx':kheel_Lx_p0, 
                 'kalman_Ry':kheel_Ry_p0, 'kalman_Ly':kheel_Ly_p0},
    }

    for joint_name, data in plot_data.items():
        # X座標のプロット
        plt.figure(figsize=(10, 6))
        plt.plot(cframe, data['raw'][:, 0], color='r', label='Raw Right', alpha=0.3)
        plt.plot(cframe, data['raw'][:, 3], color='b', label='Raw Left', alpha=0.3)
        plt.plot(cframe, data['kalman_Rx'], color='r', label='Kalman Right')
        plt.plot(cframe, data['kalman_Lx'], color='b', label='Kalman Left')
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('X Coordinate [px]', fontsize=16)
        plt.title(f'{joint_name.capitalize()} X Coordinate (Person0)', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'coord_{joint_name}_x.png'))
        plt.close()

        # Y座標のプロット
        plt.figure(figsize=(10, 6))
        plt.plot(cframe, data['raw'][:, 1], color='r', label='Raw Right', alpha=0.3)
        plt.plot(cframe, data['raw'][:, 4], color='b', label='Raw Left', alpha=0.3)
        plt.plot(cframe, data['kalman_Ry'], color='r', label='Kalman Right')
        plt.plot(cframe, data['kalman_Ly'], color='b', label='Kalman Left')
        plt.xlabel('Frame [-]', fontsize=16)
        plt.ylabel('Y Coordinate [px]', fontsize=16)
        plt.title(f'{joint_name.capitalize()} Y Coordinate (Person0)', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'coord_{joint_name}_y.png'))
        plt.close()

    print("補正後の座標グラフの作成が完了しました。")

# --- 8. 最終的な座標データの保存 ---
print("\n補正後のデータをCSVに保存中...")
df_final_p0 = df_p0.copy()

corrected_data = {
    'Ankle': (kankle_Lx_p0, kankle_Rx_p0, kankle_Ly_p0, kankle_Ry_p0),
    'Knee': (kknee_Lx_p0, kknee_Rx_p0, kknee_Ly_p0, kknee_Ry_p0),
    'Hip': (khip_Lx_p0, khip_Rx_p0, khip_Ly_p0, khip_Ry_p0),
    'BigToe': (kbigtoe_Lx_p0, kbigtoe_Rx_p0, kbigtoe_Ly_p0, kbigtoe_Ry_p0),
    'SmallToe': (ksmalltoe_Lx_p0, ksmalltoe_Rx_p0, ksmalltoe_Ly_p0, ksmalltoe_Ry_p0),
    'Heel': (kheel_Lx_p0, kheel_Rx_p0, kheel_Ly_p0, kheel_Ry_p0),
}

for col_name in df_final_p0.columns:
    for joint_name, data_tuple in corrected_data.items():
        if joint_name not in col_name:
            continue
            
        Lx, Rx, Ly, Ry = data_tuple
        
        if 'x' in col_name:
            if 'L' in col_name:
                df_final_p0.loc[start_frame:end_frame-1, col_name] = Lx
                break
            elif 'R' in col_name:
                df_final_p0.loc[start_frame:end_frame-1, col_name] = Rx
                break
        
        elif 'y' in col_name:
            if 'L' in col_name:
                df_final_p0.loc[start_frame:end_frame-1, col_name] = Ly
                break
            elif 'R' in col_name:
                df_final_p0.loc[start_frame:end_frame-1, col_name] = Ry
                break

# 保存
output_csv_path = os.path.join(path_op, f"{name}_person0_kalman.csv")
df_final_p0.to_csv(output_csv_path, index=True, header=True)

print(f"\n処理が完了しました！")
print(f"補正後のCSVファイル: {output_csv_path}")
print(f"グラフ保存先: {output_dir}")