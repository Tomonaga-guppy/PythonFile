"""
モデルフリー座標補正 - 完全版
元のスクリプトに統合可能

使い方:
1. このファイルをインポート
2. kalman2関数の代わりにmodel_free_correctionを使用
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.stats import median_abs_deviation


def detect_outliers_mad(data, confidence, threshold=3.5):
    """MADベースの外れ値検出"""
    valid_mask = confidence > 0.3
    
    if np.sum(valid_mask) < 5:
        return np.zeros_like(data, dtype=bool)
    
    valid_data = data[valid_mask]
    velocity = np.diff(valid_data)
    
    if len(velocity) < 3:
        return np.zeros_like(data, dtype=bool)
    
    median_vel = np.median(velocity)
    mad = median_abs_deviation(velocity)
    
    if mad < 1e-6:
        mad = np.std(velocity)
    
    outlier_vel_mask = np.abs(velocity - median_vel) > threshold * mad
    
    outlier_mask = np.zeros_like(data, dtype=bool)
    valid_indices = np.where(valid_mask)[0]
    
    for i, is_outlier in enumerate(outlier_vel_mask):
        if is_outlier and i < len(valid_indices) - 1:
            outlier_mask[valid_indices[i+1]] = True
    
    return outlier_mask


def model_free_correction(coord_L, coord_R, conf_L, conf_R,
                          method='savgol',
                          window_length=15,
                          polyorder=3,
                          confidence_threshold=0.3):
    """
    モデルフリーな座標補正
    
    Parameters:
    -----------
    coord_L, coord_R : array-like
        左右の座標データ
    conf_L, conf_R : array-like
        左右の信頼度 (0-1)
    method : str
        'savgol' または 'spline'
    window_length : int
        Savitzky-Golayのウィンドウサイズ（奇数）
    polyorder : int
        多項式の次数
    confidence_threshold : float
        有効データの信頼度閾値
    
    Returns:
    --------
    coord_L_corrected, coord_R_corrected : ndarray
        補正後の座標
    """
    coord_L = np.array(coord_L).copy()
    coord_R = np.array(coord_R).copy()
    conf_L = np.array(conf_L)
    conf_R = np.array(conf_R)
    
    n = len(coord_L)
    
    # 左側の補正
    outliers_L = detect_outliers_mad(coord_L, conf_L)
    invalid_L = (conf_L < confidence_threshold) | outliers_L
    valid_indices_L = np.where(~invalid_L)[0]
    
    if len(valid_indices_L) >= max(window_length, 5):
        if method == 'savgol':
            # 線形補間してからSavitzky-Golay
            coord_L_interp = np.interp(np.arange(n), valid_indices_L, 
                                      coord_L[valid_indices_L])
            if window_length % 2 == 0:
                window_length += 1
            try:
                coord_L = savgol_filter(coord_L_interp, window_length, 
                                       polyorder, mode='mirror')
            except:
                coord_L = coord_L_interp
        
        elif method == 'spline':
            # スプライン補間
            try:
                smoothing = len(valid_indices_L) * 0.1
                spline = UnivariateSpline(valid_indices_L, 
                                         coord_L[valid_indices_L],
                                         s=smoothing, k=3)
                coord_L = spline(np.arange(n))
            except:
                coord_L = np.interp(np.arange(n), valid_indices_L,
                                   coord_L[valid_indices_L])
    
    # 右側の補正（同様の処理）
    outliers_R = detect_outliers_mad(coord_R, conf_R)
    invalid_R = (conf_R < confidence_threshold) | outliers_R
    valid_indices_R = np.where(~invalid_R)[0]
    
    if len(valid_indices_R) >= max(window_length, 5):
        if method == 'savgol':
            coord_R_interp = np.interp(np.arange(n), valid_indices_R,
                                      coord_R[valid_indices_R])
            if window_length % 2 == 0:
                window_length += 1
            try:
                coord_R = savgol_filter(coord_R_interp, window_length,
                                       polyorder, mode='mirror')
            except:
                coord_R = coord_R_interp
        
        elif method == 'spline':
            try:
                smoothing = len(valid_indices_R) * 0.1
                spline = UnivariateSpline(valid_indices_R,
                                         coord_R[valid_indices_R],
                                         s=smoothing, k=3)
                coord_R = spline(np.arange(n))
            except:
                coord_R = np.interp(np.arange(n), valid_indices_R,
                                   coord_R[valid_indices_R])
    
    return coord_L, coord_R


# ===== 元のスクリプトとの統合例 =====

def apply_model_free_to_keypoints(raw_data, confidence_data,
                                  method='savgol',
                                  window_length=15):
    """
    全キーポイントに対してモデルフリー補正を適用
    
    Parameters:
    -----------
    raw_data : dict
        {'ankle': array, 'knee': array, ...}
        各配列は shape (n_frames, 6): [R_x, R_y, R_conf, L_x, L_y, L_conf]
    
    Returns:
    --------
    corrected_data : dict
        補正後のデータ
    """
    corrected_data = {}
    
    for joint_name, data in raw_data.items():
        print(f"補正中: {joint_name}...")
        
        n_frames = len(data)
        
        # X座標の補正
        x_L_corr, x_R_corr = model_free_correction(
            data[:, 3], data[:, 0],  # L_x, R_x
            data[:, 5], data[:, 2],  # L_conf, R_conf
            method=method,
            window_length=window_length
        )
        
        # Y座標の補正
        y_L_corr, y_R_corr = model_free_correction(
            data[:, 4], data[:, 1],  # L_y, R_y
            data[:, 5], data[:, 2],  # L_conf, R_conf
            method=method,
            window_length=window_length
        )
        
        corrected_data[joint_name] = {
            'R_x': x_R_corr,
            'R_y': y_R_corr,
            'L_x': x_L_corr,
            'L_y': y_L_corr
        }
    
    return corrected_data


# ===== 使用例 =====
"""
# 元のスクリプトの修正箇所（Line 527-551付近）

# === 旧コード（カルマンフィルタ）===
# kankle_Lx, kankle_Rx = kalman2(cankle[:, 3], cankle[:, 0], 200, 0.1)
# kankle_Ly, kankle_Ry = kalman2(cankle[:, 4], cankle[:, 1], 50, 0.1)

# === 新コード（モデルフリー）===
from model_free_correction_full import model_free_correction

# X座標
kankle_Lx, kankle_Rx = model_free_correction(
    cankle[:, 3], cankle[:, 0],     # 左X, 右X
    cankle[:, 5], cankle[:, 2],     # 左信頼度, 右信頼度
    method='savgol',
    window_length=15,               # データに応じて調整
    polyorder=3
)

# Y座標
kankle_Ly, kankle_Ry = model_free_correction(
    cankle[:, 4], cankle[:, 1],     # 左Y, 右Y
    cankle[:, 5], cankle[:, 2],     # 左信頼度, 右信頼度
    method='savgol',
    window_length=11,               # Y方向は変化が小さいので短めに
    polyorder=3
)

# 他の関節も同様に処理
"""

if __name__ == "__main__":
    print("モデルフリー座標補正モジュール")
    print("="*50)
    print("\n使用方法:")
    print("1. このファイルをインポート")
    print("2. model_free_correction() を使用")
    print("\nパラメータの推奨値:")
    print("- method='savgol' (最もロバスト)")
    print("- window_length=15 (X座標)")
    print("- window_length=11 (Y座標)")
    print("- polyorder=3")
    print("- confidence_threshold=0.3")