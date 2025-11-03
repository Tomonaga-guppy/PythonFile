"""
OpenCapのutilsPostProcessing.pyから移植した後処理機能
"""
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline, interp1d


def filter_nans_advanced(data: np.ndarray, method: str = 'cubic') -> np.ndarray:
    """
    欠損値を高度な補間手法で埋める
    
    Args:
        data: (num_frames, num_keypoints, 3) の3Dデータ
        method: 'cubic', 'linear', 'pchip' のいずれか
    
    Returns:
        interpolated_data: 補間後のデータ
    """
    interpolated_data = data.copy()
    num_frames, num_keypoints, _ = interpolated_data.shape
    
    for kp_idx in range(num_keypoints):
        for coord_idx in range(3):
            coord_series = interpolated_data[:, kp_idx, coord_idx]
            valid_mask = ~np.isnan(coord_series)
            num_valid = np.sum(valid_mask)
            
            # 有効なデータが少なすぎる場合はスキップ
            if num_valid < 2:
                continue
            
            valid_indices = np.where(valid_mask)[0]
            valid_values = coord_series[valid_mask]
            
            # 全フレームのインデックス
            all_indices = np.arange(num_frames)
            
            # 補間方法を選択
            if method == 'cubic' and num_valid >= 4:
                # 3次スプライン補間
                cs = CubicSpline(valid_indices, valid_values, bc_type='natural')
                interpolated_data[:, kp_idx, coord_idx] = cs(all_indices)
            elif method == 'pchip' and num_valid >= 2:
                # PCHIP補間(単調性を保持)
                from scipy.interpolate import PchipInterpolator
                pchip = PchipInterpolator(valid_indices, valid_values)
                interpolated_data[:, kp_idx, coord_idx] = pchip(all_indices)
            else:
                # 線形補間
                interp_func = interp1d(
                    valid_indices, valid_values, 
                    kind='linear', 
                    fill_value='extrapolate'
                )
                interpolated_data[:, kp_idx, coord_idx] = interp_func(all_indices)
    
    return interpolated_data


def butterworth_filter_advanced(data: np.ndarray, cutoff: float, fs: float, 
                                order: int = 4, bidirectional: bool = True) -> np.ndarray:
    """
    双方向バターワースフィルタ(位相ずれなし)
    
    Args:
        data: (num_frames, num_keypoints, 3)
        cutoff: カットオフ周波数 (Hz)
        fs: サンプリング周波数 (Hz)
        order: フィルタ次数
        bidirectional: True なら filtfilt (位相ずれなし)
    
    Returns:
        filtered_data: フィルタリング後のデータ
    """
    filtered_data = data.copy()
    num_frames, num_keypoints, _ = filtered_data.shape
    
    # バターワースフィルタの設計
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    for kp_idx in range(num_keypoints):
        for coord_idx in range(3):
            coord_series = filtered_data[:, kp_idx, coord_idx]
            
            # NaNチェック
            if np.all(np.isnan(coord_series)):
                continue
            
            valid_mask = ~np.isnan(coord_series)
            
            # 双方向フィルタを適用
            if bidirectional:
                # filtfilt: ゼロ位相フィルタ
                filtered_values = filtfilt(b, a, coord_series[valid_mask])
            else:
                # lfilter: 片方向フィルタ(位相遅れあり)
                from scipy.signal import lfilter
                filtered_values = lfilter(b, a, coord_series[valid_mask])
            
            filtered_data[valid_mask, kp_idx, coord_idx] = filtered_values
    
    return filtered_data


def interpolate_markers_temporal(data: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """
    時系列的な欠損を補間(短いギャップのみ)
    
    Args:
        data: (num_frames, num_keypoints, 3)
        max_gap: 補間する最大フレーム数
    
    Returns:
        interpolated_data: 補間後のデータ
    """
    interpolated_data = data.copy()
    num_frames, num_keypoints, _ = interpolated_data.shape
    
    for kp_idx in range(num_keypoints):
        for coord_idx in range(3):
            coord_series = interpolated_data[:, kp_idx, coord_idx]
            
            # NaNの位置を検出
            is_nan = np.isnan(coord_series)
            
            # 連続するNaNのグループを見つける
            nan_groups = []
            in_gap = False
            gap_start = 0
            
            for i in range(num_frames):
                if is_nan[i] and not in_gap:
                    gap_start = i
                    in_gap = True
                elif not is_nan[i] and in_gap:
                    nan_groups.append((gap_start, i - 1))
                    in_gap = False
            
            if in_gap:
                nan_groups.append((gap_start, num_frames - 1))
            
            # 短いギャップのみ線形補間
            for start, end in nan_groups:
                gap_length = end - start + 1
                
                if gap_length > max_gap:
                    continue  # ギャップが大きすぎる場合はスキップ
                
                # 前後の有効な値を取得
                before_idx = start - 1 if start > 0 else None
                after_idx = end + 1 if end < num_frames - 1 else None
                
                if before_idx is not None and after_idx is not None:
                    if not is_nan[before_idx] and not is_nan[after_idx]:
                        # 線形補間
                        before_val = coord_series[before_idx]
                        after_val = coord_series[after_idx]
                        
                        for i in range(start, end + 1):
                            t = (i - before_idx) / (after_idx - before_idx)
                            interpolated_data[i, kp_idx, coord_idx] = \
                                before_val * (1 - t) + after_val * t
    
    return interpolated_data


def calculate_3d_confidence_metrics(data: np.ndarray) -> dict:
    """
    3Dデータの品質メトリクスを計算
    
    Args:
        data: (num_frames, num_keypoints, 3)
    
    Returns:
        metrics: 品質メトリクスの辞書
    """
    metrics = {}
    
    # 欠損率
    total_points = data.size
    nan_points = np.sum(np.isnan(data))
    metrics['missing_rate'] = nan_points / total_points
    
    # 各キーポイントの欠損率
    num_frames, num_keypoints, _ = data.shape
    kp_missing_rates = []
    for kp_idx in range(num_keypoints):
        kp_data = data[:, kp_idx, :]
        kp_missing = np.sum(np.isnan(kp_data)) / kp_data.size
        kp_missing_rates.append(kp_missing)
    
    metrics['keypoint_missing_rates'] = np.array(kp_missing_rates)
    metrics['worst_keypoint_idx'] = np.argmax(kp_missing_rates)
    metrics['worst_keypoint_missing_rate'] = np.max(kp_missing_rates)
    
    # 時間的な平滑性(速度の変化)
    valid_data = data.copy()
    valid_data[np.isnan(valid_data)] = 0  # 一時的にNaNを0に
    
    velocity = np.diff(valid_data, axis=0)
    acceleration = np.diff(velocity, axis=0)
    
    metrics['mean_velocity'] = np.mean(np.linalg.norm(velocity, axis=2))
    metrics['mean_acceleration'] = np.mean(np.linalg.norm(acceleration, axis=2))
    
    return metrics