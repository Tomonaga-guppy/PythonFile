import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def culc_angle(vector1, vector2, n_vector, degrees=False, angle_type=None):
    """
    股関節の屈曲伸展角度を計算（矢状面への投影）
    
    Parameters:
    -----------
    vector1 : np.ndarray 
    vector2 : np.ndarray
    n_vector : np.ndarray
        法線方向ベクトル
    degrees : bool, optional
        角度を度数法で返すかどうか。デフォルトはFalse（ラジアン）。
    angle_type : str, optional
        角度の種類（将来の拡張用）。現在は未使用
    Returns:
    --------
    float : 関節角度
    """
    # 矢状面の法線を正規化
    n_norm = n_vector / np.linalg.norm(n_vector)
    
    # ベクトル1とベクトル2を矢状面に投影
    vector1_proj = vector1 - np.dot(vector1, n_norm) * n_norm
    vector2_proj = vector2 - np.dot(vector2, n_norm) * n_norm
    
    # 投影ベクトルの長さチェック
    vector1_norm = np.linalg.norm(vector1_proj)
    vector2_norm = np.linalg.norm(vector2_proj)
    
    if vector1_norm < 1e-8 or vector2_norm < 1e-8:
        return 0.0
    
    # 角度計算
    cos_angle = np.dot(vector1_proj, vector2_proj) / (vector1_norm * vector2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    
    # 符号判定：後方向ベクトルとの外積で判断
    # 矢状面内でベクトル1に直交する後方向ベクトル
    posterior_vec = np.cross(n_norm, vector1_proj)
    posterior_vec = posterior_vec / np.linalg.norm(posterior_vec)

    # ベクトル1が前方にあれば負、後方にあれば正
    sign = np.sign(np.dot(vector2_proj, posterior_vec))

    if angle_type == 'hip':
        angle = - sign * (np.pi - angle_rad)
    elif angle_type == 'knee':
        angle = - sign * angle_rad
    elif angle_type == 'ankle':
        angle = sign * (angle_rad - np.pi/2)
    elif angle_type == 'hip_adab':
        angle = np.pi - angle_rad
    elif angle_type == 'hip_inex':
        angle = angle_rad

    if degrees:
        angle = np.degrees(angle)
    
    return angle

def culc_angle_all_frames(vector1_all, vector2_all, n_vector_all, degrees=False, angle_type=None):
    """
    全フレームで関節角度を計算
    """
    angles = np.zeros(vector1_all.shape[0])
    for i in range(vector1_all.shape[0]):
        angles[i] = culc_angle(
            vector1_all[i], 
            vector2_all[i], 
            n_vector_all[i],
            degrees=degrees,
            angle_type=angle_type
        )
    return angles

def cucl_gait_event(kp3d):
    """
    3Dキーポイントから歩行イベント（IC, TO）を検出する。
    出力: event_frame_dict = {'ic_r': [], 'ic_l': [], 'to_r': [], 'to_l': []}
    """
    dist_r_midhip2hee_z = np.array(kp3d[:, 24, 2] - kp3d[:, 8, 2])  # midhip-rheeのZ距離
    dist_l_midhip2hee_z = np.array(kp3d[:, 14, 2] - kp3d[:, 8, 2])  # midhip-lheeのZ距離
    dist_r_midhip2toe_z = np.array((kp3d[:, 22, 2]+kp3d[:, 23, 2])/2 - kp3d[:, 8, 2])  # midhip-rtoeのZ距離
    dist_l_midhip2toe_z = np.array((kp3d[:, 19, 2]+kp3d[:, 20, 2])/2 - kp3d[:, 8, 2])  # midhip-ltoeのZ距離
    
    event_frame_dict = {'ic_r': [], 'ic_l': [], 'to_r': [], 'to_l': []}
    
    def remove_outlier_by_interval(frames, tol=0.1, max_iter=10):
        """
        異常な間隔が見つかったら、その区間の「前を消す」or「後ろを消す」を比較して決める。
        tol: 中央間隔 med に対する許容割合
        """
        frames = sorted(map(int, frames))
        if len(frames) < 4:
            return frames

        def deviation(fr):
            d = np.diff(fr)
            med = np.median(d)
            return np.sum(np.abs(d - med))  # 小さいほど等間隔に近い

        for _ in range(max_iter):
            if len(frames) < 4:
                break

            d = np.diff(frames)
            med = np.median(d)
            lo, hi = med * (1 - tol), med * (1 + tol)

            bad = np.where((d < lo) | (d > hi))[0]
            if len(bad) == 0:
                break

            i = int(bad[0])  # 最初の異常区間

            # 候補1: frames[i] を消す
            cand1 = frames[:i] + frames[i+1:]
            # 候補2: frames[i+1] を消す
            cand2 = frames[:i+1] + frames[i+2:]

            frames = cand1 if deviation(cand1) <= deviation(cand2) else cand2

        return frames
    
    # IC（踵：最大ピーク）
    for z, label in zip([dist_r_midhip2hee_z, dist_l_midhip2hee_z], ['ic_r', 'ic_l']):
        z = np.asarray(z, float)
        z[np.isnan(z)] = -np.inf
        peaks, _ = find_peaks(z, distance=30, prominence=0.01)
        event_frame_dict[label] = remove_outlier_by_interval(peaks.tolist())

    # TO（つま先：最小ピーク）
    for z, label in zip([dist_r_midhip2toe_z, dist_l_midhip2toe_z], ['to_r', 'to_l']):
        z = np.asarray(z, float)
        z[np.isnan(z)] = np.inf
        valleys, _ = find_peaks(-z, distance=30, prominence=0.01)
        event_frame_dict[label] = remove_outlier_by_interval(valleys.tolist())
        
        

    # print(f"Detected gait events: {event_frame_dict}")

    # frames = np.arange(len(dist_r_midhip2hee_z))
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # axes[0].plot(frames, dist_r_midhip2hee_z, label='R MidHip-Hee Z Dist', color='blue')
    # axes[0].plot(frames, dist_l_midhip2hee_z, label='L MidHip-Hee Z Dist', color='orange')
    # axes[0].set_title('MidHip to Heel Z Distance')
    # axes[0].set_xlabel('Frame [-]')
    # axes[0].set_ylabel('Distance [mm]')
    # axes[0].legend()
    
    # axes[1].plot(frames, dist_r_midhip2toe_z, label='R MidHip-Toe Z Dist', color='blue')
    # axes[1].plot(frames, dist_l_midhip2toe_z, label='L MidHip-Toe Z Dist', color='orange')
    # axes[1].set_title('MidHip to Toe Z Distance')
    # axes[1].set_xlabel('Frame [-]')
    # axes[1].set_ylabel('Distance [mm]')
    # axes[1].legend()
    
    # plt.tight_layout()
    # plt.show()
    
    return event_frame_dict

def culc_gait_cycles(event_frame_dict):
    """
    歩行イベントを歩行周期ごとに振り分ける
    returns:
    --------
    gait_cycles_r : list of list
        右足の歩行周期リスト。各要素は [IC, 対側IC, TO, 次のIC]
    gait_cycles_l : list of list
        左足の歩行周期リスト。各要素は [IC, 対側IC, TO, 次のIC]
    """
    filt_ic_r_list = event_frame_dict['ic_r']
    filt_ic_l_list = event_frame_dict['ic_l']
    filt_to_r_list = event_frame_dict['to_r']
    filt_to_l_list = event_frame_dict['to_l']
    
    # 右足の歩行周期を作成 [IC, TO, 次のIC]
    gait_cycles_r = []
    for i in range(len(filt_ic_r_list) - 1):
        ic_current = filt_ic_r_list[i]
        ic_next = filt_ic_r_list[i + 1]
        
        # 現在のICと次のICの間にある左のICを探す
        ic_l_in_cycle = [ic for ic in filt_ic_l_list if ic_current < ic < ic_next]
        # 現在のICと次のICの間にあるTOを探す
        to_in_cycle = [to for to in filt_to_r_list if ic_current < to < ic_next]
        
        if len(to_in_cycle) > 0 and len(ic_l_in_cycle) > 0:
            # 最初のTOを使用
            gait_cycles_r.append([ic_current, ic_l_in_cycle[0], to_in_cycle[0], ic_next])
    
    # 左足の歩行周期を作成 [IC, TO, 次のIC]
    gait_cycles_l = []
    for i in range(len(filt_ic_l_list) - 1):
        ic_current = filt_ic_l_list[i]
        ic_next = filt_ic_l_list[i + 1]
        
        # 現在のICと次のICの間にある右のICを探す
        ic_r_in_cycle = [ic for ic in filt_ic_r_list if ic_current < ic < ic_next]
        # 現在のICと次のICの間にあるTOを探す
        to_in_cycle = [to for to in filt_to_l_list if ic_current < to < ic_next]
        
        if len(to_in_cycle) > 0 and len(ic_r_in_cycle) > 0:
            # 最初のTOを使用
            gait_cycles_l.append([ic_current, ic_r_in_cycle[0], to_in_cycle[0], ic_next])
            
    return gait_cycles_r, gait_cycles_l