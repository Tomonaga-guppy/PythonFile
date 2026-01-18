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

def cucl_gait_event(kp3d, valid_start=0, out_root=None):
    valid_start = 0 #あくまでスクリプト内のic, toとのフレーム確認用（全体を通してのフレーム番号にする場合は消す）
    """
    3Dキーポイントから歩行イベント（IC, TO）を検出する。
    出力: event_frame_dict = {'ic_r': [], 'ic_l': [], 'to_r': [], 'to_l': []}
    """
    dist_r_midhip2hee_z = np.array(kp3d[:, 24, 2] - kp3d[:, 8, 2])  # midhip-rheeのZ距離
    dist_l_midhip2hee_z = np.array(kp3d[:, 14, 2] - kp3d[:, 8, 2])  # midhip-lheeのZ距離
    dist_r_midhip2toe_z = np.array((kp3d[:, 22, 2]+kp3d[:, 23, 2])/2 - kp3d[:, 8, 2])  # midhip-rtoeのZ距離
    dist_l_midhip2toe_z = np.array((kp3d[:, 19, 2]+kp3d[:, 20, 2])/2 - kp3d[:, 8, 2])  # midhip-ltoeのZ距離
    
    event_frame_dict = {'ic_r': [], 'ic_l': [], 'to_r': [], 'to_l': []}

    def drop_edge_peaks(peaks, values, n_check=2):
        """
        検出したピークが末端すぎる（前後に十分な有効値がない）場合は削除
        peaks: list[int] or np.ndarray
        values: 1D array (z_ori)
        n_check: ピークの前後に「有効値」が何点必要か
        """
        n = len(values)
        keep = []
        values = np.asarray(values, float)

        for p in map(int, peaks):
            # 範囲外（端に近すぎる）なら捨てる
            if p - n_check < 0 or p + n_check >= n:
                continue

            # 前後 n_check 点が有限（NaN/infでない）であることを要求
            left_ok  = np.isfinite(values[p - n_check : p]).all()
            right_ok = np.isfinite(values[p + 1 : p + 1 + n_check]).all()

            if left_ok and right_ok:
                keep.append(p)

        return keep


    def filter_by_value_mad(frames, values, k=3.0, min_keep=2, tau_min=200):
        """
        MAD（中央値±k*MAD）で値の外れ値を落とす
        """
        frames = sorted(map(int, frames))
        if len(frames) < min_keep:
            return frames

        vals = np.array([values[f] for f in frames], float)
        m = np.isfinite(vals)
        if m.sum() < min_keep:
            return frames

        med = np.nanmedian(vals[m])
        mad = np.nanmedian(np.abs(vals[m] - med))
        if not np.isfinite(mad) or mad < 1e-9:
            return frames

        tau = max(k*mad, tau_min)
        keep = [f for f, v in zip(frames, vals) if np.isfinite(v) and abs(v - med) <= tau]
        print(f"    vals :{vals}")
        print(f"    med={med:.2f}, mad={mad:.2f}, k*mad={k*mad:.2f} tau={tau:.2f}")

        return keep if len(keep) >= min_keep else frames


    def remove_outlier_by_interval(frames, tol=0.2, max_iter=10):
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
            
            # print(f"    lo<med<hi: {lo:.2f} < {med:.2f} < {hi:.2f}")
            # print(f"    frames before removal: {d}")

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
    for z_ori, label in zip([dist_r_midhip2hee_z, dist_l_midhip2hee_z], ['ic_r', 'ic_l']):
        z = np.asarray(z_ori, float)
        z[np.isnan(z)] = -np.inf
        peaks, _ = find_peaks(z, distance=30, prominence=0.0001)
        peaks = drop_edge_peaks(peaks, z_ori, n_check=2)  #端のピーク除去
        # print(f"label: {label}, ori frames: {peaks}")
        frames = filter_by_value_mad(peaks, z_ori, k=3.0)  #距離値に対しての外れ値除去
        # print(f"label: {label}, filtered frames1: {frames}")
        frames = remove_outlier_by_interval(frames)  #フレームに対しての外れ値除去
        # print(f"label: {label}, filtered frames2: {frames}")
        event_frame_dict[label] = frames

    # TO（つま先：最小ピーク）
    for z_ori, label in zip([dist_r_midhip2toe_z, dist_l_midhip2toe_z], ['to_r', 'to_l']):
        z = np.asarray(z_ori, float)
        z[np.isnan(z)] = np.inf
        valleys, _ = find_peaks(-z, distance=30, prominence=0.0001)
        valleys = drop_edge_peaks(valleys, z_ori, n_check=2)  #端のピーク除去
        # print(f"label: {label}, ori frames: {valleys}")
        frames = filter_by_value_mad(valleys, z_ori, k=3.0)  #距離値に対しての外れ値除去
        # print(f"label: {label}, filtered frames1: {frames}")
        frames = remove_outlier_by_interval(frames)  #フレームに対しての外れ値除去
        # print(f"label: {label}, filtered frames2: {frames}")
        event_frame_dict[label] = frames
        
    # 確認用に踵・つま先距離のプロットを保存
    if out_root is not None:
        # --- x軸（global表示にする）---
        n = len(dist_r_midhip2hee_z)
        frames_local = np.arange(n)
        frames_global = frames_local + int(valid_start)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # ========== Heel (IC) ==========
        axes[0].plot(frames_global, dist_r_midhip2hee_z, label='R Dist', color='blue')
        axes[0].plot(frames_global, dist_l_midhip2hee_z, label='L Dist', color='orange')

        # IC ×印
        for i, f in enumerate(event_frame_dict.get('ic_r', [])):
            fg = int(f) + int(valid_start)
            axes[0].plot(fg, dist_r_midhip2hee_z[fg - valid_start], 'o', color='blue', markersize=10, label='IC R' if i == 0 else "")
        for i, f in enumerate(event_frame_dict.get('ic_l', [])):
            fg = int(f) + int(valid_start)
            axes[0].plot(fg, dist_l_midhip2hee_z[fg - valid_start], 'o', color='orange', markersize=10, label='IC L' if i == 0 else "")

        axes[0].set_ylim(-500, 500)
        axes[0].set_title('MidHip to Heel Z Distance')
        axes[0].set_xlabel('Frame [-]')
        axes[0].set_ylabel('Distance [mm]')
        axes[0].legend()

        # ========== Toe (TO) ==========
        axes[1].plot(frames_global, dist_r_midhip2toe_z, label='R Dist', color='blue')
        axes[1].plot(frames_global, dist_l_midhip2toe_z, label='L Dist', color='orange')
        
        # TO ×印
        for i, f in enumerate(event_frame_dict.get('to_r', [])):
            fg = int(f) + int(valid_start)
            axes[1].plot(fg, dist_r_midhip2toe_z[fg - valid_start], 'o', color='blue', markersize=10, label='TO R' if i == 0 else "")
        for i, f in enumerate(event_frame_dict.get('to_l', [])):
            fg = int(f) + int(valid_start)
            axes[1].plot(fg, dist_l_midhip2toe_z[fg - valid_start], 'o', color='orange', markersize=10, label='TO L' if i == 0 else "")

        axes[1].set_title('MidHip to Toe Z Distance')
        axes[1].set_xlabel('Frame [-]')
        axes[1].set_ylabel('Distance [mm]')
        axes[1].set_ylim(-500, 500)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(out_root / "z-distance_for_gait_events.png")
        plt.show()
        plt.close(fig)
    
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