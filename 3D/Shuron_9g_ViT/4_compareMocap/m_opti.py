import pandas as pd
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
import json
from scipy.signal import resample_poly
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def butter_lowpass_filter(data, order, cutoff_freq, frame_list, sampling_freq=100):  #4次のバターワースローパスフィルタ
    # sampling_freq を可変にして、60Hz または 100Hz に対応
    nyquist_freq = sampling_freq / 2
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # print(f"data = {data}")
    # print(f"data.shape = {data.shape}")
    y = filtfilt(b, a, data[frame_list])
    data_fillter = np.copy(data)
    data_fillter[frame_list] = y
    return data_fillter

def read_3d_optitrack(csv_path, start_frame_60, end_frame_60, src_fs=100, dst_fs=60):
    """
    OptiTrackの3Dデータ(100Hz想定)を読み込み、60Hzへダウンサンプリングして返す。
    start_frame_60/end_frame_60 は「60Hzの絶対フレーム」を受け取る前提。

    返り値:
      keypoints_mocap_60: (N60, 16, 3)
      full_range_60: range(0, N60)
      abs_start_60, abs_end_60: 60Hzの絶対フレーム
      abs_frames_60: 60Hzの絶対フレーム配列（連番）
    """
    def report(df, tag):
        for m in ["MarkerSet 01:LSHO", "MarkerSet 01:RANK2"]:
            cols = [c for c in df.columns if c[0] == m and c[1] in ["X","Y","Z"]]
            nn = df[cols].notna().sum().min()
            print(f"[{tag}] {m} non-null min = {nn} / {len(df)}")
        
    # -------------------------
    # 1) 60Hzフレーム -> 100Hz行番号へ変換して切り出し
    # -------------------------
    start_100 = int(np.floor(start_frame_60 * src_fs / dst_fs))
    end_100   = int(np.ceil (end_frame_60   * src_fs / dst_fs))

    df = pd.read_csv(csv_path, skiprows=[0, 1, 2, 4], header=[0, 2])
    df = df.loc[:, df.columns.get_level_values(1).isin(['X', 'Y', 'Z'])]

    start_100 = max(0, start_100)
    end_100   = min(len(df) - 1, end_100)
    df = df.loc[start_100:end_100].reset_index(drop=True)

    # report(df, "after slice")
    
    marker_set = ["RASI", "LASI", "RPSI", "LPSI","RKNE","LKNE", "RANK","LANK","RTOE","LTOE","RHEE","LHEE",
                "RKNE2", "LKNE2", "RANK2", "LANK2", "Marker1", "Marker2", "Marker3", "Marker4", "Marker5", "Marker6"]
    marker_set_df = df[[col for col in df.columns if any(marker in col[0] for marker in marker_set)]].copy()

    if marker_set_df.empty:
        print("Error: No marker data found")
        return np.array([]), range(0), None, None, np.array([], dtype=int)
    


    # -------------------------
    # 2) RigidBody で欠損置換（現行ロジック踏襲）
    # -------------------------
    rigidbody_exists = any("RigidBody 01" in col[0] for col in marker_set_df.columns)
    if rigidbody_exists:
        print("RigidBodyデータが検出されました。欠損値の置き換え処理を実行します。")
        marker_mapping = {
            "MarkerSet 01:LPSI": "RigidBody 01:Marker2",
            "MarkerSet 01:LASI": "RigidBody 01:Marker1",
            "MarkerSet 01:RPSI": "RigidBody 01:Marker5",
            "MarkerSet 01:RASI": "RigidBody 01:Marker4"
        }
        for markerset_name, rigidbody_name in marker_mapping.items():
            markerset_cols = [col for col in marker_set_df.columns if markerset_name in col[0]]
            rigidbody_cols = [col for col in marker_set_df.columns if rigidbody_name in col[0]]
            if markerset_cols and rigidbody_cols:
                for ms_col, rb_col in zip(markerset_cols, rigidbody_cols):
                    mask = marker_set_df[ms_col].isnull()
                    marker_set_df.loc[mask, ms_col] = marker_set_df.loc[mask, rb_col]
            else:
                print(f"警告: {markerset_name}または{rigidbody_name}が見つかりません")

    final_marker_set = ["RASI", "LASI", "RPSI", "LPSI","RKNE","LKNE", "RANK","LANK","RTOE","LTOE","RHEE","LHEE",
                        "RKNE2", "LKNE2", "RANK2", "LANK2"]
    final_df = marker_set_df[[col for col in marker_set_df.columns if any(marker in col[0] for marker in final_marker_set)]].copy()

    # 欠損補間（現行の方針そのまま）
    final_df = final_df.interpolate(method="linear", limit_direction="both").ffill().bfill()
    if final_df.isnull().any().any():
        raise ValueError("NaNs remain after interpolation/fill in final_df")

    # (N100, 16, 3)
    keypoints_100 = final_df.values.reshape(-1, len(final_marker_set), 3)
    
    ####### あとでViTPoseとアニメーションで比較するようのdfを作成　################
    marker_for_vit_df = df.copy()
    rigidbody_exists_for_vit = any("RigidBody 01" in col[0] for col in df.columns)
    if rigidbody_exists_for_vit:
        marker_mapping_for_vit = {
            "MarkerSet 01:LPSI": "RigidBody 01:Marker2",
            "MarkerSet 01:LASI": "RigidBody 01:Marker1",
            "MarkerSet 01:RPSI": "RigidBody 01:Marker5",
            "MarkerSet 01:RASI": "RigidBody 01:Marker4"
        }
        for markerset_name, rigidbody_name in marker_mapping_for_vit.items():
            markerset_cols = [col for col in marker_for_vit_df.columns if markerset_name in col[0]]
            rigidbody_cols = [col for col in marker_for_vit_df.columns if rigidbody_name in col[0]]
            if markerset_cols and rigidbody_cols:
                for ms_col, rb_col in zip(markerset_cols, rigidbody_cols):
                    mask = marker_for_vit_df[ms_col].isnull()
                    marker_for_vit_df.loc[mask, ms_col] = marker_for_vit_df.loc[mask, rb_col]
            else:
                print(f"警告: {markerset_name}または{rigidbody_name}が見つかりません")
    
    final_marker_set_for_vit = ["RASI", "LASI", "RPSI", "LPSI","RKNE","LKNE", "RANK","LANK","RTOE","LTOE","RHEE","LHEE",
                                "RKNE2", "LKNE2", "RANK2", "LANK2", "VRTX", "C7", "LSHO", "RSHO", "CLAV", "T10", "STRN", "RBAK"]
    final_df_for_vit = marker_for_vit_df[[col for col in marker_for_vit_df.columns if any(marker in col[0] for marker in final_marker_set_for_vit)]].copy()
    # report(final_df_for_vit, "before interp")
    
    # 欠損補間（現行の方針そのまま）
    final_df_for_vit = final_df_for_vit.interpolate(method="linear", limit_direction="both").ffill().bfill()
    if final_df_for_vit.isnull().any().any():
        raise ValueError("NaNs remain after interpolation/fill in final_df_for_vit")
    # report(final_df_for_vit, "after interp")
    # (N100, 〇, 3)
    keypoints_100_for_vit = final_df_for_vit.values.reshape(-1, len(final_marker_set_for_vit), 3)
    #######################################################################

    # -------------------------
    # 3) ローパスフィルタ（100Hzでかける）cutoff_hz=6
    # -------------------------
    full_100 = range(keypoints_100.shape[0])
    keypoints_100_f = np.empty_like(keypoints_100)
    for j in range(keypoints_100.shape[1]):
        for ax in range(3):
            keypoints_100_f[:, j, ax] = butter_lowpass_filter(
                keypoints_100[:, j, ax],
                order=4, cutoff_freq=6,
                frame_list=full_100, sampling_freq=src_fs
            )
            
    ### あとでViTPoseとアニメーションで比較するようのdfを作成　################
    full_100_for_vit = range(final_df_for_vit.shape[0])
    keypoints_100_f_for_vit = np.empty_like(keypoints_100_for_vit)
    for j in range(keypoints_100_for_vit.shape[1]):
        for ax in range(3):
            keypoints_100_f_for_vit[:, j, ax] = butter_lowpass_filter(
                keypoints_100_for_vit[:, j, ax],
                order=4, cutoff_freq=6,
                frame_list=full_100_for_vit, sampling_freq=src_fs
            )

    # -------------------------
    # 4) 100Hz -> 60Hz (3/5) ダウンサンプリング
    # -------------------------
    keypoints_60 = resample_poly(keypoints_100_f, up=3, down=5, axis=0, padtype="line")

    # 返す絶対フレームは「60Hzの連番」
    abs_frames_60 = np.arange(start_frame_60, end_frame_60 + 1, dtype=int)

    # 長さ合わせ（端で1フレずれることがある）
    n = min(len(abs_frames_60), keypoints_60.shape[0])
    abs_frames_60 = abs_frames_60[:n]
    keypoints_60 = keypoints_60[:n]

    full_range_60 = range(n)
    print(f"元の絶対フレーム範囲(60Hz): {abs_frames_60[0]} から {abs_frames_60[-1]}")
    print(f"ダウンサンプリング前のデータ長(100Hz): {keypoints_100.shape[0]} フレーム")
    print(f"ダウンサンプリング後のデータ長(60Hz): {keypoints_60.shape[0]} フレーム")
    
    ### あとでViTPoseとアニメーションで比較するようのdfを作成　################
    keypoints_60_for_vit = resample_poly(keypoints_100_f_for_vit, up=3, down=5, axis=0, padtype="line")
    
    # 長さ合わせ（端で1フレずれることがある）
    n = min(len(abs_frames_60), keypoints_60_for_vit.shape[0])
    keypoints_60_for_vit = keypoints_60_for_vit[:n]
    
    final_df_for_vit_60 = pd.DataFrame(
        keypoints_60_for_vit.reshape(keypoints_60_for_vit.shape[0], -1),
        columns=final_df_for_vit.columns,
        index=range(keypoints_60_for_vit.shape[0])
    )
    final_df_for_vit_60.index = abs_frames_60
    # print(f"final_df_for_vit_60: {final_df_for_vit_60}")
    final_df_for_vit_60.to_csv(csv_path.parent / f"mocap_keypoints_60hz_{csv_path.stem}.csv")
    
    return keypoints_60, full_range_60, abs_frames_60



def calc_gait_events(rsac2hee_z, lsac2hee_z, rsac2toe_z, lsac2toe_z, start_frame, out_root):
        
    dist_r_sac2hee_z = np.array(rsac2hee_z)
    dist_l_sac2hee_z = np.array(lsac2hee_z)
    dist_r_sac2toe_z = np.array(rsac2toe_z)
    dist_l_sac2toe_z = np.array(lsac2toe_z)
    
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


    def filter_by_value_mad(frames, values, k=3.0, min_keep=2, tau_min=0.2):
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
        # print(f"    vals :{vals}")
        # print(f"    med={med:.2f}, mad={mad:.2f}, k*mad={k*mad:.2f} tau={tau:.2f}")

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
    for z_ori, label in zip([dist_r_sac2hee_z, dist_l_sac2hee_z], ['ic_r', 'ic_l']):
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
    for z_ori, label in zip([dist_r_sac2toe_z, dist_l_sac2toe_z], ['to_r', 'to_l']):
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
        n = len(dist_r_sac2hee_z)
        frames_local = np.arange(n)
        frames_global = frames_local + int(start_frame)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # ========== Heel (IC) ==========
        axes[0].plot(frames_global, dist_r_sac2hee_z, label='R Dist', color='blue')
        axes[0].plot(frames_global, dist_l_sac2hee_z, label='L Dist', color='orange')

        # IC ×印
        for i, f in enumerate(event_frame_dict.get('ic_r', [])):
            fg = int(f) + int(start_frame)
            axes[0].plot(fg, dist_r_sac2hee_z[fg - start_frame], 'o', color='blue', markersize=10, label='IC R' if i == 0 else "")
        for i, f in enumerate(event_frame_dict.get('ic_l', [])):
            fg = int(f) + int(start_frame)
            axes[0].plot(fg, dist_l_sac2hee_z[fg - start_frame], 'o', color='orange', markersize=10, label='IC L' if i == 0 else "")
        axes[0].set_ylim(-1, 1)
        axes[0].set_title('MidHip to Heel Z Distance')
        axes[0].set_xlabel('Frame [-]')
        axes[0].set_ylabel('Distance [m]')
        axes[0].legend() 

        # ========== Toe (TO) ==========
        axes[1].plot(frames_global, dist_r_sac2toe_z, label='R Dist', color='blue')
        axes[1].plot(frames_global, dist_l_sac2toe_z, label='L Dist', color='orange')
        
        # TO ×印
        for i, f in enumerate(event_frame_dict.get('to_r', [])):
            fg = int(f) + int(start_frame)
            axes[1].plot(fg, dist_r_sac2toe_z[fg - start_frame], 'o', color='blue', markersize=10, label='TO R' if i == 0 else "")
        for i, f in enumerate(event_frame_dict.get('to_l', [])):
            fg = int(f) + int(start_frame)
            axes[1].plot(fg, dist_l_sac2toe_z[fg - start_frame], 'o', color='orange', markersize=10, label='TO L' if i == 0 else "")

        axes[1].set_title('MidHip to Toe Z Distance')
        axes[1].set_xlabel('Frame [-]')
        axes[1].set_ylabel('Distance [m]')
        axes[1].set_ylim(-1, 1)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(out_root / "z-distance_for_gait_events.png")
        # plt.show()
        plt.close(fig)
    
    return event_frame_dict


def calc_gait_cycles(event_frame_dict):
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

def select_valid_gait_cycles(gait_cycles_abs, gait_cycles_check, tolerance=5):
    """
    歩行サイクルリストから、checkサイクルと近いサイクルのみを選択する。
    
    Parameters
    ----------
    gait_cycles_abs : list
        mocapから自動検出した歩行サイクル（絶対フレーム番号）
        [[ic, ic_opp, to, ic_end], ...]
    gait_cycles_check : list
        goproベースで計算した期待される歩行サイクル（絶対フレーム番号）
        [[ic, ic_opp, to, ic_end], ...]
    tolerance : int
        許容フレーム差（デフォルト5フレーム = 約83ms@60Hz）
    
    Returns
    -------
    valid_cycles : list
        checkサイクルと近い有効なサイクルのみ
    """
    valid_cycles = []
    
    for cycle_abs in gait_cycles_abs:
        ic_abs, ic_opp_abs, to_abs, ic_end_abs = cycle_abs
        
        # checkサイクルの中で最も近いものを探す
        for cycle_check in gait_cycles_check:
            ic_check, ic_opp_check, to_check, ic_end_check = cycle_check
            # print(f"比較中: mocapサイクル {cycle_abs} vs checkサイクル {cycle_check}")
            
            # 各イベントのフレーム差をチェック
            ic_diff = abs(ic_abs - ic_check)
            ic_end_diff = abs(ic_end_abs - ic_end_check)
            
            # IC開始と終了が許容範囲内なら有効とみなす
            if ic_diff <= tolerance and ic_end_diff <= tolerance:
                valid_cycles.append(cycle_abs)
                # print(f"    成功: フレーム差 ic_diff={ic_diff}, ic_end_diff={ic_end_diff}")
                break  # 一致したらこのcycle_absは採用済み
            # print(f"    失敗 : フレーム差 ic_diff={ic_diff}, ic_end_diff={ic_end_diff}")
    
    return valid_cycles