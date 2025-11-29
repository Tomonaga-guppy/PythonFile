import pandas as pd
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import json
import m_opti as opti  # ユーザー定義モジュールと仮定

# ---------------------------------------------------------
# ISB準拠 計算用ヘルパー関数
# ---------------------------------------------------------

def normalize(v):
    """ベクトルの正規化"""
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

def calculate_isb_angles(rot_parent, rot_child, side='R'):
    """
    ISB推奨 (Grood & Suntay) に基づく関節角度の計算
    Rotation Sequence: Z (Flex/Ext) -> X (Abd/Add) -> Y (Int/Ext)
    これはオイラー角の 'ZXY' に相当します (Wu et al., 2002)
    """
    # 相対回転行列: Parent^T * Child
    r_relative = np.dot(rot_parent.T, rot_child)
    
    # オイラー角の抽出 (Intrinsic rotations: Z->X->Y)
    angles = R.from_matrix(r_relative).as_euler('ZXY', degrees=True)
    
    flexion = angles[0]
    abduction = angles[1]
    rotation = angles[2]

    # --- 臨床的な符号の調整 ---
    # ISB定義(右向きZ軸)のままだと、左足の挙動が臨床的直感と逆になる場合があるため補正
    if side == 'L':
        # 左足の外転(左への移動)は、右向き軸基準だと符号が逆になるため反転
        abduction = -abduction
        # 回旋も同様に反転
        rotation = -rotation
        
    return flexion, rotation, abduction # 戻り値の順序: Flex, Rot(InEx), Abd(AdAb)

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------

def main():
    # パス設定 (環境に合わせて変更してください)
    csv_path_dir = Path(r"G:\gait_pattern\BR9G_shuron\sub1\thera0-3\mocap")
    
    # フレーム範囲の設定ロジック (オリジナル維持)
    start_frame = 0
    end_frame = 10000
    if str(csv_path_dir) == r"G:\gait_pattern\BR9G_shuron\sub1\thera0-2\mocap":
        start_frame = 1000
        end_frame = 1440
    elif str(csv_path_dir) == r"G:\gait_pattern\BR9G_shuron\sub1\thera0-3\mocap":
        start_frame = 943
        end_frame = 1400
    elif str(csv_path_dir) == r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\mocap":
        start_frame = 1000
    elif str(csv_path_dir) == r"G:\gait_pattern\BR9G_shuron\sub0\thera0-16\mocap":
        start_frame = 890
        end_frame = 1210
    elif str(csv_path_dir) == r"G:\gait_pattern\BR9G_shuron\sub0\thera0-15\mocap":
        start_frame = 627
        end_frame = 976
    else:
        start_frame = 0
        end_frame = 100

    csv_paths = list(csv_path_dir.glob("[0-9]*_[0-9]*_[0-9].csv"))
    geometry_json_path = Path(r"G:\gait_pattern\20250811_br\sub0\thera0-14\mocap\geometry.json")

    for i, csv_path in enumerate(csv_paths):
        print(f"Processing: {csv_path}")

        try:
            keypoints_mocap, full_range, start_frame, end_frame = opti.read_3d_optitrack(csv_path, start_frame, end_frame, geometry_path=geometry_json_path)
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            continue

        if keypoints_mocap.size == 0:
            print(f"Skipping {csv_path}: No valid data")
            continue

        sampling_freq = 100

        # --- フィルタリング (オリジナル維持) ---
        # 各マーカーのデータを取得・フィルタリング
        def get_marker(idx):
            return np.array([opti.butter_lowpass_filter(keypoints_mocap[:, idx, x], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T

        lank = get_marker(0); lank2 = get_marker(1); lasi = get_marker(2); lhee = get_marker(3)
        lknee = get_marker(4); lknee2 = get_marker(5); lpsi = get_marker(6); ltoe = get_marker(7)
        rank = get_marker(8); rank2 = get_marker(9); rasi = get_marker(10); rhee = get_marker(11)
        rknee = get_marker(12); rknee2 = get_marker(13); rpsi = get_marker(14); rtoe = get_marker(15)

        hip_list = []
        angle_list = []
        pel2hee_r_list = []
        pel2hee_l_list = []

        # --- フレームごとの処理 ---
        for frame_num in full_range:
            
            # 1. 股関節中心の推定 (Davis/Skycom)
            d_asi = np.linalg.norm(rasi[frame_num,:] - lasi[frame_num,:])
            d_leg = (np.linalg.norm(rank[frame_num,:] - rasi[frame_num,:]) + np.linalg.norm(lank[frame_num, :] - lasi[frame_num,:])) / 2
            r_marker = 0.0127; c = 0.115 * d_leg - 0.0153; x_dis = 0.1288 * d_leg - 0.04856
            beta = 0.1 * np.pi; theta = 0.496

            x_thigh = -(x_dis + r_marker) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
            y_rthigh = +(c * np.sin(theta) - d_asi/2); y_lthigh = -(c * np.sin(theta)- d_asi/2)
            z_thigh = -(x_dis + r_marker) * np.sin(beta) - c * np.cos(theta) * np.cos(beta)
            
            rthigh_pelvis = np.array([x_thigh, y_rthigh, z_thigh]).T
            lthigh_pelvis = np.array([x_thigh, y_lthigh, z_thigh]).T

            # 骨盤仮座標系 (HJC計算用)
            hip_0 = (rasi[frame_num,:] + lasi[frame_num,:]) / 2
            sacrum = (rpsi[frame_num,:] + lpsi[frame_num,:]) / 2
            
            e_x0_pelvis_0 = normalize(hip_0 - sacrum)
            e_y_pelvis_0 = normalize(lasi[frame_num,:] - rasi[frame_num,:])
            e_z_pelvis_0 = normalize(np.cross(e_x0_pelvis_0, e_y_pelvis_0))
            
            trans_mat = np.identity(4)
            trans_mat[:3, 0] = e_x0_pelvis_0; trans_mat[:3, 1] = e_y_pelvis_0
            trans_mat[:3, 2] = e_z_pelvis_0; trans_mat[:3, 3] = hip_0

            rthigh_pos = np.dot(trans_mat, np.append(rthigh_pelvis, 1))[:3]
            lthigh_pos = np.dot(trans_mat, np.append(lthigh_pelvis, 1))[:3]
            hip_center = (rthigh_pos + lthigh_pos) / 2
            hip_list.append(hip_center)

            # -------------------------------------------------------------
            # [ISB準拠] 座標系の定義
            # -------------------------------------------------------------

            # --- 骨盤 (Pelvis) ---
            # Z軸: 右向き (LASI -> RASI)
            e_z_pelvis = normalize(rasi[frame_num,:] - lasi[frame_num,:])
            # X軸: 前方 (Sacrum -> MidASIS をZ軸と直交化)
            v_sac_to_origin = normalize(hip_0 - sacrum)
            e_y_temp = normalize(np.cross(e_z_pelvis, v_sac_to_origin)) # 仮の上方向
            e_x_pelvis = normalize(np.cross(e_y_temp, e_z_pelvis))      # 前方
            e_y_pelvis = normalize(np.cross(e_z_pelvis, e_x_pelvis))    # 正式な上方向
            rot_pelvis = np.vstack([e_x_pelvis, e_y_pelvis, e_z_pelvis]).T

            # --- 大腿 (Femur) ---
            # 右大腿
            r_knee_center = (rknee[frame_num, :] + rknee2[frame_num, :]) / 2
            e_y_rfemur = normalize(rthigh_pos - r_knee_center) # 上向き
            # 膝マーカー: rknee2=外側, rknee=内側と仮定 (外-内=右)
            v_knee_vec_r = normalize(rknee2[frame_num, :] - rknee[frame_num, :])
            e_x_rfemur = normalize(np.cross(e_y_rfemur, v_knee_vec_r)) # 前方
            e_z_rfemur = normalize(np.cross(e_x_rfemur, e_y_rfemur))   # 右
            rot_rfemur = np.vstack([e_x_rfemur, e_y_rfemur, e_z_rfemur]).T

            # 左大腿
            l_knee_center = (lknee[frame_num, :] + lknee2[frame_num, :]) / 2
            e_y_lfemur = normalize(lthigh_pos - l_knee_center)
            # 左膝: 内側(lknee) - 外側(lknee2) = 右向きベクトル
            v_knee_vec_l = normalize(lknee[frame_num, :] - lknee2[frame_num, :])
            e_x_lfemur = normalize(np.cross(e_y_lfemur, v_knee_vec_l))
            e_z_lfemur = normalize(np.cross(e_x_lfemur, e_y_lfemur))
            rot_lfemur = np.vstack([e_x_lfemur, e_y_lfemur, e_z_lfemur]).T

            # --- 下腿 (Shank) ---
            # 右下腿
            r_ankle_center = (rank[frame_num,:] + rank2[frame_num,:]) / 2
            e_y_rshank = normalize(r_knee_center - r_ankle_center)
            e_x_rshank = normalize(np.cross(e_y_rshank, v_knee_vec_r))
            e_z_rshank = normalize(np.cross(e_x_rshank, e_y_rshank))
            rot_rshank = np.vstack([e_x_rshank, e_y_rshank, e_z_rshank]).T

            # 左下腿
            l_ankle_center = (lank[frame_num,:] + lank2[frame_num,:]) / 2
            e_y_lshank = normalize(l_knee_center - l_ankle_center)
            e_x_lshank = normalize(np.cross(e_y_lshank, v_knee_vec_l))
            e_z_lshank = normalize(np.cross(e_x_lshank, e_y_lshank))
            rot_lshank = np.vstack([e_x_lshank, e_y_lshank, e_z_lshank]).T

            # --- 足部 (Foot) ---
            # 右足
            e_x_rfoot = normalize(rtoe[frame_num,:] - rhee[frame_num,:]) # 前方
            v_ankle_vec_r = normalize(rank2[frame_num,:] - rank[frame_num,:]) # 外-内=右
            e_y_rfoot = normalize(np.cross(v_ankle_vec_r, e_x_rfoot)) # 上
            e_z_rfoot = normalize(np.cross(e_x_rfoot, e_y_rfoot)) # 右
            rot_rfoot = np.vstack([e_x_rfoot, e_y_rfoot, e_z_rfoot]).T

            # 左足
            e_x_lfoot = normalize(ltoe[frame_num,:] - lhee[frame_num,:])
            v_ankle_vec_l = normalize(lank[frame_num,:] - lank2[frame_num,:]) # 内-外=右
            e_y_lfoot = normalize(np.cross(v_ankle_vec_l, e_x_lfoot))
            e_z_lfoot = normalize(np.cross(e_x_lfoot, e_y_lfoot))
            rot_lfoot = np.vstack([e_x_lfoot, e_y_lfoot, e_z_lfoot]).T

            # -------------------------------------------------------------
            # [ISB準拠] 角度計算 (ZXY)
            # -------------------------------------------------------------
            
            r_hip_flex, r_hip_rot, r_hip_abd = calculate_isb_angles(rot_pelvis, rot_rfemur, 'R')
            l_hip_flex, l_hip_rot, l_hip_abd = calculate_isb_angles(rot_pelvis, rot_lfemur, 'L')

            r_knee_flex, r_knee_rot, r_knee_abd = calculate_isb_angles(rot_rfemur, rot_rshank, 'R')
            l_knee_flex, l_knee_rot, l_knee_abd = calculate_isb_angles(rot_lfemur, rot_lshank, 'L')

            r_ankle_flex, r_ankle_rot, r_ankle_abd = calculate_isb_angles(rot_rshank, rot_rfoot, 'R')
            l_ankle_flex, l_ankle_rot, l_ankle_abd = calculate_isb_angles(rot_lshank, rot_lfoot, 'L')

            angle_list.append([
                r_hip_flex, l_hip_flex, r_knee_flex, l_knee_flex, r_ankle_flex, l_ankle_flex,
                r_hip_rot, l_hip_rot, r_knee_rot, l_knee_rot, r_ankle_rot, l_ankle_rot,
                r_hip_abd, l_hip_abd, r_knee_abd, l_knee_abd, r_ankle_abd, l_ankle_abd
            ])

            # オリジナルのイベント検出用距離データ
            pel2hee_r = rhee[frame_num, :] - hip_center[:]
            pel2hee_r_list.append(pel2hee_r)
            pel2hee_l = lhee[frame_num, :] - hip_center[:]
            pel2hee_l_list.append(pel2hee_l)

        # --- データフレーム化 ---
        angle_df = pd.DataFrame(angle_list, columns=[
            "R_Hip_FlEx", "L_Hip_FlEx", "R_Knee_FlEx", "L_Knee_FlEx", "R_Ankle_PlDo", "L_Ankle_PlDo",
            "R_Hip_InEx", "L_Hip_InEx", "R_Knee_InEx", "L_Knee_InEx", "R_Ankle_InEx", "L_Ankle_InEx",
            "R_Hip_AdAb", "L_Hip_AdAb", "R_Knee_AdAb", "L_Knee_AdAb", "R_Ankle_AdAb", "L_Ankle_AdAb"
        ], index=full_range)

        # 不連続点の修正 (Unwrap)
        for col in angle_df.columns:
            angle_df[col] = np.degrees(np.unwrap(np.radians(angle_df[col])))
        
        # CSV保存
        angle_df.to_csv(csv_path.parent / f"angle_100Hz_ISB_{csv_path.name}")

        # --- 歩行イベント検出 (オリジナルロジック) ---
        pel2hee_r_array = np.array(pel2hee_r_list)
        pel2hee_l_array = np.array(pel2hee_l_list)
        
        # Z座標を使用（オリジナルのまま）
        df_IcTo = pd.DataFrame({
            "frame_100hz_rel": full_range,
            "rhee_pel_z": pel2hee_r_array[:, 2],
            "lhee_pel_z": pel2hee_l_array[:, 2],
        })
        
        # 極値検出 (簡易版)
        df_ic_r = df_IcTo.sort_values(by="rhee_pel_z", ascending=False)
        df_ic_l = df_IcTo.sort_values(by="lhee_pel_z", ascending=False)
        df_to_r = df_IcTo.sort_values(by="rhee_pel_z")
        df_to_l = df_IcTo.sort_values(by="lhee_pel_z")
        
        def get_filtered_events(df_sorted, count=30):
            raw = df_sorted.head(count)["frame_100hz_rel"].values.astype(int)
            filt = []
            skip = set()
            for v in sorted(raw):
                if v in skip: continue
                filt.append(v)
                skip.update(range(v-10, v+11))
            return sorted(filt)

        ic_r = get_filtered_events(df_ic_r); ic_l = get_filtered_events(df_ic_l)
        to_r = get_filtered_events(df_to_r); to_l = get_filtered_events(df_to_l)

        # サイクル作成
        gait_cycles_r = []
        for i in range(len(ic_r) - 1):
            if any(ic_r[i] < t < ic_r[i+1] for t in to_r):
                gait_cycles_r.append([ic_r[i], next(t for t in to_r if ic_r[i] < t < ic_r[i+1]), ic_r[i+1]])

        gait_cycles_l = []
        for i in range(len(ic_l) - 1):
            if any(ic_l[i] < t < ic_l[i+1] for t in to_l):
                gait_cycles_l.append([ic_l[i], next(t for t in to_l if ic_l[i] < t < ic_l[i+1]), ic_l[i+1]])

        # --- 正規化と平均化 ---
        normalized_percentage = np.linspace(0, 100, 101)
        
        def process_cycles_and_stats(cycles, side_prefix):
            norm_cycles = []
            if not cycles: return None
            
            # 全サイクルのデータを格納する辞書
            data_collection = {
                f"{side_prefix}_Hip_FlEx": [], f"{side_prefix}_Knee_FlEx": [], f"{side_prefix}_Ankle_PlDo": [],
                f"{side_prefix}_Hip_InEx": [], f"{side_prefix}_Knee_InEx": [], f"{side_prefix}_Ankle_InEx": [],
                f"{side_prefix}_Hip_AdAb": [], f"{side_prefix}_Knee_AdAb": [], f"{side_prefix}_Ankle_AdAb": []
            }
            stance_percents = []

            for idx, (start, to, end) in enumerate(cycles):
                length = end - start
                orig_frames = np.arange(start, end + 1)
                stance_percents.append(((to - start)/length)*100)
                
                # 各関節角度を正規化
                for col in data_collection.keys():
                    if col in angle_df.columns:
                        vals = angle_df.loc[orig_frames, col].values
                        interp_vals = np.interp(normalized_percentage, np.linspace(0, 100, len(orig_frames)), vals)
                        data_collection[col].append(interp_vals)

            # 平均と標準偏差を計算
            stats_df = pd.DataFrame({'Percentage': normalized_percentage})
            for col, values in data_collection.items():
                arr = np.array(values)
                stats_df[f"{col}_mean"] = np.mean(arr, axis=0)
                stats_df[f"{col}_std"] = np.std(arr, axis=0)
            
            return stats_df, np.mean(stance_percents)

        stats_r, to_percent_r = process_cycles_and_stats(gait_cycles_r, "R")
        stats_l, to_percent_l = process_cycles_and_stats(gait_cycles_l, "L")

        # 保存
        if stats_r is not None: stats_r.to_csv(csv_path.parent / f"norm_stats_R_ISB_{csv_path.stem}.csv")
        if stats_l is not None: stats_l.to_csv(csv_path.parent / f"norm_stats_L_ISB_{csv_path.stem}.csv")

# --- グラフ描画 (Plotting) ---
        def plot_side(stats, to_percent, side_char):
            if stats is None: return
            
            # 1. 屈曲伸展 (Flex/Ext)
            fig, axes = plt.subplots(3, 1, figsize=(8, 10))
            # (関節名, Y軸最小, Y軸最大, 接尾辞)
            joints_config = [
                ('Hip', -40, 60, 'FlEx'), 
                ('Knee', -10, 80, 'FlEx'), 
                ('Ankle', -30, 40, 'PlDo') # 足首だけ PlDo
            ]
            
            for i, (joint, ymin, ymax, suffix) in enumerate(joints_config):
                col = f"{side_char}_{joint}_{suffix}"
                
                # データが存在するか確認
                if f"{col}_mean" not in stats.columns:
                    print(f"Warning: Column {col}_mean not found. Skipping plot.")
                    continue

                mean = stats[f"{col}_mean"]
                std = stats[f"{col}_std"]
                
                axes[i].plot(normalized_percentage, mean, 'b-', label='Mean')
                axes[i].fill_between(normalized_percentage, mean-std, mean+std, alpha=0.3, color='b')
                axes[i].axvline(x=to_percent, color='r', linestyle='--', label='Toe Off')
                
                # タイトル設定
                title_suffix = "Flex(+)/Ext(-)"
                if suffix == 'PlDo':
                    title_suffix = "Dorsi(+)/Plant(-)"
                axes[i].set_title(f"{side_char} {joint} {title_suffix}")
                
                axes[i].set_ylim(ymin, ymax)
                axes[i].grid(True)
            
            plt.tight_layout()
            plt.savefig(csv_path.parent / f"gait_cycle_FlEx_{side_char}_ISB_{csv_path.stem}.png")
            plt.close()

            # 2. 回旋 (Int/Ext Rotation)
            fig, axes = plt.subplots(3, 1, figsize=(8, 10))
            joints_simple = ['Hip', 'Knee', 'Ankle']
            
            for i, joint in enumerate(joints_simple):
                col = f"{side_char}_{joint}_InEx"
                if f"{col}_mean" not in stats.columns: continue

                mean = stats[f"{col}_mean"]
                std = stats[f"{col}_std"]
                
                axes[i].plot(normalized_percentage, mean, 'g-', label='Mean')
                axes[i].fill_between(normalized_percentage, mean-std, mean+std, alpha=0.3, color='g')
                axes[i].axvline(x=to_percent, color='r', linestyle='--')
                axes[i].set_title(f"{side_char} {joint} Int(+)/Ext(-)")
                axes[i].set_ylim(-30, 30)
                axes[i].grid(True)
            plt.tight_layout()
            plt.savefig(csv_path.parent / f"gait_cycle_InEx_{side_char}_ISB_{csv_path.stem}.png")
            plt.close()

            # 3. 外転内転 (Abd/Add)
            fig, axes = plt.subplots(3, 1, figsize=(8, 10))
            
            for i, joint in enumerate(joints_simple):
                col = f"{side_char}_{joint}_AdAb"
                if f"{col}_mean" not in stats.columns: continue

                mean = stats[f"{col}_mean"]
                std = stats[f"{col}_std"]
                
                axes[i].plot(normalized_percentage, mean, 'm-', label='Mean')
                axes[i].fill_between(normalized_percentage, mean-std, mean+std, alpha=0.3, color='m')
                axes[i].axvline(x=to_percent, color='r', linestyle='--')
                axes[i].set_title(f"{side_char} {joint} Abd(+)/Add(-)")
                axes[i].set_ylim(-30, 30)
                axes[i].grid(True)
            plt.tight_layout()
            plt.savefig(csv_path.parent / f"gait_cycle_AdAb_{side_char}_ISB_{csv_path.stem}.png")
            plt.close()

        plot_side(stats_r, to_percent_r, "R")
        plot_side(stats_l, to_percent_l, "L")

        print(f"Finished processing: {csv_path.name}")

if __name__ == "__main__":
    main()