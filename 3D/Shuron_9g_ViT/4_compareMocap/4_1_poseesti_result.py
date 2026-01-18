"""
目的:
- 3_3_reconstruct3d.py が出力した 3D姿勢(OpenPose/ViTPose等) の解析を行う

入出力（想定フォルダ）:
ROOT/
  subX/
    theraX-0/
      mocap/
        *_opti_intermediate.npz  (4_1が作成)
      3d_kp_<method>_<csv_tag>.npz  (3_3が作成)

出力:
theraX-0/
  Pose3D_results/
    <method>/
      normalized_cycle_R_mean_<tag>_<method>.csv
      normalized_cycle_L_mean_<tag>_<method>.csv
      gait_parameters_R_<tag>_<method>.csv
      gait_parameters_L_<tag>_<method>.csv
      angle_error_summary_<tag>_<method>.csv
      gait_cycle_FlEx_R_<tag>_<method>.png など（主要角度プロット）

重要:
- 3_3の出力（3D再計算・補間・フィルタ）には一切手を入れない（解析だけ）
- タイミング合わせは同期用IMUで行う
依存:
- numpy, pandas, matplotlib
- m_openpose.py（culc_angle_all_frames を使用）
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import m_openpose as op  # culc_angle_all_frames

# =========================
# 設定（ここだけ調整）
# =========================
ROOT_DIR = Path(r"G:\gait_pattern\2025_shuron_BR9G")

# 3_3出力npzを使う優先順位（最初に見つかったものを使う）
PREFERRED_3D_KEY_ORDER = ["butter", "spline", "conf_filt", "raw"]

# 3Dのサンプリング周波数
FS_3D = 60.0

# 麻痺側情報
PARALYZED_SIDE = "R"  # 9号館では全員右麻痺と設定

# =========================
# 角度プロットのY範囲（関節ごとに設定）
# =========================
YLIM_BY_ANGLE = {
    # Hip
    "Hip_FlEx":  (-40,  50),
    "Hip_AdAb":  (-20,  20),
    "Hip_InEx":  (-20,  20),

    # Knee
    "Knee_FlEx": (-10, 70),

    # Ankle  ※あなたの系列名は Ankle_PlDo (背屈/底屈)
    "Ankle_PlDo": (-30,  50),
}

def ylims_for_angle_key(k: str):
    """
    例: 'R_Hip_FlEx' -> (-70, 70)
        'L_Knee_FlEx' -> (0, 120)
    見つからなければ None
    """
    base = k.split("_", 1)[1]  # 'Hip_FlEx' など
    return YLIM_BY_ANGLE.get(base, None)


# =========================
# 小物
# =========================
def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def pick_3d_array(npz, preferred_keys):
    # 3_3: butterが無効のとき np.array([]) を入れていることがある
    for k in preferred_keys:
        if k in npz.files:
            arr = npz[k]
            if isinstance(arr, np.ndarray) and arr.size == 0:
                continue
            return k, arr
    raise KeyError(f"no usable 3d key found. available={npz.files}")

# =========================
# 角度計算（BODY_25 index前提）
# =========================
def compute_angles_from_body25(kp3d: np.ndarray):
    """
    kp3d: (n_frames, 25, 3)
    returns dict of angle series (deg)
    """
    n = kp3d.shape[0]
    # 必要点抽出（4_ref_opti_opの定義に合わせる）
    neck = kp3d[:, 1, :]
    midhip = kp3d[:, 8, :]

    rhip = kp3d[:, 9, :]
    rknee = kp3d[:, 10, :]
    rankle = kp3d[:, 11, :]
    rhee = kp3d[:, 24, :]
    rtoe = (kp3d[:, 22, :] + kp3d[:, 23, :]) / 2.0

    lhip = kp3d[:, 12, :]
    lknee = kp3d[:, 13, :]
    lankle = kp3d[:, 14, :]
    lhee = kp3d[:, 21, :]
    ltoe = (kp3d[:, 19, :] + kp3d[:, 20, :]) / 2.0

    # ベクトル
    pel_up = neck - midhip
    thigh_r = rknee - rhip
    shank_r = rankle - rknee
    foot_r = rtoe - rhee

    thigh_l = lknee - lhip
    shank_l = lankle - lknee
    foot_l = ltoe - lhee

    # 回転軸など
    n_axis = rhip - lhip  # 左→右

    # 股関節屈伸/膝屈伸/足関節背屈底屈
    rhip_flex = op.culc_angle_all_frames(pel_up, thigh_r, n_axis, degrees=True, angle_type="hip")
    lhip_flex = op.culc_angle_all_frames(pel_up, thigh_l, n_axis, degrees=True, angle_type="hip")

    rknee_flex = op.culc_angle_all_frames(thigh_r, shank_r, n_axis, degrees=True, angle_type="knee")
    lknee_flex = op.culc_angle_all_frames(thigh_l, shank_l, n_axis, degrees=True, angle_type="knee")

    rankle_pldo = op.culc_angle_all_frames(shank_r, foot_r, n_axis, degrees=True, angle_type="ankle")
    lankle_pldo = op.culc_angle_all_frames(shank_l, foot_l, n_axis, degrees=True, angle_type="ankle")

    # 内転外転: 進行方向相当の軸を cross(pel_up, n_axis) とする（旧スクリプト互換）
    n_axis_adab = np.cross(pel_up, n_axis)
    # 0割対策（長さ0のときは少しだけ逃がす）
    eps = 1e-9
    n_norm = np.linalg.norm(n_axis_adab, axis=1)
    bad = n_norm < eps
    if np.any(bad):
        n_axis_adab[bad, :] = np.array([0.0, 0.0, 1.0])

    rhip_adab = op.culc_angle_all_frames(pel_up, thigh_r, n_axis_adab, degrees=True, angle_type="hip_adab")
    lhip_adab = op.culc_angle_all_frames(pel_up, thigh_l, n_axis_adab, degrees=True, angle_type="hip_adab")

    # 内旋外旋: （旧スクリプト互換）vector1=thigh, vector2=n_axis_adab, n=pel_up
    rhip_inex = op.culc_angle_all_frames(thigh_r, n_axis_adab, pel_up, degrees=True, angle_type="hip_inex")
    lhip_inex = op.culc_angle_all_frames(thigh_l, n_axis_adab, pel_up, degrees=True, angle_type="hip_inex")

    # 返す（必要なら増やせる）
    return dict(
        # angles
        R_Hip_FlEx=rhip_flex,
        R_Knee_FlEx=rknee_flex,
        R_Ankle_PlDo=rankle_pldo,
        R_Hip_AdAb=rhip_adab,
        R_Hip_InEx=rhip_inex,
        L_Hip_FlEx=lhip_flex,
        L_Knee_FlEx=lknee_flex,
        L_Ankle_PlDo=lankle_pldo,
        L_Hip_AdAb=lhip_adab,
        L_Hip_InEx=lhip_inex,
    )


# =========================
# 歩行パラメータ（3D）
# =========================
def calculate_gait_parameters_3d(gait_cycles, midhip, rhee, lhee, side, sampling_freq=60):
    """
    回帰に使用するパラメータを算出する
    gait_cycles: [[ic, ic_opp, to, ic_end], ...] (int index)
    
    算出する指標:
    - gait_speed: 歩行速度 [m/s]
    - swing_stance_ratio: 時間的対称性（遊脚期比）ここではまだ算出不可
    - stride_time: ストライド時間 [s]
    - stride_width: 歩隔 [m]
    """
    gait_params = []
    
    for cycle_idx, (ic_start, ic_opp, to, ic_end) in enumerate(gait_cycles):
        # 整数インデックスに変換
        ic_start_idx = int(np.round(ic_start))
        ic_opp_idx = int(np.round(ic_opp))
        to_idx = int(np.round(to))
        ic_end_idx = int(np.round(ic_end))
        
        # ストライド時間 [s]
        stride_time = (ic_end_idx - ic_start_idx) / sampling_freq
        
        # 遊脚期時間 [s]
        swing_duration = (ic_end_idx - to_idx) / sampling_freq
        
        # 歩行速度 [m/s]
        hip_displacement = np.linalg.norm(midhip[ic_end_idx] - midhip[ic_start_idx]) / 1000 # mmからmに変換
        gait_speed = hip_displacement / stride_time

        # 歩隔[m](とステップ長[m])
        if side == 'R':
            a = np.linalg.norm(lhee[ic_opp_idx] - rhee[ic_start_idx]) / 1000 # mmからmに変換
            b = np.linalg.norm(rhee[ic_end_idx] - lhee[ic_opp_idx]) / 1000 # mmからmに変換
            c = np.linalg.norm(rhee[ic_end_idx] - rhee[ic_start_idx]) / 1000 # mmからmに変換
            step_length = (b**2 + c**2 -a**2) / (2*c)
            stride_width = np.sqrt(b**2 - step_length**2)
        else:
            a = np.linalg.norm(rhee[ic_opp_idx] - lhee[ic_start_idx]) / 1000 # mmからmに変換
            b = np.linalg.norm(lhee[ic_end_idx] - rhee[ic_opp_idx]) / 1000 # mmからmに変換
            c = np.linalg.norm(lhee[ic_end_idx] - lhee[ic_start_idx]) / 1000 # mmからmに変換
            step_length = (b**2 + c**2 -a**2) / (2*c)
            stride_width = np.sqrt(b**2 - step_length**2)

        params = {
            'cycle_index': cycle_idx,
            'gait_speed': gait_speed,
            'stride_time': stride_time,
            'swing_duration': swing_duration,
            'stride_width': stride_width,
        }
        gait_params.append(params)
    
    return gait_params


# =========================
# 本体（1 trial x 1 method）
# =========================
def process_one_method(thera_dir: Path, npz_3d_path: Path):
    """
    1条件分をここで処理する
    """
    # ---------- 被験者ID・麻痺側 ----------
    sub_id = thera_dir.parent.name  # subX
    paralyzed_side = PARALYZED_SIDE
    
    if paralyzed_side is None:
        print(f"[SKIP] {sub_id} not in PARALYZED_SIDES dict. Skipping {thera_dir.name}")
        return
    
    # ---------- 3Dnpzファイルの読み込み ----------
    npz = np.load(npz_3d_path, allow_pickle=True)
    used_key, kp3d = pick_3d_array(npz, PREFERRED_3D_KEY_ORDER)
    
    # ---------- 有効フレーム範囲の決定（MidHip Z） ----------
    # midhipが-2mから2mの範囲を有効範囲とするための処理
    valid_frame = npz["valid_frame_range"] if "valid_frame_range" in npz.files else None
    valid_start, valid_end = valid_frame[0], valid_frame[1] if valid_frame is not None else (valid_start, valid_end)

    print(f"[INFO] Valid frame range by midhip z: {valid_start} -> {valid_end} "
        f"(total {valid_end - valid_start + 1} frames)")

    npz_pt_path = Path(str(npz_3d_path).replace("_PA", "_PT"))
    npz_pt = np.load(npz_pt_path, allow_pickle=True)
    _, kp3d_pt = pick_3d_array(npz_pt, PREFERRED_3D_KEY_ORDER)

    # kp3d: (n_frames, 25, 3)
    n_frames = len(npz["frame"])
    print(f"n_frames: {n_frames}, used_key: {used_key}")
    method_tag = npz_3d_path.stem  # 3d_kp_<method(ViTPose)>_<csv_tag>
    # method名を雑に切り出し
    # 例: 3d_kp_ViTPose_pa_spline など
    m = re.match(r"3d_kp_(.+)$", method_tag)
    # 例: ViTPose_pa_spline など
    method_tag = m.group(1) if m else method_tag
    
    # ---------- 結果の保存先ディレクトリ ----------
    tag_short = npz_3d_path.stem.replace("3d_kp_", "")

    out_root = thera_dir / "ViTPose_results"
    out_root.mkdir(parents=True, exist_ok=True)
    
    #  --------- 歩行イベント検出 ----------
    # 基本的に解析は有効フレーム範囲内で行う（local）・デバック時は全体通してのフレーム番号(global)へ変換
    event_frame_dict_local = op.cucl_gait_event(kp3d[valid_start:valid_end+1], valid_start, out_root)

    gait_cycles_r, gait_cycles_l = op.culc_gait_cycles(event_frame_dict_local)
    print(f"Detected {len(gait_cycles_r)} right cycles, {len(gait_cycles_l)} left cycles")
    print(f"gait_cycles_r: {gait_cycles_r}")
    print(f"gait_cycles_l: {gait_cycles_l}")
    
    if len(gait_cycles_r) == 0 and len(gait_cycles_l) == 0:
        print(f"[SKIP] no valid cycles after conversion: {thera_dir.name} / {npz_3d_path.name}")
        return
    
    # --- プロット用にグローバル版を別で作る ---
    event_frame_dict_global = {
        k: [int(f) + valid_start for f in v]
        for k, v in event_frame_dict_local.items()
    }

    # デバック用に(全体通しての)初期接地フレームを取得しておく
    ic_r = event_frame_dict_global.get("ic_r", [])
    ic_l = event_frame_dict_global.get("ic_l", [])
    
    print(f"ic_r: {ic_r}, ic_l: {ic_l}")

    # ---------- 関節角度算出 ----------
    angles_dict = compute_angles_from_body25(kp3d[valid_start:valid_end+1])

    # プロット・解析用にフレーム軸を元に戻す
    for k in angles_dict:
        if isinstance(angles_dict[k], np.ndarray):
            pad = np.full((valid_start,), np.nan)
            angles_dict[k] = np.concatenate([pad, angles_dict[k]])
        
    def plot_three_timeseries(title, keys_r, keys_l, angles_dict, fname, ic_r=None, ic_l=None):
        """
        keys_r: ["R_Hip_FlEx","R_Knee_FlEx","R_Ankle_PlDo"] など
        keys_l: ["L_Hip_FlEx","L_Knee_FlEx","L_Ankle_PlDo"] など
        ic_rやic_l: 左右の初期接地フレームリスト（プロットに縦線を入れる）
        3段：股→膝→足
        横軸：フレーム番号
        各段に R と L を重ね描き
        """
        assert len(keys_r) == 3 and len(keys_l) == 3

        # フレーム数（R/Lで長さが違う可能性に備えて最小に合わせる）
        n_r = len(angles_dict[keys_r[0]]) if keys_r[0] in angles_dict else 0
        n_l = len(angles_dict[keys_l[0]]) if keys_l[0] in angles_dict else 0
        n = min(n_r, n_l) if (n_r > 0 and n_l > 0) else max(n_r, n_l)
        frames = np.arange(n)

        # 初期接地フレーム（範囲外は除外、重複も潰す）
        ic_r = [] if ic_r is None else sorted({int(x) for x in ic_r if 0 <= int(x) < n})
        ic_l = [] if ic_l is None else sorted({int(x) for x in ic_l if 0 <= int(x) < n})

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

        for ax, kr, kl in zip(axes, keys_r, keys_l):
            has_r = kr in angles_dict
            has_l = kl in angles_dict

            if not has_r and not has_l:
                ax.set_title(f"{kr} / {kl} (missing)")
                ax.grid(True)
                continue

            if has_r:
                y_r = np.asarray(angles_dict[kr], dtype=float)[:n]
                ax.plot(frames, y_r, label=kr, color="tab:orange")

            if has_l:
                y_l = np.asarray(angles_dict[kl], dtype=float)[:n]
                ax.plot(frames, y_l, label=kl, color="tab:blue")

            # ---- IC 縦線（破線） ----
            # 凡例が増えすぎないよう「最初の1本だけ label を付ける」
            for i, f in enumerate(ic_r):
                ax.axvline(f, linestyle="--", linewidth=1.0,
                        color="tab:orange", alpha=0.6,
                        label="IC_R" if i == 0 else None)
            for i, f in enumerate(ic_l):
                ax.axvline(f, linestyle="--", linewidth=1.0,
                        color="tab:blue", alpha=0.6,
                        label="IC_L" if i == 0 else None)
            
            ax.set_ylabel("Angle [deg]")

            # y軸範囲は「R_」「L_」を外したベース名で共通設定（既存関数を流用）
            # 例: R_Hip_FlEx / L_Hip_FlEx のどちらでも同じYlimになる
            ref_key = kr if has_r else kl
            ylims = ylims_for_angle_key(ref_key)
            if ylims is not None:
                ax.set_ylim(*ylims)

            ax.grid(True)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Frame")
        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(out_root / fname)
        plt.close()
    
    plot_three_timeseries(
        f"Timeseries Flex/Ext (R & L) - {method_tag} ({used_key})",
        ["R_Hip_FlEx", "R_Knee_FlEx", "R_Ankle_PlDo"],
        ["L_Hip_FlEx", "L_Knee_FlEx", "L_Ankle_PlDo"],
        angles_dict,
        f"timeseries_FlEx_{tag_short}.png",
        ic_r=ic_r, ic_l=ic_l
    )
    plot_three_timeseries(
        f"Timeseries Ad/Ab (R & L) - {method_tag} ({used_key})",
        ["R_Hip_AdAb", "R_Knee_AdAb", "R_Ankle_AdAb"],
        ["L_Hip_AdAb", "L_Knee_AdAb", "L_Ankle_AdAb"],
        angles_dict,
        f"timeseries_AdAb_{tag_short}.png",
        ic_r=ic_r, ic_l=ic_l
    )
    plot_three_timeseries(
        f"Timeseries In/Ex (R & L) - {method_tag} ({used_key})",
        ["R_Hip_InEx", "R_Knee_InEx", "R_Ankle_InEx"],
        ["L_Hip_InEx", "L_Knee_InEx", "L_Ankle_InEx"],
        angles_dict,
        f"timeseries_InEx_{tag_short}.png",
        ic_r=ic_r, ic_l=ic_l
    )

    # ---------- 歩行パラメータの計算 ----------
    midhip = kp3d[valid_start:valid_end+1, 8, :]
    rhee = kp3d[valid_start:valid_end+1, 24, :]
    lhee = kp3d[valid_start:valid_end+1, 21, :]

    gait_params_r = calculate_gait_parameters_3d(gait_cycles_r, midhip, rhee, lhee, side="R", sampling_freq=FS_3D)
    gait_params_l = calculate_gait_parameters_3d(gait_cycles_l, midhip, rhee, lhee, side="L", sampling_freq=FS_3D)

    # 時間的対称性指標（遊脚期比のSI）を算出
    if paralyzed_side == "R":
        swing_duration_para = np.array([p['swing_duration'] for p in gait_params_r]).mean()
        swing_duration_nonpara = np.array([p['swing_duration'] for p in gait_params_l]).mean()
        symmetry_index_sw = (swing_duration_para - swing_duration_nonpara) / (0.5 * (swing_duration_para + swing_duration_nonpara)) * 100
    elif paralyzed_side == "L":
        swing_duration_para = np.array([p['swing_duration'] for p in gait_params_l]).mean()
        swing_duration_nonpara = np.array([p['swing_duration'] for p in gait_params_r]).mean()
        symmetry_index_sw = (swing_duration_para - swing_duration_nonpara) / (0.5 * (swing_duration_para + swing_duration_nonpara)) * 100
    
    # 関節角度情報をまとめる
    if paralyzed_side == "R":
        hip_flex = angles_dict["R_Hip_FlEx"][gait_cycles_r[0][0]:gait_cycles_r[0][3]+1] if len(gait_cycles_r) > 0 else np.array([])
        hip_max_flex = np.max(hip_flex) if hip_flex.size > 0 else np.nan
        hip_max_ext = np.min(hip_flex) if hip_flex.size > 0 else np.nan
        knee_max_flex = np.max(angles_dict["R_Knee_FlEx"][gait_cycles_r[0][0]:gait_cycles_r[0][3]+1]) if len(gait_cycles_r) > 0 else np.nan
        ankle_max_pl = np.min(angles_dict["R_Ankle_PlDo"][gait_cycles_r[0][0]:gait_cycles_r[0][3]+1]) if len(gait_cycles_r) > 0 else np.nan
        hip_max_ab = np.max(angles_dict["R_Hip_AdAb"][gait_cycles_r[0][0]:gait_cycles_r[0][3]+1]) if len(gait_cycles_r) > 0 else np.nan
    elif paralyzed_side == "L":
        hip_flex = angles_dict["L_Hip_FlEx"][gait_cycles_l[0][0]:gait_cycles_l[0][3]+1] if len(gait_cycles_l) > 0 else np.array([])
        hip_max_flex = np.max(hip_flex) if hip_flex.size > 0 else np.nan
        hip_max_ext = np.min(hip_flex) if hip_flex.size > 0 else np.nan
        knee_max_flex = np.max(angles_dict["L_Knee_FlEx"][gait_cycles_l[0][0]:gait_cycles_l[0][3]+1]) if len(gait_cycles_l) > 0 else np.nan
        ankle_max_pl = np.min(angles_dict["L_Ankle_PlDo"][gait_cycles_l[0][0]:gait_cycles_l[0][3]+1]) if len(gait_cycles_l) > 0 else np.nan
        hip_max_ab = np.max(angles_dict["L_Hip_AdAb"][gait_cycles_l[0][0]:gait_cycles_l[0][3]+1]) if len(gait_cycles_l) > 0 else np.nan
    max_angle_list = [hip_max_flex, hip_max_ext, knee_max_flex, ankle_max_pl, hip_max_ab]
    
    # 歩行パラメータのまとめ
    def summarize_gait_parameters(gait_params_r, gait_params_l, symmetry_index_sw, max_angle_list, paralyzed_side):
        """
        歩行パラメータのまとめを作成
        """
        if paralyzed_side == "R":
            gait_speed = np.mean([p['gait_speed'] for p in gait_params_r])
            stride_time = np.mean([p['stride_time'] for p in gait_params_r])
            stride_width = np.mean([p['stride_width'] for p in gait_params_r])
        elif paralyzed_side == "L":
            gait_speed = np.mean([p['gait_speed'] for p in gait_params_l])
            stride_time = np.mean([p['stride_time'] for p in gait_params_l])
            stride_width = np.mean([p['stride_width'] for p in gait_params_l])
        summary = {
            'gait_speed': gait_speed,
            'symmetry_index_sw': symmetry_index_sw,
            'stride_time': stride_time,
            'stride_width': stride_width,
            'hip_max_flex': max_angle_list[0],
            'hip_max_ext': max_angle_list[1],
            'knee_max_flex': max_angle_list[2],
            'ankle_max_pl': max_angle_list[3],
            'hip_max_ab': max_angle_list[4],
        }
        return summary
    
    pa_gait_params = summarize_gait_parameters(gait_params_r, gait_params_l, symmetry_index_sw, max_angle_list, paralyzed_side)
    
    # ---------- 結果保存 ----------
    # 歩行パラメータ保存
    pa_gait_params_df = pd.DataFrame([pa_gait_params])
    pa_gait_params_df.to_csv(out_root / f"pa_gait_parameters.csv", index=False)

    # PTの介助指標を算出  ##############################################################################
    def _nan_safe_zscore(x):
        x = np.asarray(x, float)
        m = np.nanmean(x)
        s = np.nanstd(x)
        if not np.isfinite(s) or s < 1e-12:
            return np.full_like(x, np.nan)
        return (x - m) / s

    def _norm_xcorr_max(a, b, max_lag, def_lag=None):
        a = _nan_safe_zscore(a)
        b = _nan_safe_zscore(b)

        mask = np.isfinite(a) & np.isfinite(b)
        a = a[mask]
        b = b[mask]
        n = len(a)
        if n < 5:
            return np.nan

        if def_lag is not None:
            lag = def_lag
            if lag < 0:
                aa, bb = a[-lag:], b[:n+lag]
            elif lag > 0:
                aa, bb = a[:n-lag], b[lag:]
            else:
                aa, bb = a, b
            if len(aa) < 5:
                return np.nan
            cc = np.nanmean(aa * bb)
            return float(cc), int(lag)
        else:
            best = -np.inf
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    aa, bb = a[-lag:], b[:n+lag]
                elif lag > 0:
                    aa, bb = a[:n-lag], b[lag:]
                else:
                    aa, bb = a, b
                if len(aa) < 5:
                    continue
                cc = np.nanmean(aa * bb)
                if np.isfinite(cc):
                    if best < cc:
                        best = cc
                        best_lag = lag

        return np.nan if best == -np.inf else float(best), int(best_lag)

    def calculate_pt_assist_metrics_by_cycle(gait_cycles,pa_midhip,pt_midhip,sampling_freq=60.0,max_lag=30):
        """
        gait_cycles: [[ic, ic_opp, to, ic_end], ...]
        pa_midhip, pt_midhip: (N,3) [mm]
        """
        max_lag_frames = int(max_lag)

        hip_dist_cycle = []
        hip_cc_x_cycle = []
        hip_cc_y_cycle = []
        hip_cc_z_cycle = []
        hip_cc_lag_cycle = []

        for ic, _, _, ic_end in gait_cycles:
            if ic_end <= ic:
                continue

            pa = pa_midhip[ic:ic_end]
            pt = pt_midhip[ic:ic_end]
            if len(pa) < 5:
                continue

            # --- hip_dist (m) ---
            dist = np.linalg.norm(pa - pt, axis=1) / 1000.0
            hip_dist_cycle.append(np.nanmedian(dist))

            # --- hip_cc (x,y,z), lag ---
            hip_cc_x_cycle_, lag = _norm_xcorr_max(pa[:,0], pt[:,0], max_lag_frames)
            hip_cc_y_cycle_, _ = _norm_xcorr_max(pa[:,1], pt[:,1], max_lag_frames, def_lag=lag)
            hip_cc_z_cycle_, _ = _norm_xcorr_max(pa[:,2], pt[:,2], max_lag_frames, def_lag=lag)
            hip_cc_x_cycle.append(hip_cc_x_cycle_)
            hip_cc_y_cycle.append(hip_cc_y_cycle_)
            hip_cc_z_cycle.append(hip_cc_z_cycle_)
            hip_cc_lag_cycle.append(lag / sampling_freq)
            
            save_hip_x_fig = True  # デバッグ用にTrueにすると各周期のヒップ座標をプロット
            if save_hip_x_fig:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                frames = np.arange(len(pa))
                
                # X座標
                axes[0].plot(frames, pa[:, 0], label='PA', color='tab:red', linewidth=2)
                axes[0].plot(frames, pt[:, 0], label='PT', color='tab:blue', linewidth=2)
                axes[0].set_xlabel('Frame')
                axes[0].set_ylabel('X [mm]')
                axes[0].set_title(f'Hip X')
                axes[0].legend()
                axes[0].grid(True)
                
                # Y座標
                axes[1].plot(frames, pa[:, 1], label='PA', color='tab:red', linewidth=2)
                axes[1].plot(frames, pt[:, 1], label='PT', color='tab:blue', linewidth=2)
                axes[1].set_xlabel('Frame')
                axes[1].set_ylabel('Y [mm]')
                axes[1].set_title('Hip Y')
                axes[1].legend()
                axes[1].grid(True)
                
                # Z座標
                axes[2].plot(frames, pa[:, 2], label='PA', color='tab:red', linewidth=2)
                axes[2].plot(frames, pt[:, 2], label='PT', color='tab:blue', linewidth=2)
                axes[2].set_xlabel('Frame')
                axes[2].set_ylabel('Z [mm]')
                axes[2].set_title('Hip Z')
                axes[2].legend()
                axes[2].grid(True)
                
                cycle_idx = len(hip_dist_cycle)
                plt.suptitle(f'MidHip Trajectory (Cycle {cycle_idx}): PA vs PT')
                plt.tight_layout()
                
                # 保存先（out_root はこの関数のスコープ外なので thera_dir から構築）
                debug_dir = thera_dir / "ViTPose_results"
                plt.savefig(debug_dir / f"hip_xyz_cycle_{cycle_idx:02d}.png", dpi=150)
                plt.close()
                
        return dict(
            hip_dist=np.nanmean(hip_dist_cycle),
            hip_cc_x=np.nanmean(hip_cc_x_cycle),
            hip_cc_y=np.nanmean(hip_cc_y_cycle),
            hip_cc_z=np.nanmean(hip_cc_z_cycle),
            hip_cc_lag=np.nanmean(hip_cc_lag_cycle),
        )
        
    if paralyzed_side == "R":
        gait_cycles = gait_cycles_r
    elif paralyzed_side == "L":
        gait_cycles = gait_cycles_l
        
    assist_metrics = calculate_pt_assist_metrics_by_cycle(
    gait_cycles=gait_cycles,
    pa_midhip=kp3d[valid_start:valid_end+1, 8, :],
    pt_midhip=kp3d_pt[valid_start:valid_end+1, 8, :],
    sampling_freq=FS_3D,)

    pd.DataFrame([assist_metrics]).to_csv(
        out_root / f"pt_assist_metrics.csv",
        index=False
    )
    
    print(f"[OK] {thera_dir.name} / {npz_3d_path.name}")
    print(f"     -> {out_root}")

def main():
    for sub_dir in sorted(ROOT_DIR.glob("sub*")):
        if not sub_dir.exists():
            print(f"[SKIP] missing: {sub_dir}")
            continue

        for thera_dir in sorted(sub_dir.glob("thera*-0")):
            if not thera_dir.exists():
                print(f"[SKIP] missing: {thera_dir}")
                continue
            
            # 3_3の3D npz（複数methodがある前提で全部回す）
            npz_pa = list(thera_dir.glob("3d_kp_*_PA.npz"))
            npz_list = sorted(npz_pa, key=lambda p: natural_sort_key(p.name))
            if len(npz_list) == 0:
                print(f"[SKIP] 3D npz not found (run 3_3 first): {thera_dir}")
                continue

            print("=" * 90)
            print(f"[RUN] {thera_dir}  (methods={len(npz_list)})")
            print("=" * 90)

            for npz_3d in npz_list:
                try:
                    process_one_method(thera_dir, npz_3d)
                except Exception as e:
                    print(f"[ERROR] {thera_dir.name} / {npz_3d.name}: {e}")


if __name__ == "__main__":
    main()