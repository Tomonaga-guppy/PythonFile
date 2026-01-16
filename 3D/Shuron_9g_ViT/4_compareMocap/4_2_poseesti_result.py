"""
目的:
- 3_3_reconstruct3d.py が出力した 3D姿勢(OpenPose/ViTPose等) の解析のみを行う（= Mocap解析はしない）
- 4_1.py が保存した Mocap中間ファイル（_intermediate_ref_opti_openpose/*.npz）を参照し、
  「タイミング合わせ（100Hz<->60Hz）」と「角度・歩行周期の正規化・平均・誤差指標」を作る。

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
ROOT_DIR = Path(r"G:\gait_pattern\BR9G_shuron")

# 解析対象: sub1~sub10, thera1-0~thera10-0 を総当たり（必要なら範囲変更）
SUB_RANGE = range(1, 7)

# 3_3出力npzを使う優先順位（最初に見つかったものを使う）
PREFERRED_3D_KEY_ORDER = ["butter", "spline", "conf_filt", "raw"]

# 3Dのサンプリング周波数（3_3のFRAME_RATE=60前提）
FS_3D = 60.0
FS_MOCAP = 100.0

# 正規化歩行周期 0..100 [%] を何点で表すか（101点=0..100の整数%）
NORM_POINTS = 101

INTERMEDIATE_DIRNAME = "_intermediate_ref_opti_openpose"

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


def _intermediate_dir(base_dir: Path) -> Path:
    d = base_dir / INTERMEDIATE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_opti_intermediate_npz(mocap_dir: Path) -> Path | None:
    inter_dir = _intermediate_dir(mocap_dir)
    cands = sorted(inter_dir.glob("*_opti_intermediate.npz"), key=lambda p: natural_sort_key(p.name))
    return cands[0] if cands else None


def pick_3d_array(npz, preferred_keys):
    # 3_3: butterが無効のとき np.array([]) を入れていることがある
    for k in preferred_keys:
        if k in npz.files:
            arr = npz[k]
            if isinstance(arr, np.ndarray) and arr.size == 0:
                continue
            return k, arr
    raise KeyError(f"no usable 3d key found. available={npz.files}")


def first_crossing_index(z_series: np.ndarray, threshold: float) -> int:
    """
    z_series が threshold を初めて超えた index。無ければ -1。
    """
    if z_series is None or len(z_series) == 0:
        return -1
    m = np.isfinite(z_series)
    if not np.any(m):
        return -1
    idx = np.where(m & (z_series > threshold))[0]
    return int(idx[0]) if len(idx) else -1


def hz100_abs_to_hz60_idx(frame_100_abs: float, mc_frame_offset_100hz: float) -> float:
    """
    100Hz absolute frame を 60Hz index（3D npzの配列index）へ変換
    - mc_frame_offset_100hz は「100Hzの原点（3D index=0 に相当する時刻）」のabsoluteフレーム
    - t = (frame_100_abs - mc_frame_offset_100hz)/100
    - idx60 = t*60 = (frame_100_abs - mc_frame_offset_100hz)*0.6
    """
    return (frame_100_abs + mc_frame_offset_100hz) * (FS_3D / FS_MOCAP)  # 0.6


def clamp_cycle_indices(cycle, n_frames):
    """
    cycle = [ic, ic_opp, to, ic_end] (float index)
    0..n_frames-1 に丸め、単調増加が破綻するものは None
    """
    ic, ic_opp, to, ic_end = [int(np.round(x)) for x in cycle]
    ic = max(0, min(n_frames - 1, ic))
    ic_opp = max(0, min(n_frames - 1, ic_opp))
    to = max(0, min(n_frames - 1, to))
    ic_end = max(0, min(n_frames - 1, ic_end))
    if not (ic < ic_opp < to < ic_end):
        return None
    return [ic, ic_opp, to, ic_end]


def normalize_cycle(angle_series: np.ndarray, ic: int, to: int, ic_end: int, n_points=101):
    """
    角度時系列 angle_series から [ic..ic_end] を 0..100% に線形補間して返す。
    """
    if ic_end <= ic:
        return None
    orig_frames = np.arange(ic, ic_end + 1)
    x_new = np.linspace(0, 100, n_points)
    x_old = np.linspace(0, 100, len(orig_frames))
    y_old = angle_series[orig_frames]
    # NaNを含む場合は補間のために埋める（解析用、3D自体は変更しない）
    if np.any(~np.isfinite(y_old)):
        s = pd.Series(y_old).interpolate(limit_direction="both").to_numpy()
        y_old = s
    y_new = np.interp(x_new, x_old, y_old)
    stance_pct = ((to - ic) / (ic_end - ic)) * 100.0
    return x_new, y_new, stance_pct


def mean_std_cycles(cycles_ys: list[np.ndarray]):
    arr = np.array(cycles_ys, dtype=float)  # (n_cycles, n_points)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    return float(np.mean(np.abs(a[m] - b[m])))


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
        midhip=midhip,
        rhee=rhee,
        lhee=lhee,
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
def calculate_gait_parameters_3d(gait_cycles, midhip, rhee, lhee, side, sampling_freq=60.0):
    """
    4_1の mocap計算と同型の項目を 3D(60Hz) から作る
    gait_cycles: [[ic, ic_opp, to, ic_end], ...] (int index)
    """
    gait_params = []
    for cycle_idx, (ic, ic_opp, to, ic_end) in enumerate(gait_cycles):
        if ic_end <= ic:
            continue

        cycle_duration = (ic_end - ic) / sampling_freq
        stance_duration = (to - ic) / sampling_freq
        swing_duration = (ic_end - to) / sampling_freq

        hip_disp = np.linalg.norm(midhip[ic_end] - midhip[ic]) / 1000.0  # mm->m（3_3はmm）
        gait_speed = hip_disp / cycle_duration if cycle_duration > 0 else np.nan

        # stride length（踵）
        if side == "R":
            stride_length = np.linalg.norm(rhee[ic_end] - rhee[ic]) / 1000.0
            v_stride = (rhee[ic_end] - rhee[ic]) / 1000.0
            v_step = (lhee[ic_opp] - rhee[ic]) / 1000.0
            step_length_opposite = np.linalg.norm(lhee[ic_opp] - rhee[ic]) / 1000.0
        else:
            stride_length = np.linalg.norm(lhee[ic_end] - lhee[ic]) / 1000.0
            v_stride = (lhee[ic_end] - lhee[ic]) / 1000.0
            v_step = (rhee[ic_opp] - lhee[ic]) / 1000.0
            step_length_opposite = np.linalg.norm(rhee[ic_opp] - lhee[ic]) / 1000.0

        # step width（ベクトル外積の距離）
        denom = np.linalg.norm(v_stride)
        if denom < 1e-9:
            step_width = np.nan
        else:
            step_width = abs(np.linalg.norm(np.cross(v_stride, v_step))) / denom

        gait_params.append(
            dict(
                cycle_index=cycle_idx,
                cycle_duration=cycle_duration,
                stance_duration=stance_duration,
                swing_duration=swing_duration,
                stance_phase_percent=(stance_duration / cycle_duration) * 100 if cycle_duration > 0 else np.nan,
                swing_phase_percent=(swing_duration / cycle_duration) * 100 if cycle_duration > 0 else np.nan,
                gait_speed=gait_speed,
                stride_length=stride_length,
                step_width=step_width,
                step_length_opposite=step_length_opposite,
            )
        )
    return gait_params


# =========================
# 本体（1 trial x 1 method）
# =========================
def process_one_method(thera_dir: Path, mocap_intermediate: Path, npz_3d_path: Path):
    # ---------- load mocap intermediate ----------
    mid = np.load(mocap_intermediate, allow_pickle=True)
    gait_cycles_r_abs = mid["gait_cycles_r_abs"].astype(float)  # (n,4)
    gait_cycles_l_abs = mid["gait_cycles_l_abs"].astype(float)

    mean_cycle_r_cols = list(mid["mean_cycle_r_cols"].tolist())
    mean_cycle_l_cols = list(mid["mean_cycle_l_cols"].tolist())
    mean_cycle_r_mocap = pd.DataFrame(mid["mean_cycle_r"], columns=mean_cycle_r_cols)
    mean_cycle_l_mocap = pd.DataFrame(mid["mean_cycle_l"], columns=mean_cycle_l_cols)

    # ---------- load 3D npz ----------
    npz = np.load(npz_3d_path, allow_pickle=True)
    used_key, kp3d = pick_3d_array(npz, PREFERRED_3D_KEY_ORDER)
    
    angles_dict = compute_angles_from_body25(kp3d)

    # kp3d: (n_frames, 25, 3)
    n_frames = kp3d.shape[0]
    method_tag = npz_3d_path.stem  # 3d_kp_<method>_<csv_tag>
    # method名を雑に切り出し
    # 例: 3d_kp_ViTPose_openpose_facemasked_pa_spline など
    m = re.match(r"3d_kp_(.+)$", method_tag)
    method_tag = m.group(1) if m else method_tag

    # ---------- timing alignment ----------
    # 同期用IMuを読み込んでmocapとgoproのタイミング合わせを行う
    sync_imu_dir = thera_dir / "IMU"
    sync_imu = None
    if sync_imu_dir.exists():
        # 同期用IMUファイルの取得
        sync_imu = next(sync_imu_dir.glob("*SYNC*.csv"), None)

    if sync_imu is None:
        print(f"[WARN] no sync IMU found: {thera_dir.name} / {npz_3d_path.name}")
        return
    sync_imu_df = pd.read_csv(sync_imu, sep=",", header=None)
    # ext data 行のみ抽出
    ext_df = sync_imu_df[sync_imu_df[0] == "ext data"].reset_index(drop=True)

    # 念のため int 化
    ext_df[2] = ext_df[2].astype(int)
    ext_df[3] = ext_df[3].astype(int)

    # 変化点検出
    # 2列目 LED: 1 → 0
    col2_fall = ext_df.index[
        (ext_df[2].shift(1) == 1) & (ext_df[2] == 0)
    ]
    # 3列目 Mocap: 0 → 1
    col3_rise = ext_df.index[
        (ext_df[3].shift(1) == 0) & (ext_df[3] == 1)
    ]

    # フレームずれ量の計算
    if len(col2_fall) > 0 and len(col3_rise) > 0:
        # 最初のイベント同士で比較（一般的）
        sync_frame_diff_100hz = (col3_rise[0] - col2_fall[0])
        print(f"Frame difference (col3 - col2): {sync_frame_diff_100hz}")
    else:
        print("変化点が検出できませんでした")
        
    gopro_trimming_json = thera_dir / "gopro" / "trimming_info.json"
    with open(str(gopro_trimming_json), "r", encoding="utf-8") as f:
        data = json.load(f)

    gopro_cut_frame = data["trimming_settings"]["start_frame_relative"]
    print(f"GoPro trimming reference frame: {gopro_cut_frame}")
    sync_frame_diff_100hz -= gopro_cut_frame * (FS_MOCAP / FS_3D)  # ゴープロカット分を引く
    print(f"Adjusted frame difference after GoPro trimming: {sync_frame_diff_100hz}")

    # ---------- convert gait cycles to 60Hz indices ----------
    gait_cycles_r_60 = []
    gait_cycles_l_60 = []

    for (ic, ic_opp, to, ic_end) in gait_cycles_r_abs:
        cyc60 = [
            hz100_abs_to_hz60_idx(ic, sync_frame_diff_100hz),
            hz100_abs_to_hz60_idx(ic_opp, sync_frame_diff_100hz),
            hz100_abs_to_hz60_idx(to, sync_frame_diff_100hz),
            hz100_abs_to_hz60_idx(ic_end, sync_frame_diff_100hz),
        ]
        cc = clamp_cycle_indices(cyc60, n_frames)
        if cc is not None:
            gait_cycles_r_60.append(cc)

    for (ic, ic_opp, to, ic_end) in gait_cycles_l_abs:
        cyc60 = [
            hz100_abs_to_hz60_idx(ic, sync_frame_diff_100hz),
            hz100_abs_to_hz60_idx(ic_opp, sync_frame_diff_100hz),
            hz100_abs_to_hz60_idx(to, sync_frame_diff_100hz),
            hz100_abs_to_hz60_idx(ic_end, sync_frame_diff_100hz),
        ]
        cc = clamp_cycle_indices(cyc60, n_frames)
        if cc is not None:
            gait_cycles_l_60.append(cc)

    if len(gait_cycles_r_60) == 0 and len(gait_cycles_l_60) == 0:
        print(f"[SKIP] no valid cycles after conversion: {thera_dir.name} / {npz_3d_path.name}")
        return

    # ---------- normalize cycles (angles) ----------
    # right
    norm_x = np.linspace(0, 100, NORM_POINTS)
    norm_cycles_r = []
    for cyc_idx, (ic, ic_opp, to, ic_end) in enumerate(gait_cycles_r_60):
        row = {"cycle_index": cyc_idx, "ic_start": ic, "to": to, "ic_end": ic_end, "percentage": norm_x}
        for key in ["R_Hip_FlEx", "R_Knee_FlEx", "R_Ankle_PlDo", "R_Hip_InEx", "R_Hip_AdAb"]:
            out = normalize_cycle(angles_dict[key], ic, to, ic_end, n_points=NORM_POINTS)
            if out is None:
                break
            _, y, stance_pct = out
            row[key] = y
            row["stance_phase_percentage"] = stance_pct
        if "R_Hip_FlEx" in row:
            norm_cycles_r.append(row)

    # left
    norm_cycles_l = []
    for cyc_idx, (ic, ic_opp, to, ic_end) in enumerate(gait_cycles_l_60):
        row = {"cycle_index": cyc_idx, "ic_start": ic, "to": to, "ic_end": ic_end, "percentage": norm_x}
        for key in ["L_Hip_FlEx", "L_Knee_FlEx", "L_Ankle_PlDo", "L_Hip_InEx", "L_Hip_AdAb"]:
            out = normalize_cycle(angles_dict[key], ic, to, ic_end, n_points=NORM_POINTS)
            if out is None:
                break
            _, y, stance_pct = out
            row[key] = y
            row["stance_phase_percentage"] = stance_pct
        if "L_Hip_FlEx" in row:
            norm_cycles_l.append(row)

    # ---------- output dirs ----------
    # ---------- output dirs ----------
    tag_short = npz_3d_path.stem.replace("3d_kp_", "")

    out_root = thera_dir / "Pose3D_results" / method_tag
    out_root.mkdir(parents=True, exist_ok=True)



    # ---------- mean/std cycles -> CSV ----------
    def save_mean_cycle(side: str, norm_cycles: list[dict], keys: list[str]):
        if len(norm_cycles) == 0:
            return None

        stance_mean = float(np.mean([c["stance_phase_percentage"] for c in norm_cycles]))
        df = pd.DataFrame({"Percentage": norm_x})
        for k in keys:
            ys = [c[k] for c in norm_cycles if k in c]
            if len(ys) == 0:
                df[f"{k}_mean"] = np.nan
                df[f"{k}_std"] = np.nan
            else:
                m, s = mean_std_cycles(ys)
                df[f"{k}_mean"] = m
                df[f"{k}_std"] = s

        df.attrs["stance_mean"] = stance_mean
        return df

    mean_r = save_mean_cycle("R", norm_cycles_r, ["R_Hip_FlEx", "R_Knee_FlEx", "R_Ankle_PlDo", "R_Hip_InEx", "R_Hip_AdAb"])
    mean_l = save_mean_cycle("L", norm_cycles_l, ["L_Hip_FlEx", "L_Knee_FlEx", "L_Ankle_PlDo", "L_Hip_InEx", "L_Hip_AdAb"])

    tag_short = npz_3d_path.stem.replace("3d_kp_", "")
    if mean_r is not None:
        mean_r.to_csv(out_root / f"normalized_cycle_R_mean_{tag_short}.csv", index=False)
    if mean_l is not None:
        mean_l.to_csv(out_root / f"normalized_cycle_L_mean_{tag_short}.csv", index=False)

    # ---------- gait parameters (3D) ----------
    midhip = angles_dict["midhip"]
    rhee = angles_dict["rhee"]
    lhee = angles_dict["lhee"]

    gait_params_r = calculate_gait_parameters_3d(gait_cycles_r_60, midhip, rhee, lhee, side="R", sampling_freq=FS_3D)
    gait_params_l = calculate_gait_parameters_3d(gait_cycles_l_60, midhip, rhee, lhee, side="L", sampling_freq=FS_3D)

    pd.DataFrame(gait_params_r).to_csv(out_root / f"gait_parameters_R_{tag_short}.csv", index=False)
    pd.DataFrame(gait_params_l).to_csv(out_root / f"gait_parameters_L_{tag_short}.csv", index=False)

    # ---------- compare with mocap mean cycles (error summary) ----------
    # mocap平均CSVには、4_1由来の列名が入っている（例: R_Hip_FlEx_mean 等）
    # ここでは「mean同士」を比較する。キーが無い場合はスキップ。
    errors = []
    def add_err(side, joint_key_base, mean_df_3d, mean_df_mocap):
        """
        joint_key_base: "R_Hip_FlEx" のようなベース
        compare: <base>_mean
        """
        col_m = f"{joint_key_base}_mean"
        if mean_df_3d is None or col_m not in mean_df_3d.columns:
            return
        if col_m not in mean_df_mocap.columns:
            return
        e = dict(
            joint=joint_key_base,
            mae=mae(mean_df_mocap[col_m].to_numpy(), mean_df_3d[col_m].to_numpy()),
            rmse=rmse(mean_df_mocap[col_m].to_numpy(), mean_df_3d[col_m].to_numpy()),
        )
        errors.append(e)

    if mean_r is not None:
        for k in ["R_Hip_FlEx", "R_Knee_FlEx", "R_Ankle_PlDo", "R_Hip_InEx", "R_Hip_AdAb"]:
            add_err("R", k, mean_r, mean_cycle_r_mocap)
    if mean_l is not None:
        for k in ["L_Hip_FlEx", "L_Knee_FlEx", "L_Ankle_PlDo", "L_Hip_InEx", "L_Hip_AdAb"]:
            add_err("L", k, mean_l, mean_cycle_l_mocap)

    err_df = pd.DataFrame(errors)
    err_df.to_csv(out_root / f"angle_error_summary_{tag_short}.csv", index=False)

    def plot_three_timeseries(title, keys, angles_dict, fname):
        """
        keys: ["R_Hip_FlEx","R_Knee_FlEx","R_Ankle_PlDo"] など
        3段：股→膝→足
        横軸：フレーム番号
        """
        n = len(angles_dict[keys[0]])
        frames = np.arange(n)

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

        for ax, k in zip(axes, keys):
            if k not in angles_dict:
                ax.set_title(f"{k} (missing)")
                ax.grid(True)
                continue

            y = np.asarray(angles_dict[k], dtype=float)
            ax.plot(frames, y, label=k)

            ax.set_ylabel("Angle [deg]")

            # 角度種別ごとのY範囲
            ylims = ylims_for_angle_key(k)
            if ylims is not None:
                ax.set_ylim(*ylims)

            ax.grid(True)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Frame")
        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(out_root / fname)
        plt.close()




    # ---------- plots（主要角度） ----------
    def plot_three(title, keys, mean_df, fname):
        if mean_df is None:
            return
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        stance = float(mean_df.attrs.get("stance_mean", 60.0))

        for ax, k in zip(axes, keys):
            mcol = f"{k}_mean"
            scol = f"{k}_std"

            if mcol not in mean_df.columns:
                ax.set_title(f"{k} (missing)")
                continue

            ax.plot(mean_df["Percentage"], mean_df[mcol], label="Mean")
            if scol in mean_df.columns:
                ax.fill_between(
                    mean_df["Percentage"],
                    mean_df[mcol] - mean_df[scol],
                    mean_df[mcol] + mean_df[scol],
                    alpha=0.3,
                )

            ax.axvline(x=stance, linestyle="--", label="Toe Off (mean)")
            ax.set_ylabel("Angle [deg]")

            ylims = ylims_for_angle_key(k)
            if ylims is not None:
                ax.set_ylim(*ylims)

            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Gait Cycle [%]")
        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(out_root / fname)
        plt.close()


    plot_three(
        f"Flex/Ext (Right) - {method_tag} ({used_key})",
        ["R_Hip_FlEx", "R_Knee_FlEx", "R_Ankle_PlDo"],
        mean_r,
        f"gait_cycle_FlEx_R_{tag_short}.png",
    )
    plot_three(
        f"Flex/Ext (Left) - {method_tag} ({used_key})",
        ["L_Hip_FlEx", "L_Knee_FlEx", "L_Ankle_PlDo"],
        mean_l,
        f"gait_cycle_FlEx_L_{tag_short}.png",
    )
    plot_three(
        f"In/Ex (Right) - {method_tag} ({used_key})",
        ["R_Hip_InEx", "R_Hip_InEx", "R_Hip_InEx"],  # 1軸だけだが3段に合わせる（見た目統一）
        mean_r,
        f"gait_cycle_InEx_R_{tag_short}.png",
    )
    plot_three(
        f"In/Ex (Left) - {method_tag} ({used_key})",
        ["L_Hip_InEx", "L_Hip_InEx", "L_Hip_InEx"],
        mean_l,
        f"gait_cycle_InEx_L_{tag_short}.png",
    )
    plot_three(
        f"Ad/Ab (Right) - {method_tag} ({used_key})",
        ["R_Hip_AdAb", "R_Hip_AdAb", "R_Hip_AdAb"],
        mean_r,
        f"gait_cycle_AdAb_R_{tag_short}.png",
    )
    plot_three(
        f"Ad/Ab (Left) - {method_tag} ({used_key})",
        ["L_Hip_AdAb", "L_Hip_AdAb", "L_Hip_AdAb"],
        mean_l,
        f"gait_cycle_AdAb_L_{tag_short}.png",
    )

    # ---------- timeseries plots（主要角度：3段レイアウト） ----------
    plot_three_timeseries(
        f"Timeseries Flex/Ext (Right) - {method_tag} ({used_key})",
        ["R_Hip_FlEx", "R_Knee_FlEx", "R_Ankle_PlDo"],
        angles_dict,
        f"timeseries_FlEx_R_{tag_short}.png",
    )
    plot_three_timeseries(
        f"Timeseries Flex/Ext (Left) - {method_tag} ({used_key})",
        ["L_Hip_FlEx", "L_Knee_FlEx", "L_Ankle_PlDo"],
        angles_dict,
        f"timeseries_FlEx_L_{tag_short}.png",
    )
    
    plot_three_timeseries(
        f"Timeseries Ad/Ab (Right) - {method_tag} ({used_key})",
        ["R_Hip_AdAb", "R_Knee_AdAb", "R_Ankle_AdAb"],
        angles_dict,
        f"timeseries_AdAb_R_{tag_short}.png",
    )
    plot_three_timeseries(
        f"Timeseries Ad/Ab (Left) - {method_tag} ({used_key})",
        ["L_Hip_AdAb", "L_Knee_AdAb", "L_Ankle_AdAb"],
        angles_dict,
        f"timeseries_AdAb_L_{tag_short}.png",
    )

    plot_three_timeseries(
        f"Timeseries In/Ex (Right) - {method_tag} ({used_key})",
        ["R_Hip_InEx", "R_Knee_InEx", "R_Ankle_InEx"],
        angles_dict,
        f"timeseries_InEx_R_{tag_short}.png",
    )
    plot_three_timeseries(
        f"Timeseries In/Ex (Left) - {method_tag} ({used_key})",
        ["L_Hip_InEx", "L_Knee_InEx", "L_Ankle_InEx"],
        angles_dict,
        f"timeseries_InEx_L_{tag_short}.png",
    )



    print(f"[OK] {thera_dir.name} / {npz_3d_path.name}")
    print(f"     -> {out_root}")


def main():
    for sub_i in SUB_RANGE:
        sub_dir = ROOT_DIR / f"sub{sub_i}"
        if not sub_dir.exists():
            print(f"[SKIP] missing: {sub_dir}")
            continue

        thera_dir = sub_dir / f"thera{sub_i}-0"
        mocap_dir = thera_dir / "mocap"
        if not mocap_dir.exists():
            print(f"[SKIP] missing mocap dir: {mocap_dir}")
            continue

        # 4_1の中間npz
        mid_npz = find_opti_intermediate_npz(mocap_dir)
        if mid_npz is None or not mid_npz.exists():
            print(f"[SKIP] mocap intermediate not found (run 4_1 first): {mocap_dir}")
            continue

        # 3_3の3D npz（複数methodがある前提で全部回す）
        npz_list = sorted(thera_dir.glob("3d_kp_*.npz"), key=lambda p: natural_sort_key(p.name))
        if len(npz_list) == 0:
            print(f"[SKIP] 3D npz not found (run 3_3 first): {thera_dir}")
            continue

        print("=" * 90)
        print(f"[RUN] {thera_dir}  (methods={len(npz_list)})")
        print(f"  mocap intermediate: {mid_npz.name}")
        print("=" * 90)

        for npz_3d in npz_list:
            try:
                process_one_method(thera_dir, mid_npz, npz_3d)
            except Exception as e:
                print(f"[ERROR] {thera_dir.name} / {npz_3d.name}: {e}")


if __name__ == "__main__":
    main()
