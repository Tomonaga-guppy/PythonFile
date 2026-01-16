"""
4_angle_timeseries_compare_mocap_openposeSeg_vitpose.py
======================================================
4_1 (mocap) と 4_2 (Pose3D) の出力を使って、
Mocap / openpose_seg / ViTPose の関節角度を「フレーム時系列」で重ね描きする。

前提（あなたの既存フロー）:
- 4_1 が thera*/mocap/ に以下を作成済み:
    - _intermediate_ref_opti_openpose/*_opti_intermediate.npz
        start_frame, end_frame など
    - angle_100Hz_<元mocapCSV名>.csv  （関節角度: R/L_Hip_FlEx 等）
- 4_2 が thera*/ 直下に 3_3 出力の 3D npz を作成済み:
    - 3d_kp_openpose_seg_*.npz
    - 3d_kp_ViTPose_*.npz
- 同期:
    - thera*/IMU/*SYNC*.csv の ext data から LED(2列目) と Mocap(3列目) の立下り/立上りで同期
    - thera*/gopro/trimming_info.json の start_frame_relative で GoPro カット分を補正

出力:
theraX-0/
  Pose3D_results/
    _compare_with_mocap/
      _timeseries_openpose_seg_ViTPose/
        compare_timeseries_FlEx_R_<tag>.png
        compare_timeseries_FlEx_L_<tag>.png
        compare_timeseries_Hip_AdAb_R_<tag>.png
        compare_timeseries_Hip_AdAb_L_<tag>.png
        compare_timeseries_Hip_InEx_R_<tag>.png
        compare_timeseries_Hip_InEx_L_<tag>.png
        _summary_<tag>.txt

注意:
- ここでは「描画と比較用の補間」だけ行い、3Dやmocapの計算結果自体は一切変更しない。
"""

import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import m_openpose as op  # culc_angle_all_frames


# =========================
# 設定（ここだけ調整）
# =========================
ROOT_DIR = Path(r"G:\gait_pattern\BR9G_shuron")

SUB_RANGE = range(5, 6)  # sub5

# 3D npz を使う優先順位（3_3出力のキー）
PREFERRED_3D_KEY_ORDER = ["butter", "spline", "conf_filt", "raw"]

# sampling
FS_3D = 60.0
FS_MOCAP = 100.0

INTERMEDIATE_DIRNAME = "_intermediate_ref_opti_openpose"

# 対象method（3d_kp_<method>_...）
METHOD_OP3D = "openpose_seg"
METHOD_VIT3D = "ViTPose"

# 角度Yレンジ（あなたの 4_1 / 4_2 と同系）
YLIM_BY_ANGLE = {
    "Hip_FlEx":  (-40,  50),
    "Hip_AdAb":  (-20,  20),
    "Hip_InEx":  (-20,  20),
    "Knee_FlEx": (-10,  70),
    "Ankle_PlDo": (-30,  50),
}


# =========================
# 小物
# =========================
def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _intermediate_dir(mocap_dir: Path) -> Path:
    d = mocap_dir / INTERMEDIATE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_opti_intermediate_npz(mocap_dir: Path) -> Path | None:
    inter_dir = _intermediate_dir(mocap_dir)
    cands = sorted(inter_dir.glob("*_opti_intermediate.npz"), key=lambda p: natural_sort_key(p.name))
    return cands[0] if cands else None


def pick_3d_array(npz, preferred_keys):
    for k in preferred_keys:
        if k in npz.files:
            arr = npz[k]
            if isinstance(arr, np.ndarray) and arr.size == 0:
                continue
            return k, arr
    raise KeyError(f"no usable 3d key found. available={npz.files}")


def ylims_for_angle_key(k: str):
    # 例: 'R_Hip_FlEx' -> 'Hip_FlEx'
    if "_" not in k:
        return None
    base = k.split("_", 1)[1]
    return YLIM_BY_ANGLE.get(base, None)


def safe_interp_nan(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if np.all(~np.isfinite(y)):
        return y
    s = pd.Series(y)
    return s.interpolate(limit_direction="both").to_numpy()


def mocap_to_pose_frames(mocap_y_rel: np.ndarray,
                         start_frame_abs_100hz: int,
                         sync_offset_100hz: float,
                         n_pose_frames: int) -> np.ndarray:
    """
    mocap_y_rel: angle series indexed by "relative mocap frame" (0..N-1) where abs = start_frame_abs + rel
    sync_offset_100hz: 4_2と同じ定義（hz100_abs_to_hz60_idxで使う offset）
    変換:
        idx60 = (abs + offset) * (FS_3D/FS_MOCAP)
        => abs = idx60*(FS_MOCAP/FS_3D) - offset
        => rel = abs - start_frame_abs
    """
    y_old = safe_interp_nan(mocap_y_rel)
    x_old = np.arange(len(y_old), dtype=float)

    idx60 = np.arange(n_pose_frames, dtype=float)
    abs_100 = idx60 * (FS_MOCAP / FS_3D) - float(sync_offset_100hz)
    rel = abs_100 - float(start_frame_abs_100hz)

    y_new = np.full(n_pose_frames, np.nan, dtype=float)
    m = (rel >= 0) & (rel <= (len(y_old) - 1))
    if np.any(m):
        y_new[m] = np.interp(rel[m], x_old, y_old)
    return y_new


def compute_angles_from_body25(kp3d: np.ndarray):
    """
    kp3d: (n_frames, 25, 3)
    returns dict of angle series (deg)
    ※ 4_2 と同じ計算（m_openpose.culc_angle_all_frames）を使う
    """
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

    pel_up = neck - midhip
    thigh_r = rknee - rhip
    shank_r = rankle - rknee
    foot_r = rtoe - rhee

    thigh_l = lknee - lhip
    shank_l = lankle - lknee
    foot_l = ltoe - lhee

    n_axis = rhip - lhip  # 左→右

    rhip_flex = op.culc_angle_all_frames(pel_up, thigh_r, n_axis, degrees=True, angle_type="hip")
    lhip_flex = op.culc_angle_all_frames(pel_up, thigh_l, n_axis, degrees=True, angle_type="hip")

    rknee_flex = op.culc_angle_all_frames(thigh_r, shank_r, n_axis, degrees=True, angle_type="knee")
    lknee_flex = op.culc_angle_all_frames(thigh_l, shank_l, n_axis, degrees=True, angle_type="knee")

    rankle_pldo = op.culc_angle_all_frames(shank_r, foot_r, n_axis, degrees=True, angle_type="ankle")
    lankle_pldo = op.culc_angle_all_frames(shank_l, foot_l, n_axis, degrees=True, angle_type="ankle")

    n_axis_adab = np.cross(pel_up, n_axis)
    eps = 1e-9
    n_norm = np.linalg.norm(n_axis_adab, axis=1)
    bad = n_norm < eps
    if np.any(bad):
        n_axis_adab[bad, :] = np.array([0.0, 0.0, 1.0])

    rhip_adab = op.culc_angle_all_frames(pel_up, thigh_r, n_axis_adab, degrees=True, angle_type="hip_adab")
    lhip_adab = op.culc_angle_all_frames(pel_up, thigh_l, n_axis_adab, degrees=True, angle_type="hip_adab")

    rhip_inex = op.culc_angle_all_frames(thigh_r, n_axis_adab, pel_up, degrees=True, angle_type="hip_inex")
    lhip_inex = op.culc_angle_all_frames(thigh_l, n_axis_adab, pel_up, degrees=True, angle_type="hip_inex")

    return dict(
        R_Hip_FlEx=np.asarray(rhip_flex, dtype=float),
        R_Knee_FlEx=np.asarray(rknee_flex, dtype=float),
        R_Ankle_PlDo=np.asarray(rankle_pldo, dtype=float),
        R_Hip_AdAb=np.asarray(rhip_adab, dtype=float),
        R_Hip_InEx=np.asarray(rhip_inex, dtype=float),
        L_Hip_FlEx=np.asarray(lhip_flex, dtype=float),
        L_Knee_FlEx=np.asarray(lknee_flex, dtype=float),
        L_Ankle_PlDo=np.asarray(lankle_pldo, dtype=float),
        L_Hip_AdAb=np.asarray(lhip_adab, dtype=float),
        L_Hip_InEx=np.asarray(lhip_inex, dtype=float),
    )


def find_3d_npz(thera_dir: Path, method: str) -> Path | None:
    cands = sorted(thera_dir.glob(f"3d_kp_{method}_*.npz"), key=lambda p: natural_sort_key(p.name))
    return cands[0] if cands else None


def get_tag_from_npz(npz_path: Path, method: str) -> str:
    stem = npz_path.stem
    prefix = f"3d_kp_{method}_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    # fallback
    return stem.replace("3d_kp_", "")


def load_mocap_angles_csv(mocap_dir: Path, mid_npz: Path) -> Path | None:
    base = mid_npz.stem.replace("_opti_intermediate", "")
    exact = mocap_dir / f"angle_100Hz_{base}.csv"
    if exact.exists():
        return exact
    # fallback: first match
    cands = sorted(mocap_dir.glob("angle_100Hz_*.csv"), key=lambda p: natural_sort_key(p.name))
    return cands[0] if cands else None


def compute_sync_offset_100hz(thera_dir: Path) -> float | None:
    """
    4_2 と同様に、IMU ext data の変化点から同期差分(100Hz)を求め、
    trimming_info.json の start_frame_relative で補正して返す。
    """
    imu_dir = thera_dir / "IMU"
    sync_csv = None
    if imu_dir.exists():
        sync_csv = next(imu_dir.glob("*SYNC*.csv"), None)
    if sync_csv is None or (not sync_csv.exists()):
        return None

    df = pd.read_csv(sync_csv, header=None)
    ext_df = df[df[0] == "ext data"].reset_index(drop=True)
    if len(ext_df) == 0:
        return None

    ext_df[2] = ext_df[2].astype(int)
    ext_df[3] = ext_df[3].astype(int)

    col2_fall = ext_df.index[(ext_df[2].shift(1) == 1) & (ext_df[2] == 0)]
    col3_rise = ext_df.index[(ext_df[3].shift(1) == 0) & (ext_df[3] == 1)]
    if len(col2_fall) == 0 or len(col3_rise) == 0:
        return None

    sync_frame_diff_100hz = float(col3_rise[0] - col2_fall[0])

    # GoPro trimming 補正
    trim_json = thera_dir / "gopro" / "trimming_info.json"
    if trim_json.exists():
        with open(trim_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        gopro_cut = float(data["trimming_settings"]["start_frame_relative"])
        sync_frame_diff_100hz -= gopro_cut * (FS_MOCAP / FS_3D)

    return sync_frame_diff_100hz


# =========================
# プロット
# =========================
def plot_three_rows(title: str,
                    out_path: Path,
                    frames: np.ndarray,
                    series_dict: dict,
                    keys: list[str]):
    """
    keys: 例 ["R_Hip_FlEx","R_Knee_FlEx","R_Ankle_PlDo"]
    series_dict[k] = {"Mocap": y1, "openpose_seg": y2, "ViTPose": y3}
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, k in zip(axes, keys):
        if k not in series_dict:
            ax.set_title(f"{k} (missing)")
            ax.grid(True, alpha=0.3)
            continue

        for label, y in series_dict[k].items():
            ax.plot(frames, y, label=label, linewidth=1.2)

        ax.set_title(k)
        ax.set_ylabel("Angle [deg]")

        ylims = ylims_for_angle_key(k)
        if ylims is not None:
            ax.set_ylim(*ylims)

        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Frame (Pose3D 60Hz index)")
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_single(title: str,
                out_path: Path,
                frames: np.ndarray,
                series: dict,
                angle_key: str):
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    for label, y in series.items():
        ax.plot(frames, y, label=label, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Frame (Pose3D 60Hz index)")
    ax.set_ylabel("Angle [deg]")

    ylims = ylims_for_angle_key(angle_key)
    if ylims is not None:
        ax.set_ylim(*ylims)

    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =========================
# 本体
# =========================
def process_one_trial(thera_dir: Path):
    mocap_dir = thera_dir / "mocap"
    if not mocap_dir.exists():
        print(f"[SKIP] no mocap dir: {thera_dir}")
        return

    mid_npz = find_opti_intermediate_npz(mocap_dir)
    if mid_npz is None or (not mid_npz.exists()):
        print(f"[SKIP] no intermediate npz (run 4_1): {thera_dir}")
        return

    # mocap angles csv
    angle_csv = load_mocap_angles_csv(mocap_dir, mid_npz)
    if angle_csv is None or (not angle_csv.exists()):
        print(f"[SKIP] no mocap angle csv (run 4_1): {thera_dir}")
        return

    # 3D npz
    npz_op = find_3d_npz(thera_dir, METHOD_OP3D)
    npz_vit = find_3d_npz(thera_dir, METHOD_VIT3D)
    if npz_op is None or npz_vit is None:
        print(f"[SKIP] missing 3d npz (run 3_3/4_2): {thera_dir}")
        return

    tag = get_tag_from_npz(npz_vit, METHOD_VIT3D)  # 代表tag（例: pa_spline）
    out_dir = thera_dir / "Pose3D_results" / "_compare_with_mocap" / "_timeseries_openpose_seg_ViTPose"
    out_dir.mkdir(parents=True, exist_ok=True)

    # load intermediate (start_frame)
    mid = np.load(mid_npz, allow_pickle=True)
    start_frame = int(mid["start_frame"]) if "start_frame" in mid.files else 0

    # sync offset
    sync_offset = compute_sync_offset_100hz(thera_dir)
    if sync_offset is None:
        print(f"[SKIP] sync offset not found (need IMU SYNC): {thera_dir}")
        return

    # load mocap angle df (index_col=0 to drop unnamed index column)
    mocap_df = pd.read_csv(angle_csv, index_col=0)
    # 欲しい角度列
    angle_keys = [
        "R_Hip_FlEx", "R_Knee_FlEx", "R_Ankle_PlDo", "R_Hip_AdAb", "R_Hip_InEx",
        "L_Hip_FlEx", "L_Knee_FlEx", "L_Ankle_PlDo", "L_Hip_AdAb", "L_Hip_InEx",
    ]
    for k in angle_keys:
        if k not in mocap_df.columns:
            # 欠けていても続行（plotでmissing扱い）
            pass

    # load pose3d angles (openpose_seg)
    npzA = np.load(npz_op, allow_pickle=True)
    used_key_A, kpA = pick_3d_array(npzA, PREFERRED_3D_KEY_ORDER)
    angA = compute_angles_from_body25(kpA)

    # vitpose
    npzB = np.load(npz_vit, allow_pickle=True)
    used_key_B, kpB = pick_3d_array(npzB, PREFERRED_3D_KEY_ORDER)
    angB = compute_angles_from_body25(kpB)

    n_pose = min(len(next(iter(angA.values()))), len(next(iter(angB.values()))))
    frames = np.arange(n_pose, dtype=int)

    # mocap -> pose frame mapping
    series_dict = {}
    for k in angle_keys:
        mocap_y = mocap_df[k].to_numpy(dtype=float) if k in mocap_df.columns else np.full(len(mocap_df), np.nan)
        mocap_on_pose = mocap_to_pose_frames(
            mocap_y_rel=mocap_y,
            start_frame_abs_100hz=start_frame,
            sync_offset_100hz=sync_offset,
            n_pose_frames=n_pose
        )

        yA = angA.get(k, np.full(n_pose, np.nan))[:n_pose]
        yB = angB.get(k, np.full(n_pose, np.nan))[:n_pose]
        series_dict[k] = {
            "Mocap": mocap_on_pose,
            METHOD_OP3D: yA,
            METHOD_VIT3D: yB,
        }

    # ---- plots ----
    plot_three_rows(
        title=f"FlEx Right (Mocap vs {METHOD_OP3D} vs {METHOD_VIT3D}) tag={tag}",
        out_path=out_dir / f"compare_timeseries_FlEx_R_{tag}.png",
        frames=frames,
        series_dict=series_dict,
        keys=["R_Hip_FlEx", "R_Knee_FlEx", "R_Ankle_PlDo"],
    )
    plot_three_rows(
        title=f"FlEx Left (Mocap vs {METHOD_OP3D} vs {METHOD_VIT3D}) tag={tag}",
        out_path=out_dir / f"compare_timeseries_FlEx_L_{tag}.png",
        frames=frames,
        series_dict=series_dict,
        keys=["L_Hip_FlEx", "L_Knee_FlEx", "L_Ankle_PlDo"],
    )

    plot_single(
        title=f"Hip Ad/Ab Right (Mocap vs {METHOD_OP3D} vs {METHOD_VIT3D}) tag={tag}",
        out_path=out_dir / f"compare_timeseries_Hip_AdAb_R_{tag}.png",
        frames=frames,
        series=series_dict.get("R_Hip_AdAb", {}),
        angle_key="R_Hip_AdAb",
    )
    plot_single(
        title=f"Hip Ad/Ab Left (Mocap vs {METHOD_OP3D} vs {METHOD_VIT3D}) tag={tag}",
        out_path=out_dir / f"compare_timeseries_Hip_AdAb_L_{tag}.png",
        frames=frames,
        series=series_dict.get("L_Hip_AdAb", {}),
        angle_key="L_Hip_AdAb",
    )

    plot_single(
        title=f"Hip In/Ex Right (Mocap vs {METHOD_OP3D} vs {METHOD_VIT3D}) tag={tag}",
        out_path=out_dir / f"compare_timeseries_Hip_InEx_R_{tag}.png",
        frames=frames,
        series=series_dict.get("R_Hip_InEx", {}),
        angle_key="R_Hip_InEx",
    )
    plot_single(
        title=f"Hip In/Ex Left (Mocap vs {METHOD_OP3D} vs {METHOD_VIT3D}) tag={tag}",
        out_path=out_dir / f"compare_timeseries_Hip_InEx_L_{tag}.png",
        frames=frames,
        series=series_dict.get("L_Hip_InEx", {}),
        angle_key="L_Hip_InEx",
    )

    # ---- summary ----
    summary = []
    summary.append(f"thera_dir: {thera_dir}")
    summary.append(f"mocap intermediate: {mid_npz}")
    summary.append(f"mocap angles csv: {angle_csv}")
    summary.append(f"start_frame(abs 100Hz): {start_frame}")
    summary.append(f"sync_offset_100hz(after trimming): {sync_offset}")
    summary.append(f"pose3d openpose_seg npz: {npz_op} (used_key={used_key_A})")
    summary.append(f"pose3d vitpose npz: {npz_vit} (used_key={used_key_B})")
    summary.append(f"n_pose_frames_used: {n_pose}")
    (out_dir / f"_summary_{tag}.txt").write_text("\n".join(summary), encoding="utf-8")

    print(f"[OK] {thera_dir.name} -> {out_dir}")


def main():
    for sub_i in SUB_RANGE:
        thera_dir = ROOT_DIR / f"sub{sub_i}" / f"thera{sub_i}-0"
        if not thera_dir.exists():
            continue
        print("=" * 90)
        print(f"[RUN] {thera_dir}")
        print("=" * 90)
        try:
            process_one_trial(thera_dir)
        except Exception as e:
            print(f"[ERROR] {thera_dir.name}: {e}")


if __name__ == "__main__":
    main()
