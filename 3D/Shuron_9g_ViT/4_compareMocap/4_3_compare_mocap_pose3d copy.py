"""
4_3_compare_mocap_pose3d.py
==========================
目的:
- 4_1_mocap_result.py（または rewrite版）が保存した Mocap中間ファイル
    mocap/_intermediate_ref_opti_openpose/*_opti_intermediate.npz
  から Mocap の正規化歩行周期平均（mean_cycle_r/l）とイベント（gait_cycles_*_abs）を読み込む。
- 4_2_poseesti_result.py が保存した Pose3D_results/<method>/ 配下の
    normalized_cycle_{R/L}_mean_<tag>.csv
    gait_parameters_{R/L}_<tag>.csv
  を読み込み、Mocap vs Pose3D（OpenPose / ViTPose 等）の比較を行う。

出力（例）:
theraX-0/
  Pose3D_results/
    _compare_with_mocap/
      angle_error_summary_<tag>.csv
      gait_param_comparison_stats_<tag>.csv
      gait_param_mae_<tag>.csv
      compare_cycle_FlEx_R_<tag>.png など

重要:
- 3_3 / 4_1 / 4_2 の結果ファイルは一切変更しない（読み込みのみ）
- 比較は「平均波形どうし」「歩行パラメータ（周期ごと）どうし」で行う

依存:
- numpy, pandas, matplotlib
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 設定（ここだけ調整）
# =========================
ROOT_DIR = Path(r"G:\gait_pattern\BR9G_shuron")

# 解析対象 sub
SUB_RANGE = range(5, 6)

# 正規化歩行周期の点数（4_1/4_2 と合わせる: 101点=0..100%）
NORM_POINTS = 101

INTERMEDIATE_DIRNAME = "_intermediate_ref_opti_openpose"

# 角度プロットのY範囲（関節ごと）
YLIM_BY_ANGLE = {
    # Hip
    "Hip_FlEx":  (-40,  50),
    "Hip_AdAb":  (-20,  20),
    "Hip_InEx":  (-20,  20),

    # Knee
    "Knee_FlEx": (-10,  70),

    # Ankle  ※系列名: Ankle_PlDo (背屈/底屈)
    "Ankle_PlDo": (-30,  50),
}

# 比較する角度系列（4_2 側に存在する範囲に合わせる）
ANGLE_BASE_KEYS = [
    "Hip_FlEx",
    "Knee_FlEx",
    "Ankle_PlDo",
    "Hip_InEx",
    "Hip_AdAb",
]

# 比較する歩行パラメータ（4_1/4_2 共通で出ている想定）
GAIT_PARAM_KEYS = [
    ("gait_speed", "Gait Speed [m/s]"),
    ("stride_length", "Stride Length [m]"),
    ("step_width", "Step Width [m]"),
]
# GAIT_PARAM_KEYS = [
#     ("gait_speed", "Gait Speed [m/s]"),
#     ("stride_length", "Stride Length [m]"),
#     ("step_width", "Step Width [m]"),
#     ("step_length_opposite", "Step Length Opposite [m]"),
#     ("stance_phase_percent", "Stance Phase [%]"),
#     ("swing_phase_percent", "Swing Phase [%]"),
# ]



import ast

def _coerce_scalar_cell(x):
    """CSVにベクトル文字列が混ざっていても、比較用にスカラーへ落とす。
    - '[a b c]' / '[a, b, c]' / 'a b c' など -> np.linalg.norm([a,b,c])
    - list/tuple/np.ndarray -> 同様
    - それ以外 -> float 変換（失敗は NaN）
    ※4_1 側で step_width がベクトルで保存されてしまうケースの救済が主目的。
    """
    if x is None:
        return np.nan
    # already numeric
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    # numpy array / list
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float).ravel()
        if arr.size == 0:
            return np.nan
        if arr.size == 1:
            return float(arr[0])
        return float(np.linalg.norm(arr))
    # string
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return np.nan
        # try direct float
        try:
            return float(s)
        except Exception:
            pass
        # try parse like "[0.1 0.2 0.3]" or "[0.1,0.2,0.3]" or "0.1 0.2 0.3"
        s2 = s
        if s2.startswith("[") and s2.endswith("]"):
            s2 = s2[1:-1]
        s2 = s2.replace(",", " ")
        # np.fromstring is robust for space-separated floats
        arr = np.fromstring(s2, sep=" ")
        if arr.size == 0:
            # last resort: ast.literal_eval
            try:
                v = ast.literal_eval(s)
                return _coerce_scalar_cell(v)
            except Exception:
                return np.nan
        if arr.size == 1:
            return float(arr[0])
        return float(np.linalg.norm(arr))
    # fallback
    try:
        return float(x)
    except Exception:
        return np.nan


def coerce_scalar_series(series: pd.Series) -> np.ndarray:
    """Series を比較用 float 配列に変換（ベクトル文字列も救済）。"""
    return series.apply(_coerce_scalar_cell).to_numpy(dtype=float)


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


def load_mocap_mean_cycles(mid_npz_path: Path):
    """
    4_1 intermediate npz から mean_cycle_r/l を DataFrame 復元して返す。
    """
    mid = np.load(mid_npz_path, allow_pickle=True)
    cols_r = list(mid["mean_cycle_r_cols"].tolist())
    cols_l = list(mid["mean_cycle_l_cols"].tolist())
    mean_r = pd.DataFrame(mid["mean_cycle_r"], columns=cols_r)
    mean_l = pd.DataFrame(mid["mean_cycle_l"], columns=cols_l)
    stem = mid_npz_path.stem.replace("_opti_intermediate", "")
    return mean_r, mean_l, stem


def ylims_for_angle_base(base: str):
    return YLIM_BY_ANGLE.get(base, None)


def mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    return float(np.mean(np.abs(a[m] - b[m])))


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_pose_mean_cycle_pairs(method_dir: Path):
    """
    Pose3D_results/<method>/normalized_cycle_R_mean_*.csv を走査して、
    tag_short ごとに (R_csv, L_csv) を返す。
    """
    pairs = []
    r_files = sorted(method_dir.glob("normalized_cycle_R_mean_*.csv"), key=lambda p: natural_sort_key(p.name))
    for r_csv in r_files:
        tag = r_csv.name.replace("normalized_cycle_R_mean_", "").replace(".csv", "")
        l_csv = method_dir / f"normalized_cycle_L_mean_{tag}.csv"
        pairs.append((tag, r_csv, l_csv if l_csv.exists() else None))
    return pairs


def compare_mean_cycles(mean_mocap: pd.DataFrame, mean_pose: pd.DataFrame, side: str):
    """
    平均波形（_mean列）を Mocap vs Pose3D で比較し、角度ごとの MAE/RMSE を返す。
    """
    rows = []
    for base in ANGLE_BASE_KEYS:
        col = f"{side}_{base}_mean"
        if col not in mean_mocap.columns:
            continue
        if col not in mean_pose.columns:
            continue
        rows.append(
            dict(
                joint=f"{side}_{base}",
                mae=mae(mean_mocap[col].to_numpy(), mean_pose[col].to_numpy()),
                rmse=rmse(mean_mocap[col].to_numpy(), mean_pose[col].to_numpy()),
            )
        )
    return rows


def plot_compare_cycle_flexext(out_dir: Path, tag: str, side: str,
                              mean_mocap: pd.DataFrame, mean_pose: pd.DataFrame,
                              pose_label: str):
    """
    屈曲伸展（股・膝・足）を3段で比較プロット（Mocap vs Pose）。
    """
    keys = [f"{side}_Hip_FlEx", f"{side}_Knee_FlEx", f"{side}_Ankle_PlDo"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for ax, k in zip(axes, keys):
        mcol = f"{k}_mean"
        scol = f"{k}_std"
        if mcol not in mean_mocap.columns or mcol not in mean_pose.columns:
            ax.set_title(f"{k} (missing)")
            ax.grid(True)
            continue

        x = mean_mocap["Percentage"] if "Percentage" in mean_mocap.columns else np.arange(len(mean_mocap))
        ax.plot(x, mean_mocap[mcol], label="Mocap")
        if scol in mean_mocap.columns:
            ax.fill_between(x, mean_mocap[mcol] - mean_mocap[scol], mean_mocap[mcol] + mean_mocap[scol], alpha=0.2)

        x2 = mean_pose["Percentage"] if "Percentage" in mean_pose.columns else np.arange(len(mean_pose))
        ax.plot(x2, mean_pose[mcol], label=pose_label)
        if scol in mean_pose.columns:
            ax.fill_between(x2, mean_pose[mcol] - mean_pose[scol], mean_pose[mcol] + mean_pose[scol], alpha=0.2)

        base = k.split("_", 1)[1]
        ylims = ylims_for_angle_base(base)
        if ylims is not None:
            ax.set_ylim(*ylims)

        ax.set_ylabel("Angle [deg]")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Gait Cycle [%]")
    fig.suptitle(f"Cycle Compare Fl/Ex ({side}) - {tag}")
    plt.tight_layout()
    plt.savefig(out_dir / f"compare_cycle_FlEx_{side}_{tag}.png")
    plt.close()


def plot_compare_cycle_single(out_dir: Path, tag: str, side: str,
                             base: str,
                             mean_mocap: pd.DataFrame, mean_pose: pd.DataFrame,
                             pose_label: str):
    """
    単一角度（例: Hip_InEx / Hip_AdAb）を1段で比較プロット
    """
    k = f"{side}_{base}"
    mcol = f"{k}_mean"
    scol = f"{k}_std"

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    if mcol not in mean_mocap.columns or mcol not in mean_pose.columns:
        ax.set_title(f"{k} (missing)")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / f"compare_cycle_{base}_{side}_{tag}.png")
        plt.close()
        return

    x = mean_mocap["Percentage"] if "Percentage" in mean_mocap.columns else np.arange(len(mean_mocap))
    ax.plot(x, mean_mocap[mcol], label="Mocap")
    if scol in mean_mocap.columns:
        ax.fill_between(x, mean_mocap[mcol] - mean_mocap[scol], mean_mocap[mcol] + mean_mocap[scol], alpha=0.2)

    x2 = mean_pose["Percentage"] if "Percentage" in mean_pose.columns else np.arange(len(mean_pose))
    ax.plot(x2, mean_pose[mcol], label=pose_label)
    if scol in mean_pose.columns:
        ax.fill_between(x2, mean_pose[mcol] - mean_pose[scol], mean_pose[mcol] + mean_pose[scol], alpha=0.2)

    ylims = ylims_for_angle_base(base)
    if ylims is not None:
        ax.set_ylim(*ylims)

    ax.set_xlabel("Gait Cycle [%]")
    ax.set_ylabel("Angle [deg]")
    ax.grid(True)
    ax.legend()
    ax.set_title(f"Cycle Compare {k} - {tag}")
    plt.tight_layout()
    plt.savefig(out_dir / f"compare_cycle_{base}_{side}_{tag}.png")
    plt.close()


def load_gait_params_pair(mocap_dir: Path, mocap_stem: str, method_dir: Path, tag: str):
    """
    gait_parameters_{R/L}_*.csv の読み込み（Mocap / Pose）を返す
    """
    mocap_r = mocap_dir / f"gait_parameters_R_{mocap_stem}.csv"
    mocap_l = mocap_dir / f"gait_parameters_L_{mocap_stem}.csv"
    pose_r = method_dir / f"gait_parameters_R_{tag}.csv"
    pose_l = method_dir / f"gait_parameters_L_{tag}.csv"

    return (
        safe_read_csv(mocap_r),
        safe_read_csv(mocap_l),
        safe_read_csv(pose_r),
        safe_read_csv(pose_l),
    )


def calculate_gait_param_mae(params_mocap: pd.DataFrame, params_pose: pd.DataFrame, key: str):
    """
    各歩行周期（cycle_index）ごとに Mocap を正解として Pose の MAE を計算
    - cycle_index が無い場合は先頭から min(len) で対応
    """
    if params_mocap is None or params_pose is None:
        return np.nan, 0

    if key not in params_mocap.columns or key not in params_pose.columns:
        return np.nan, 0

    if "cycle_index" in params_mocap.columns and "cycle_index" in params_pose.columns:
        m = params_mocap.set_index("cycle_index")[key]
        p = params_pose.set_index("cycle_index")[key]
        common = sorted(set(m.index.tolist()) & set(p.index.tolist()))
        if len(common) == 0:
            return np.nan, 0
        err = (coerce_scalar_series(p.loc[common]) - coerce_scalar_series(m.loc[common]))
        return float(np.mean(np.abs(err))), len(common)

    # fallback: 順番で合わせる
    n = min(len(params_mocap), len(params_pose))
    if n == 0:
        return np.nan, 0
    err = coerce_scalar_series(params_pose[key])[:n] - coerce_scalar_series(params_mocap[key])[:n]
    return float(np.mean(np.abs(err))), n


def gait_param_mean_std(df: pd.DataFrame, key: str):
    if df is None or key not in df.columns:
        return np.nan, np.nan
    v = coerce_scalar_series(df[key])
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return float(np.mean(v)), float(np.std(v))


def run_one_pair(thera_dir: Path, mocap_mid_npz: Path, method_dir: Path, tag: str,
                 r_csv: Path, l_csv: Path | None):
    """
    1 method_dir x 1 tag（=1 つの 3d_kp_*.npz に対応）を比較
    """
    mocap_dir = thera_dir / "mocap"
    mean_r_mocap, mean_l_mocap, mocap_stem = load_mocap_mean_cycles(mocap_mid_npz)

    mean_r_pose = safe_read_csv(r_csv)
    mean_l_pose = safe_read_csv(l_csv) if l_csv is not None else None

    if mean_r_pose is None and mean_l_pose is None:
        return

    pose_label = method_dir.name  # <method>
    compare_dir = thera_dir / "Pose3D_results" / "_compare_with_mocap" / pose_label
    compare_dir.mkdir(parents=True, exist_ok=True)

    # ---------- angle errors ----------
    errors = []
    if mean_r_pose is not None:
        errors += compare_mean_cycles(mean_r_mocap, mean_r_pose, "R")
    if mean_l_pose is not None:
        errors += compare_mean_cycles(mean_l_mocap, mean_l_pose, "L")

    err_df = pd.DataFrame(errors)
    err_df.to_csv(compare_dir / f"angle_error_summary_{tag}.csv", index=False)

    # ---------- plots (cycle) ----------
    if mean_r_pose is not None:
        plot_compare_cycle_flexext(compare_dir, tag, "R", mean_r_mocap, mean_r_pose, pose_label)
        plot_compare_cycle_single(compare_dir, tag, "R", "Hip_InEx", mean_r_mocap, mean_r_pose, pose_label)
        plot_compare_cycle_single(compare_dir, tag, "R", "Hip_AdAb", mean_r_mocap, mean_r_pose, pose_label)

    if mean_l_pose is not None:
        plot_compare_cycle_flexext(compare_dir, tag, "L", mean_l_mocap, mean_l_pose, pose_label)
        plot_compare_cycle_single(compare_dir, tag, "L", "Hip_InEx", mean_l_mocap, mean_l_pose, pose_label)
        plot_compare_cycle_single(compare_dir, tag, "L", "Hip_AdAb", mean_l_mocap, mean_l_pose, pose_label)

    # ---------- gait parameter comparison ----------
    mocap_r, mocap_l, pose_r, pose_l = load_gait_params_pair(mocap_dir, mocap_stem, method_dir, tag)

    comp_rows = []
    mae_rows = []

    for side, df_m, df_p in [("R", mocap_r, pose_r), ("L", mocap_l, pose_l)]:
        if df_m is None or df_p is None:
            continue

        for key, label in GAIT_PARAM_KEYS:
            m_mean, m_std = gait_param_mean_std(df_m, key)
            p_mean, p_std = gait_param_mean_std(df_p, key)

            comp_rows.append(
                dict(
                    side=side,
                    key=key,
                    label=label,
                    mocap_mean=m_mean,
                    mocap_std=m_std,
                    pose_mean=p_mean,
                    pose_std=p_std,
                    abs_diff_mean=(abs(p_mean - m_mean) if np.isfinite(p_mean) and np.isfinite(m_mean) else np.nan),
                )
            )

            v_mae, ncyc = calculate_gait_param_mae(df_m, df_p, key)
            mae_rows.append(
                dict(
                    side=side,
                    key=key,
                    label=label,
                    mae=v_mae,
                    n_cycles=ncyc,
                )
            )

    pd.DataFrame(comp_rows).to_csv(compare_dir / f"gait_param_comparison_stats_{tag}.csv", index=False)
    pd.DataFrame(mae_rows).to_csv(compare_dir / f"gait_param_mae_{tag}.csv", index=False)

    # MAE bar plot（主要パラメータ）
    if len(mae_rows) > 0:
        dfp = pd.DataFrame(mae_rows)
        for side in ["R", "L"]:
            sub = dfp[dfp["side"] == side].copy()
            if len(sub) == 0:
                continue
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.bar(sub["label"], sub["mae"])
            ax.set_title(f"Gait Param MAE ({side}) - {tag}")
            ax.set_ylabel("MAE")
            ax.tick_params(axis="x", rotation=30)
            ax.grid(True, axis="y")
            plt.tight_layout()
            plt.savefig(compare_dir / f"gait_param_mae_{side}_{tag}.png")
            plt.close()

    print(f"[OK] {thera_dir.name} / {method_dir.name} / {tag}")
    print(f"     -> {compare_dir}")


def main():
    for sub_i in SUB_RANGE:
        sub_dir = ROOT_DIR / f"sub{sub_i}"
        if not sub_dir.exists():
            continue

        thera_dir = sub_dir / f"thera{sub_i}-0"
        mocap_dir = thera_dir / "mocap"
        if not mocap_dir.exists():
            continue

        mid_npz = find_opti_intermediate_npz(mocap_dir)
        if mid_npz is None or not mid_npz.exists():
            print(f"[SKIP] mocap intermediate not found (run 4_1 first): {mocap_dir}")
            continue

        pose_root = thera_dir / "Pose3D_results"
        if not pose_root.exists():
            print(f"[SKIP] Pose3D_results not found (run 4_2 first): {thera_dir}")
            continue

        method_dirs = sorted([d for d in pose_root.iterdir() if d.is_dir() and d.name != "_compare_with_mocap"],
                             key=lambda p: natural_sort_key(p.name))
        if len(method_dirs) == 0:
            print(f"[SKIP] no method dirs: {pose_root}")
            continue

        print("=" * 90)
        print(f"[RUN] {thera_dir}  (methods={len(method_dirs)})")
        print(f"  mocap intermediate: {mid_npz.name}")
        print("=" * 90)

        for method_dir in method_dirs:
            pairs = find_pose_mean_cycle_pairs(method_dir)
            if len(pairs) == 0:
                continue

            for tag, r_csv, l_csv in pairs:
                try:
                    run_one_pair(thera_dir, mid_npz, method_dir, tag, r_csv, l_csv)
                except Exception as e:
                    print(f"[ERROR] {thera_dir.name} / {method_dir.name} / {tag}: {e}")


if __name__ == "__main__":
    main()
