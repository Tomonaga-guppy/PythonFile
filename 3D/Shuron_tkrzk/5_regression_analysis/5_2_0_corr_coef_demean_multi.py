#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5_2_0_corr_coef_demean_multi_targets.py

患者内 demean（within-patient）後に、目的変数（target）を複数切り替えて
Spearman相関 + 散布図を自動保存する版。

出力構造:
  correlation_analysis_demean_multi/
    y_speed_delta/
      scatter/
      spearman_speed_delta_vs_predictors__demean.csv
    y_stride_time_delta/
      scatter/
      spearman_stride_time_delta_vs_predictors__demean.csv
    ...
  + 全部まとめ: spearman_all_targets__demean.csv
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# 1) 設定
# ============================================================
CSV_PATH = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression\regression_dataset.csv")

# 出力先（目的変数ごとにサブフォルダを作る）
OUT_ROOT = CSV_PATH.parent / "correlation_analysis_demean_multi"

PATIENT_ID_COL = "pa_id"

# ---- 目的変数（ここを自動で切り替えて回す） ----
TARGET_COLS = [
    "speed_delta",
    "SI_sw_delta",
    "stride_time_delta",
    "stride_width_delta",
    "hip_fl_max_delta",
    "hip_ex_max_delta",
    "kne_fl_max_delta",
    "ank_do_max_delta",
    "hip_ab_max_delta",
]

# ---- 相関を取りたい説明側（predictor） ----
PREDICTOR_COLS = [
    "hip_dist",
    "wri_para_s",
    "wri_nonpara_s",
    "cos_sim",
    "hip_cc_x",
    "hip_cc_y",
    "hip_cc_z",
    "hip_cc_3d",
    "hip_cc_lag",
]

# 目的変数同士の相関（例: speed_delta vs stride_time_delta）も一緒に見たいなら True
INCLUDE_OTHER_TARGETS_AS_PREDICTORS = False

MAKE_PLOTS = True
PLOT_DPI = 200

UNITS = {
    "speed_delta": "m/s",
    "SI_sw_delta": "-",
    "stride_time_delta": "s",
    "stride_width_delta": "m",
    "hip_fl_max_delta": "deg",
    "hip_ex_max_delta": "deg",
    "kne_fl_max_delta": "deg",
    "ank_do_max_delta": "deg",
    "hip_ab_max_delta": "deg",
    "hip_dist": "m",
    "wri_para_s": "-",
    "wri_nonpara_s": "-",
    "cos_sim": "-",
    "hip_cc_x": "-",
    "hip_cc_y": "-",
    "hip_cc_z": "-",
    "hip_cc_3d": "-",
    "hip_cc_lag": "frames",
}


# ============================================================
# 2) 関数
# ============================================================
def safe_spearman(x: pd.Series, y: pd.Series):
    """NaN除外後の Spearman rho, p, n"""
    xy = pd.concat([x, y], axis=1).dropna()
    n = len(xy)
    if n < 3:
        return np.nan, np.nan, int(n)
    rho, p = spearmanr(xy.iloc[:, 0], xy.iloc[:, 1])
    return float(rho), float(p), int(n)


def demean_within_patient(df: pd.DataFrame, patient_col: str, cols: list[str]) -> pd.DataFrame:
    """
    患者内 demean: x_ij -> x_ij - mean_i
    NaNはそのまま（後段でdropna）
    """
    out = df.copy()
    out[cols] = df.groupby(patient_col)[cols].transform(lambda s: s - s.mean())
    return out


def pretty_name(col: str) -> str:
    """表示用: *_delta は Δ を付けて見やすく"""
    if col.endswith("_delta"):
        return "Δ" + col.replace("_delta", "")
    return col


def _fmt_num(v: float, kind: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan"
    if kind == "rho":
        return f"{v:.3f}"
    if kind == "p":
        return f"{v:.3g}"
    return str(v)


def save_scatter(df: pd.DataFrame, x_col: str, y_col: str, out_png: Path, rho: float, p: float, n: int, suffix_title: str = ""):
    xy = df[[x_col, y_col]].dropna()

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)

    ax.scatter(xy[x_col], xy[y_col])

    title_text = f"{pretty_name(y_col)} vs {pretty_name(x_col)}{suffix_title}"
    ax.set_title(title_text)

    ax.set_xlabel(f"{pretty_name(x_col)} [{UNITS.get(x_col, '-')}]")
    ax.set_ylabel(f"{pretty_name(y_col)} [{UNITS.get(y_col, '-')}]")

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)

    txt = f"rho={_fmt_num(rho,'rho')}\np={_fmt_num(p,'p')}\nn={n}"
    fig.text(
        0.78, 0.94,
        txt,
        ha="left",
        va="center",
        fontsize=10,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black",
            alpha=0.9,
        )
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=PLOT_DPI)
    plt.close(fig)


def unique_keep_order(xs: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# ============================================================
# 3) メイン処理
# ============================================================
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # --- 列チェック ---
    needed = set([PATIENT_ID_COL]) | set(TARGET_COLS) | set(PREDICTOR_COLS)
    if INCLUDE_OTHER_TARGETS_AS_PREDICTORS:
        needed |= set(TARGET_COLS)

    missing = [c for c in sorted(needed) if c not in df.columns]
    if missing:
        raise ValueError(f"CSVに列がありません: {missing}")

    # --- demeanは1回でまとめてやる（全ターゲット&予測子の和集合） ---
    cols_to_demean = unique_keep_order(TARGET_COLS + PREDICTOR_COLS)
    if INCLUDE_OTHER_TARGETS_AS_PREDICTORS:
        cols_to_demean = unique_keep_order(cols_to_demean + TARGET_COLS)

    df_dm = demean_within_patient(df, PATIENT_ID_COL, cols_to_demean)

    all_rows = []

    for target in TARGET_COLS:
        # 保存先（目的変数ごと）
        out_dir = OUT_ROOT / f"y_{target}"
        out_dir.mkdir(exist_ok=True)

        plot_dir = out_dir / "scatter"
        if MAKE_PLOTS:
            plot_dir.mkdir(exist_ok=True)

        # このターゲットに対して相関を取る相手
        predictors = list(PREDICTOR_COLS)
        if INCLUDE_OTHER_TARGETS_AS_PREDICTORS:
            predictors += [t for t in TARGET_COLS if t != target]
        predictors = unique_keep_order([c for c in predictors if c != target])

        results = []

        for pred in predictors:
            rho, p, n = safe_spearman(df_dm[pred], df_dm[target])

            row = {
                "target": target,
                "predictor": pred,
                "n_pairwise": n,
                "spearman_rho": rho,
                "p_value": p,
                "note": "within-patient demean",
            }
            results.append(row)
            all_rows.append(row)

            if MAKE_PLOTS:
                out_png = plot_dir / f"{target}_vs_{pred}__demean.png"
                save_scatter(df_dm, pred, target, out_png, rho, p, n, suffix_title=" (demean)")

        res_df = pd.DataFrame(results).sort_values("p_value")
        out_csv = out_dir / f"spearman_{target}_vs_predictors__demean.csv"
        res_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        print(f"=== DONE target={target} ===")
        print(f"CSV : {out_csv}")

    # 全部まとめ
    all_df = pd.DataFrame(all_rows).sort_values(["target", "p_value"])
    out_all = OUT_ROOT / "spearman_all_targets__demean.csv"
    all_df.to_csv(out_all, index=False, encoding="utf-8-sig")
    print("=== DONE ALL ===")
    print(f"CSV : {out_all}")


if __name__ == "__main__":
    main()
