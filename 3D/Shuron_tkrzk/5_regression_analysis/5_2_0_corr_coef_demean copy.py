"""
5_2_0_corr_coef_demean.py

患者内 demean（within-patient）後に相関をプロット・計算（Spearman）
- regression_dataset.csv を読み込み
- 各患者(pa_id)内で [speed_delta と各 *_delta] を demean
- speed_delta vs (他の *_delta) の散布図を保存
- Spearman相関係数 rho と p値、ペア数 n を算出
- 結果CSVを保存

ちなみに目的変数をspeed_deltaではなくspeedにしてもdemeanしているので結果は同じ
    speed_demean = speed - mean_speed_within_patient
    speed_delta_demean = speed_delta - mean_speed_delta_within_patient
        = speed - speed_baseline - (mean_speed_within_patient - mean_speed_baseline_within_patient)
        = speed - mean_speed_within_patient - (speed_baseline - mean_speed_baseline_within_patient)
        = speed - mean_speed_within_patient
        = speed_demean
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

# 出力先（demean版はフォルダ名を分ける）
OUT_DIR  = CSV_PATH.parent / "correlation_analysis_demean"

PATIENT_ID_COL = "pa_id"   # ← 患者ID列（あなたのdatasetでは pa_id）

X_COL = "SI_sw_delta"      # ← 横軸
# X_COL = "speed_delta"      # ← 横軸
Y_COLS = [
    # "speed_delta",
    "SI_sw_delta",
    "stride_time_delta",
    "stride_width_delta",
    "hip_fl_max_delta",
    "hip_ex_max_delta",
    "kne_fl_max_delta",
    "ank_do_max_delta",
    "hip_ab_max_delta",
    'hip_dist', 
    'wri_para_s', 
    'wri_nonpara_s', 
    'cos_sim', 
    'hip_cc_x', 
    'hip_cc_y', 
    'hip_cc_z', 
    'hip_cc_3d', 
    'hip_cc_lag'
]

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
    'hip_dist': 'm',
    'wri_para_s': '-',
    'wri_nonpara_s': '-',
    'cos_sim': '-',
    'hip_cc_x': '-',
    'hip_cc_y': '-',
    'hip_cc_z': '-',
    'hip_cc_3d': '-',
    'hip_cc_lag': 'frames',
}

MAKE_PLOTS = True
PLOT_DPI = 200


# ============================================================
# 2) 関数
# ============================================================
def safe_spearman(x: pd.Series, y: pd.Series):
    """NaN除外後の Spearman rho, p, n"""
    xy = pd.concat([x, y], axis=1).dropna()
    n = len(xy)
    if n < 3:
        return np.nan, np.nan, n
    rho, p = spearmanr(xy.iloc[:, 0], xy.iloc[:, 1])
    return float(rho), float(p), int(n)


def save_scatter(df, x_col, y_col, out_png, rho, p, n, suffix_title: str = ""):
    xy = df[[x_col, y_col]].dropna()

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)

    # 散布図
    ax.scatter(xy[x_col], xy[y_col])

    # タイトル
    title_text = f"ΔSpeed vs {y_col}{suffix_title}"
    ax.set_title(title_text)

    # 軸ラベル（単位つき）
    ax.set_xlabel(f"ΔSpeed [{UNITS.get(x_col, '-')}]")
    y_label_base = y_col.replace("_delta", "")
    ax.set_ylabel(f"{y_label_base} [{UNITS.get(y_col, '-')}]")

    # グリッド
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)

    # タイトル横（figure座標）に rho / p / n
    txt = f"rho={rho:.3f}\np={p:.3g}\nn={n}"
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
            alpha=0.9
        )
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=PLOT_DPI)
    plt.close(fig)


def demean_within_patient(df: pd.DataFrame, patient_col: str, cols: list[str]) -> pd.DataFrame:
    """
    患者内 demean: x_ij -> x_ij - mean_i
    NaNはそのまま（後段でdropna）
    """
    out = df.copy()
    out[cols] = df.groupby(patient_col)[cols].transform(lambda s: s - s.mean())
    return out


# ============================================================
# 3) メイン処理
# ============================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_dir = OUT_DIR / "scatter"
    plot_dir.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # 列チェック
    for c in [PATIENT_ID_COL, X_COL] + Y_COLS:
        if c not in df.columns:
            raise ValueError(f"CSVに列がありません: {c}")

    # demean
    cols_to_demean = [X_COL] + Y_COLS
    df_dm = demean_within_patient(df, PATIENT_ID_COL, cols_to_demean)

    results = []

    for y_col in Y_COLS:
        rho, p, n = safe_spearman(df_dm[X_COL], df_dm[y_col])

        results.append({
            "x": X_COL,
            "y": y_col,
            "n_pairwise": n,
            "spearman_rho": rho,
            "p_value": p,
            "note": "within-patient demean",
        })

        if MAKE_PLOTS:
            out_png = plot_dir / f"{y_col}_vs_{X_COL}__demean.png"
            save_scatter(df_dm, X_COL, y_col, out_png, rho, p, n, suffix_title=" (demean)")

    res_df = pd.DataFrame(results).sort_values("p_value")
    out_csv = OUT_DIR / "spearman_speed_delta_vs_others__demean.csv"
    res_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("=== DONE (demean) ===")
    print(f"CSV : {out_csv}")
    print(res_df)


if __name__ == "__main__":
    main()
