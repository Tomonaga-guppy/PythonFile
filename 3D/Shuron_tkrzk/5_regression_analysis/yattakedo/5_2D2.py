import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =========================
# 設定
# =========================
ROOT_DIR = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression")
CSV_PATH = ROOT_DIR / "regression_dataset.csv"

PATIENT_COL = "pa_id"
YCOL = "speed_delta"
XCOLS = ['hip_dist', 'wri_para_s', 'wri_nonpara_s', 'cos_sim', 'hip_cc_x', 'hip_cc_y', 'hip_cc_z', 'hip_cc_3d', 'hip_cc_lag']

OUT_DIR = ROOT_DIR / "reg_single_assist" / "fig_demeaned_scatter"
OUT_DIR.mkdir(exist_ok=True)

# 患者ごとのマーカー（最大4人想定）
MARKER_MAP = {
    2: "o",   # circle
    3: "s",   # square
    15: "^",  # triangle
    16: "D",  # diamond
}

# 患者ごとのカラー（最大4人想定）
COLOR_MAP = {
    2: "#1f77b4",   # blue
    3: "#ff7f0e",   # orange
    15: "#2ca02c",  # green
    16: "#d62728",  # red
}

# =========================
# デミーン関数
# =========================
def demean_by_group(df, cols, group_col):
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out.groupby(group_col)[c].transform("mean")
    return out

# =========================
# データ読み込み & デミーン
# =========================
df = pd.read_csv(CSV_PATH)

df_dm = demean_by_group(
    df,
    cols=[YCOL] + XCOLS,
    group_col=PATIENT_COL
)

# =========================
# プロット
# =========================
sns.set(style="whitegrid", context="talk")

for xcol in XCOLS:
    plt.figure(figsize=(7, 6))

    # 患者ごとに marker と color を変えて描画
    for pid, m in MARKER_MAP.items():
        sub = df_dm[df_dm[PATIENT_COL] == pid]
        if len(sub) == 0:
            continue

        plt.scatter(
            sub[xcol],
            sub[YCOL],
            marker=m,
            s=120,
            # facecolors="none",   # 白抜き
            edgecolors=COLOR_MAP.get(pid, "black"),
            linewidths=1.8,
            label=f"PA {pid}"
        )

    # 全体回帰直線（参考）
    sns.regplot(
        data=df_dm,
        x=xcol,
        y=YCOL,
        scatter=False,
        color="black",
        ci=None,
        line_kws={"linewidth": 2, "linestyle": "--"}
    )

    # 原点ガイド
    plt.axhline(0, color="gray", linestyle=":", linewidth=1)
    plt.axvline(0, color="gray", linestyle=":", linewidth=1)

    plt.xlabel(f"{xcol} (demeaned)")
    plt.ylabel("speed_delta (demeaned)")
    plt.title(f"Demeaned {xcol} vs Demeaned speed_delta")

    plt.legend(title="Patient", frameon=True)
    plt.tight_layout()

    out_path = OUT_DIR / f"scatter_demeaned_{xcol}_marker.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")
