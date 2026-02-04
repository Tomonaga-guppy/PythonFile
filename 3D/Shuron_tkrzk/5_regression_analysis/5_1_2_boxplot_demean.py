from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# 設定
# =========================
CSV_PATH = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression\regression_dataset.csv")
OUT_DIR = CSV_PATH.parent / "boxplots_each_column_demean_by_pa_id"
OUT_DIR.mkdir(exist_ok=True)

EXCLUDE_COLS = ["pa_id", "pt_id"]

# ptが多い可能性を考えて、マーカー候補を多めに用意
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "p", "8", "H", "d"]


ASSIST_COLS = [
    'hip_dist', 
    'hip_dist_n',
    'wri_para_s', 
    'wri_nonpara_s', 
    'cos_sim', 
    'hip_cc_x', 
    'hip_cc_y', 
    'hip_cc_z', 
    'hip_cc_3d', 
    'hip_cc_lag'
]
GAIT_PARAM_COLS = [
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
PA_BASE_COLS = [
    "pa_age", 
    "pa_height", 
    "pa_weight", 
    "fac", 
    "brs_lower", 
    "sias_m_hip", 
    "sias_m_knee", 
    "sias_m_ankle", 
    "sias_m_total", 
    "sias_sens_sole", 
    "sias_prop_toe", 
    "mi_hip", 
    "mi_knee", 
    "mi_ankle", 
    "mi_total", 
    "days_post_onset", 
    "fim_walk", 
    "fim_motor", 
    "fim_cog", 
    "mmse"
]

PT_BASE_COLS = [
    "pt_age", 
    "pt_height", 
    "pt_weight", 
    "grip_power"
]



# =========================
# CSV 読み込み
# =========================
df = pd.read_csv(CSV_PATH)

DEMEAN_GROUP_COL = "pa_id"   # 必要なら "pt_id" に変えるだけ

# 数値列だけを対象（pa_id, pt_id は除外）
demean_cols = [
    c for c in df.columns
    if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])
]

# group内平均との差分へ（dfを上書き：以降のコードを一切いじらないため）
df[demean_cols] = df.groupby(DEMEAN_GROUP_COL)[demean_cols].transform(lambda s: s - s.mean())

# プロット対象（数値列のみ）
plot_cols = [
    c for c in df.columns
    if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])
]

# =========================
# ここが重要：列ループ前にマッピングを確定
# =========================
pa_vals = sorted(df["pa_id"].dropna().unique().tolist())
pt_vals = sorted(df["pt_id"].dropna().unique().tolist())

# 色は matplotlib のデフォルト色循環 C0, C1, ... を固定割当
pa_to_color = {pa: f"C{i % 10}" for i, pa in enumerate(pa_vals)}  # 10色を周回（必要なら増やす）

# マーカーを pt_id に固定割当
pt_to_marker = {pt: MARKERS[i % len(MARKERS)] for i, pt in enumerate(pt_vals)}

print("PA mapping (pa_id -> color):")
for pa in pa_vals:
    print(pa, "->", pa_to_color[pa])

print("\nPT mapping (pt_id -> marker):")
for pt in pt_vals:
    print(pt, "->", pt_to_marker[pt])

# 凡例用ハンドル（毎回作ると重い＆ブレるので固定で作る）
pa_legend_handles = [
    Line2D([0], [0], marker="o", linestyle="", markersize=7,
           markerfacecolor=pa_to_color[pa], markeredgecolor="k",
           label=f"pa_id={pa}")
    for pa in pa_vals
]
pt_legend_handles = [
    Line2D([0], [0], marker=pt_to_marker[pt], linestyle="", markersize=7,
           markerfacecolor="white", markeredgecolor="k",
           label=f"pt_id={pt}")
    for pt in pt_vals
]

# =========================
# 各列ごとにプロット
# =========================
for col in plot_cols:
    if col in ASSIST_COLS:
        team_flag = 0
    elif col in GAIT_PARAM_COLS:
        team_flag = 1
    elif col in PA_BASE_COLS:
        team_flag = 2
    elif col in PT_BASE_COLS:
        team_flag = 3
    else:
        team_flag = 4
            
    sub = df[["pa_id", "pt_id", col]].dropna()  # 行ごとに揃えて落とす
    if len(sub) == 0:
        print(f"[SKIP] {col} (no data)")
        continue

    x_all = sub[col].values

    fig, ax = plt.subplots(figsize=(8, 3.5))

    # boxplot（横向き）
    bp = ax.boxplot(
        x_all,
        vert=False,
        patch_artist=False,
        showfliers=False,
        boxprops=dict(color="black", linewidth=1.2),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
    )
    for box in bp["boxes"]:
        box.set_alpha(0.6)

    # --- 点プロット：行ごとに「固定マッピング」を参照して描画 ---
    # （点数が少ない想定ならこれが一番確実）
    y_base = 1.0
    jitter = np.random.uniform(-0.06, 0.06, size=len(sub))
    y = y_base + jitter

    for i, row in enumerate(sub.itertuples(index=False)):
        pa = row.pa_id
        pt = row.pt_id
        x = getattr(row, col)

        ax.scatter(
            x, y[i],
            s=45,
            alpha=0.85,
            marker=pt_to_marker[pt],
            facecolors=pa_to_color[pa],   # paで色固定
            edgecolors="k",
            linewidths=0.4
        )

    # 軸等
    ax.set_yticks([])
    ax.set_xlabel(col)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # 凡例：PA（色）とPT（形）を別々に表示（右側に2段）
    leg1 = ax.legend(
        handles=pa_legend_handles,
        title="pa_id (color)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=pt_legend_handles,
        title="pt_id (marker)",
        loc="lower left",
        bbox_to_anchor=(1.02, 0),
        borderaxespad=0.0,
        frameon=True
    )

    plt.tight_layout()

    out_path = OUT_DIR / f"{team_flag}_{col}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {out_path}")
