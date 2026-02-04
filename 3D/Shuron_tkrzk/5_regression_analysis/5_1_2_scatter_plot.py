from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# 設定
# =========================
CSV_PATH = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression\regression_dataset.csv")

# 横軸に使いたい指標（for文で回す）
X_LIST = [
    "speed_delta", "SI_sw_delta", "stride_time_delta", "stride_width_delta",
    "hip_ex_max_delta", "kne_fl_max_delta", "ank_do_max_delta", "hip_ab_max_delta"
]

EXCLUDE_COLS = ["pa_id", "pt_id"]

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

# =========================
# PA / PT マッピング（1回だけ決める）
# =========================
pa_vals = sorted(df["pa_id"].dropna().unique().tolist())
pt_vals = sorted(df["pt_id"].dropna().unique().tolist())

pa_to_color = {pa: f"C{i % 10}" for i, pa in enumerate(pa_vals)}
pt_to_marker = {pt: MARKERS[i % len(MARKERS)] for i, pt in enumerate(pt_vals)}

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
# X_COL を for で回す
# =========================
for X_COL in X_LIST:

    if X_COL not in df.columns:
        print(f"[SKIP] X_COL='{X_COL}' not found")
        continue

    # Xごとに保存フォルダを分ける
    out_dir = CSV_PATH.parent / f"scatter_x_delta"
    out_dir.mkdir(exist_ok=True)
    OUT_DIR = out_dir / X_COL
    OUT_DIR.mkdir(exist_ok=True)

    # Y候補（数値のみ）
    y_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and c != X_COL
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    print(f"\n=== X_COL = {X_COL} ===")
    print("Y columns:", len(y_cols))

    for y_col in y_cols:
        if y_col in ASSIST_COLS:
            team_flag = 0
        elif y_col in GAIT_PARAM_COLS:
            team_flag = 1
        elif y_col in PA_BASE_COLS:
            team_flag = 2
        elif y_col in PT_BASE_COLS:
            team_flag = 3
        else:
            team_flag = 4
        sub = df[["pa_id", "pt_id", X_COL, y_col]].dropna()
        if len(sub) == 0:
            continue

        fig, ax = plt.subplots(figsize=(6.8, 5.2))

        for row in sub.itertuples(index=False):
            ax.scatter(
                getattr(row, X_COL),
                getattr(row, y_col),
                s=55,
                alpha=0.85,
                marker=pt_to_marker[row.pt_id],
                facecolors=pa_to_color[row.pa_id],
                edgecolors="k",
                linewidths=0.4,
                zorder=3
            )

        ax.set_xlabel(X_COL)
        ax.set_ylabel(y_col)
        ax.grid(True, linestyle="--", alpha=0.5)

        # 凡例（PA=色、PT=形）
        leg1 = ax.legend(
            handles=pa_legend_handles,
            title="pa_id (color)",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.00)
        )
        ax.add_artist(leg1)

        ax.legend(
            handles=pt_legend_handles,
            title="pt_id (marker)",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.30)
        )

        ax.set_title(f"{y_col} vs {X_COL}")

        plt.tight_layout()
        out_path = OUT_DIR / f"{team_flag}_{y_col}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[SAVED] {out_path}")
