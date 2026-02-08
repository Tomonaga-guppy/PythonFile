from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# フォント・見た目（全体設定）
# =========================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 20,          # 全体
    "axes.titlesize": 20,    # タイトル
    "axes.labelsize": 20,     # 軸ラベル
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
})

# =========================
# 設定
# =========================
CSV_PATH = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression\regression_dataset.csv")
OUT_DIR = CSV_PATH.parent / "boxplots_each_column"
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
    'hip_cc_lag',
]
GAIT_PARAM_ORIGINAL_COLS = [
    "speed_ori",
    "SI_sw_ori",
    "stride_time_ori",
    "stride_width_ori",
    "hip_fl_max_ori",
    "hip_ex_max_ori",
    "kne_fl_max_ori",
    "ank_do_max_ori",
    "hip_ab_max_ori",
]
GAIT_PARAM_COLS = [
    "speed",
    "SI_sw",
    "stride_time",
    "stride_width",
    "hip_fl_max",
    "hip_ex_max",
    "kne_fl_max",
    "ank_do_max",
    "hip_ab_max",
]
GAIT_PARAM_DELTA_COLS = [
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
    "mmse",
]
PT_BASE_COLS = [
    "pt_age",
    "pt_height",
    "pt_weight",
    "grip_power",
    "exp",
]

# =========================
# 単位マップ（必要に応じて増やしてOK）
# =========================
UNITS = {
    # ---- assist系（あなたの定義に合わせて微調整してOK）----
    "hip_dist": "m",          # もしmmで保存してるなら "mm" に変更
    "hip_dist_n": "-",        # 正規化なら無次元
    "wri_para_s": "m",        # 距離/位置なら m（あなたの定義次第）
    "wri_nonpara_s": "m",
    "cos_sim": "-",           # cos類似度
    "hip_cc_x": "-",          # 相関係数
    "hip_cc_y": "-",
    "hip_cc_z": "-",
    "hip_cc_3d": "-",
    "hip_cc_lag": "s",        # もしフレームなら "frames"

    # ---- gait系（一般的な想定：必要なら変更）----
    "speed_ori": "m/s",
    "speed": "m/s",
    "speed_delta": "m/s",

    "SI_sw_ori": "-",
    "SI_sw": "-",
    "SI_sw_delta": "-",

    "stride_time_ori": "s",
    "stride_time": "s",
    "stride_time_delta": "s",

    "stride_width_ori": "m",
    "stride_width": "m",
    "stride_width_delta": "m",

    "hip_fl_max_ori": "deg",
    "hip_ex_max_ori": "deg",
    "kne_fl_max_ori": "deg",
    "ank_do_max_ori": "deg",
    "hip_ab_max_ori": "deg",

    "hip_fl_max": "deg",
    "hip_ex_max": "deg",
    "kne_fl_max": "deg",
    "ank_do_max": "deg",
    "hip_ab_max": "deg",

    "hip_fl_max_delta": "deg",
    "hip_ex_max_delta": "deg",
    "kne_fl_max_delta": "deg",
    "ank_do_max_delta": "deg",
    "hip_ab_max_delta": "deg",

    # ---- base系 ----
    "pa_age": "yr",
    "pt_age": "yr",
    "pa_height": "cm",  # もしmなら "m"
    "pt_height": "cm",
    "pa_weight": "kg",
    "pt_weight": "kg",
    "grip_power": "kgf",  # Nで記録してるなら "N"
    "exp": "yr",

    "days_post_onset": "day",
    "fac": "-", "brs_lower": "-",
    "fim_walk": "-", "fim_motor": "-", "fim_cog": "-", "mmse": "-",

    "sias_m_hip": "-", "sias_m_knee": "-", "sias_m_ankle": "-", "sias_m_total": "-",
    "sias_sens_sole": "-", "sias_prop_toe": "-",
    "mi_hip": "-", "mi_knee": "-", "mi_ankle": "-", "mi_total": "-",
}

def label_with_unit(col: str) -> str:
    u = UNITS.get(col, None)
    if u is None:
        return f"{col} [unit?]"   # 未登録を見つけやすくする
    if u == "-" or u == "":
        return f"{col} [-]"
    return f"{col} [{u}]"

# =========================
# CSV 読み込み
# =========================
df = pd.read_csv(CSV_PATH)

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
    Line2D([0], [0], marker="o", linestyle="", markersize=8,
           markerfacecolor=pa_to_color[pa], markeredgecolor="k",
           label=f"PA {pa}")
    for pa in pa_vals
]
pt_legend_handles = [
    Line2D([0], [0], marker=pt_to_marker[pt], linestyle="", markersize=8,
           markerfacecolor="white", markeredgecolor="k",
           label=f"PT {pt}")
    for pt in pt_vals
]

# =========================
# 各列ごとにプロット
# =========================
for col in plot_cols:
    if col in ASSIST_COLS:
        team_flag = 0
    elif col in GAIT_PARAM_DELTA_COLS:
        team_flag = 1
    elif col in GAIT_PARAM_COLS:
        team_flag = 2
    elif col in GAIT_PARAM_ORIGINAL_COLS:
        team_flag = 3
    elif col in PA_BASE_COLS:
        team_flag = 4
    elif col in PT_BASE_COLS:
        team_flag = 5
    else:
        team_flag = 6

    sub = df[["pa_id", "pt_id", col]].dropna()  # 行ごとに揃えて落とす
    if len(sub) == 0:
        print(f"[SKIP] {col} (no data)")
        continue

    x_all = sub[col].values

    fig, ax = plt.subplots(figsize=(9.5, 4.4))  # ちょい大きめ

    # boxplot（横向き）
    bp = ax.boxplot(
        x_all,
        vert=False,
        patch_artist=False,
        showfliers=False,
        boxprops=dict(color="black", linewidth=1.4),
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(color="black", linewidth=1.4),
        capprops=dict(color="black", linewidth=1.4),
    )
    for box in bp["boxes"]:
        box.set_alpha(0.6)

    # --- 点プロット：行ごとに「固定マッピング」を参照して描画 ---
    y_base = 1.0
    jitter = np.random.uniform(-0.06, 0.06, size=len(sub))
    y = y_base + jitter

    for i, row in enumerate(sub.itertuples(index=False)):
        pa = row.pa_id
        pt = row.pt_id
        x = getattr(row, col)

        ax.scatter(
            x, y[i],
            s=60,
            alpha=0.85,
            marker=pt_to_marker[pt],
            facecolors=pa_to_color[pa],   # paで色固定
            edgecolors="k",
            linewidths=0.5
        )

    # 軸等
    ax.set_yticks([])
    ax.set_xlabel(label_with_unit(col))
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # 凡例：PA（色）とPT（形）を別々に表示（右側に2段）
    leg1 = ax.legend(
        handles=pa_legend_handles,
        title="Patient (color)       ",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=pt_legend_handles,
        title="Therapist (marker)",
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
