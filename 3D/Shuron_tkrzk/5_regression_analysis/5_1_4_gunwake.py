"""
5_1_3_gunwake.py
=================
speed_delta >= 0.1 の2群で
- Mann–Whitney U検定（両側）
- 効果量 r（rank-biserial correlation; r = Z / sqrt(N)）
- 箱ひげ図 + p値 + r 表示（PNG保存）
- 結果CSV出力

★ 5_1_2_scatter_plot.py と同じルールで色分け・形分け
  - pa_id : 色 (C0, C1, ...)
  - pt_id : マーカー (o, s, ^, ...)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu

# =========================
# 設定
# =========================
CSV_PATH = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression\regression_dataset.csv")

THRESHOLD = 0.1  # speed_delta の閾値（改善あり）
OUT_DIR = CSV_PATH.parent / f"mw_box_speed_ge{str(THRESHOLD).replace('.', 'p')}"
OUT_DIR.mkdir(exist_ok=True)

OUT_CSV = OUT_DIR / "mw_test_results_with_r.csv"

ASSIST_COLS = [
    "hip_dist",
    "hip_dist_n",
    "wri_para_s",
    "wri_nonpara_s",
    "cos_sim",
    "hip_cc_x",
    "hip_cc_y",
    "hip_cc_z",
    "hip_cc_3d",
    "hip_cc_lag",
]

PT_COLS = [
    "exp"
]

MIN_N_PER_GROUP = 3  # 小さすぎると不安定なのでスキップ

# 5_1_2 と同じマーカー候補
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "p", "8", "H", "d"]

# =========================
# 関数
# =========================
def format_p(p: float) -> str:
    if p < 1e-4:
        return "p < 1e-4"
    return f"p = {p:.4f}"


def rank_biserial_r_from_u(u_stat: float, n1: int, n2: int) -> tuple[float, float]:
    """r（rank-biserial correlation）を Z/√N で計算する（ties補正なし）"""
    mu_u = n1 * n2 / 2.0
    sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma_u == 0:
        return np.nan, np.nan
    z = (u_stat - mu_u) / sigma_u
    r = z / np.sqrt(n1 + n2)
    return float(r), float(z)


def add_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=1.2, color="k")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom")


def significance_star(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# =========================
# CSV 読み込み
# =========================
df = pd.read_csv(CSV_PATH)

required_cols = {"speed_delta", "pa_id", "pt_id"}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise KeyError(f"CSVに必要な列がありません: {missing}")

# pt_id==15を除外
df = df[df["pt_id"] != 15]

# =========================
# PA / PT マッピング（1回だけ決める）
# =========================
pa_vals = sorted(df["pa_id"].dropna().unique().tolist())
pt_vals = sorted(df["pt_id"].dropna().unique().tolist())

# pt_vals = pt_vals[:-1] 

print(f"pt__valis = {pt_vals}")

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
# 群分け
# =========================
df["speed_group"] = np.where(df["speed_delta"] >= THRESHOLD, f">={THRESHOLD}", f"<{THRESHOLD}")

# =========================
# 検定 + 箱ひげ図
# =========================
results = []
rng = np.random.default_rng(0)

for col in ASSIST_COLS + PT_COLS:
    if col not in df.columns:
        print(f"[SKIP] {col}: column not found")
        continue

    sub_all = df[["pa_id", "pt_id", "speed_group", col]].dropna()
    
    g_hi = sub_all.loc[sub_all["speed_group"] == f">={THRESHOLD}", col]
    g_lo = sub_all.loc[sub_all["speed_group"] == f"<{THRESHOLD}", col]

    n_hi, n_lo = len(g_hi), len(g_lo)
    if n_hi < MIN_N_PER_GROUP or n_lo < MIN_N_PER_GROUP:
        print(f"[SKIP] {col}: sample too small (>=:{n_hi}, <:{n_lo})")
        continue

    # U検定（両側）
    u_stat, p_val = mannwhitneyu(g_hi, g_lo, alternative="two-sided")

    # 効果量 r（rank-biserial）
    r, z = rank_biserial_r_from_u(u_stat, n_hi, n_lo)

    # --- 箱ひげ図 ---
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    data = [g_lo.values, g_hi.values]  # 左: <, 右: >=
    labels = [f"<{THRESHOLD} (n={n_lo})", f">={THRESHOLD} (n={n_hi})"]

    ax.boxplot(
        data,
        labels=labels,
        widths=0.55,
        showfliers=True,
        medianprops=dict(color="k", linewidth=1.0),
    )

    # 生データ点（PA=色、PT=形） + jitter
    sub_lo = sub_all.loc[sub_all["speed_group"] == f"<{THRESHOLD}", ["pa_id", "pt_id", col]]
    sub_hi = sub_all.loc[sub_all["speed_group"] == f">={THRESHOLD}", ["pa_id", "pt_id", col]]

    for i, sub_g in enumerate([sub_lo, sub_hi], start=1):  # x=1,2
        if len(sub_g) == 0:
            continue
        x = rng.normal(loc=i, scale=0.06, size=len(sub_g))
        for xi, (pa, pt, val) in zip(x, sub_g.itertuples(index=False, name=None)):
            ax.scatter(
                xi, val,
                s=45,
                alpha=0.85,
                marker=pt_to_marker.get(pt, "o"),
                facecolors=pa_to_color.get(pa, "C0"),
                edgecolors="k",
                linewidths=0.4,
                zorder=3,
            )

    # Y軸範囲（上の注釈が詰まらないように余白）
    y_all = np.concatenate([g_lo.values, g_hi.values])
    ymin_data = np.nanmin(y_all)
    ymax_data = np.nanmax(y_all)

    yr = ymax_data - ymin_data
    if yr == 0:
        yr = max(abs(ymax_data), 1.0) * 0.01

    top_pad = yr * 0.40
    bottom_pad = yr * 0.06
    ax.set_ylim(ymin_data - bottom_pad, ymax_data + top_pad)

    ax.set_title(f"{col} by speed_delta group")
    ax.set_ylabel(col)
    ax.grid(True, linestyle="--", alpha=0.5)

    # p値 + r を上に括弧表示
    ymin, ymax = ax.get_ylim()
    data_max = np.nanmax(y_all)
    bracket_y = data_max + (ymax - ymin) * 0.06
    bracket_h = (ymax - ymin) * 0.03

    star = significance_star(p_val)

    add_bracket(
        ax,
        1, 2,
        bracket_y,
        bracket_h,
        f"{format_p(p_val)}, r = {r:.3f}",
    )

    if star:
        ax.text(
            1.5,
            bracket_y + bracket_h * 1.4,
            star,
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    # 凡例（PA=色、PT=形）
    leg1 = ax.legend(
        handles=pa_legend_handles,
        title="pa_id (color)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=pt_legend_handles,
        title="pt_id (marker)",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.30),
        borderaxespad=0.0,
    )

    plt.tight_layout()
    out_png = OUT_DIR / f"box_{col}.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_png}")

    # 結果
    results.append(
        {
            "variable": col,
            "threshold": THRESHOLD,
            "n_<": int(n_lo),
            "n_>=": int(n_hi),
            "median_<": float(np.median(g_lo.values)),
            "median_>=": float(np.median(g_hi.values)),
            "U_statistic": float(u_stat),
            "Z_value": float(z),
            "p_value": float(p_val),
            "rank_biserial_r": float(r),
        }
    )

# =========================
# 保存（CSV）
# =========================
res_df = pd.DataFrame(results).sort_values("p_value")
res_df.to_csv(OUT_CSV, index=False)
print(f"[SAVED] {OUT_CSV}")
print(res_df)
