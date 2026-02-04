"""
歩行速度の変化量とほかの歩行指標変化量との相関をプロット・計算
サンプル数が少ないのであくまで参考
- regression_dataset.csv を読み込み
- speed_delta vs (他の *_delta) の散布図を保存
- Spearman相関係数 rho と p値、ペア数 n を算出
- 結果CSVを保存
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
OUT_DIR  = CSV_PATH.parent / "correlation_analysis"

X_COL = "speed_delta"   # ← 横軸
Y_COLS = [
    "SI_sw_delta",
    "stride_time_delta",
    "stride_width_delta",
    "hip_fl_max_delta",
    "hip_ex_max_delta",
    "kne_fl_max_delta",
    "ank_do_max_delta",
    "hip_ab_max_delta",
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


def save_scatter(df, x_col, y_col, out_png, rho, p, n):
    xy = df[[x_col, y_col]].dropna()

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)

    # 散布図
    ax.scatter(xy[x_col], xy[y_col])

    # タイトル（中央）
    title_text = f"ΔSpeed vs {y_col}"
    ax.set_title(title_text)

    # 軸ラベル（単位つき）
    ax.set_xlabel(f"ΔSpeed [{UNITS.get(x_col, '-')}]")
    y_label_base = y_col.replace("_delta", "")
    ax.set_ylabel(f"{y_label_base} [{UNITS.get(y_col, '-')}]")

    # グリッド
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)

    # ===== タイトル横（figure座標）に rho / p =====
    txt = f"rho={rho:.3f}\np={p:.3g}"
    fig.text(
        0.78, 0.94,               # ← タイトル右横
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


# ============================================================
# 3) メイン処理
# ============================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_dir = OUT_DIR / "scatter"
    plot_dir.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # 列チェック
    for c in [X_COL] + Y_COLS:
        if c not in df.columns:
            raise ValueError(f"CSVに列がありません: {c}")

    results = []

    for y_col in Y_COLS:
        rho, p, n = safe_spearman(df[X_COL], df[y_col])

        results.append({
            "x": X_COL,
            "y": y_col,
            "n_pairwise": n,
            "spearman_rho": rho,
            "p_value": p,
        })

        if MAKE_PLOTS:
            out_png = plot_dir / f"{y_col}_vs_{X_COL}.png"
            save_scatter(df, X_COL, y_col, out_png, rho, p, n)

    res_df = pd.DataFrame(results).sort_values("p_value")
    out_csv = OUT_DIR / "spearman_speed_delta_vs_others.csv"
    res_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("=== DONE ===")
    print(f"CSV : {out_csv}")
    print(res_df)


if __name__ == "__main__":
    main()
