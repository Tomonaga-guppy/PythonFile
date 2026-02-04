"""
pa_gait_parameters.csv を読み込み，
各変数について「ヒストグラム」と「Q-Qプロット」を別々に保存する。

- Histogram（人数分布）
- Q-Q plot（正規分布）
  左上に
    - Shapiro-Wilk p
    - Jarque-Bera p
  を枠付きで表示

実行:
  python plot_hist_qq_shapiro_jb.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import MaxNLocator

# ==========================
# ハードコーディング設定
# ==========================
CSV_PATH = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression\pa_gait_parameters.csv")
COLS = ["gait_speed", "gait_speed_delta"]

OUT_DIR = CSV_PATH.parent / "gait_speed_qq_normality"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY_NAME = {
    "gait_speed": "Speed",
    "gait_speed_delta": "ΔSpeed",
}
UNITS = {
    "gait_speed": "m/s",
    "gait_speed_delta": "m/s",
}

HIST_BINS = 6
# HIST_BINS = 18
GRID_KW = dict(linestyle="--", linewidth=0.8, alpha=0.4)


def clean_series(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise KeyError(f"Column not found: {col}\nAvailable: {list(df.columns)}")
    return pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)


def p_shapiro(x: np.ndarray) -> float | None:
    # Shapiro は n<3 で不可
    if len(x) < 3:
        return None
    return float(stats.shapiro(x).pvalue)


def p_jarque_bera(x: np.ndarray) -> float | None:
    # JB は歪度・尖度ベース。nが極端に小さいと不安定なのでガード（目安）
    if len(x) < 5:
        return None
    return float(stats.jarque_bera(x).pvalue)


def save_histogram(x: np.ndarray, col: str) -> Path:
    label = DISPLAY_NAME.get(col, col)
    unit = UNITS.get(col, "-")

    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)

    ax.hist(x, bins=HIST_BINS, alpha=0.75)
    ax.set_xlabel(f"{label} [{unit}]")
    ax.set_ylabel("Number of subjects [-]")
    ax.grid(True, **GRID_KW)
    
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    out = OUT_DIR / f"{col}__hist.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def save_qqplot(x: np.ndarray, col: str) -> Path:
    label = DISPLAY_NAME.get(col, col)
    unit = UNITS.get(col, "-")

    sh_p = p_shapiro(x)
    jb_p = p_jarque_bera(x)

    sh_txt = "NA" if sh_p is None else f"{sh_p:.3f}"
    jb_txt = "NA" if jb_p is None else f"{jb_p:.3f}"

    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)

    stats.probplot(x, dist="norm", plot=ax)

    ax.set_xlabel("Theoretical quantiles [-]")
    ax.set_ylabel(f"Ordered values of {label} [{unit}]")
    ax.grid(True, **GRID_KW)

    txt = f"Shapiro-Wilk p={sh_txt}\nJarque-Bera p={jb_txt}"
    ax.text(
        0.03, 0.97, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.95)
    )

    out = OUT_DIR / f"{col}__qq.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    for col in COLS:
        x = clean_series(df, col)

        out_hist = save_histogram(x, col)
        out_qq = save_qqplot(x, col)

        print("Saved:", out_hist)
        print("Saved:", out_qq)


if __name__ == "__main__":
    main()
