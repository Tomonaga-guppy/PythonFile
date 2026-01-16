#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ridge回帰（存在する指標はすべて使う / ハードコーディング版）

- 4つのCSVを merge
- 説明変数: 目的変数(Y_COL)とID列(pa_id, pt_id)を除く「存在する全列」
- カテゴリ列は get_dummies でダミー化
- 定数列（全員同じ値）は自動で除去
- RidgeCV + LOOCV（Leave-One-Out）で alpha を選択
- LOOCV予測の R2/RMSE/MAE を出力（過学習の目安）
- 係数（標準化後のスケール）を abs_coef 降順でCSV出力

注意:
- Ridgeはp値を出しません（係数の符号・大きさ・LOOCV指標で解釈）
- nが小さいので、係数は「傾向」として扱うのが安全
- 目的変数に近い派生量（リークになる列）が入っている可能性がある場合は除外推奨
  → 必要なら EXCLUDE_COLS に列名を追加
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =========================
# 固定パス（あなたの環境）
# =========================
ROOT_DIR = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression")

PA_GAIT_CSV   = ROOT_DIR / "pa_gait_parameters.csv"
PA_BASIC_CSV  = ROOT_DIR / "pa_basic_data.csv"
PT_BASIC_CSV  = ROOT_DIR / "pt_basic_data.csv"
PT_ASSIST_CSV = ROOT_DIR / "pt_assist_parameters.csv"

OUTDIR = ROOT_DIR / "reg_Ridge_all"

# 目的変数
Y_COL = "gait_speed_delta"

# 除外したい列があればここに追加（例：目的変数の別表現や、明らかなリーク列など）
EXCLUDE_COLS = [
    "gait_speed",  # gait_speed_deltaは自動で除外されるので不要
]

# alpha探索（罰則の強さ）
ALPHAS = np.logspace(-3, 3, 61)
# =========================


def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # 1) load
    pa_gait = read_csv_safely(PA_GAIT_CSV)
    pa_basic = read_csv_safely(PA_BASIC_CSV)
    pt_basic = read_csv_safely(PT_BASIC_CSV)
    pt_assist = read_csv_safely(PT_ASSIST_CSV)

    # 2) keys check
    for k in ["pa_id", "pt_id"]:
        if k not in pa_gait.columns:
            raise KeyError(f"{PA_GAIT_CSV} must contain '{k}'")
    if "pa_id" not in pa_basic.columns:
        raise KeyError(f"{PA_BASIC_CSV} must contain 'pa_id'")
    if "pt_id" not in pt_basic.columns:
        raise KeyError(f"{PT_BASIC_CSV} must contain 'pt_id'")
    if not set(["pa_id", "pt_id"]).issubset(pt_assist.columns):
        raise KeyError(f"{PT_ASSIST_CSV} must contain 'pa_id' and 'pt_id'")

    # 3) merge
    df = (
        pa_gait
        .merge(pa_basic, on="pa_id", how="left", suffixes=("", "_pa"))
        .merge(pt_basic, on="pt_id", how="left", suffixes=("", "_pt"))
        .merge(pt_assist, on=["pa_id", "pt_id"], how="left", suffixes=("", "_assist"))
    )

    if Y_COL not in df.columns:
        raise KeyError(f"目的変数 '{Y_COL}' が見つかりません。df.columns を確認してください。")

    # 4) choose ALL predictors (exist all) except IDs & target & excludes
    id_cols = ["pa_id", "pt_id"]
    exclude = set(id_cols + [Y_COL] + EXCLUDE_COLS)

    X_cols = [c for c in df.columns if c not in exclude]

    if len(X_cols) == 0:
        raise ValueError("説明変数が0です。除外設定を確認してください。")

    # 5) prepare dataset: drop missing rows on (Y + X)
    df_use = df[[Y_COL] + X_cols].copy().dropna(axis=0, how="any")
    df_use.to_csv(OUTDIR / "merged_dataset_used.csv", index=False)

    y = df_use[Y_COL].astype(float).to_numpy()

    # 6) dummy encode categorical columns
    X_raw = df_use[X_cols]
    X = pd.get_dummies(X_raw, drop_first=True)

    # 7) drop constant columns (all same)
    constant_like = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_like:
        X = X.drop(columns=constant_like)

    X_mat = X.to_numpy()
    feature_names = list(X.columns)

    if X_mat.shape[0] < 3:
        raise ValueError(f"有効サンプルが少なすぎます: n={X_mat.shape[0]}（欠損で落ちすぎ）")

    # 8) RidgeCV with LOOCV to choose alpha
    loo = LeaveOneOut()
    ridgecv = Pipeline([
        ("scaler", StandardScaler()),
        ("ridgecv", RidgeCV(alphas=ALPHAS, cv=loo, scoring="neg_mean_squared_error"))
    ])

    ridgecv.fit(X_mat, y)
    best_alpha = ridgecv.named_steps["ridgecv"].alpha_

    # 9) LOOCV prediction metrics (generalization-ish)
    ridge_fixed = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha))
    ])
    y_pred = cross_val_predict(ridge_fixed, X_mat, y, cv=loo)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    # 10) Fit final model on all data to extract coefficients
    ridge_fixed.fit(X_mat, y)
    coefs = ridge_fixed.named_steps["ridge"].coef_

    coef_df = pd.DataFrame({
        "variable": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    coef_df.to_csv(OUTDIR / "ridge_coefficients.csv", index=False)

    # 11) metrics output
    with open(OUTDIR / "ridge_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Ridge Regression (ALL available predictors)\n")
        f.write(f"target={Y_COL}\n")
        f.write(f"n_samples(after dropna)={len(y)}\n")
        f.write(f"n_features(after dummies & drop-constant)={X_mat.shape[1]}\n")
        f.write(f"best_alpha(LOOCV)={best_alpha}\n\n")
        f.write("LOOCV prediction metrics:\n")
        f.write(f"  R2   = {r2:.4f}\n")
        f.write(f"  RMSE = {rmse:.4f}\n")
        f.write(f"  MAE  = {mae:.4f}\n\n")
        f.write("Top 15 coefficients by |coef|:\n")
        f.write(coef_df.head(15).to_string(index=False))
        f.write("\n")

    print("[OK] Ridge (ALL predictors) finished.")
    print(f"Output folder: {OUTDIR}")
    print(f"- {OUTDIR / 'ridge_coefficients.csv'}")
    print(f"- {OUTDIR / 'ridge_metrics.txt'}")
    print(f"- {OUTDIR / 'merged_dataset_used.csv'}")
    print(f"best_alpha={best_alpha}, LOOCV_R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")


if __name__ == "__main__":
    main()
