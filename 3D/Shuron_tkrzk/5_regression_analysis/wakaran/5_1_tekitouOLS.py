#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PA歩行指標（目的変数）を、PT介助指標＋基礎情報で回帰するスクリプト
ただやろうとすると特徴量pよりもサンプル数nが少なくてうまくいかない（n<p）

入力（固定パス）:
- G:\gait_pattern\2025_shuron_tkrzk\regression\pa_gait_parameters.csv
- G:\gait_pattern\2025_shuron_tkrzk\regression\pa_basic_data.csv
- G:\gait_pattern\2025_shuron_tkrzk\regression\pt_basic_data.csv
- G:\gait_pattern\2025_shuron_tkrzk\regression\pt_assist_parameters.csv
出力（固定フォルダ）:
- reg_OLS_tekitou/results_summary.txt
- reg_OLS_tekitou/coefficients_model1.csv
- reg_OLS_tekitou/coefficients_model2.csv
- reg_OLS_tekitou/merged_dataset_used_model2.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# =========================
# ここだけ編集すればOK
# =========================
ROOT_DIR = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression")

PA_GAIT_CSV   = ROOT_DIR / "pa_gait_parameters.csv"
PA_BASIC_CSV  = ROOT_DIR / "pa_basic_data.csv"
PT_BASIC_CSV  = ROOT_DIR / "pt_basic_data.csv"
PT_ASSIST_CSV = ROOT_DIR / "pt_assist_parameters.csv"

OUTDIR = ROOT_DIR / "reg_OLS_tekitou"

# 目的変数（例: 最小介助時からの歩行速度変化量）
Y_COL = "gait_speed_delta"

# 数値説明変数だけ標準化するか（係数比較しやすい）
STANDARDIZE_X_NUMERIC = True

# 階層回帰:
# Model1 = 基礎情報のみ
# Model2 = Model1 + 介助指標（hip_dist, hip_cc_*）
ASSIST_COLS = ["hip_dist", "hip_cc_x", "hip_cc_y", "hip_cc_z", "hip_cc_lag"]
# =========================


def read_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df

def dedupe_keep_order(cols: list[str]) -> list[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def standardize_numeric_only(X: pd.DataFrame) -> pd.DataFrame:
    Xs = X.copy()
    for c in Xs.columns:
        if pd.api.types.is_numeric_dtype(Xs[c]):
            mu = Xs[c].mean()
            sd = Xs[c].std(ddof=0)
            if sd == 0 or np.isnan(sd):
                continue
            Xs[c] = (Xs[c] - mu) / sd
    return Xs


def make_design_matrix(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    standardize_x_numeric: bool = False,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    returns:
      y: Series(float)
      X: DataFrame (dummy-coded + const + (optional) standardized numeric)
      used: DataFrame (y/x only, after listwise deletion)
    """
    missing = [c for c in [y_col] + x_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in merged df: {missing}")

    used = df[[y_col] + x_cols].copy().dropna(axis=0, how="any")

    y = used[y_col].astype(float)

    X_raw = used[x_cols].copy()

    # boolは0/1へ
    for c in X_raw.columns:
        if pd.api.types.is_bool_dtype(X_raw[c]):
            X_raw[c] = X_raw[c].astype(int)

    # カテゴリ/文字列はダミー化
    X = pd.get_dummies(X_raw, drop_first=True)

    # 定数列は落とす
    constant_like = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_like:
        warnings.warn(f"Dropping constant columns: {constant_like}")
        X = X.drop(columns=constant_like)

    if standardize_x_numeric:
        X = standardize_numeric_only(X)

    X = sm.add_constant(X, has_constant="add")
    return y, X, used


def fit_ols(y: pd.Series, X: pd.DataFrame):
    return sm.OLS(y, X).fit()


def coef_table(model) -> pd.DataFrame:
    ci = model.conf_int()
    out = pd.DataFrame(
        {
            "coef": model.params,
            "std_err": model.bse,
            "t": model.tvalues,
            "p": model.pvalues,
            "ci_low": ci[0],
            "ci_high": ci[1],
        }
    )
    out.index.name = "term"
    return out.reset_index()


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # 1) load
    pa_gait = read_csv_safely(PA_GAIT_CSV)
    pa_basic = read_csv_safely(PA_BASIC_CSV)
    pt_basic = read_csv_safely(PT_BASIC_CSV)
    pt_assist = read_csv_safely(PT_ASSIST_CSV)

    # 2) validate keys
    for key in ["pa_id", "pt_id"]:
        if key not in pa_gait.columns:
            raise KeyError(f"{PA_GAIT_CSV} must contain '{key}'")
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

    # 4) choose X columns (existing only)

    # PA（患者）
    PA_REP_COLS_CANDIDATES = [
        "pa_age",
        "mi_total",
        "sias_m_total",
        "fim_motor",
        "mmse",
    ]

    # PT（療法士）
    PT_REP_COLS_CANDIDATES = [
        "pt_age",
        "pt_sex",
        "exp",
        "certified",
    ]

    # 実際に存在する列だけ採用
    X_base = [c for c in (PA_REP_COLS_CANDIDATES + PT_REP_COLS_CANDIDATES) if c in df.columns]
    X_base = dedupe_keep_order(X_base)

    # 介助指標（これは研究的に固定）
    X_assist = [c for c in ASSIST_COLS if c in df.columns]

    if len(X_base) == 0:
        warnings.warn("基礎情報が見つからないため、Model1は定数のみになります。")
    if len(X_assist) == 0:
        warnings.warn("介助指標が見つからないため、Model2はModel1と同じになります。")

    # 5) Model1: base covariates only (or const-only)
    X1_cols = X_base
    y1, X1, used1 = make_design_matrix(
        df=df, y_col=Y_COL, x_cols=X1_cols, standardize_x_numeric=STANDARDIZE_X_NUMERIC
    )
    model1 = fit_ols(y1, X1)

    # 6) Model2: base + assist
    X2_cols = dedupe_keep_order(X_base + X_assist)
    y2, X2, used2 = make_design_matrix(
        df=df, y_col=Y_COL, x_cols=X2_cols, standardize_x_numeric=STANDARDIZE_X_NUMERIC
    )
    model2 = fit_ols(y2, X2)

    # 7) outputs
    used2.to_csv(os.path.join(OUTDIR, "merged_dataset_used_model2.csv"), index=False)
    coef_table(model1).to_csv(os.path.join(OUTDIR, "coefficients_model1.csv"), index=False)
    coef_table(model2).to_csv(os.path.join(OUTDIR, "coefficients_model2.csv"), index=False)

    summary_path = os.path.join(OUTDIR, "results_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Model 1 (Base covariates) ===\n")
        f.write(model1.summary().as_text())
        f.write("\n\n")
        f.write("=== Model 2 (Base + Assist parameters) ===\n")
        f.write(model2.summary().as_text())
        f.write("\n\n")
        f.write("=== Fit metrics (note: n may differ due to missing) ===\n")
        f.write(f"Model1: n={int(model1.nobs)}, R2={model1.rsquared:.4f}, adjR2={model1.rsquared_adj:.4f}\n")
        f.write(f"Model2: n={int(model2.nobs)}, R2={model2.rsquared:.4f}, adjR2={model2.rsquared_adj:.4f}\n")
        f.write(f"Delta R2 (Model2-Model1): {(model2.rsquared - model1.rsquared):.4f}\n")

    print("[OK] Finished.")
    print(f"Output folder: {os.path.abspath(OUTDIR)}")
    print(f"- {os.path.abspath(summary_path)}")


if __name__ == "__main__":
    main()
