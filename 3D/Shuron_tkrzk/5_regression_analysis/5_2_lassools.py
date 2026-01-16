#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
regression_dataset.csv + regression_groups.json を入力にして
LassoCVで変数選択 → 選択変数のみでOLS（statsmodels）を実行する。

- 目的変数: speed, speed_delta（jsonのspeedグループから自動）
- 説明変数: PT_assist / PA_gait / PA_basic / PT_basic を組み合わせて複数モデルを回す
- 毎回、出力はタイムスタンプ付き新規フォルダ（過去結果を参照しない）

出力:
  reg_LassoOLS_grouped/YYYYMMDD_HHMMSS/
    summary_models.csv
    lasso_select_long.csv
    ols_coef_long.csv
    adoption_rate_<group>.csv
    各モデルの *_ols_summary.txt, *_ols_coef.csv, *_lasso_select_long.csv
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


# ===== 入出力（あなたの環境に合わせて固定）=====
INDIR = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression")
DATA_CSV = INDIR / "regression_dataset.csv"
GROUPS_JSON = INDIR / "regression_groups.json"

BASE_OUTDIR = INDIR / "reg_LassoOLS_grouped"
# ===============================================

# ===== Lasso設定 =====
RANDOM_STATE = 0
LASSO_CV = 3
LASSO_MAX_ITER = 200000
COEF_ZERO_TOL = 1e-12

# 強制的に常に入れたい説明変数があれば
FORCE_INCLUDE = [
    # "hip_dist",
]

# p_keptがこれ以上だとOLSが不安定になりやすいので制限（必要なら変更）
MAX_OLS_P = 8


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _sanitize(s: str) -> str:
    s = re.sub(r"\s+", "_", str(s))
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "tag"

def load_groups() -> dict:
    with open(GROUPS_JSON, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j["groups"]

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

def lasso_then_ols(df: pd.DataFrame, y_col: str, x_cols: list[str], tag: str, outdir: Path):
    x_cols = [c for c in x_cols if c in df.columns and c != y_col]
    if (y_col not in df.columns) or len(x_cols) == 0:
        return None, None, None

    # 欠損除去（このモデルで使う列のみ）
    d = df[[y_col] + x_cols].dropna(axis=0, how="any").reset_index(drop=True)
    n_used = len(d)
    if n_used < 8:
        print(f"SKIP {tag}: too few rows (n={n_used})")
        return None, None, None

    d = coerce_numeric(d, [y_col] + x_cols).dropna(axis=0, how="any").reset_index(drop=True)
    n_used = len(d)
    if n_used < 8:
        print(f"SKIP {tag}: too few rows after numeric (n={n_used})")
        return None, None, None

    y = d[y_col].to_numpy(dtype=float)
    X_raw = d[x_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    lasso = LassoCV(
        cv=min(LASSO_CV, n_used),
        random_state=RANDOM_STATE,
        max_iter=LASSO_MAX_ITER,
        fit_intercept=True,
        alphas=100,
    )
    lasso.fit(X, y)

    coef = lasso.coef_
    intercept = float(lasso.intercept_)
    alpha = float(lasso.alpha_)
    r2_train = float(lasso.score(X, y))

    kept = (np.abs(coef) > COEF_ZERO_TOL)
    kept_cols = [c for c, m in zip(x_cols, kept) if m]

    for c in FORCE_INCLUDE:
        if c in x_cols and c not in kept_cols:
            kept_cols.append(c)
    kept_cols = list(dict.fromkeys(kept_cols))

    # OLSの次元制限（n_usedに対して多すぎると破綻しやすい）
    if len(kept_cols) > MAX_OLS_P:
        # |coef|の大きい順に上位だけ使う（Lasso係数で順位付け）
        coef_abs = [(c, abs(v)) for c, v in zip(x_cols, coef) if c in kept_cols]
        coef_abs.sort(key=lambda x: x[1], reverse=True)
        kept_cols = [c for c, _ in coef_abs[:MAX_OLS_P]]

    p_kept = len(kept_cols)

    # Lasso long 出力（モデル単体）
    lasso_rows = [{
        "tag": tag, "y": y_col, "x": "const",
        "coef": intercept, "alpha": alpha, "r2_train": r2_train,
        "kept": 1, "n_used": n_used, "p_all": len(x_cols), "p_kept": p_kept
    }]
    for c, v in zip(x_cols, coef):
        lasso_rows.append({
            "tag": tag, "y": y_col, "x": c,
            "coef": float(v), "alpha": alpha, "r2_train": r2_train,
            "kept": int(c in kept_cols), "n_used": n_used, "p_all": len(x_cols), "p_kept": p_kept
        })
    lasso_df = pd.DataFrame(lasso_rows)
    lasso_df.to_csv(outdir / f"{_sanitize(tag)}_lasso_select_long.csv", index=False, encoding="utf-8-sig")

    if p_kept == 0:
        print(f"SKIP {tag}: lasso selected 0 (alpha={alpha})")
        return None, None, lasso_df

    # ======================
    # 非標準化OLS（解釈用）
    # ======================
    X_ols = sm.add_constant(d[kept_cols], has_constant="add")
    model = sm.OLS(y, X_ols).fit()

    # ======================
    # 標準化OLS（真の標準化β）
    # ======================
    X_std = StandardScaler().fit_transform(d[kept_cols])
    y_std = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

    X_std = pd.DataFrame(X_std, columns=kept_cols)
    X_std = sm.add_constant(X_std, has_constant="add")

    model_std = sm.OLS(y_std, X_std).fit()

    ols_std_df = pd.DataFrame({
        "tag": tag,
        "y": y_col,
        "x": model_std.params.index.astype(str),
        "coef_std": model_std.params.to_numpy(),
        "p_std": model_std.pvalues.to_numpy(),
    })

    conf = model.conf_int()
    ols_df = pd.DataFrame({
        "tag": tag,
        "y": y_col,
        "x": model.params.index.astype(str),
        "coef": model.params.to_numpy(),
        "p": model.pvalues.to_numpy(),
        "ci_low": conf[0].to_numpy(),
        "ci_high": conf[1].to_numpy(),
        "ols_rsq_adj": float(model.rsquared_adj),
        "ols_rsq": float(model.rsquared),
        "ols_aic": float(model.aic),
        "ols_bic": float(model.bic),
        "ols_f_pvalue": float(model.f_pvalue) if np.isfinite(model.f_pvalue) else np.nan,
        "alpha": alpha,
        "r2_lasso_train": r2_train,
        "p_all": len(x_cols),
        "p_kept": p_kept,
        "n_used": n_used,
    })
    
    ols_df = ols_df.merge(ols_std_df,on=["tag", "y", "x"],how="left")

    ols_df.to_csv(outdir / f"{_sanitize(tag)}_ols_coef.csv", index=False, encoding="utf-8-sig")
    with open(outdir / f"{_sanitize(tag)}_ols_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(model.summary()))
        
    with open(outdir / f"{_sanitize(tag)}_ols_summary_std.txt", "w", encoding="utf-8") as f:
        f.write(str(model_std.summary()))

    summary = {
        "tag": tag, "y": y_col,
        "n_used": int(n_used),
        "p_all": int(len(x_cols)),
        "p_kept": int(p_kept),
        "alpha": alpha,
        "r2_lasso_train": r2_train,
        "ols_rsq": float(model.rsquared),
        "ols_rsq_adj": float(model.rsquared_adj),
        "ols_f_pvalue": float(model.f_pvalue) if np.isfinite(model.f_pvalue) else np.nan,
    }

    print(f"OK {tag}: p_kept={p_kept}, ols_rsq_adj={model.rsquared_adj:.3f}")
    return summary, ols_df, lasso_df


def adoption_rate(lasso_long: pd.DataFrame, groups: dict, group_name: str) -> pd.DataFrame:
    d = lasso_long.copy()
    d = d[d["x"] != "const"]
    group_set = set(groups.get(group_name, []))
    sub = d[d["x"].isin(group_set)]
    if len(sub) == 0:
        return pd.DataFrame()

    den = sub.groupby("x")["tag"].nunique().rename("n_models_used")
    num = sub[sub["kept"] == 1].groupby("x")["tag"].nunique().rename("n_models_kept")
    out = pd.concat([den, num], axis=1).fillna(0)
    out["n_models_used"] = out["n_models_used"].astype(int)
    out["n_models_kept"] = out["n_models_kept"].astype(int)
    out["adoption_rate"] = out["n_models_kept"] / out["n_models_used"].replace(0, np.nan)
    out = out.reset_index().sort_values(["adoption_rate", "n_models_kept"], ascending=[False, False])
    return out


def main():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"not found: {DATA_CSV}")
    if not GROUPS_JSON.exists():
        raise FileNotFoundError(f"not found: {GROUPS_JSON}")

    groups = load_groups()
    df = pd.read_csv(DATA_CSV)

    # 出力フォルダ（毎回新規）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = BASE_OUTDIR / ts
    _mkdir(outdir)

    # 目的変数
    y_list = [c for c in groups.get("speed", []) if c in df.columns]
    if len(y_list) == 0:
        raise KeyError("speed group is empty in json or columns missing.")

    # 説明変数グループ
    X_PT = [c for c in groups.get("PT_assist", []) if c in df.columns]
    X_PG = [c for c in groups.get("PA_gait", []) if c in df.columns]
    X_PB = [c for c in groups.get("PA_basic", []) if c in df.columns]
    X_TB = [c for c in groups.get("PT_basic", []) if c in df.columns]

    model_specs = [
        ("PTassist", X_PT),
        ("PTassist+PAgait", X_PT + X_PG),
        ("PTassist+PAbasic", X_PT + X_PB),
        ("PTassist+PTbasic", X_PT + X_TB),
        ("PTassist+PAgait+PAbasic", X_PT + X_PG + X_PB),
        ("PTassist+PAgait+PAbasic+PTbasic", X_PT + X_PG + X_PB + X_TB),
    ]

    summaries, all_ols, all_lasso = [], [], []

    for y in y_list:
        for name, xcols in model_specs:
            xcols = list(dict.fromkeys([c for c in xcols if c != y]))
            tag = f"y_{y}__{name}"
            summary, ols_df, lasso_df = lasso_then_ols(df, y, xcols, tag, outdir)
            if summary is not None:
                summaries.append(summary)
            if ols_df is not None:
                all_ols.append(ols_df)
            if lasso_df is not None:
                all_lasso.append(lasso_df)

    if summaries:
        pd.DataFrame(summaries).to_csv(outdir / "summary_models.csv", index=False, encoding="utf-8-sig")
    if all_ols:
        pd.concat(all_ols, ignore_index=True).to_csv(outdir / "ols_coef_long.csv", index=False, encoding="utf-8-sig")
    if all_lasso:
        lasso_long = pd.concat(all_lasso, ignore_index=True)
        lasso_long.to_csv(outdir / "lasso_select_long.csv", index=False, encoding="utf-8-sig")

        for gname in ["PT_assist", "PA_gait", "PA_basic", "PT_basic"]:
            rate = adoption_rate(lasso_long, groups, gname)
            if len(rate) > 0:
                rate.to_csv(outdir / f"adoption_rate_{gname}.csv", index=False, encoding="utf-8-sig")

    print(f"done. output: {outdir}")


if __name__ == "__main__":
    main()
