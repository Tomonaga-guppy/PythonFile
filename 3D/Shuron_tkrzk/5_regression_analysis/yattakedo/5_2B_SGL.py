#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5_2B_SGL_fixed.py
=================
Sparse Group Lasso (SGL) 回帰 with LOOCV + GridSearch
- group_lasso のバージョン差（scale_l2_by の有無）を吸収
- 出力: G:\\gait_pattern\\2025_shuron_tkrzk\\regression\\reg_SGL\\YYYYMMDD_HHMMSS\\...

Install:
  (.easyvit_env) pip install group-lasso
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# SGL (group-lasso)
# -------------------------
try:
    from group_lasso import GroupLasso
except Exception as e:
    raise ImportError(
        "GroupLasso が import できませんでした。\n"
        "`.easyvit_env` で `pip install group-lasso` を実行してください。\n"
        f"Original error: {e}"
    )


# =========================
# 1) 設定（ここだけ触ればOK）
# =========================
ROOT_DIR = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression")
IN_DATASET = ROOT_DIR / "regression_dataset.csv"
IN_GROUPS  = ROOT_DIR / "regression_groups.json"

OUT_PARENT = ROOT_DIR / "reg_SGL"

# 目的変数（必要に応じて増やす）
Y_COLS = [
    "speed_delta",
    "speed",
    # "stride_width",
]

# どのグループを説明変数に含めるか（モデル比較）
MODEL_SPECS: Dict[str, List[str]] = {
    "PTassist": ["PT_assist"],
    "PTassist_PT_basic": ["PT_assist", "PT_basic"],
    "PTassist_PAbasic_PT_basic": ["PT_assist", "PA_basic", "PT_basic"],
}

# 欠損の扱い
DROPNA_STRATEGY = "drop"  # "drop" or "impute_mean"

# ハイパラ探索
GRID_group_reg = [0.01, 0.1, 0.5]
GRID_l1_reg    = [0.01, 0.1, 0.5]

# 収束設定（ConvergenceWarning が多いなら増やす）
MAX_ITER = 5000
TOL = 1e-6

# 係数ゼロ判定閾値（数値誤差用）
COEF_EPS = 1e-8

# 警告をうるさくしない（必要なら False に）
SUPPRESS_FISTA_WARNING = True


# =========================
# 2) ユーティリティ
# =========================
def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out


def load_groups_json(p: Path) -> Dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_X_y(
    df: pd.DataFrame,
    groups_def: Dict,
    y_col: str,
    use_group_names: List[str],
) -> Tuple[pd.DataFrame, pd.Series, List[int], List[str]]:
    """
    groups_def 例:
      {
        "id_cols": ["pa_id","pt_id"],
        "groups": {"PT_assist":[...], "PA_gait":[...], ...}
      }
    """
    gdict: Dict[str, List[str]] = groups_def["groups"]

    X_cols: List[str] = []
    group_ids: List[int] = []
    group_name_of_col: List[str] = []

    gid = 1
    for gname in use_group_names:
        cols = [c for c in gdict.get(gname, []) if c in df.columns]
        for c in cols:
            X_cols.append(c)
            group_ids.append(gid)
            group_name_of_col.append(gname)
        gid += 1

    if len(X_cols) == 0:
        raise ValueError(f"X columns are empty. use_group_names={use_group_names}")
    if y_col not in df.columns:
        raise KeyError(f"y_col not found in dataset: {y_col}")

    X = df[X_cols].copy()
    y = df[y_col].copy()
    return X, y, group_ids, group_name_of_col


def handle_missing(X: pd.DataFrame, y: pd.Series, strategy: str) -> Tuple[pd.DataFrame, pd.Series]:
    if strategy == "drop":
        m = X.notna().all(axis=1) & y.notna()
        return X.loc[m].reset_index(drop=True), y.loc[m].reset_index(drop=True)

    if strategy == "impute_mean":
        X2 = X.copy()
        for c in X2.columns:
            if X2[c].isna().any():
                X2[c] = X2[c].fillna(X2[c].mean())
        y2 = y.copy()
        if y2.isna().any():
            y2 = y2.fillna(y2.mean())
        return X2.reset_index(drop=True), y2.reset_index(drop=True)

    raise ValueError(f"Unknown missing strategy: {strategy}")


def make_model(groups: np.ndarray, group_reg: float, l1_reg: float) -> GroupLasso:
    """
    group_lasso のバージョン差を吸収して GroupLasso を作る．
    - scale_l2_by が無い版でも落ちない
    """
    common_kwargs = dict(
        groups=groups,
        group_reg=group_reg,
        l1_reg=l1_reg,
        n_iter=MAX_ITER,
        tol=TOL,
        fit_intercept=True,
        # NOTE: ライブラリ側の綴りが supress_warning のことが多い
        supress_warning=True,
    )

    # scale_l2_by が通る版
    try:
        return GroupLasso(**common_kwargs, scale_l2_by=None)
    except TypeError:
        return GroupLasso(**common_kwargs)


@dataclass
class CVResult:
    group_reg: float
    l1_reg: float
    mae: float
    rmse: float
    r2: float
    y_true: np.ndarray
    y_pred: np.ndarray


def loocv_score_sgl(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    group_reg: float,
    l1_reg: float,
) -> CVResult:
    loo = LeaveOneOut()
    y_true_all: List[float] = []
    y_pred_all: List[float] = []

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # 標準化（リーク防止で train fit）
        sx = StandardScaler()
        X_tr_s = sx.fit_transform(X_tr)
        X_te_s = sx.transform(X_te)

        # y も標準化（ペナルティ安定化）
        sy = StandardScaler()
        y_tr_s = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()

        model = make_model(groups=groups, group_reg=group_reg, l1_reg=l1_reg)

        if SUPPRESS_FISTA_WARNING:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr_s, y_tr_s)
        else:
            model.fit(X_tr_s, y_tr_s)

        pred_s = np.asarray(model.predict(X_te_s)).ravel()
        pred = sy.inverse_transform(pred_s.reshape(-1, 1)).ravel()

        y_true_all.append(float(y_te.item()))
        y_pred_all.append(float(pred.item()))

    y_true = np.array(y_true_all, dtype=float)
    y_pred = np.array(y_pred_all, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return CVResult(
        group_reg=group_reg,
        l1_reg=l1_reg,
        mae=mae,
        rmse=rmse,
        r2=r2,
        y_true=y_true,
        y_pred=y_pred,
    )


def fit_final_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    group_reg: float,
    l1_reg: float,
) -> Tuple[GroupLasso, StandardScaler, StandardScaler]:
    sx = StandardScaler()
    Xs = sx.fit_transform(X)

    sy = StandardScaler()
    ys = sy.fit_transform(y.reshape(-1, 1)).ravel()

    model = make_model(groups=groups, group_reg=group_reg, l1_reg=l1_reg)

    if SUPPRESS_FISTA_WARNING:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Xs, ys)
    else:
        model.fit(Xs, ys)

    return model, sx, sy


def save_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_png: Path, title: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =========================
# 3) メイン
# =========================
def main() -> None:
    if not IN_DATASET.exists():
        raise FileNotFoundError(f"not found: {IN_DATASET}")
    if not IN_GROUPS.exists():
        raise FileNotFoundError(f"not found: {IN_GROUPS}")

    out_dir = OUT_PARENT / now_stamp()
    safe_mkdir(out_dir)

    # 入力
    df = pd.read_csv(IN_DATASET)
    df = to_numeric_df(df)
    groups_def = load_groups_json(IN_GROUPS)

    # 設定保存
    config = {
        "IN_DATASET": str(IN_DATASET),
        "IN_GROUPS": str(IN_GROUPS),
        "Y_COLS": Y_COLS,
        "MODEL_SPECS": MODEL_SPECS,
        "DROPNA_STRATEGY": DROPNA_STRATEGY,
        "GRID_group_reg": GRID_group_reg,
        "GRID_l1_reg": GRID_l1_reg,
        "MAX_ITER": MAX_ITER,
        "TOL": TOL,
        "COEF_EPS": COEF_EPS,
        "SUPPRESS_FISTA_WARNING": SUPPRESS_FISTA_WARNING,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    all_summary_rows = []

    for y_col in Y_COLS:
        for spec_name, use_groups in MODEL_SPECS.items():
            run_name = f"y_{y_col}__{spec_name}"
            print("=" * 90)
            print(f"[RUN] {run_name}  (X groups={use_groups})")
            print("=" * 90)

            X_df, y_ser, group_ids, group_name_of_col = build_X_y(
                df=df,
                groups_def=groups_def,
                y_col=y_col,
                use_group_names=use_groups,
            )
            X_df, y_ser = handle_missing(X_df, y_ser, DROPNA_STRATEGY)

            n = len(y_ser)
            p = X_df.shape[1]
            if n < 5 or p == 0:
                print(f"SKIP {run_name}: n={n}, p={p}")
                continue

            X = X_df.to_numpy(dtype=float)
            y = y_ser.to_numpy(dtype=float)
            groups_arr = np.array(group_ids, dtype=int)

            # LOOCV GridSearch
            cv_rows = []
            best: Optional[CVResult] = None

            for gr in GRID_group_reg:
                for l1 in GRID_l1_reg:
                    res = loocv_score_sgl(
                        X=X,
                        y=y,
                        groups=groups_arr,
                        group_reg=gr,
                        l1_reg=l1,
                    )
                    cv_rows.append({
                        "group_reg": gr,
                        "l1_reg": l1,
                        "MAE": res.mae,
                        "RMSE": res.rmse,
                        "R2": res.r2,
                    })
                    if best is None or res.mae < best.mae:
                        best = res

            assert best is not None
            cv_df = (
                pd.DataFrame(cv_rows)
                .sort_values(["MAE", "RMSE"], ascending=[True, True])
                .reset_index(drop=True)
            )

            # 最良ハイパラで fit
            model, sx, sy = fit_final_model(
                X=X,
                y=y,
                groups=groups_arr,
                group_reg=best.group_reg,
                l1_reg=best.l1_reg,
            )

            coef = np.asarray(model.coef_, dtype=float).ravel()
            intercept = float(np.asarray(model.intercept_).ravel()[0])  # DeprecationWarning 対策

            coef_df = pd.DataFrame({
                "feature": X_df.columns.tolist(),
                "group_name": group_name_of_col,
                "group_id": group_ids,
                "coef_std_space": coef,
                "abs_coef": np.abs(coef),
                "selected": (np.abs(coef) > COEF_EPS),
            }).sort_values(["selected", "abs_coef"], ascending=[False, False]).reset_index(drop=True)

            selected_df = coef_df[coef_df["selected"]].copy()

            pred_df = pd.DataFrame({
                "y_true": best.y_true,
                "y_pred": best.y_pred,
                "error": best.y_pred - best.y_true,
                "abs_error": np.abs(best.y_pred - best.y_true),
            })

            # 保存
            run_dir = out_dir / run_name
            safe_mkdir(run_dir)

            cv_df.to_csv(run_dir / "cv_results.csv", index=False, encoding="utf-8-sig")
            coef_df.to_csv(run_dir / "coef.csv", index=False, encoding="utf-8-sig")
            selected_df.to_csv(run_dir / "selected_features.csv", index=False, encoding="utf-8-sig")
            pred_df.to_csv(run_dir / "pred_loocv.csv", index=False, encoding="utf-8-sig")

            with open(run_dir / "best_params.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_group_reg": best.group_reg,
                        "best_l1_reg": best.l1_reg,
                        "LOOCV_MAE": best.mae,
                        "LOOCV_RMSE": best.rmse,
                        "LOOCV_R2": best.r2,
                        "n_samples": int(n),
                        "n_features": int(p),
                        "intercept_std_space": intercept,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            save_scatter(
                y_true=best.y_true,
                y_pred=best.y_pred,
                out_png=run_dir / "scatter_true_pred.png",
                title=f"{run_name} (LOOCV)  MAE={best.mae:.3g}, R2={best.r2:.3g}",
            )

            all_summary_rows.append({
                "run": run_name,
                "y": y_col,
                "spec": spec_name,
                "X_groups": "+".join(use_groups),
                "n": int(n),
                "p": int(p),
                "best_group_reg": best.group_reg,
                "best_l1_reg": best.l1_reg,
                "MAE": best.mae,
                "RMSE": best.rmse,
                "R2": best.r2,
                "n_selected": int(selected_df.shape[0]),
                "selected_groups": ",".join(sorted(selected_df["group_name"].unique().tolist())) if selected_df.shape[0] else "",
            })

            print(f"[OK] {run_name}")
            print(f"  best: group_reg={best.group_reg}, l1_reg={best.l1_reg}")
            print(f"  LOOCV: MAE={best.mae:.4f}, RMSE={best.rmse:.4f}, R2={best.r2:.4f}")
            print(f"  selected: {int(selected_df.shape[0])} / {p}")
            if selected_df.shape[0] > 0:
                print("  top selected:")
                print(selected_df.head(10)[["feature", "group_name", "coef_std_space"]].to_string(index=False))

    # 全体サマリ
    if len(all_summary_rows) > 0:
        summary_df = (
            pd.DataFrame(all_summary_rows)
            .sort_values(["y", "MAE"], ascending=[True, True])
            .reset_index(drop=True)
        )
        summary_df.to_csv(out_dir / "summary_all_runs.csv", index=False, encoding="utf-8-sig")

        plt.figure(figsize=(10, 5))
        plt.scatter(np.arange(len(summary_df)), summary_df["MAE"].to_numpy())
        plt.xticks(np.arange(len(summary_df)), summary_df["run"].tolist(), rotation=90)
        plt.ylabel("MAE (LOOCV)")
        plt.tight_layout()
        plt.savefig(out_dir / "MAE_all_runs.png", dpi=200)
        plt.close()

    print("\nDONE")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
