#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
5_3_ridge_multi_y_auto.py
========================
regression_dataset.csv と regression_groups.json を使い，
複数のXセット（グループ組み合わせ）× 複数のy（*_delta）を
Ridge回帰で一括評価する。

- Xセットごとに OUT_DIR/<X_SET_NAME>/ に保存
  - summary.csv
  - coef__<y>.csv
  - pred__<y>.csv（LOO予測; 任意で便利）

- 最後に OUT_DIR/_comparison_summary.csv に全条件まとめ表を保存

評価:
- alpha選択: KFold（最大5分割）で neg-MAE を最小化するalphaを選択（全データで1回）
- そのbest_alpha固定で LOOCV 予測 -> MAE/RMSE/corr

注意:
- N=10の小標本なので、結果解釈は「傾向・比較」が中心（p値議論はしない）
"""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, GridSearchCV, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 0) パス設定
# ============================================================
REG_DIR = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression")
DATASET_CSV = REG_DIR / "regression_dataset.csv"
GROUPS_JSON = REG_DIR / "regression_groups.json"

OUT_BASE = REG_DIR / "ridge_multi_y_auto"
OUT_BASE.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) Xセット定義（ここだけ触ればOK）
# ============================================================
X_GROUP_SETS = {
    # 主解析（おすすめ）
    "PT_only": ["PT_assist"],
    "PT_plus_PAgait": ["PT_assist", "PA_gait"],
    "PT_plus_PAgait_PAbasic": ["PT_assist", "PA_gait", "PA_basic"],

    # 比較・補足（入れておくと便利）
    "PAgait_only": ["PA_gait"],

    # フル（参考）
    "ALL": ["PT_assist", "PA_gait", "PA_basic", "PT_basic"],
}

# y は *_delta を自動検出（推奨）
AUTO_Y_DELTA = True

# 自動検出せず固定したい場合はこちらを使う（AUTO_Y_DELTA=Falseにする）
Y_DELTA_LIST = [
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

# alpha候補（小標本なので多すぎないのが無難）
ALPHAS = np.logspace(-4, 4, 41)

# 欠損がある行を落とす
DROP_NAN_ROWS = True

# LOO予測値CSVも保存するか（便利だがファイル数増える）
SAVE_PRED = True


# ============================================================
# 2) ヘルパ関数
# ============================================================
def load_groups(groups_json: Path) -> tuple[dict, list[str]]:
    with open(groups_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    groups = d["groups"]
    id_cols = d.get("id_cols", ["pa_id", "pt_id"])
    return groups, id_cols


def make_X_cols(groups: dict, group_names: list[str], df_cols: list[str]) -> list[str]:
    cols: list[str] = []
    for g in group_names:
        if g not in groups:
            raise KeyError(f"Group '{g}' not found in regression_groups.json")
        cols.extend(groups[g])
    # 重複除去（順序保持）
    cols = list(dict.fromkeys(cols))
    # 実在列だけに絞る
    cols = [c for c in cols if c in df_cols]
    return cols


def detect_y_delta_cols(df: pd.DataFrame, x_cols: list[str], id_cols: list[str]) -> list[str]:
    y_cols = [c for c in df.columns if c.endswith("_delta")]
    # idや説明変数と衝突しそうなものを除外
    y_cols = [c for c in y_cols if (c not in x_cols and c not in id_cols)]
    return y_cols


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def fit_and_eval_ridge(X: pd.DataFrame, y: pd.Series, alphas: np.ndarray) -> dict:
    """
    1) KFoldでalpha選択（neg-MAE最大化＝MAE最小）
    2) best_alpha固定でLOO予測 -> MAE/RMSE/corr
    3) 全データfitして標準化係数を返す
    """
    n = len(y)
    n_splits = min(5, n)
    if n_splits < 2:
        raise RuntimeError("Not enough samples for CV.")

    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    cv_alpha = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=base_pipe,
        param_grid={"ridge__alpha": alphas},
        scoring="neg_mean_absolute_error",
        cv=cv_alpha,
        n_jobs=1,
        refit=True,
    )
    gs.fit(X, y)
    best_alpha = float(gs.best_params_["ridge__alpha"])

    loo = LeaveOneOut()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha))
    ])
    y_pred = cross_val_predict(model, X, y, cv=loo, n_jobs=1)

    mae = float(mean_absolute_error(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    corr = safe_corr(np.asarray(y), np.asarray(y_pred))

    # 係数（標準化空間）
    model.fit(X, y)
    coef = model.named_steps["ridge"].coef_.ravel()

    return {
        "best_alpha": best_alpha,
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "coef": coef,
        "y_pred": y_pred,
    }


# ============================================================
# 3) メイン
# ============================================================
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Not found: {DATASET_CSV}")
    if not GROUPS_JSON.exists():
        raise FileNotFoundError(f"Not found: {GROUPS_JSON}")

    df = pd.read_csv(DATASET_CSV)
    groups, id_cols = load_groups(GROUPS_JSON)

    print(f"[INFO] dataset: {DATASET_CSV}")
    print(f"[INFO] df shape: {df.shape}")
    print(f"[INFO] output dir: {OUT_BASE}")

    all_summary_rows = []

    # ------------------------------------------------------------
    # Xセットごとに回す
    # ------------------------------------------------------------
    for x_set_name, x_group_list in X_GROUP_SETS.items():
        out_dir = OUT_BASE / x_set_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # X列を group から生成
        x_cols = make_X_cols(groups, x_group_list, df.columns.tolist())
        if len(x_cols) == 0:
            print(f"[SKIP] {x_set_name}: X cols empty")
            continue

        # y列決定
        if AUTO_Y_DELTA:
            y_cols = detect_y_delta_cols(df, x_cols, id_cols)
        else:
            y_cols = [c for c in Y_DELTA_LIST if c in df.columns]

        if len(y_cols) == 0:
            print(f"[SKIP] {x_set_name}: no y_delta columns")
            continue

        print("\n" + "=" * 70)
        print(f"[X SET] {x_set_name}: {x_group_list}")
        print(f"X cols ({len(x_cols)}): {x_cols}")
        print(f"y cols ({len(y_cols)}): {y_cols}")

        summary_rows = []

        for y_name in y_cols:
            X = df[x_cols].copy()
            y = df[y_name].copy()

            # 欠損除外
            if DROP_NAN_ROWS:
                valid_mask = X.notna().all(axis=1) & y.notna()
                X = X.loc[valid_mask].reset_index(drop=True)
                y = y.loc[valid_mask].reset_index(drop=True)

            if len(y) < 3:
                print(f"[SKIP] {x_set_name}/{y_name}: too few rows (n={len(y)})")
                continue

            res = fit_and_eval_ridge(X, y, ALPHAS)

            # 係数CSV
            coef_df = pd.DataFrame({
                "feature": x_cols,
                "coef_std": res["coef"],
                "abs_coef_std": np.abs(res["coef"]),
            }).sort_values("abs_coef_std", ascending=False)
            coef_path = out_dir / f"coef__{y_name}.csv"
            coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")

            # 予測CSV（任意）
            if SAVE_PRED:
                pred_df = pd.DataFrame({
                    "y_true": y.values,
                    "y_pred": res["y_pred"],
                    "error": (y.values - res["y_pred"]),
                })
                pred_path = out_dir / f"pred__{y_name}.csv"
                pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

            row = {
                "x_set": x_set_name,
                "y": y_name,
                "n": int(len(y)),
                "best_alpha": res["best_alpha"],
                "MAE": res["mae"],
                "RMSE": res["rmse"],
                "corr": res["corr"],
            }
            summary_rows.append(row)
            all_summary_rows.append(row)

            print(f"[DONE] {x_set_name}/{y_name}  n={len(y)}  alpha={res['best_alpha']:.6g}  MAE={res['mae']:.6g}")

        if len(summary_rows) == 0:
            print(f"[WARN] {x_set_name}: no y processed")
            continue

        # summary保存（Xセットごと）
        summary_df = pd.DataFrame(summary_rows).sort_values(["MAE", "y"], ascending=[True, True])
        summary_path = out_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        # 上位係数top10をまとめ（Xセットごと）
        top_rows = []
        for y_name in summary_df["y"].tolist():
            coef_df = pd.read_csv(out_dir / f"coef__{y_name}.csv")
            coef_df = coef_df.head(10).copy()
            coef_df.insert(0, "y", y_name)
            top_rows.append(coef_df)
        top10_df = pd.concat(top_rows, axis=0, ignore_index=True)
        top10_df.to_csv(out_dir / "coef_top10_all_y.csv", index=False, encoding="utf-8-sig")

        print(f"[SAVE] {summary_path}")
        print(f"[SAVE] {out_dir / 'coef_top10_all_y.csv'}")

    # ------------------------------------------------------------
    # 全条件まとめ比較表
    # ------------------------------------------------------------
    if len(all_summary_rows) == 0:
        print("[ERROR] no results.")
        return

    comp_df = pd.DataFrame(all_summary_rows)

    # まとめ表：MAEを条件×yでピボット（比較が超ラク）
    pivot_mae = comp_df.pivot_table(index="y", columns="x_set", values="MAE", aggfunc="mean")
    pivot_rmse = comp_df.pivot_table(index="y", columns="x_set", values="RMSE", aggfunc="mean")
    pivot_alpha = comp_df.pivot_table(index="y", columns="x_set", values="best_alpha", aggfunc="mean")

    comp_path = OUT_BASE / "_comparison_summary_long.csv"
    pivot_mae_path = OUT_BASE / "_comparison_MAE_pivot.csv"
    pivot_rmse_path = OUT_BASE / "_comparison_RMSE_pivot.csv"
    pivot_alpha_path = OUT_BASE / "_comparison_alpha_pivot.csv"

    comp_df.to_csv(comp_path, index=False, encoding="utf-8-sig")
    pivot_mae.to_csv(pivot_mae_path, encoding="utf-8-sig")
    pivot_rmse.to_csv(pivot_rmse_path, encoding="utf-8-sig")
    pivot_alpha.to_csv(pivot_alpha_path, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("[OK] ALL DONE. Saved comparison files:")
    print(f"- {comp_path}")
    print(f"- {pivot_mae_path}")
    print(f"- {pivot_rmse_path}")
    print(f"- {pivot_alpha_path}")


if __name__ == "__main__":
    main()
