#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sparse Group Lasso (SGL) - ハードコーディング版

入力:
- G:\\gait_pattern\\2025_shuron_tkrzk\\regression\\reg_Ridge_all\\merged_dataset_used.csv
  ※あなたがRidgeで作った「欠損除去済み」データをそのまま利用

出力:
- G:\\gait_pattern\\2025_shuron_tkrzk\\regression\\reg_SGL\\sgl_best_params.txt
- G:\\gait_pattern\\2025_shuron_tkrzk\\regression\\reg_SGL\\sgl_coefficients.csv
- G:\\gait_pattern\\2025_shuron_tkrzk\\regression\\reg_SGL\\sgl_group_summary.csv
- G:\\gait_pattern\\2025_shuron_tkrzk\\regression\\reg_SGL\\sgl_predictions_loocv.csv

注意:
- nが小さいため、CVはLOOCV、指標はMSE/MAE中心で見る（R2は補助）
- SGLは "どのグループが効いたか"+"その中でどの変数が効いたか" を説明しやすい
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ====== ここはあなたの環境に固定 ======
IN_CSV = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression\reg_Ridge_all\merged_dataset_used.csv")
OUTDIR = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression\reg_SGL")

Y_COL = "gait_speed_delta"

# 目的変数の計算に使っていてリークっぽい列があればここに追加して除外
EXCLUDE_COLS = [
    # 例: "gait_speed"  # gait_speed_delta が速度差分なら、gait_speedは除外推奨
]

# グリッド（まずは粗め。n=10なので探索しすぎない）
GROUP_REG_GRID = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0]
L1_REG_GRID    = [0.0,   0.001,0.01,0.03,0.1, 0.3, 1.0]

RANDOM_SEED = 0
# =====================================

def ensure_group_lasso_available():
    try:
        # 多くの環境ではこっち
        from group_lasso import GroupLasso  # noqa
        return "GroupLasso"
    except Exception as e1:
        try:
            # 万一SparseGroupLassoが存在する環境もあるかもしれないので保険
            from group_lasso import SparseGroupLasso  # noqa
            return "SparseGroupLasso"
        except Exception as e2:
            msg = (
                "\n[ERROR] group_lasso が import できません。\n"
                "以下でインストール/更新を試してください:\n\n"
                "  C:/Users/Tomson/StrokeProject/easy_ViTPose/.easyvit_env/Scripts/python.exe -m pip install -U group-lasso\n\n"
                "確認:\n"
                "  C:/Users/Tomson/StrokeProject/easy_ViTPose/.easyvit_env/Scripts/python.exe -m pip show group-lasso\n"
            )
            raise RuntimeError(msg) from e2


def assign_group(feature_name: str) -> str:
    """
    変数名からグループ名を推定（ルールは必要に応じて増やしてOK）
    優先順位: 重症度 -> 介助 -> PT基礎 -> PA基礎 -> 歩行アウトカム/歩行指標 -> その他
    """
    n = feature_name.lower()

    # 重症度・評価スケール（患者）
    severity_keys = ["fac", "brs", "sias", "mi_", "fma", "nihss", "mmt"]
    if any(k in n for k in severity_keys):
        return "patient_severity"

    # 介助指標
    assist_keys = ["hip_cc", "hip_dist", "assist", "sync", "distance", "corr"]
    if any(k in n for k in assist_keys):
        return "assist_features"

    # PT基礎
    if n.startswith("pt_") or "therapist" in n or "exp" in n or "cert" in n:
        return "therapist_basic"

    # PA基礎
    if n.startswith("pa_") or "patient" in n:
        return "patient_basic"

    # 歩行系（歩行指標やアウトカム）
    gait_keys = [
        "gait", "stride", "step", "cadence", "symmetry", "swing", "stance",
        "speed", "width", "length", "time"
    ]
    if any(k in n for k in gait_keys):
        return "gait_features"

    return "other"


def main():
    klass = ensure_group_lasso_available()

    if klass == "GroupLasso":
        from group_lasso import GroupLasso as SGLModel
    else:
        from group_lasso import SparseGroupLasso as SGLModel

    os.makedirs(OUTDIR, exist_ok=True)

    if not IN_CSV.exists():
        raise FileNotFoundError(f"Input not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    if Y_COL not in df.columns:
        raise KeyError(f"'{Y_COL}' が見つかりません。CSV列名を確認してください。")

    # 説明変数候補（目的変数除外 + EXCLUDE）
    X_cols = [c for c in df.columns if c != Y_COL and c not in EXCLUDE_COLS]
    if len(X_cols) == 0:
        raise ValueError("説明変数が0です。")

    # ダミー化（カテゴリがあれば）
    X_raw = df[X_cols]
    X_df = pd.get_dummies(X_raw, drop_first=True)

    # 定数列は除外
    constant_like = [c for c in X_df.columns if X_df[c].nunique(dropna=True) <= 1]
    if constant_like:
        X_df = X_df.drop(columns=constant_like)

    feature_names = list(X_df.columns)
    y = df[Y_COL].astype(float).to_numpy()

    # グループ割当
    group_names = [assign_group(f) for f in feature_names]
    unique_groups = sorted(list(dict.fromkeys(group_names)))  # order-preserving unique
    group_to_id = {g:i for i, g in enumerate(unique_groups)}
    groups = np.array([group_to_id[g] for g in group_names], dtype=int)

    # 標準化（SGLでも重要）
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.to_numpy(dtype=float))

    loo = LeaveOneOut()

    # ===== グリッドサーチ（LOOCVでMSE最小） =====
    best = None
    best_info = None

    for group_reg in GROUP_REG_GRID:
        for l1_reg in L1_REG_GRID:
            preds = []
            trues = []

            for train_idx, test_idx in loo.split(X):
                Xtr, Xte = X[train_idx], X[test_idx]
                ytr, yte = y[train_idx], y[test_idx]

                model = SGLModel(
                    groups=groups,
                    group_reg=group_reg,
                    l1_reg=l1_reg,
                    n_iter=5000,
                    tol=1e-5,
                    scale_reg="inverse_group_size",
                    fit_intercept=True,
                    random_state=RANDOM_SEED,
                    supress_warning=True,
                )
                model.fit(Xtr, ytr)
                yhat = float(np.asarray(model.predict(Xte)).ravel()[0])

                preds.append(float(yhat))
                trues.append(float(yte[0]))

            mse = mean_squared_error(trues, preds)
            mae = mean_absolute_error(trues, preds)
            r2  = r2_score(trues, preds)  # LOOCVの全予測に対するR2（参考）

            if (best is None) or (mse < best):
                best = mse
                best_info = {
                    "group_reg": group_reg,
                    "l1_reg": l1_reg,
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                }

    # ===== 最良パラメータで全データ学習 =====
    best_group_reg = best_info["group_reg"]
    best_l1_reg = best_info["l1_reg"]

    final_model = SGLModel(
        groups=groups,
        group_reg=best_group_reg,
        l1_reg=best_l1_reg,
        n_iter=10000,
        tol=1e-6,
        scale_reg="inverse_group_size",
        fit_intercept=True,
        random_state=RANDOM_SEED,
        supress_warning=True,
    )
    final_model.fit(X, y)

    coef = np.asarray(final_model.coef_).reshape(-1)   # 1次元に潰す
    intercept = float(np.asarray(final_model.intercept_).reshape(-1)[0])  # スカラ化

    # ===== LOOCV予測（最良パラメータ固定） =====
    preds = []
    trues = []

    for train_idx, test_idx in loo.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        m = SGLModel(
            groups=groups,
            group_reg=best_group_reg,
            l1_reg=best_l1_reg,
            n_iter=10000,
            tol=1e-6,
            scale_reg="inverse_group_size",
            fit_intercept=True,
            random_state=RANDOM_SEED,
            supress_warning=True,
        )

        m.fit(Xtr, ytr)

        yhat = float(np.asarray(m.predict(Xte)).reshape(-1)[0])

        preds.append(yhat)
        trues.append(float(yte[0]))


    loocv_mse = mean_squared_error(trues, preds)
    loocv_mae = mean_absolute_error(trues, preds)
    loocv_r2  = r2_score(trues, preds)

    # ===== 出力: best params =====
    with open(OUTDIR / "sgl_best_params.txt", "w", encoding="utf-8") as f:
        f.write("Sparse Group Lasso (SGL)\n")
        f.write(f"target={Y_COL}\n")
        f.write(f"n_samples={len(y)}\n")
        f.write(f"n_features={len(feature_names)}\n")
        f.write(f"n_groups={len(unique_groups)}\n\n")
        f.write("[Best params by LOOCV MSE]\n")
        f.write(f"group_reg={best_group_reg}\n")
        f.write(f"l1_reg={best_l1_reg}\n")
        f.write(f"grid_best_mse={best_info['mse']:.6f}\n")
        f.write(f"grid_best_mae={best_info['mae']:.6f}\n")
        f.write(f"grid_best_r2 ={best_info['r2']:.6f}\n\n")
        f.write("[LOOCV metrics with best params]\n")
        f.write(f"LOOCV_MSE={loocv_mse:.6f}\n")
        f.write(f"LOOCV_MAE={loocv_mae:.6f}\n")
        f.write(f"LOOCV_R2 ={loocv_r2:.6f}\n\n")
        f.write(f"intercept={intercept}\n")

    # ===== 出力: 係数 =====
    coef_df = pd.DataFrame({
        "variable": feature_names,
        "group": [assign_group(v) for v in feature_names],
        "coef": coef,
        "abs_coef": np.abs(coef),
        "selected": (np.abs(coef) > 1e-12),
    }).sort_values("abs_coef", ascending=False)

    coef_df.to_csv(OUTDIR / "sgl_coefficients.csv", index=False)

    # ===== 出力: グループ要約 =====
    group_summary = []
    for g in unique_groups:
        idx = [i for i, gg in enumerate(coef_df["group"].tolist()) if gg == g]
        # coef_dfは並び替え済みなので、元feature順のidxが欲しい
    # 元feature順で計算する
    coef_by_feature = pd.DataFrame({
        "variable": feature_names,
        "group": [assign_group(v) for v in feature_names],
        "coef": coef,
    })
    for g in unique_groups:
        sub = coef_by_feature[coef_by_feature["group"] == g]
        l2 = float(np.sqrt(np.sum(sub["coef"].to_numpy() ** 2)))
        n_sel = int(np.sum(np.abs(sub["coef"].to_numpy()) > 1e-12))
        group_summary.append({
            "group": g,
            "group_id": group_to_id[g],
            "l2_norm": l2,
            "n_features": int(len(sub)),
            "n_selected": n_sel,
            "selected": (n_sel > 0),
        })

    group_df = pd.DataFrame(group_summary).sort_values("l2_norm", ascending=False)
    group_df.to_csv(OUTDIR / "sgl_group_summary.csv", index=False)

    # ===== 出力: LOOCV予測 =====
    pred_df = pd.DataFrame({
        "y_true": trues,
        "y_pred": preds,
        "error": np.array(preds) - np.array(trues),
    })
    pred_df.to_csv(OUTDIR / "sgl_predictions_loocv.csv", index=False)

    print("[OK] Sparse Group Lasso finished.")
    print(f"Output folder: {OUTDIR}")
    print(f"- {OUTDIR / 'sgl_best_params.txt'}")
    print(f"- {OUTDIR / 'sgl_coefficients.csv'}")
    print(f"- {OUTDIR / 'sgl_group_summary.csv'}")
    print(f"- {OUTDIR / 'sgl_predictions_loocv.csv'}")
    print(f"best group_reg={best_group_reg}, l1_reg={best_l1_reg}, LOOCV_R2={loocv_r2:.3f}, MAE={loocv_mae:.3f}")


if __name__ == "__main__":
    main()
