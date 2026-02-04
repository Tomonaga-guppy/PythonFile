"""
5_3_ridge_multi_y.py
====================
- regression_dataset.csv から X を作り
- y は (1) *_delta 列を自動検出、無ければ (2) *delta を探す（例: speed_delta）
- RidgeCVでalpha選択（全データで一度だけ）
- そのalpha固定でLOOCV予測 -> MAE/RMSE を算出（R^2は使わない）
- yごとの係数（標準化空間）と summary をCSV保存

ポイント
- LOOCVのR^2はtestが1点なので未定義 -> MAE/RMSE推奨
- 入れ子CV（cross_val_scoreの中でGridSearch等）は避ける（重くて止まりやすい）
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# # ============================================================
# # 0) 警告を抑える（デバッグ出力を見やすく）
# # ============================================================
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)

# # sklearnの UndefinedMetricWarning（R^2など）も抑制したい場合
# try:
#     from sklearn.exceptions import UndefinedMetricWarning
#     warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# except Exception:
#     pass


# ============================================================
# 1) パス
# ============================================================
ROOT_DIR = Path(r"G:\gait_pattern\2025_shuron_tkrzk")
REG_DIR = ROOT_DIR / "regression"
DATASET_PATH = REG_DIR / "regression_dataset.csv"

OUT_DIR = REG_DIR / "ridge_multi_y"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2) X列のセット（必要に応じて編集）
#    例: PT_assistのみ / PT_assist+PAgait / +basic など
# ============================================================
# X_COLS = [
#     "hip_dist",
#     "wri_para_s",
#     "wri_nonpara_s",
#     "cos_sim",
#     "hip_cc_x",
#     "hip_cc_y",
#     "hip_cc_z",
#     "hip_cc_lag",
#     # ここから必要なら追加（regression_dataset.csvに存在する列名に合わせる）
#     # "SI_sw", "stride_time", ...
# ]

X_COLS = [
    # --- PT_assist ---
    "hip_dist",
    "wri_para_s",
    "wri_nonpara_s",
    "cos_sim",
    "hip_cc_x",
    "hip_cc_y",
    "hip_cc_z",
    "hip_cc_lag",

    # --- PA_gait（介助中の歩行状態）---
    "SI_sw",
    "stride_time",
    "stride_width",
    "hip_fl_max",
    "hip_ex_max",
    "kne_fl_max",
    "ank_do_max",
    "hip_ab_max",
]

# y候補のエイリアス（古い列名 -> 新しい列名）
Y_ALIASES = {
    "speed_delta": "gait_speed_delta",
    "symmetry_index_sw_delta": "symmetry_index_sw_delta",  # 例（同名ならそのまま）
    "stride_time_delta": "stride_time_delta",
    "stride_width_delta": "stride_width_delta",
}


# ============================================================
# 3) ヘルパ
# ============================================================
def detect_y_columns(df: pd.DataFrame) -> list[str]:
    """y列（delta系）を自動検出。*_delta が無ければ *delta を探す。"""
    y_cols = [c for c in df.columns if c.endswith("_delta")]
    if len(y_cols) > 0:
        return y_cols

    # fallback: "delta" を含む列
    y_cols = [c for c in df.columns if "delta" in c.lower()]
    return y_cols


def loo_predict_with_fixed_alpha(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """alpha固定でLOOCV予測（標準化込み）"""
    loo = LeaveOneOut()
    preds = np.full_like(y, fill_value=np.nan, dtype=float)

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        # fold内で標準化（リーク防止）
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = Ridge(alpha=alpha)
        model.fit(X_tr_s, y_tr)
        preds[test_idx[0]] = model.predict(X_te_s)[0]

    return preds


# ============================================================
# 4) メイン
# ============================================================
def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    print(f"[INFO] dataset: {DATASET_PATH}")
    print(f"[INFO] df shape: {df.shape}")

    # X列の存在チェック
    missing_x = [c for c in X_COLS if c not in df.columns]
    if missing_x:
        print("[WARN] Missing X columns:", missing_x)
        print("[INFO] Available columns example:", df.columns.tolist()[:30])
        # ここで止めるかは好みだが、わかりやすく止める
        return

    # y列の自動検出
    y_cols = detect_y_columns(df)

    # エイリアス補完（speed_deltaしか無い等の救済）
    # 例: speed_delta があればそれをyとして使う
    # 例: gait_speed_delta があるならそれ優先
    fixed_y_cols = []
    for yc in y_cols:
        if yc in df.columns:
            fixed_y_cols.append(yc)

    # エイリアスで追加
    for old, new in Y_ALIASES.items():
        if new in df.columns and new not in fixed_y_cols:
            fixed_y_cols.append(new)
        elif old in df.columns and old not in fixed_y_cols and new not in df.columns:
            fixed_y_cols.append(old)

    # 重複除去（順序保持）
    seen = set()
    y_cols_final = []
    for c in fixed_y_cols:
        if c not in seen:
            seen.add(c)
            y_cols_final.append(c)

    if len(y_cols_final) == 0:
        print("[ERROR] delta系のy列が見つかりませんでした．")
        print("[INFO] df columns with 'delta':", [c for c in df.columns if "delta" in c.lower()])
        return

    print("[INFO] X cols:", X_COLS)
    print("[INFO] y cols:", y_cols_final)

    summary_rows = []

    # Xの生配列
    X_all = df[X_COLS].to_numpy(dtype=float)

    for y_col in y_cols_final:
        if y_col not in df.columns:
            print(f"[SKIP] {y_col} not found")
            continue

        y_series = df[y_col]
        valid_mask = y_series.notna()

        X = X_all[valid_mask.values]
        y = y_series[valid_mask].to_numpy(dtype=float)

        n = len(y)
        if n < 3:
            print(f"[SKIP] {y_col}: too few samples (n={n})")
            continue

        # -----------------------------
        # (A) alpha選択：全データで一度だけ（LOOでneg-MAE最大化）
        # -----------------------------
        alphas = np.logspace(-4, 4, 200)

        # 標準化 + RidgeCV
        ridgecv = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("ridgecv", RidgeCV(alphas=alphas, cv=LeaveOneOut(), scoring="neg_mean_absolute_error"))
        ])
        ridgecv.fit(X, y)

        best_alpha = float(ridgecv.named_steps["ridgecv"].alpha_)

        # -----------------------------
        # (B) そのalpha固定でLOOCV予測 -> MAE/RMSE
        # -----------------------------
        preds = loo_predict_with_fixed_alpha(X, y, alpha=best_alpha)

        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))

        # -----------------------------
        # (C) 全データで係数（標準化空間）
        # -----------------------------
        # 標準化して Ridge(alpha=best_alpha) をfit
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        ridge = Ridge(alpha=best_alpha)
        ridge.fit(Xs, y)

        coef = ridge.coef_.ravel()
        coef_df = pd.DataFrame({
            "feature": X_COLS,
            "coef_std": coef,
            "abs_coef_std": np.abs(coef),
        }).sort_values("abs_coef_std", ascending=False)

        coef_path = OUT_DIR / f"coef__{y_col}.csv"
        coef_df.to_csv(coef_path, index=False)

        print("\n==================================================")
        print(f"[Y] {y_col} (n={n})")
        print(f"best_alpha: {best_alpha:.6g}")
        print(f"MAE : {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"coef saved: {coef_path}")
        print("top5 coef:")
        print(coef_df.head(5).to_string(index=False))

        summary_rows.append({
            "y": y_col,
            "n": n,
            "best_alpha": best_alpha,
            "MAE": mae,
            "RMSE": rmse,
        })

    if len(summary_rows) == 0:
        print("[ERROR] 有効なyが無く，summaryが空でした．")
        return

    summary_df = pd.DataFrame(summary_rows).sort_values("MAE")
    summary_path = OUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n===== DONE =====")
    print(summary_df.to_string(index=False))
    print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()
