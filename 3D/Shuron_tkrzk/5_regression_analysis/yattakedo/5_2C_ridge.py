import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_score

import warnings
from sklearn.exceptions import UndefinedMetricWarning
# R^2が定義できない系の大量ワーニングだけ消す
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# RidgeCV内部の "test scores are non-finite" も消したいなら追加
warnings.filterwarnings("ignore", message="One or more of the test scores are non-finite*")

# ===============================
# 設定
# ===============================
ROOT = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression")
DATASET = ROOT / "regression_dataset.csv"
GROUPS_JSON = ROOT / "regression_groups.json"


BASE_OUTDIR = ROOT / "reg_Ridge"

TARGET = "speed_delta"   # 目的変数
USE_GROUPS = [
    "PT_assist",
    "PA_gait",
    "PA_basic",
    "PT_basic",
]

ALPHAS = np.logspace(-3, 3, 50)  # Ridgeの正則化強度

# ===============================
# load
# ===============================
df = pd.read_csv(DATASET)
with open(GROUPS_JSON, "r", encoding="utf-8") as f:
    groups = json.load(f)["groups"]

# ===============================
# X, y 作成
# ===============================
X_cols = []
for g in USE_GROUPS:
    X_cols += groups[g]

X_cols = [c for c in X_cols if c != TARGET]

X = df[X_cols].values
y = df[TARGET].values

print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
print(f"[INFO] X columns: {X_cols}")

print("X shape:", X.shape)
y_s = df[TARGET]   # pandas Series
print("y shape:", y_s.shape, " y non-NaN:", y_s.notna().sum())
print("rows after valid:", y_s.dropna().shape[0])

# ===============================
# Ridge + 標準化
# ===============================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=ALPHAS, cv=LeaveOneOut()))
])

# ===============================
# CV R^2
# ===============================
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring="r2")

print("\n===== Ridge CV result =====")
print(f"R^2 (mean) : {scores.mean():.3f}")
print(f"R^2 (std)  : {scores.std():.3f}")

# scores = cross_val_score(model, X, y, cv=loo, scoring="neg_mean_absolute_error")
# mae = -scores

# print("\n===== Ridge CV result =====")
# print(f"MAE (mean) : {mae.mean():.6f}")
# print(f"MAE (std)  : {mae.std():.6f}")

# ===============================
# Fit & 係数確認
# ===============================
model.fit(X, y)
ridge = model.named_steps["ridge"]

coef = ridge.coef_
coef_df = pd.DataFrame({
    "feature": X_cols,
    "coef": coef,
    "abs_coef": np.abs(coef)
}).sort_values("abs_coef", ascending=False)

print("\n===== Ridge coefficients (standardized) =====")
print(coef_df)

# 出力フォルダ（毎回新規）
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = BASE_OUTDIR / ts
outdir.mkdir(parents=True, exist_ok=True)
    
# 保存
coef_df.to_csv(outdir / "ridge_coefficients.csv", index=False)
