#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5_2C_ridge_demean.py

患者内 demean（within-patient）した Ridge 回帰
- 目的変数: speed_delta
- 説明変数: 介助指標 + 歩行指標（speed系は除外）
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_score

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="One or more of the test scores are non-finite*")


# ===============================
# 設定
# ===============================
ROOT = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression")
DATASET = ROOT / "regression_dataset.csv"
GROUPS_JSON = ROOT / "regression_groups.json"

BASE_OUTDIR = ROOT / "reg_Ridge_demean"

TARGET = "speed_delta"
PATIENT_ID = "pa_id"   # ← 患者ID列（要確認）

USE_GROUPS = [
    "PT_assist",
    # "PA_gait",
    # "PA_basic",
    "PT_basic",
]

ALPHAS = np.logspace(-3, 3, 50)


# ===============================
# load
# ===============================
df = pd.read_csv(DATASET)

with open(GROUPS_JSON, "r", encoding="utf-8") as f:
    groups = json.load(f)["groups"]


# ===============================
# X columns 作成（speed系除外）
# ===============================
X_cols = []
for g in USE_GROUPS:
    X_cols += groups[g]

X_cols = [c for c in X_cols if c != TARGET]


# ===============================
# demean（患者内）
# ===============================
demean_cols = X_cols + [TARGET]

df_demean = df.copy()
df_demean[demean_cols] = (
    df.groupby(PATIENT_ID)[demean_cols]
      .transform(lambda x: x - x.mean())
)

print("[INFO] demean applied within patient")


# ===============================
# X, y
# ===============================
X = df_demean[X_cols].values
y = df_demean[TARGET].values

print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
print(f"[INFO] X columns: {X_cols}")


# ===============================
# Ridge（※ 標準化なし）
# demeanしているので StandardScaler は不要
# ===============================
model = RidgeCV(
    alphas=ALPHAS,
    cv=LeaveOneOut()
)


# ===============================
# CV R^2
# ===============================
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring="r2")

print("\n===== Ridge demean CV result =====")
print(f"R^2 (mean) : {scores.mean():.3f}")
print(f"R^2 (std)  : {scores.std():.3f}")


# ===============================
# Fit & 係数
# ===============================
model.fit(X, y)

coef = model.coef_
coef_df = pd.DataFrame({
    "feature": X_cols,
    "coef": coef,
    "abs_coef": np.abs(coef)
}).sort_values("abs_coef", ascending=False)

print("\n===== Ridge coefficients (demeaned) =====")
print(coef_df)


# ===============================
# save
# ===============================
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = BASE_OUTDIR / ts
outdir.mkdir(parents=True, exist_ok=True)

coef_df.to_csv(outdir / "ridge_coefficients_demean.csv", index=False)

print(f"[INFO] saved to {outdir}")
