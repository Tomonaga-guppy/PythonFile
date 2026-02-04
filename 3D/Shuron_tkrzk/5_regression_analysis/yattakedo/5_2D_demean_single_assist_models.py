"""
============================================
患者固定（デミーン）したデータで、介助指標を1つずつ入れた単回帰をすべて実行し、
結果を1枚の表にまとめるスクリプト。

- 入力: regression_dataset.csv（ROOT_DIR 配下）
- 目的変数: YCOL
- 患者固定: PATIENT_COL ごとにデミーン（平均との差）
- 説明変数: ASSIST_COLS（"auto" の場合は ASSIST_GROUP_MAP から存在する列を拾う）
- 出力: OUT_DIR 配下
  - single_assist_models__summary.csv
  - single_assist_models__config.json

指標:
- 標準化β（デミーン後のx,yを全体でz-scoreしてOLS）
- 95%CI（患者クラスタ・ブートストラップ：患者ID単位で再標本化）
- Spearman ρ（参考） + 95%CI（同じブートストラップ）

注意:
- n が非常に小さいため、p値の解釈は推奨しません（探索的）。
- ブートストラップは「患者単位」で行い、患者内の条件点はまとめて扱います。

"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
except Exception as e:
    raise ImportError("statsmodels が必要です: pip install statsmodels") from e


# ============================
# 設定
# ============================
ROOT_DIR   = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression")
IN_DATASET = ROOT_DIR / "regression_dataset.csv"

# 出力フォルダ（好きに変更OK）
OUT_DIR    = ROOT_DIR / "reg_single_assist"
OUT_PREFIX = "single_assist_models"

# 目的変数と患者ID列
YCOL        = "speed_delta"
PATIENT_COL = "pa_id"

# 介助指標列（"auto" ならASSIST_GROUP_MAP順に存在する列を拾う）
ASSIST_COLS = "auto"

# ブートストラップ設定
N_BOOT = 5000
SEED   = 42

# 患者固定（デミーン）する列：YCOL + 介助指標
# （自動で組み立てるので、ここは触らなくてOK）


# ----------------------------
# 介助指標の概念グループ（表示用）
# ----------------------------
ASSIST_GROUP_MAP = {
    "position": ["hip_dist", "wri_para_s", "wri_nonpara_s"],
    "posture_sync": ["cos_sim"],
    "motion_sync": ["hip_cc_x", "hip_cc_y", "hip_cc_z", "hip_cc_3d", "hip_cc_lag"],
}
GROUP_ORDER = ["position", "posture_sync", "motion_sync"]


def zscore_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    m = np.nanmean(x)
    s = np.nanstd(x, ddof=0)
    if not np.isfinite(s) or s == 0:
        return x * np.nan
    return (x - m) / s


def demean_by_group(df: pd.DataFrame, cols: List[str], group_col: str) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[c] = out[c] - out.groupby(group_col)[c].transform("mean")
    return out


def assist_group_name(col: str) -> str:
    for g, cols in ASSIST_GROUP_MAP.items():
        if col in cols:
            return g
    return "other"


def cluster_bootstrap_patient(
    df: pd.DataFrame,
    patient_col: str,
    n_boot: int,
    seed: int,
) -> List[pd.DataFrame]:
    """
    患者ID単位で再標本化するクラスタ・ブートストラップ。
    返り値は resampled dataframe のリスト（長さ n_boot）。
    """
    rng = np.random.default_rng(seed)
    pats = df[patient_col].dropna().unique().tolist()
    if len(pats) == 0:
        return []
    out = []
    for _ in range(n_boot):
        sampled = rng.choice(pats, size=len(pats), replace=True)
        parts = []
        for pid in sampled:
            part = df[df[patient_col] == pid]
            parts.append(part)
        out.append(pd.concat(parts, ignore_index=True))
    return out


def fit_std_beta(x: np.ndarray, y: np.ndarray) -> float:
    """
    x, y は同じ長さの1D。NaN除外後、zscoreしてOLS（切片あり）し、標準化βを返す。
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    zx = zscore_nan(x[mask])
    zy = zscore_nan(y[mask])
    if not (np.isfinite(np.nanstd(zx)) and np.isfinite(np.nanstd(zy))):
        return np.nan
    X = sm.add_constant(zx, has_constant="add")
    model = sm.OLS(zy, X).fit()
    # 標準化後なので係数が標準化β
    return float(model.params[1])


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    rho, _ = stats.spearmanr(x[mask], y[mask])
    return float(rho)


def percentile_ci(vals: List[float], alpha: float = 0.05) -> Tuple[float, float]:
    a = np.asarray([v for v in vals if np.isfinite(v)], float)
    if len(a) == 0:
        return (np.nan, np.nan)
    lo = np.percentile(a, 100 * (alpha / 2))
    hi = np.percentile(a, 100 * (1 - alpha / 2))
    return (float(lo), float(hi))


def resolve_assist_cols(df: pd.DataFrame) -> List[str]:
    if isinstance(ASSIST_COLS, str) and ASSIST_COLS.strip().lower() == "auto":
        candidate: List[str] = []
        for g in GROUP_ORDER:
            candidate += ASSIST_GROUP_MAP[g]
        assist_cols = [c for c in candidate if c in df.columns]
        if len(assist_cols) == 0:
            assist_cols = [
                c for c in df.columns
                if any(k in c for k in ["hip_", "wri_", "cos_sim", "cc", "lag"])
            ]
        return assist_cols
    elif isinstance(ASSIST_COLS, (list, tuple)):
        return [c for c in ASSIST_COLS if c in df.columns]
    else:
        cols = [c.strip() for c in str(ASSIST_COLS).split(",") if c.strip()]
        return [c for c in cols if c in df.columns]


def main() -> None:
    if not IN_DATASET.exists():
        raise FileNotFoundError(f"not found: {IN_DATASET}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_DATASET)

    # 必須列チェック
    for c in [PATIENT_COL, YCOL]:
        if c not in df.columns:
            raise KeyError(f"missing column: {c} in {IN_DATASET}\navailable: {list(df.columns)}")

    assist_cols = resolve_assist_cols(df)
    if len(assist_cols) == 0:
        raise ValueError("assist_cols is empty. ASSIST_COLS を見直してください。")

    # デミーン（患者固定）
    cols_to_demean = [YCOL] + assist_cols
    df_dm = demean_by_group(df, cols_to_demean, PATIENT_COL)

    # ブートストラップ用に、必要列だけ
    use_cols = [PATIENT_COL, YCOL] + assist_cols
    df_dm_use = df_dm[use_cols].copy()

    boot_dfs = cluster_bootstrap_patient(df_dm_use, PATIENT_COL, N_BOOT, SEED)

    y = df_dm_use[YCOL].to_numpy(float)

    records = []
    for xcol in assist_cols:
        x = df_dm_use[xcol].to_numpy(float)

        beta = fit_std_beta(x, y)
        rho = spearman_rho(x, y)

        boot_beta = []
        boot_rho = []
        for bdf in boot_dfs:
            xb = bdf[xcol].to_numpy(float)
            yb = bdf[YCOL].to_numpy(float)
            boot_beta.append(fit_std_beta(xb, yb))
            boot_rho.append(spearman_rho(xb, yb))

        b_lo, b_hi = percentile_ci(boot_beta, alpha=0.05)
        r_lo, r_hi = percentile_ci(boot_rho, alpha=0.05)

        n_pair = int(np.sum(np.isfinite(x) & np.isfinite(y)))
        records.append(dict(
            x=xcol,
            assist_group=assist_group_name(xcol),
            n=n_pair,
            n_patients=int(df_dm_use[PATIENT_COL].nunique()),
            std_beta=beta,
            beta_ci_low=b_lo,
            beta_ci_high=b_hi,
            spearman_rho=rho,
            rho_ci_low=r_lo,
            rho_ci_high=r_hi,
        ))

    res = pd.DataFrame.from_records(records)

    # 表示用の並び
    res["group_order"] = res["assist_group"].apply(lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else 999)
    res["abs_beta"] = res["std_beta"].abs()
    res = res.sort_values(by=["group_order", "abs_beta"], ascending=[True, False]).reset_index(drop=True)
    res = res.drop(columns=["group_order"])

    # 本文で深掘りする代表2指標（abs(std_beta)上位、ただし概念グループ被り回避）
    picked = []
    used_groups = set()
    for _, row in res.sort_values("abs_beta", ascending=False).iterrows():
        g = row["assist_group"]
        if g in used_groups:
            continue
        picked.append(row["x"])
        used_groups.add(g)
        if len(picked) >= 2:
            break

    out_csv  = OUT_DIR / f"{OUT_PREFIX}__summary.csv"
    out_json = OUT_DIR / f"{OUT_PREFIX}__config.json"

    res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            dict(
                root_dir=str(ROOT_DIR),
                dataset=str(IN_DATASET),
                out_dir=str(OUT_DIR),
                ycol=YCOL,
                patient_col=PATIENT_COL,
                assist_cols=assist_cols,
                assist_group_map=ASSIST_GROUP_MAP,
                recommended_for_main_text=picked,
                n_boot=N_BOOT,
                seed=SEED,
                note="患者固定（デミーン）後に、介助指標を1つずつ入れた単回帰（標準化β）とSpearmanρを計算。CIは患者クラスタ・ブートストラップ（百分位）。",
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("OK")
    print(f"- input : {IN_DATASET}")
    print(f"- out   : {OUT_DIR}")
    print(f"- ycol  : {YCOL}")
    print(f"- assist: {assist_cols}")
    print(f"- recommended_for_main_text: {picked}")


if __name__ == "__main__":
    main()
