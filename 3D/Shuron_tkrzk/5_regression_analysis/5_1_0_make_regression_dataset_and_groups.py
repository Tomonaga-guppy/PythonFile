#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
regressionフォルダの4CSVから、解析用データセットを作成し、
指定のグループ（speed / PA歩行 / PT介助 / PA基本 / PT基本）で列を管理できるようにする。

入力（同一フォルダ想定）:
- pa_gait_parameters.csv
- pt_assist_parameters.csv
- pa_basic_data.csv
- pt_basic_data.csv

出力（同一フォルダに保存）:
- regression_dataset.csv            : マージ済みデータ

使い方（例）:
python 0_make_regression_dataset_and_groups.py --indir G:\\...\\regression
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ===== ユーザー指定のグループ（列名） =====
GROUPS = {
    "PA_gait": [
        "speed_ori",
        "SI_sw_ori",
        "stride_time_ori",
        "stride_width_ori",
        "hip_fl_max_ori",
        "hip_ex_max_ori",
        "kne_fl_max_ori",
        "ank_do_max_ori",
        "hip_ab_max_ori",
        "speed",
        "SI_sw",
        "stride_time",
        "stride_width",
        "hip_fl_max",
        "hip_ex_max",
        "kne_fl_max",
        "ank_do_max",
        "hip_ab_max",
        "speed_delta",
        "SI_sw_delta",
        "stride_time_delta",
        "stride_width_delta",
        "hip_ex_max_delta",
        "kne_fl_max_delta",
        "ank_do_max_delta",
        "hip_ab_max_delta",
    ],
    "PT_assist": [
        "hip_dist",
        "hip_dist_n",
        "wri_para_s",
        "wri_nonpara_s",
        "cos_sim",
        "hip_cc_x",
        "hip_cc_y",
        "hip_cc_z",
        "hip_cc_3d",
        "hip_cc_lag",
    ],
    "PA_basic": [
        "pa_age",
        "pa_height",
        "pa_weight",
        "fac",
        "brs_lower",
        "sias_m_hip",
        "sias_m_knee",
        "sias_m_ankle",
        "sias_m_total",
        "sias_sens_sole",
        "sias_prop_toe",
        "mi_hip",
        "mi_knee",
        "mi_ankle",
        "mi_total",
        "days_post_onset",
        "fim_walk",
        "fim_motor",
        "fim_cog",
        "mmse",
    ],
    "PT_basic": [
        "pt_age",
        "pt_height",
        "pt_weight",
        "grip_power",
        "exp",
    ],
}

ID_COLS = ["pa_id", "pt_id"]

# ===== 入力CSVの列名を統一するためのrename =====
RENAME_PA_GAIT = {
    "gait_speed": "speed",
    "gait_speed_ori": "speed_ori",
    "gait_speed_delta": "speed_delta",

    "symmetry_index_sw": "SI_sw",
    "symmetry_index_sw_ori": "SI_sw_ori",
    "symmetry_index_sw_delta": "SI_sw_delta",

    "stride_time": "stride_time",
    "stride_time_ori": "stride_time_ori",
    "stride_time_delta": "stride_time_delta",

    "stride_width": "stride_width",
    "stride_width_ori": "stride_width_ori",
    "stride_width_delta": "stride_width_delta",

    "hip_max_ext": "hip_ex_max",
    "hip_max_ext_ori": "hip_ex_max_ori",
    "hip_max_ext_delta": "hip_ex_max_delta",

    "knee_max_flex": "kne_fl_max",
    "knee_max_flex_ori": "kne_fl_max_ori",
    "knee_max_flex_delta": "kne_fl_max_delta",

    "ankle_max_do": "ank_do_max",
    "ankle_max_do_ori": "ank_do_max_ori",
    "ankle_max_do_delta": "ank_do_max_delta",

    "hip_max_ab": "hip_ab_max",
    "hip_max_ab_ori": "hip_ab_max_ori",
    "hip_max_ab_delta": "hip_ab_max_delta",
}

def read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"not found: {p}")
    return pd.read_csv(p)

def ensure_ids(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in ID_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: missing id cols {missing} (need {ID_COLS})")


def main():
    ROOT_DIR = Path(r"G:\gait_pattern\2025_shuron_tkrzk\regression")

    # 入力
    pa_gait = read_csv(ROOT_DIR / "pa_gait_parameters.csv").rename(columns=RENAME_PA_GAIT)
    pt_assist = read_csv(ROOT_DIR / "pt_assist_parameters.csv")
    pa_basic = read_csv(ROOT_DIR / "pa_basic_data.csv")
    pt_basic = read_csv(ROOT_DIR / "pt_basic_data.csv")

    # ID確認
    ensure_ids(pa_gait, "pa_gait_parameters.csv")
    ensure_ids(pt_assist, "pt_assist_parameters.csv")
    if "pa_id" not in pa_basic.columns:
        raise KeyError("pa_basic_data.csv: missing 'pa_id'")
    if "pt_id" not in pt_basic.columns:
        raise KeyError("pt_basic_data.csv: missing 'pt_id'")

    # マージ（pa_id, pt_idがある表同士）
    df = pd.merge(pa_gait, pt_assist, on=ID_COLS, how="inner", validate="one_to_one")
    df = pd.merge(df, pa_basic, on="pa_id", how="left", validate="many_to_one")
    df = pd.merge(df, pt_basic, on="pt_id", how="left", validate="many_to_one")

    # グループ列の存在チェック（無い列は落とす：列名揺れがあっても止めない）
    available = set(df.columns)
    groups_available = {}
    for gname, cols in GROUPS.items():
        cols2 = [c for c in cols if c in available]
        groups_available[gname] = cols2

    # 出力：グループ順に並べた列（ID→speed→PT_assist→PA_gait→PA_basic→PT_basic→その他）
    ordered = []
    ordered += [c for c in ID_COLS if c in df.columns]
    for g in ["PT_assist", "PA_gait", "PA_basic", "PT_basic"]:
        ordered += groups_available.get(g, [])
    ordered = list(dict.fromkeys(ordered))

    other = [c for c in df.columns if c not in ordered]
    df_ordered = df[ordered + other].copy()
    
    print(f"df_ordered = {df_ordered}")

    # 保存
    out_csv = ROOT_DIR / "regression_dataset.csv"
    df_ordered.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 画面表示
    print("OK")
    print(f"- saved: {out_csv}")
    print("\n[groups]")
    for k, v in groups_available.items():
        print(f"{k}: {len(v)} cols -> {v}")

if __name__ == "__main__":
    main()
