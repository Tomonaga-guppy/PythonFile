"""
rigid_fill_markers.py
=====================
Vicon/Nexus等のCSV (Multi-row header) から指定マーカーの3D座標を読み取り、
6点剛体（LASI,LPSI,LILC,RASI,RPSI,RILC）を用いて欠損を剛体補間します。

要件
- 6点すべて存在するフレーム群からロバストな剛体テンプレートを作成
- 各フレームで3点以上存在すれば、その点で剛体フィットして欠損点を復元
- 補間前後の座標を確認できるよう、差分レポートCSVを出力

出力（入力CSVと同じフォルダ）
- <stem>_rigidfilled.csv      : 対象6マーカーのみ（Frame,Time付き）の補間後座標
- <stem>_rigidfill_report.csv : 欠損が埋まった行だけ（frame, marker, before, after）
- <stem>_rigidfill_summary.txt: ざっくり統計
"""

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


MARKERS = ["LASI", "LPSI", "LILC", "RASI", "RPSI", "RILC"]
PREFIX  = "MarkerSet 01:"


# ----------------------------
# CSV（Vicon系 multi-row header）を読む
# ----------------------------
def read_vicon_like_csv(csv_path: Path):
    """
    Returns:
      df: pandas.DataFrame with columns:
        - Frame
        - Time (Seconds)
        - "<MarkerName>_<Axis>"  (Axis in X,Y,Z)
    """
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for r in reader:
            rows.append(r)

    # ヘッダ行探索: Axis行（Frame,Time (Seconds),X,Y,Z,...）を探す
    axis_row_idx = None
    for i, r in enumerate(rows[:50]):
        if len(r) >= 6 and r[0].strip() == "Frame" and "Time" in (r[1] if len(r) > 1 else ""):
            if any(x.strip() == "X" for x in r) and any(x.strip() == "Y" for x in r) and any(x.strip() == "Z" for x in r):
                axis_row_idx = i
                break
    if axis_row_idx is None:
        raise ValueError("Axis行（Frame, Time (Seconds), X,Y,Z...）が見つかりませんでした。")

    # マーカー名行探索（環境差に備えて上方向に探索）
    name_row_idx = None
    for j in range(max(0, axis_row_idx - 10), axis_row_idx):
        r = rows[j]
        if any(PREFIX in c for c in r):
            name_row_idx = j
            break
    if name_row_idx is None:
        raise ValueError("マーカー名行（MarkerSet 01:...）が見つかりませんでした。")

    name_row = rows[name_row_idx]
    axis_row = rows[axis_row_idx]

    max_len = max(len(name_row), len(axis_row))
    name_row = name_row + [""] * (max_len - len(name_row))
    axis_row = axis_row + [""] * (max_len - len(axis_row))

    cols = []
    for k in range(max_len):
        if k == 0:
            cols.append("Frame")
        elif k == 1:
            cols.append("Time (Seconds)")
        else:
            mname = name_row[k].strip()
            ax = axis_row[k].strip()
            if mname == "" and ax == "":
                cols.append(f"col{k}")
            elif mname == "":
                cols.append(f"col{k}_{ax}")
            else:
                cols.append(f"{mname}_{ax}")

    data_start = axis_row_idx + 1
    data_rows = []
    for r in rows[data_start:]:
        if len(r) == 0:
            continue
        if len(r) < max_len:
            r = r + [""] * (max_len - len(r))
        elif len(r) > max_len:
            r = r[:max_len]
        data_rows.append(r)

    df = pd.DataFrame(data_rows, columns=cols)

    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
    df["Time (Seconds)"] = pd.to_numeric(df["Time (Seconds)"], errors="coerce")

    for c in df.columns[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Frame"]).reset_index(drop=True)
    df["Frame"] = df["Frame"].astype(int)
    return df


def get_marker_xyz_cols(df: pd.DataFrame, marker: str):
    base = f"{PREFIX}{marker}_"
    cols = {ax: None for ax in ["X", "Y", "Z"]}
    for c in df.columns:
        if c.startswith(base):
            ax = c[len(base):].strip()
            if ax in cols:
                cols[ax] = c

    if any(cols[ax] is None for ax in cols):
        for ax in ["X", "Y", "Z"]:
            pat = re.compile(re.escape(f"{PREFIX}{marker}") + r".*_" + re.escape(ax) + r"$")
            for c in df.columns:
                if pat.search(c):
                    cols[ax] = c
                    break

    if any(cols[ax] is None for ax in cols):
        raise ValueError(f"{marker} のX/Y/Z列が見つかりませんでした: {cols}")
    return cols["X"], cols["Y"], cols["Z"]


# ----------------------------
# 剛体変換（Kabsch）+ ロバストIRLS
# ----------------------------
def rigid_transform_kabsch(A, B, w=None):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    N = A.shape[0]
    if w is None:
        w = np.ones(N, float)
    w = w / (np.sum(w) + 1e-12)

    a0 = np.sum(A * w[:, None], axis=0)
    b0 = np.sum(B * w[:, None], axis=0)
    Ac = A - a0
    Bc = B - b0

    H = (Ac * w[:, None]).T @ Bc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = b0 - R @ a0
    return R, t


def is_noncollinear(P, eps=1e-9):
    Pc = P - P.mean(axis=0, keepdims=True)
    return np.linalg.matrix_rank(Pc, tol=eps) >= 2


def robust_rigid_fit_irls(A, B, max_iter=20, huber_k=1.345, tol=1e-6):
    N = A.shape[0]
    w = np.ones(N, float)
    R, t = rigid_transform_kabsch(A, B, w=w)

    for _ in range(max_iter):
        B_hat = (R @ A.T).T + t
        r = np.linalg.norm(B - B_hat, axis=1)

        med = np.median(r)
        mad = np.median(np.abs(r - med)) + 1e-12
        sigma = 1.4826 * mad + 1e-12

        u = r / (sigma * huber_k)
        w_new = np.ones_like(u)
        mask = u > 1.0
        w_new[mask] = 1.0 / (u[mask] + 1e-12)
        w_new = np.clip(w_new, 1e-6, 1.0)

        R_new, t_new = rigid_transform_kabsch(A, B, w=w_new)
        dR = np.linalg.norm(R_new - R)
        dt = np.linalg.norm(t_new - t)
        R, t = R_new, t_new
        w = w_new
        if dR + dt < tol:
            break

    B_hat = (R @ A.T).T + t
    rms = np.sqrt(np.mean(np.sum((B - B_hat) ** 2, axis=1)))
    return R, t, w, rms


def build_template_from_full_frames(X_full, n_iter=10):
    Tpl = X_full[0].copy()
    for _ in range(n_iter):
        aligned = []
        for P in X_full:
            R, t = rigid_transform_kabsch(Tpl, P)
            P_al = (R.T @ (P - t).T).T
            aligned.append(P_al)
        aligned = np.stack(aligned, axis=0)
        Tpl_new = aligned.mean(axis=0)
        if np.linalg.norm(Tpl_new - Tpl) < 1e-6:
            Tpl = Tpl_new
            break
        Tpl = Tpl_new
    return Tpl


def rigid_body_fill(X, min_markers=3, rms_bad=None):
    X = np.asarray(X, float)
    T = X.shape[0]
    full_mask = ~np.isnan(X).any(axis=(1, 2))
    X_full = X[full_mask]
    if X_full.shape[0] < 3:
        raise ValueError("6点すべて存在するフレームが少なすぎます（最低でも数フレーム必要）")

    template = build_template_from_full_frames(X_full)

    X_filled = X.copy()
    rms_list = np.full(T, np.nan)
    used_counts = np.zeros(T, dtype=int)
    Rt_list = [None] * T

    for i in range(T):
        avail = ~np.isnan(X[i, :, 0])
        idx = np.where(avail)[0]
        used_counts[i] = len(idx)
        if len(idx) < min_markers:
            continue

        B = X[i, idx, :]
        if len(idx) >= 3 and (not is_noncollinear(B)):
            continue

        A = template[idx, :]
        R, t, w, rms = robust_rigid_fit_irls(A, B)
        rms_list[i] = rms
        if rms_bad is not None and rms > rms_bad:
            continue

        Rt_list[i] = (R, t)
        Xhat = (R @ template.T).T + t
        miss = ~avail
        X_filled[i, miss, :] = Xhat[miss, :]

    # 3点未満などは簡易補間（回転は近い方、並進は線形）
    good = np.array([rt is not None for rt in Rt_list], dtype=bool)
    good_idx = np.where(good)[0]
    if len(good_idx) >= 2:
        for i in range(T):
            if Rt_list[i] is not None:
                continue
            j_prev = good_idx[good_idx < i]
            j_next = good_idx[good_idx > i]
            if len(j_prev) == 0 or len(j_next) == 0:
                continue
            a = j_prev[-1]
            b = j_next[0]
            Ra, ta = Rt_list[a]
            Rb, tb = Rt_list[b]
            R = Ra if (i - a) <= (b - i) else Rb
            alpha = (i - a) / (b - a)
            t = (1 - alpha) * ta + alpha * tb

            Xhat = (R @ template.T).T + t
            avail = ~np.isnan(X[i, :, 0])
            miss = ~avail
            X_filled[i, miss, :] = Xhat[miss, :]

    info = dict(template=template, rms=rms_list, used_counts=used_counts, full_mask=full_mask)
    return X_filled, info


def main():
    # ★ここだけ自分の環境に合わせて固定パスを指定
    csv_path = Path(r"G:\gait_pattern\2025_shuron_BR9G\sub1\thera1-0\mocap\tst\1-1-0.csv")

    df = read_vicon_like_csv(csv_path)

    T = len(df)
    X = np.full((T, len(MARKERS), 3), np.nan, float)
    for mi, m in enumerate(MARKERS):
        cx, cy, cz = get_marker_xyz_cols(df, m)
        X[:, mi, 0] = df[cx].to_numpy()
        X[:, mi, 1] = df[cy].to_numpy()
        X[:, mi, 2] = df[cz].to_numpy()

    # 0埋めが欠損扱いのデータなら必要に応じて有効化
    # zero_mask = np.isfinite(X).all(axis=2) & (np.linalg.norm(X, axis=2) < 1e-12)
    # X[zero_mask] = np.nan

    before_missing = np.isnan(X[..., 0])
    filled, info = rigid_body_fill(X, min_markers=3, rms_bad=None)
    after_missing = np.isnan(filled[..., 0])

    out_dir = csv_path.parent
    stem = csv_path.stem
    out_filled = out_dir / f"{stem}_rigidfilled.csv"
    out_report = out_dir / f"{stem}_rigidfill_report.csv"
    out_summary = out_dir / f"{stem}_rigidfill_summary.txt"

    out_df = pd.DataFrame({
        "Frame": df["Frame"].to_numpy(),
        "Time (Seconds)": df["Time (Seconds)"].to_numpy(),
    })
    for mi, m in enumerate(MARKERS):
        out_df[f"{m}_X"] = filled[:, mi, 0]
        out_df[f"{m}_Y"] = filled[:, mi, 1]
        out_df[f"{m}_Z"] = filled[:, mi, 2]
    out_df.to_csv(out_filled, index=False)

    rows = []
    for i in range(T):
        for mi, m in enumerate(MARKERS):
            if before_missing[i, mi] and (not after_missing[i, mi]):
                ax, ay, az = filled[i, mi, :]
                rows.append({
                    "Frame": int(df.loc[i, "Frame"]),
                    "Time (Seconds)": float(df.loc[i, "Time (Seconds)"]) if pd.notna(df.loc[i, "Time (Seconds)"]) else np.nan,
                    "Marker": m,
                    "Before_X": np.nan, "Before_Y": np.nan, "Before_Z": np.nan,
                    "After_X": ax, "After_Y": ay, "After_Z": az,
                })
    rep = pd.DataFrame(rows)
    rep.to_csv(out_report, index=False)

    n_total = T * len(MARKERS)
    n_miss_before = int(np.sum(before_missing))
    n_miss_after  = int(np.sum(after_missing))
    n_filled = n_miss_before - n_miss_after
    n_full_frames = int(np.sum(info["full_mask"]))

    txt = []
    txt.append(f"Input: {csv_path}")
    txt.append(f"Frames: {T}")
    txt.append(f"Full frames (all 6 present): {n_full_frames}")
    txt.append(f"Missing before: {n_miss_before} / {n_total}")
    txt.append(f"Missing after : {n_miss_after} / {n_total}")
    txt.append(f"Filled points : {n_filled}")
    txt.append("")
    txt.append("Note: report CSV contains only entries that were missing and got filled.")
    out_summary.write_text("\n".join(txt), encoding="utf-8")

    print("\n".join(txt))
    if len(rep) > 0:
        print("\n--- First 20 filled entries (from report) ---")
        print(rep.head(20).to_string(index=False))
    else:
        print("\nNo missing markers were filled (either no missing or fill conditions not met).")

    print(f"\nSaved:\n- {out_filled}\n- {out_report}\n- {out_summary}")


if __name__ == "__main__":
    main()
