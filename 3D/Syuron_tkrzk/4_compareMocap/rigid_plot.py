import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 設定（ここだけ変える）
# =========================
ORIG_CSV   = Path(r"G:\gait_pattern\2025_shuron_BR9G\sub1\thera1-0\mocap\tst\1-1-0.csv")
FILLED_CSV = ORIG_CSV.with_name(f"{ORIG_CSV.stem}_rigidfilled.csv")

MARKERS = ["LASI", "LPSI", "LILC", "RASI", "RPSI", "RILC"]
AXES = ["X", "Y", "Z"]


# =========================
# 元CSV（Vicon multi-row header）から対象6マーカーだけ読む
# ※ rigid_fill_markers.py と同等の "列名生成" を使う簡易版
# =========================
import csv, re

PREFIX = "MarkerSet 01:"

def read_vicon_like_csv(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for r in reader:
            rows.append(r)

    axis_row_idx = None
    for i, r in enumerate(rows[:50]):
        if len(r) >= 6 and r[0].strip() == "Frame" and "Time" in (r[1] if len(r) > 1 else ""):
            if any(x.strip() == "X" for x in r) and any(x.strip() == "Y" for x in r) and any(x.strip() == "Z" for x in r):
                axis_row_idx = i
                break
    if axis_row_idx is None:
        raise ValueError("Axis行が見つかりませんでした（Frame, Time (Seconds), X,Y,Z...）。")

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

    data_rows = []
    for r in rows[axis_row_idx + 1:]:
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
    cols = {ax: None for ax in AXES}
    for c in df.columns:
        if c.startswith(base):
            ax = c[len(base):].strip()
            if ax in cols:
                cols[ax] = c
    if any(cols[ax] is None for ax in cols):
        for ax in AXES:
            pat = re.compile(re.escape(f"{PREFIX}{marker}") + r".*_" + re.escape(ax) + r"$")
            for c in df.columns:
                if pat.search(c):
                    cols[ax] = c
                    break
    if any(cols[ax] is None for ax in cols):
        raise ValueError(f"{marker} のX/Y/Z列が見つかりません: {cols}")
    return cols["X"], cols["Y"], cols["Z"]


# =========================
# 読み込み
# =========================
df_before = read_vicon_like_csv(ORIG_CSV)
df_after  = pd.read_csv(FILLED_CSV)

# Frameが合っている前提（念のためチェック）
if not np.array_equal(df_before["Frame"].to_numpy(), df_after["Frame"].to_numpy()):
    raise ValueError("Frame列が一致しません。元CSVとrigidfilled.csvが同じFrame系列か確認してください。")

frames = df_after["Frame"].to_numpy()

# =========================
# プロット関数（補間前 vs 補間で追加された部分）
# =========================
def plot_marker(marker, save_dir, use_time=False, frame_range=None):
    """
    marker: "LASI" etc
    use_time: Trueなら横軸Time、FalseならFrame
    frame_range: (start_frame, end_frame) で表示範囲を限定
    """
    # before（元データ）
    cx, cy, cz = get_marker_xyz_cols(df_before, marker)
    before = np.stack([df_before[cx].to_numpy(),
                       df_before[cy].to_numpy(),
                       df_before[cz].to_numpy()], axis=1)  # (T,3)

    # after（補間後データ）
    after = np.stack([df_after[f"{marker}_X"].to_numpy(),
                      df_after[f"{marker}_Y"].to_numpy(),
                      df_after[f"{marker}_Z"].to_numpy()], axis=1)

    # 補間で追加された部分のマスク（元データがNaNだった場所）
    was_missing = np.isnan(before).any(axis=1)
    
    x = df_after["Time (Seconds)"].to_numpy() if use_time else frames

    # 範囲指定
    if frame_range is not None:
        f0, f1 = frame_range
        mask = (frames >= f0) & (frames <= f1)
        x = x[mask]
        before = before[mask]
        after = after[mask]
        was_missing = was_missing[mask]
        

    # 3行1列のサブプロットを作成
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle(f"{marker} (Original vs Filled points)", fontsize=14, fontweight='bold')
    
    xlabel = "Time (s)" if use_time else "Frame"
    
    for ai, ax_name in enumerate(AXES):
        ax = axes[ai]
        
        # 元データ（欠損部分は表示されない）
        ax.scatter(x, before[:, ai], label="Original (before)", s=10, color='blue')
        
        # 補間で追加された部分のみ
        filled_x = x[was_missing]
        filled_y = after[was_missing, ai]
        ax.scatter(filled_x, filled_y, label="Filled (interpolated)", s=10, color='red')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{ax_name} [mm]")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f"{marker}_filled.png", dpi=150)
    plt.close()


# =========================
# 統計情報を表示する関数
# =========================
def print_fill_statistics(marker):
    """補間された点の統計情報を表示"""
    cx, cy, cz = get_marker_xyz_cols(df_before, marker)
    before = np.stack([df_before[cx].to_numpy(),
                       df_before[cy].to_numpy(),
                       df_before[cz].to_numpy()], axis=1)
    
    was_missing = np.isnan(before).any(axis=1)
    total_frames = len(frames)
    filled_frames = np.sum(was_missing)
    
    print(f"\n{marker}:")
    print(f"  Total frames: {total_frames}")
    print(f"  Filled frames: {filled_frames} ({filled_frames/total_frames*100:.1f}%)")
    
    if filled_frames > 0:
        # 連続する欠損区間を検出
        gaps = []
        in_gap = False
        gap_start = None
        for i, missing in enumerate(was_missing):
            if missing and not in_gap:
                gap_start = frames[i]
                in_gap = True
            elif not missing and in_gap:
                gaps.append((gap_start, frames[i-1]))
                in_gap = False
        if in_gap:
            gaps.append((gap_start, frames[-1]))
        
        print(f"  Number of gap regions: {len(gaps)}")
        for i, (start, end) in enumerate(gaps, 1):
            print(f"    Gap {i}: Frame {start} to {end} ({end-start+1} frames)")


# =========================
# 実行例
# =========================
# 1) 各マーカーの統計情報を表示
print("=== Fill Statistics ===")
for m in MARKERS:
    print_fill_statistics(m)

# 2) 各マーカーをプロット
print("\n=== Generating plots ===")
for m in MARKERS:
    print(f"Plotting {m}...")
    plot_marker(m, save_dir=ORIG_CSV.parent, use_time=False, frame_range=(632,1156))

# 3) 特定の欠損区間だけ拡大表示したい場合
# plot_marker("LPSI", save_dir=ORIG_CSV.parent, use_time=False, frame_range=(380, 520))

print("\n=== Done ===")