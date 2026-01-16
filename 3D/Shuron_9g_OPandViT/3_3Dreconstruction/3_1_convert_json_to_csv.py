"""
OpenPose / ViTPose の JSON 出力（people[...].pose_keypoints_2d）を CSV に変換し、
0欠損 → 3次スプライン補間 を行うスクリプト（最大1人のみ）。

仕様（要望反映）:
- 人物は各フレームで最大1人のみ採用（= bbox面積が最大の人を選ぶ）
- 採用条件:
    1) bbox面積 >= AREA_TH
    2) bbox計算に使える有効キーポイント数 >= MIN_VALID_KPTS
- シーケンス全体で「採用できたフレーム数」が全体の MIN_DET_RATIO 未満なら、そのフォルダは skip
- 外れ値除去は行わない（後段3Dで対処する前提）
    - 座標が0の点のみ NaN 化 → 3次スプライン補間
    - プロットの×印は「0だった点」のみ
- 出力は pa のみ:
    - {base}_pa.csv           （Raw: frame列drop）
    - {base}_pa_spline.csv    （Spline: frame列drop）
    - pa_plots フォルダ
- allpersons_plots などは作らない
"""

import json
import glob
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================
# 設定（ここだけ調整）
# =========================
root_dir = Path(r"G:\gait_pattern\BR9G_shuron")
directions = ["fl", "fr", "sagi"]

# 解析対象被験者
SUB_GLOB = "sub[1-6]"   # 例: sub1〜sub6（必要に応じて sub* などに）

# 人物は最大1人として扱う（各フレームで候補を面積最大で選ぶ）
AREA_TH = 200 * 300       # 60,000 px^2
MIN_VALID_KPTS = 5        # bbox計算に使える点がこれ未満なら無効扱い
MIN_DET_RATIO = 0.10      # 採用できたフレーム比率がこれ未満ならフォルダごとskip

OUT_PERSON_TAG = "pa"     # p0 の代わりに pa

# 画像座標レンジ（4K想定）
XLIM = (0, 3840)
YLIM = (0, 2160)

# 既にcsvがあればスキップ（同名の pa.csv/pa_spline.csv が揃っていれば処理しない）
SKIP_IF_CSV_EXISTS = False

# OpenPose BODY_25
keypoint_names = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
    "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel"
]

# 個別プロット不要
SKIP_KEYPOINTS = {
    "Nose", "Neck",
    "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist",
    "REye", "REar", "LEye", "LEar",
}

# 並列処理数
MAX_WORKERS = 4

# =========================
# 探索
# =========================
def iter_poseestimate_json_dirs(root: Path, directions_list):
    for sub in sorted(root.glob(SUB_GLOB)):
        if not sub.is_dir():
            continue
        for thera in sub.iterdir():
            if not (thera.is_dir() and thera.name.startswith("thera")):
                continue
            for d in directions_list:
                gopro_dir = thera / "gopro" / d
                if not gopro_dir.exists():
                    continue
                for poseestimate_dir in gopro_dir.iterdir():
                    if not poseestimate_dir.is_dir():
                        continue
                    json_dir = poseestimate_dir / "json"
                    if json_dir.exists() and json_dir.is_dir():
                        yield json_dir, d

# =========================
# 変換コア（1人だけ選ぶ）
# =========================
def person_bbox_area_from_kpts(kpts, p_th=0.0):
    """
    pose_keypoints_2d (x,y,p)*25 から bbox area を計算。
    - x,y が 0 でない
    - p > p_th
    の点だけで bbox を作る
    戻り値: (area, valid_count)
    """
    if not kpts or len(kpts) < len(keypoint_names) * 3:
        return 0.0, 0

    xs, ys = [], []
    for i in range(len(keypoint_names)):
        x = kpts[i * 3 + 0]
        y = kpts[i * 3 + 1]
        p = kpts[i * 3 + 2]
        if x is None or y is None:
            continue
        x = float(x)
        y = float(y)
        p = float(p)

        if x == 0 or y == 0:
            continue
        if p <= p_th:
            continue

        xs.append(x)
        ys.append(y)

    valid = len(xs)
    if valid < MIN_VALID_KPTS:
        return 0.0, valid

    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    area = float(max(0.0, w) * max(0.0, h))
    return area, valid


def load_sequence_single_person(json_dir: Path):
    """
    各フレームで
      - bbox area >= AREA_TH を満たす候補だけ残す
      - その中で bbox area 最大の 1 名を採用
    として、1人分の DF を作る（採用できないフレームは0埋め）。
    """
    json_files = sorted(glob.glob(str(json_dir / "*.json")))
    frame_count = len(json_files)
    if frame_count == 0:
        return None, 0, 0

    rows = []
    det_frames = 0

    for frame_idx, fp in enumerate(json_files):
        with open(fp, "r") as f:
            val = json.load(f)

        people = val.get("people", []) or []

        best_person = None
        best_area = -1.0

        for person in people:
            kpts = person.get("pose_keypoints_2d", []) or []
            area, _valid = person_bbox_area_from_kpts(kpts, p_th=0.0)
            if area < AREA_TH:
                continue
            if area > best_area:
                best_area = area
                best_person = person

        row = {"frame": frame_idx}
        if best_person is None:
            for name in keypoint_names:
                row[f"{name}_x"] = 0
                row[f"{name}_y"] = 0
                row[f"{name}_p"] = 0
        else:
            det_frames += 1
            kpts = best_person.get("pose_keypoints_2d", []) or []
            if len(kpts) >= len(keypoint_names) * 3:
                for i, name in enumerate(keypoint_names):
                    row[f"{name}_x"] = kpts[i * 3 + 0]
                    row[f"{name}_y"] = kpts[i * 3 + 1]
                    row[f"{name}_p"] = kpts[i * 3 + 2]
            else:
                for name in keypoint_names:
                    row[f"{name}_x"] = 0
                    row[f"{name}_y"] = 0
                    row[f"{name}_p"] = 0

        rows.append(row)

    df = pd.DataFrame(rows)
    return df, frame_count, det_frames

# =========================
# 0→NaN → 3次スプライン補間（×は0由来のみ）
# =========================
def zero_nan_and_spline(df_raw: pd.DataFrame):
    """
    - x==0 または y==0 のフレームを欠損扱い → x,y を NaN
    - 3次スプラインで補間
    - ×プロット用に「0由来マスク」を返す
    """
    df0 = df_raw.copy()
    mask0_by_kp = {}

    for k in keypoint_names:
        xcol = f"{k}_x"
        ycol = f"{k}_y"
        x = df0[xcol].astype(float).to_numpy()
        y = df0[ycol].astype(float).to_numpy()

        m0 = (x == 0) | (y == 0)
        if m0.any():
            df0.loc[m0, xcol] = np.nan
            df0.loc[m0, ycol] = np.nan
        mask0_by_kp[k] = m0

    df_s = df0.copy()
    xy_cols = [c for c in df_s.columns if c.endswith("_x") or c.endswith("_y")]

    for col in xy_cols:
        s = df_s[col].astype(float)
        if s.notna().sum() >= 4:
            df_s[col] = s.interpolate(method="spline", order=3, limit_direction="both")
        else:
            df_s[col] = s.fillna(0)

    df_s[xy_cols] = df_s[xy_cols].fillna(0)
    return df_s, mask0_by_kp

# =========================
# プロット
# =========================
def _save_plot(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def kp_pair_name(rkp: str, lkp: str) -> str:
    if rkp.startswith("R") and lkp.startswith("L") and rkp[1:] == lkp[1:]:
        return rkp[1:]
    return f"{rkp}_{lkp}"


def save_plots_single_person(df_raw, df_spline, mask0_by_kp, out_root: Path, plot_name: str):
    """
    - 不要KPはスキップ
    - KP別: raw/spline + 0由来×、3段(x,y,p)
    - 左右セット: raw/spline を左右同時、2段(x,y)
    """
    plot_folder = out_root / f"{plot_name}_plots"
    plot_folder.mkdir(parents=True, exist_ok=True)
    frames = df_raw["frame"].to_numpy()

    # ----- A) KP別（raw/spline + 0由来×、3段: x,y,p） -----
    for kp in keypoint_names:
        if kp in SKIP_KEYPOINTS:
            continue

        m0 = mask0_by_kp.get(kp, np.zeros_like(frames, dtype=bool))

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # X
        x_col = f"{kp}_x"
        axes[0].plot(frames, df_raw[x_col], label="Raw", alpha=0.6, linewidth=1)
        axes[0].plot(frames, df_spline[x_col], label="Spline", alpha=0.9, linewidth=1.5)
        if m0.any():
            axes[0].scatter(frames[m0], df_raw.loc[m0, x_col], marker="x", s=35, label="Zero", zorder=4)
        axes[0].set_ylabel("X (px)")
        axes[0].set_ylim(*XLIM)
        axes[0].set_title(f"{kp} - X")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Y
        y_col = f"{kp}_y"
        axes[1].plot(frames, df_raw[y_col], label="Raw", alpha=0.6, linewidth=1)
        axes[1].plot(frames, df_spline[y_col], label="Spline", alpha=0.9, linewidth=1.5)
        if m0.any():
            axes[1].scatter(frames[m0], df_raw.loc[m0, y_col], marker="x", s=35, label="Zero", zorder=4)
        axes[1].set_ylabel("Y (px)")
        axes[1].set_ylim(*YLIM)
        axes[1].set_title(f"{kp} - Y")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # p（confidenceはRawのみ）
        p_col = f"{kp}_p"
        axes[2].plot(frames, df_raw[p_col].astype(float), label="Confidence", linewidth=1.2)
        axes[2].set_xlabel("Frame")
        axes[2].set_ylabel("p")
        axes[2].set_ylim(0, 1.05)
        axes[2].set_title(f"{kp} - Confidence")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        _save_plot(fig, plot_folder / f"{kp}.png")

    # ----- B) 左右セット（raw と spline を左右同時描画、2段: x,y） -----
    lr_pairs = []
    for kp in keypoint_names:
        if kp.startswith("R"):
            lk = "L" + kp[1:]
            if lk in keypoint_names:
                lr_pairs.append((kp, lk))

    for rkp, lkp in lr_pairs:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        rx, lx = f"{rkp}_x", f"{lkp}_x"
        ry, ly = f"{rkp}_y", f"{lkp}_y"

        # 右=赤、左=青、raw=薄、spline=濃
        axes[0].plot(frames, df_raw[rx],    label="R raw",    color="red",  alpha=0.35, linewidth=1)
        axes[0].plot(frames, df_spline[rx], label="R spline", color="red",  alpha=0.95, linewidth=1.6)
        axes[0].plot(frames, df_raw[lx],    label="L raw",    color="blue", alpha=0.35, linewidth=1)
        axes[0].plot(frames, df_spline[lx], label="L spline", color="blue", alpha=0.95, linewidth=1.6)
        axes[0].set_ylabel("X (px)")
        axes[0].set_ylim(*XLIM)
        axes[0].set_title(f"{rkp} / {lkp} - X")
        axes[0].legend(ncol=2)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(frames, df_raw[ry],    label="R raw",    color="red",  alpha=0.35, linewidth=1)
        axes[1].plot(frames, df_spline[ry], label="R spline", color="red",  alpha=0.95, linewidth=1.6)
        axes[1].plot(frames, df_raw[ly],    label="L raw",    color="blue", alpha=0.35, linewidth=1)
        axes[1].plot(frames, df_spline[ly], label="L spline", color="blue", alpha=0.95, linewidth=1.6)
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Y (px)")
        axes[1].set_ylim(*YLIM)
        axes[1].set_title(f"{rkp} / {lkp} - Y")
        axes[1].legend(ncol=2)
        axes[1].grid(True, alpha=0.3)

        name = kp_pair_name(rkp, lkp)
        _save_plot(fig, plot_folder / f"{name}_LR.png")

# =========================
# 1フォルダ処理
# =========================
def process_one_json_dir(json_dir: Path, direction: str):
    out_root = json_dir.parent
    base = out_root.name

    title = f"{base}_{OUT_PERSON_TAG}"
    csv_raw = out_root / f"{title}.csv"
    csv_spline = out_root / f"{title}_spline.csv"

    if SKIP_IF_CSV_EXISTS and csv_raw.exists() and csv_spline.exists():
        return "skip_exists"

    # 1人だけ選ぶDF
    df_raw, frame_count, det_frames = load_sequence_single_person(json_dir)
    if df_raw is None:
        return "skip"

    # 検出率が低すぎるフォルダはスキップ
    if det_frames < max(1, int(np.ceil(frame_count * MIN_DET_RATIO))):
        return "skip_low_detection"

    df_orig = df_raw.copy()  # ×用（0保持）

    # 0→NaN→Spline
    df_spline, mask0_by_kp = zero_nan_and_spline(df_raw)

    # 保存（frame列はdrop）
    df_raw.to_csv(csv_raw, index=False)
    df_spline.to_csv(csv_spline, index=False)

    # プロット（pa_plots）
    save_plots_single_person(
        df_orig, df_spline,
        mask0_by_kp,
        out_root, plot_name=OUT_PERSON_TAG
    )

    return "ok"

# =========================
# 並列処理用
# =========================
def worker(args):
    json_dir, direction = args
    try:
        rel = json_dir.relative_to(root_dir)
        parts = rel.parts
        display = " / ".join(parts[:5])
        print(f"[START] {display}")
        return process_one_json_dir(json_dir, direction)
    except Exception as e:
        return f"error: {json_dir} {e}"

# =========================
# メイン
# =========================
def main():
    targets = list(iter_poseestimate_json_dirs(root_dir, directions))
    print(f"root: {root_dir}")
    print(f"found json dirs: {len(targets)}")
    if not targets:
        print("No target json directories found.")
        return

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(worker, t): t for t in targets}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="All poseestimate folders"):
            res = fut.result()
            if isinstance(res, str) and res.startswith("error"):
                print(res)

    dt = time.time() - t0
    print("\n" + "=" * 60)
    print("done.")
    print(f"folders: {len(targets)}")
    print(f"elapsed: {dt:.1f} sec")
    print("=" * 60)

if __name__ == "__main__":
    main()
