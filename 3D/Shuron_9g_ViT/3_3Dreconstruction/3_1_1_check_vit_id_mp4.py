"""
本スクリプトは実行しなくても3_2_3d_reconstruct3d.py で3D再構成は可能

ROOT_DIR 以下を再帰探索し、各ディレクトリで

- undistorted/ : 連続画像（背景）
- ViTPose/     : <base>_PA(_spline).csv と <base>_PT(_spline).csv

を見つけ、背景画像に PA/PT のスケルトンを重ねた
HD(1280x720) mp4 を生成する
"""

from __future__ import annotations

from pathlib import Path
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


# =========================
# 設定（ここだけ変更）
# =========================
ROOT_DIR = Path(r"G:\gait_pattern\2025_shuron_BR9G")

USE_SPLINE = False                  # 三次元化時はスプライン補間を使用していないので基本はFalseでよい．　True: *_spline.csv を使用
OUT_W, OUT_H = 1280, 720           # HD固定
FPS = 60.0                         # 出力fps
P_TH = 0.0                         # confidence閾値（例: 0.2）
OVERWRITE = True                  # 既に出力があればスキップ

MAX_WORKERS = 4

COLOR_PA = (0, 0, 255)             # BGR
COLOR_PT = (255, 0, 0)             # BGR
# =========================


# tqdm.write が並列で混線しないように
_LOG_LOCK = threading.Lock()


def log(msg: str):
    with _LOG_LOCK:
        tqdm.write(msg)


# BODY25
KEYPOINT_NAMES = [
    "Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist",
    "MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle",
    "REye","LEye","REar","LEar","LBigToe","LSmallToe","LHeel",
    "RBigToe","RSmallToe","RHeel"
]
KP_IDX = {k: i for i, k in enumerate(KEYPOINT_NAMES)}

EDGES = [
    ("Nose","Neck"),
    ("Neck","RShoulder"),("RShoulder","RElbow"),("RElbow","RWrist"),
    ("Neck","LShoulder"),("LShoulder","LElbow"),("LElbow","LWrist"),
    ("Neck","MidHip"),
    ("MidHip","RHip"),("RHip","RKnee"),("RKnee","RAnkle"),
    ("MidHip","LHip"),("LHip","LKnee"),("LKnee","LAnkle"),
    ("Nose","REye"),("REye","REar"),
    ("Nose","LEye"),("LEye","LEar"),
    ("LAnkle","LHeel"),("LHeel","LBigToe"),("LHeel","LSmallToe"),
    ("RAnkle","RHeel"),("RHeel","RBigToe"),("RHeel","RSmallToe"),
]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "frame" not in df.columns:
        df.insert(0, "frame", np.arange(len(df), dtype=int))
    return df


def get_xy(df: pd.DataFrame, i: int):
    row = df.iloc[i]
    xy = np.zeros((25, 2), dtype=float)
    pp = np.zeros(25, dtype=float)
    for j, k in enumerate(KEYPOINT_NAMES):
        xy[j, 0] = float(row.get(f"{k}_x", 0.0))
        xy[j, 1] = float(row.get(f"{k}_y", 0.0))
        pp[j]    = float(row.get(f"{k}_p", 0.0))
    return xy, pp


def valid(pt, conf) -> bool:
    return (pt[0] > 0.0) and (pt[1] > 0.0) and (conf >= P_TH)


def draw_skeleton(img, xy, pp, color, kp_radius=3, line_thickness=2):
    for a, b in EDGES:
        ia, ib = KP_IDX[a], KP_IDX[b]
        if valid(xy[ia], pp[ia]) and valid(xy[ib], pp[ib]):
            ax, ay = int(round(xy[ia, 0])), int(round(xy[ia, 1]))
            bx, by = int(round(xy[ib, 0])), int(round(xy[ib, 1]))
            cv2.line(img, (ax, ay), (bx, by), color, line_thickness, cv2.LINE_AA)

    for i in range(25):
        if valid(xy[i], pp[i]):
            x, y = int(round(xy[i, 0])), int(round(xy[i, 1]))
            cv2.circle(img, (x, y), kp_radius, color, -1, cv2.LINE_AA)

def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in _num_pat.split(str(p))]

_num_pat = re.compile(r"(\d+)")


def frame_index_from_name(p: Path) -> int:
    m = _num_pat.findall(p.stem)
    return int(m[-1]) if m else 10**18


def list_image_files(undistorted_dir: Path):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files = []
    for e in exts:
        files.extend(undistorted_dir.glob(e))
    return sorted(files, key=frame_index_from_name)


def find_bases(vitpose_dir: Path, use_spline: bool):
    bases = []
    if use_spline:
        for pa in vitpose_dir.glob("*_PA_spline.csv"):
            base = pa.name.replace("_PA_spline.csv", "")
            if (vitpose_dir / f"{base}_PT_spline.csv").exists():
                bases.append(base)
    else:
        for pa in vitpose_dir.glob("*_PA.csv"):
            base = pa.name.replace("_PA.csv", "")
            if (vitpose_dir / f"{base}_PT.csv").exists():
                bases.append(base)
    return sorted(set(bases))


def process_one_trial(undistorted_dir: Path, vitpose_dir: Path, base: str, use_spline: bool) -> tuple[bool, str]:
    """
    戻り値: (success, message)
    """
    tag = f"{vitpose_dir.parent} | {base}"
    log(f"[START] {tag}")

    try:
        out_path = vitpose_dir / f"{base}_overlay_HD_{'spline' if use_spline else 'raw'}.mp4"
        if out_path.exists() and not OVERWRITE:
            log(f"[END]   {tag} -> SKIP (exists)")
            return True, f"SKIP: {out_path}"

        pa_csv = vitpose_dir / (f"{base}_PA_spline.csv" if use_spline else f"{base}_PA.csv")
        pt_csv = vitpose_dir / (f"{base}_PT_spline.csv" if use_spline else f"{base}_PT.csv")

        if not pa_csv.exists():
            log(f"[END]   {tag} -> FAIL (PA missing)")
            return False, f"PA not found: {pa_csv}"
        if not pt_csv.exists():
            log(f"[END]   {tag} -> FAIL (PT missing)")
            return False, f"PT not found: {pt_csv}"

        df_pa = load_csv(pa_csv)
        df_pt = load_csv(pt_csv)

        img_files = list_image_files(undistorted_dir)
        if not img_files:
            log(f"[END]   {tag} -> FAIL (no images)")
            return False, f"no images: {undistorted_dir}"

        # 画像解像度（1枚目で固定想定）
        first = cv2.imread(str(img_files[0]))
        if first is None:
            log(f"[END]   {tag} -> FAIL (cannot read first)")
            return False, f"cannot read first image: {img_files[0]}"
        src_h, src_w = first.shape[:2]
        sx = OUT_W / src_w
        sy = OUT_H / src_h

        n = min(len(img_files), len(df_pa), len(df_pt))
        if n <= 0:
            log(f"[END]   {tag} -> FAIL (no frames)")
            return False, f"no frames: {tag}"

        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OUT_W, OUT_H))
        if not vw.isOpened():
            log(f"[END]   {tag} -> FAIL (VideoWriter)")
            return False, f"VideoWriter open failed: {out_path}"

        for i in range(n):
            img = cv2.imread(str(img_files[i]))
            if img is None:
                vw.release()
                log(f"[END]   {tag} -> FAIL (cannot read image)")
                return False, f"cannot read image: {img_files[i]}"

            img_hd = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

            xy_pa, pp_pa = get_xy(df_pa, i)
            xy_pt, pp_pt = get_xy(df_pt, i)

            # 座標をHDへ同率スケール（レターボックスなし）
            xy_pa[:, 0] *= sx
            xy_pa[:, 1] *= sy
            xy_pt[:, 0] *= sx
            xy_pt[:, 1] *= sy

            draw_skeleton(img_hd, xy_pa, pp_pa, COLOR_PA)
            draw_skeleton(img_hd, xy_pt, pp_pt, COLOR_PT)

            cv2.putText(img_hd, f"Frame {i+1}/{n}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img_hd, "PA", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_PA, 2, cv2.LINE_AA)
            cv2.putText(img_hd, "PT", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_PT, 2, cv2.LINE_AA)

            vw.write(img_hd)

        vw.release()
        log(f"[END]   {tag} -> OK")
        return True, f"OK: {out_path}"

    except Exception as e:
        log(f"[END]   {tag} -> ERROR")
        return False, f"ERROR: {tag} -> {e}"


def build_tasks(root: Path, use_spline: bool):
    """
    ROOT_DIR 以下の ViTPose/ と undistorted/ のペアを探し、
    (undistorted_dir, vitpose_dir, base) のタスクを列挙
    """
    tasks = []
    vitpose_dirs = sorted([p for p in root.rglob("ViTPose") if p.is_dir()])
    for vitpose_dir in vitpose_dirs:
        # subの対象を絞る場合は以下の3行を有効化して調整（今はsub6のみ扱う場合）
        target_sub_dir = vitpose_dir.parent.parent.parent.parent
        if not target_sub_dir.name.startswith("sub10"):
            continue
        undistorted_dir = vitpose_dir.parent / "undistorted"
        if not undistorted_dir.exists():
            continue
        bases = find_bases(vitpose_dir, use_spline=use_spline)
        for base in bases:
            tasks.append((undistorted_dir, vitpose_dir, base))
    
    # sub1から処理が始まるようにソート
    tasks.sort(key=lambda t: natural_key(t[1].parent) + [t[2]])
    
    
    
    return tasks


def main():
    tasks = build_tasks(ROOT_DIR, use_spline=USE_SPLINE)
    print(f"Total tasks found: {len(tasks)}")
    for task in tasks:
        print(f"    task :{task}")
    if not tasks:
        print("[DONE] task not found (ViTPose + undistorted).")
        return

    ok = 0
    ng = 0
    msgs_ng = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(process_one_trial, und_dir, vit_dir, base, USE_SPLINE)
            for (und_dir, vit_dir, base) in tasks
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Overlay (trials)"):
            success, msg = fut.result()
            if success:
                ok += 1
            else:
                ng += 1
                msgs_ng.append(msg)

    print(f"\n[DONE] success={ok}  failed={ng}  total={len(tasks)}")
    if msgs_ng:
        print("\n--- Failed list (first 50) ---")
        for m in msgs_ng[:50]:
            print(m)


if __name__ == "__main__":
    main()
