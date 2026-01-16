#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3_4_3d_animate_dual.py
======================
PA / PT の 3Dキーポイントnpzを「同時に」読み込み、2人同時の3Dアニメーション(mp4)を出力する。

- PA: 赤
- PT: 青
- npz 内の ANIM_TARGET_KEY (例: "butter") を描画
- conf がある場合、CONF_DRAW_TH 未満の点は NaN 扱いで描画しない
- 両者の valid_frame_range の共通区間（intersection）を使って同期させる
- 出力先: 各npzの親フォルダ / 3d_anim_{ANIM_TARGET_KEY}_PA-PT / *_PA-PT_{ANIM_TARGET_KEY}.mp4

注意:
- PAファイル群/PTファイル群の「対応付け」はファイル名から行う（"_PA" <-> "_PT" を置換）
- 片方しか存在しないペアはスキップ
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
from collections import Counter

# =========================
# 設定
# =========================
ROOT_DIR = Path(r"G:\gait_pattern\2025_shuron_tkrzk")
FRAME_RATE = 60
ANIM_TARGET_KEY = "butter"
CONF_DRAW_TH = 0.4
MAX_WORKERS = 4

# どれを対象にするか（例）
#   - "sub*/thera*/3d_kp_*PA.npz"
#   - "sub*/thera*/3d_kp_*PA_spline.npz"
PA_GLOB = "sub*/thera*/3d_kp_*PA.npz"

# ログ設定（親で確実に出す）
SHOW_TARGET_LIST = True
LIST_PREVIEW_N = 10
PRINT_QUEUE_LIST = True
SHOW_RUNNING_POSTFIX = True


def get_skeleton_connections():
    # BODY25想定（元スクリプト準拠）
    return [
        (1, 8), (1, 2), (1, 5), (2, 3), (3, 4),
        (5, 6), (6, 7), (8, 9), (8, 12), (9, 10),
        (10, 11), (12, 13), (13, 14), (1, 0),
        (0, 15), (15, 17), (0, 16), (16, 18),
        (11, 24), (11, 22), (22, 23),
        (14, 21), (14, 19), (19, 20)
    ]


# =========================
# 2人描画用 update
# =========================
def update_frame_dual(i, data_pa, conf_pa, data_pt, conf_pt,
                      scat_pa, lines_pa, scat_pt, lines_pt, ax, frame_text):
    kp_pa = data_pa[i].copy()
    kp_pt = data_pt[i].copy()

    if conf_pa is not None:
        kp_pa[conf_pa[i] < CONF_DRAW_TH] = np.nan
    if conf_pt is not None:
        kp_pt[conf_pt[i] < CONF_DRAW_TH] = np.nan

    # 元スクリプトと同じ軸変換: [x,y,z] -> (z, x, y)
    xyz_pa = np.column_stack([kp_pa[:, 2], kp_pa[:, 0], kp_pa[:, 1]])
    xyz_pt = np.column_stack([kp_pt[:, 2], kp_pt[:, 0], kp_pt[:, 1]])

    valid_pa = ~np.isnan(xyz_pa).any(axis=1)
    valid_pt = ~np.isnan(xyz_pt).any(axis=1)

    # scatter 更新
    if valid_pa.any():
        scat_pa._offsets3d = (xyz_pa[valid_pa, 0], xyz_pa[valid_pa, 1], xyz_pa[valid_pa, 2])
    else:
        scat_pa._offsets3d = ([], [], [])

    if valid_pt.any():
        scat_pt._offsets3d = (xyz_pt[valid_pt, 0], xyz_pt[valid_pt, 1], xyz_pt[valid_pt, 2])
    else:
        scat_pt._offsets3d = ([], [], [])

    # line 更新
    conns = get_skeleton_connections()
    for l, (a, b) in zip(lines_pa, conns):
        if not np.isnan(kp_pa[a]).any() and not np.isnan(kp_pa[b]).any():
            l.set_data_3d(
                [xyz_pa[a, 0], xyz_pa[b, 0]],
                [xyz_pa[a, 1], xyz_pa[b, 1]],
                [xyz_pa[a, 2], xyz_pa[b, 2]],
            )
        else:
            l.set_data_3d([], [], [])

    for l, (a, b) in zip(lines_pt, conns):
        if not np.isnan(kp_pt[a]).any() and not np.isnan(kp_pt[b]).any():
            l.set_data_3d(
                [xyz_pt[a, 0], xyz_pt[b, 0]],
                [xyz_pt[a, 1], xyz_pt[b, 1]],
                [xyz_pt[a, 2], xyz_pt[b, 2]],
            )
        else:
            l.set_data_3d([], [], [])

    # 表示範囲（元スクリプト準拠）
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 2000)
    ax.set_zlim(0, 2000)
    
    # フレーム番号更新
    frame_text.set_text(f"Frame: {i:04d}")
    
    return [scat_pa, *lines_pa, scat_pt, *lines_pt]


def make_animation_dual(data_pa, conf_pa, data_pt, conf_pt, out_mp4, title):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # PA: red
    scat_pa = ax.scatter([], [], [], s=40, c="r", depthshade=True)
    lines_pa = [ax.plot([], [], [], lw=2)[0] for _ in get_skeleton_connections()]

    # PT: blue
    scat_pt = ax.scatter([], [], [], s=40, c="b", depthshade=True)
    lines_pt = [ax.plot([], [], [], lw=2)[0] for _ in get_skeleton_connections()]

    # フレーム番号表示
    frame_text = ax.text(
        0.02, 0.95, 0.98, "",              # (x,y,z) は表示用なので相対位置感覚でOK
        transform=ax.transAxes,            # 画面固定（カメラが動いても追従）
        fontsize=14,
        color="black"
    )
    
    ax.set_title(title)
    ax.set_xlabel("Z (Forward)")
    ax.set_ylabel("X (Side)")
    ax.set_zlabel("Y (Up)")
    ax.view_init(elev=10, azim=45)
    ax.set_autoscale_on(False)

    n_frames = min(len(data_pa), len(data_pt))

    ani = animation.FuncAnimation(
        fig,
        update_frame_dual,
        frames=n_frames,
        fargs=(data_pa, conf_pa, data_pt, conf_pt, scat_pa, lines_pa, scat_pt, lines_pt, ax, frame_text),
        interval=1000 / FRAME_RATE,
        blit=False
    )

    writer = animation.FFMpegWriter(fps=FRAME_RATE, bitrate=3600)
    ani.save(out_mp4, writer=writer)
    plt.close(fig)


# =========================
# ペア探索 & ログ
# =========================
def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT_DIR))
    except Exception:
        return str(p)


def to_pt_path(pa_path: Path) -> Path:
    # ファイル名の "_PA" を "_PT" に置換して対応付け
    name = pa_path.name.replace("_PA", "_PT")
    return pa_path.with_name(name)


def plan_out_mp4(pa_path: Path, pt_path: Path) -> Path:
    out_dir = pa_path.parent / f"3d_anim_{ANIM_TARGET_KEY}_PA-PT"
    # stemは拡張子除去。末尾の _PA を落として統一名にする
    base = pa_path.stem.replace("_PA", "")
    return out_dir / f"{base}_PA-PT_{ANIM_TARGET_KEY}.mp4"


def print_target_summary(pairs):
    print("\n[Targets] (PA, PT) -> mp4 (planned)")
    print("-" * 80)
    n = len(pairs)

    def show_one(pa, pt):
        print(f"- {rel(pa)} + {rel(pt)} -> {rel(plan_out_mp4(pa, pt))}")

    if n <= LIST_PREVIEW_N * 2:
        for pa, pt in pairs:
            show_one(pa, pt)
    else:
        for pa, pt in pairs[:LIST_PREVIEW_N]:
            show_one(pa, pt)
        print(f"... ({n - LIST_PREVIEW_N * 2} more)")
        for pa, pt in pairs[-LIST_PREVIEW_N:]:
            show_one(pa, pt)
    print("-" * 80)


# =========================
# 1 pair = 1 job
# =========================
def run_one_pair(pa_path_str: str, pt_path_str: str):
    pa_path = Path(pa_path_str)
    pt_path = Path(pt_path_str)
    t0 = time.time()

    try:
        d_pa = np.load(pa_path, allow_pickle=True)
        d_pt = np.load(pt_path, allow_pickle=True)

        if ANIM_TARGET_KEY not in d_pa or ANIM_TARGET_KEY not in d_pt:
            return False, "skip_no_key", pa_path_str, pt_path_str, "", 0.0

        data_pa = d_pa[ANIM_TARGET_KEY]
        data_pt = d_pt[ANIM_TARGET_KEY]

        conf_pa = d_pa["conf"] if "conf" in d_pa.files else None
        conf_pt = d_pt["conf"] if "conf" in d_pt.files else None

        # valid_frame_range は共通区間を使う
        if "valid_frame_range" in d_pa.files and "valid_frame_range" in d_pt.files:
            s_pa, e_pa = d_pa["valid_frame_range"].astype(int)
            s_pt, e_pt = d_pt["valid_frame_range"].astype(int)
        else:
            # ない場合は全域
            s_pa, e_pa = 0, data_pa.shape[0] - 1
            s_pt, e_pt = 0, data_pt.shape[0] - 1

        if getattr(data_pa, "size", 0) == 0 or getattr(data_pt, "size", 0) == 0:
            return False, "skip_empty", pa_path_str, pt_path_str, "", 0.0

        n_pa = data_pa.shape[0]
        n_pt = data_pt.shape[0]

        # 範囲を各データ内にクリップ
        s_pa = max(0, min(s_pa, n_pa - 1))
        e_pa = max(0, min(e_pa, n_pa - 1))
        s_pt = max(0, min(s_pt, n_pt - 1))
        e_pt = max(0, min(e_pt, n_pt - 1))

        # 共通区間
        s = max(s_pa, s_pt)
        e = min(e_pa, e_pt)
        if s > e:
            return False, "skip_bad_range", pa_path_str, pt_path_str, "", 0.0

        data_pa = data_pa[s:e+1]
        data_pt = data_pt[s:e+1]

        if conf_pa is not None:
            conf_pa = conf_pa[s:e+1]
        if conf_pt is not None:
            conf_pt = conf_pt[s:e+1]

        if data_pa.shape[0] == 0 or data_pt.shape[0] == 0:
            return False, "skip_zero_frames", pa_path_str, pt_path_str, "", 0.0

        out_dir = pa_path.parent / f"3d_anim_{ANIM_TARGET_KEY}_PA-PT"
        out_dir.mkdir(exist_ok=True)

        out_mp4 = plan_out_mp4(pa_path, pt_path)
        title = f"{pa_path.stem} + {pt_path.stem} ({ANIM_TARGET_KEY})"

        make_animation_dual(data_pa, conf_pa, data_pt, conf_pt, out_mp4, title)

        return True, "ok", pa_path_str, pt_path_str, str(out_mp4), time.time() - t0

    except Exception as e:
        return False, "err", pa_path_str, pt_path_str, repr(e), time.time() - t0


# =========================
# メイン
# =========================
def main():
    pa_files = sorted(ROOT_DIR.glob(PA_GLOB))
    if not pa_files:
        print(f"PA npz が見つかりません: {PA_GLOB}")
        return

    pairs = []
    missing_pt = 0
    for pa in pa_files:
        pt = to_pt_path(pa)
        if pt.exists():
            pairs.append((pa, pt))
        else:
            missing_pt += 1

    if not pairs:
        print("PA/PT のペアが見つかりません（PTが存在しない可能性）")
        return

    cpu = os.cpu_count() or 1
    workers = MAX_WORKERS if MAX_WORKERS and MAX_WORKERS > 0 else max(1, cpu - 1)

    print("=" * 80)
    print(f"PA files    : {len(pa_files)} (missing PT for {missing_pt})")
    print(f"pairs       : {len(pairs)}")
    print(f"workers     : {workers}")
    print(f"target      : {ANIM_TARGET_KEY}")
    print(f"PA_GLOB     : {PA_GLOB}")
    print("=" * 80)

    if SHOW_TARGET_LIST:
        print_target_summary(pairs)

    counts = Counter()
    running = set()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {}

        for pa, pt in pairs:
            fut = ex.submit(run_one_pair, str(pa), str(pt))
            futures[fut] = (pa, pt)
            running.add((str(pa), str(pt)))
            if PRINT_QUEUE_LIST:
                print(f"[QUEUED] {rel(pa)} + {rel(pt)}")

        pbar = tqdm(as_completed(futures), total=len(futures), desc="3D dual animation", unit="pair")

        for fut in pbar:
            ok, kind, pa_str, pt_str, extra, sec = fut.result()
            counts[kind] += 1
            running.discard((pa_str, pt_str))

            pa = Path(pa_str)
            pt = Path(pt_str)

            if ok:
                out_mp4 = Path(extra)
                tqdm.write(f"[OK] {rel(pa)} + {rel(pt)} -> {rel(out_mp4)} [{sec:.2f}s]")
            else:
                if kind == "err":
                    tqdm.write(f"[ERR] {rel(pa)} + {rel(pt)} : {extra} [{sec:.2f}s]")
                else:
                    tqdm.write(f"[SKIP] {rel(pa)} + {rel(pt)} ({kind}) [{sec:.2f}s]")

            if SHOW_RUNNING_POSTFIX:
                pbar.set_postfix_str(f"running={len(running)}")

        pbar.close()

    print("=" * 80)
    print("SUMMARY")
    for k in ["ok", "skip_no_key", "skip_empty", "skip_bad_range", "skip_zero_frames", "err"]:
        if k in counts:
            print(f"  {k:15s}: {counts[k]}")
    print("=" * 80)


if __name__ == "__main__":
    main()
