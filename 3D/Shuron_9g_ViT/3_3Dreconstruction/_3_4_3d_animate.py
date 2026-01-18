"""
3_4_3d_animate.py (parallel, visible logging)
=============================================
- 結果に影響する処理は変更しない
- 親プロセス側で「対象/投入/処理中/完了」を確実に表示する
Mocapとの比較はなし
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
ROOT_DIR = Path(r"G:\gait_pattern\2025_shuron_BR9G")
FRAME_RATE = 60
ANIM_TARGET_KEY = "butter"
CONF_DRAW_TH = 0.4
MAX_WORKERS = 4

# ログ設定（親で確実に出す）
SHOW_TARGET_LIST = True
LIST_PREVIEW_N = 10
PRINT_QUEUE_LIST = True          # 投入時に表示（多いと長い）
SHOW_RUNNING_POSTFIX = True      # tqdm postfix に running 数を出す


def get_skeleton_connections():
    return [
        (1, 8), (1, 2), (1, 5), (2, 3), (3, 4),
        (5, 6), (6, 7), (8, 9), (8, 12), (9, 10),
        (10, 11), (12, 13), (13, 14), (1, 0),
        (0, 15), (15, 17), (0, 16), (16, 18),
        (11, 24), (11, 22), (22, 23),
        (14, 21), (14, 19), (19, 20)
    ]


# =========================
# アニメーション（結果に直結：変更しない）
# =========================
def update_frame(i, data_3d, conf, scat, lines, ax):
    kp = data_3d[i].copy()

    if conf is not None:
        kp[conf[i] < CONF_DRAW_TH] = np.nan

    xyz = np.column_stack([kp[:, 2], kp[:, 0], kp[:, 1]])
    valid = ~np.isnan(xyz).any(axis=1)

    if valid.any():
        scat._offsets3d = (xyz[valid, 0], xyz[valid, 1], xyz[valid, 2])
    else:
        scat._offsets3d = ([], [], [])

    for l, (a, b) in zip(lines, get_skeleton_connections()):
        if not np.isnan(kp[a]).any() and not np.isnan(kp[b]).any():
            l.set_data_3d(
                [xyz[a, 0], xyz[b, 0]],
                [xyz[a, 1], xyz[b, 1]],
                [xyz[a, 2], xyz[b, 2]],
            )
        else:
            l.set_data_3d([], [], [])

    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 2000)
    ax.set_zlim(0, 2000)

    return [scat, *lines]


def make_animation(data_3d, conf, out_mp4, title):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    scat = ax.scatter([], [], [], s=40, c="r", depthshade=True)
    lines = [ax.plot([], [], [], lw=2)[0] for _ in get_skeleton_connections()]

    ax.set_title(title)
    ax.set_xlabel("Z (Forward)")
    ax.set_ylabel("X (Side)")
    ax.set_zlabel("Y (Up)")
    ax.view_init(elev=10, azim=45)
    ax.set_autoscale_on(False)

    ani = animation.FuncAnimation(
        fig,
        update_frame,
        frames=len(data_3d),
        fargs=(data_3d, conf, scat, lines, ax),
        interval=1000 / FRAME_RATE,
        blit=False
    )

    writer = animation.FFMpegWriter(fps=FRAME_RATE, bitrate=3600)
    ani.save(out_mp4, writer=writer)
    plt.close(fig)


# =========================
# ログ用
# =========================
def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT_DIR))
    except Exception:
        return str(p)

def plan_out_mp4(npz_path: Path) -> Path:
    out_dir = npz_path.parent / f"3d_anim_{ANIM_TARGET_KEY}"
    return out_dir / f"{npz_path.stem}_{ANIM_TARGET_KEY}.mp4"

def print_target_summary(npz_files):
    print("\n[Targets] npz -> mp4 (planned)")
    print("-" * 80)
    n = len(npz_files)

    def show_one(p: Path):
        print(f"- {rel(p)} -> {rel(plan_out_mp4(p))}")

    if n <= LIST_PREVIEW_N * 2:
        for p in npz_files:
            show_one(p)
    else:
        for p in npz_files[:LIST_PREVIEW_N]:
            show_one(p)
        print(f"... ({n - LIST_PREVIEW_N * 2} more)")
        for p in npz_files[-LIST_PREVIEW_N:]:
            show_one(p)
    print("-" * 80)


# =========================
# 1 npz = 1 job
# =========================
def run_one_npz(npz_path_str: str):
    npz_path = Path(npz_path_str)
    t0 = time.time()
    try:
        data = np.load(npz_path, allow_pickle=True)

        if ANIM_TARGET_KEY not in data:
            return False, "skip_no_key", npz_path_str, "", 0.0

        data_3d = data[ANIM_TARGET_KEY]
        conf = data["conf"]
        s, e = data["valid_frame_range"].astype(int)

        if getattr(data_3d, "size", 0) == 0:
            return False, "skip_empty", npz_path_str, "", 0.0

        n = data_3d.shape[0]
        s = max(0, min(s, n - 1))
        e = max(0, min(e, n - 1))
        if s > e:
            return False, "skip_bad_range", npz_path_str, "", 0.0

        data_3d = data_3d[s:e+1]
        conf = conf[s:e+1]

        if data_3d.shape[0] == 0:
            return False, "skip_zero_frames", npz_path_str, "", 0.0

        out_dir = npz_path.parent / f"3d_anim_{ANIM_TARGET_KEY}"
        out_dir.mkdir(exist_ok=True)

        out_mp4 = out_dir / f"{npz_path.stem}_{ANIM_TARGET_KEY}.mp4"
        title = f"{npz_path.stem} ({ANIM_TARGET_KEY})"

        make_animation(data_3d, conf, out_mp4, title)

        return True, "ok", npz_path_str, str(out_mp4), time.time() - t0

    except Exception as e:
        return False, "err", npz_path_str, repr(e), time.time() - t0


# =========================
# メイン
# =========================
def main():
    npz_files = sorted(ROOT_DIR.glob("sub*/thera*/3d_kp_*.npz"))
    if not npz_files:
        print("3_3 の出力 npz が見つかりません")
        return

    cpu = os.cpu_count() or 1
    workers = MAX_WORKERS if MAX_WORKERS and MAX_WORKERS > 0 else max(1, cpu - 1)

    print("=" * 80)
    print(f"npz files : {len(npz_files)}")
    print(f"workers   : {workers}")
    print(f"target    : {ANIM_TARGET_KEY}")
    print("=" * 80)

    if SHOW_TARGET_LIST:
        print_target_summary(npz_files)

    counts = Counter()
    running = set()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {}

        # --- 投入（ここは親プロセスなので必ず表示される）---
        for p in npz_files:
            fut = ex.submit(run_one_npz, str(p))
            futures[fut] = p
            running.add(str(p))
            if PRINT_QUEUE_LIST:
                print(f"[QUEUED] {rel(p)}")

        pbar = tqdm(as_completed(futures), total=len(futures), desc="3D animation", unit="npz")

        for fut in pbar:
            ok, kind, npz_path_str, extra, sec = fut.result()
            counts[kind] += 1
            running.discard(npz_path_str)

            p = Path(npz_path_str)

            if ok:
                out_mp4 = Path(extra)
                tqdm.write(f"[OK] {rel(p)} -> {rel(out_mp4)} [{sec:.2f}s]")
            else:
                if kind == "err":
                    tqdm.write(f"[ERR] {rel(p)} : {extra} [{sec:.2f}s]")
                else:
                    tqdm.write(f"[SKIP] {rel(p)} ({kind}) [{sec:.2f}s]")

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
