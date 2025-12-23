"""
余分な顔部分を黒塗りするスクリプト
現状FRのみ実施
"""


from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# 設定
# =========================
root_dir = Path(r"G:\gait_pattern\BR9G_shuron")
directions = ["fl", "fr"]

EXCLUDE_THERA0_1 = True  #除外する条件を設定
SKIP_IF_OUTPUT_EXISTS = True
EXPAND_SCALE = 2.0

# マルチスレッド設定
IO_WORKERS = 8
PREFETCH = 16

ROI_BY_DIRECTION = {
    "fr": dict(th_x1=2100, th_x2=2800, th_y1=900, th_y2=1400),
}

#出力ディレクトリ名
OUT_DIR_NAME = "undistorted_facemasked"


def apply_black_fill_expand(img, x1, y1, x2, y2, expand_scale=2.0):
    h, w = img.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    new_bw = int(bw * expand_scale)
    new_bh = int(bh * expand_scale)

    nx1 = max(0, cx - new_bw // 2)
    ny1 = max(0, cy - new_bh // 2)
    nx2 = min(w, cx + new_bw // 2)
    ny2 = min(h, cy + new_bh // 2)

    cv2.rectangle(img, (nx1, ny1), (nx2, ny2), (0, 0, 0), thickness=-1)
    return img


def should_fill_box(direction: str, box_xyxy: tuple[int, int, int, int]) -> bool:
    if direction not in ROI_BY_DIRECTION:
        return False
    x1, y1, x2, y2 = box_xyxy
    roi = ROI_BY_DIRECTION[direction]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (roi["th_x1"] < cx < roi["th_x2"] and roi["th_y1"] < cy < roi["th_y2"])


def list_subject_dirs(root: Path):
    return [d for d in root.iterdir() if d.is_dir() and d.name.startswith("sub")]


def list_thera_dirs(subject_dir: Path):
    thera = [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("thera")]
    if EXCLUDE_THERA0_1:
        thera = [d for d in thera if not d.name.startswith("thera0-1")]
        thera = [d for d in thera if not d.name.startswith("thera0-2_1")]
        # thera1-0_1 から thera6-0_1 も除外
        exclude_patterns = [f"thera{i}-1" for i in range(1, 7)]
        for pattern in exclude_patterns:
            thera = [d for d in thera if not d.name.startswith(pattern)]
    return thera


def get_undistorted_dir(thera_dir: Path, direction: str) -> Path:
    return thera_dir / "gopro" / direction / "undistorted"


# ---------- I/O worker ----------
def read_image(path: Path):
    img = cv2.imread(str(path))
    return path, img


def write_image(path: Path, img):
    cv2.imwrite(str(path), img)
    return path


def copy_one(src: Path, dst: Path):
    shutil.copy2(src, dst)
    return src.name


def copy_pngs_multithread(src_dir: Path, dst_dir: Path, workers: int):
    dst_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted(src_dir.glob("*.png"))
    if not pngs:
        return 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(copy_one, p, dst_dir / p.name) for p in pngs]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="copy(fl)", unit="file", leave=False):
            pass
    return len(pngs)


def process_fr_with_prefetch(model: YOLO, pngs: list[Path], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=IO_WORKERS) as io_ex:
        read_futs = {}
        write_futs = []

        it = iter(pngs)
        for _ in range(min(PREFETCH, len(pngs))):
            p = next(it, None)
            if p is None:
                break
            read_futs[p] = io_ex.submit(read_image, p)

        for i in tqdm(range(len(pngs)), desc="frames(fr)", unit="frame", leave=False):
            p0 = pngs[i]
            fut = read_futs.pop(p0, None)
            if fut is None:
                path, img = read_image(p0)
            else:
                path, img = fut.result()

            p_next = next(it, None)
            if p_next is not None:
                read_futs[p_next] = io_ex.submit(read_image, p_next)

            if img is None:
                continue

            # 推論は1本（GPU競合回避）
            results = model(img, verbose=False)

            boxes = []
            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                    boxes.append((x1, y1, x2, y2))

            boxes_to_fill = [bb for bb in boxes if should_fill_box("fr", bb)]
            for (x1, y1, x2, y2) in boxes_to_fill:
                img = apply_black_fill_expand(img, x1, y1, x2, y2, expand_scale=EXPAND_SCALE)

            out_path = out_dir / path.name
            write_futs.append(io_ex.submit(write_image, out_path, img))
            
            # 書き込みfutureを溜めすぎない（メモリ安定化）
            if len(write_futs) >= 128:
                for fut in as_completed(write_futs[:64]):
                    fut.result()
                write_futs = write_futs[64:]

        for fut in tqdm(as_completed(write_futs), total=len(write_futs), desc="write(fr)", unit="file", leave=False):
            _ = fut.result()


def main():
    # OpenCV内部のスレッド競合を抑える（スレッドI/Oと喧嘩しやすい）
    cv2.setNumThreads(0)

    model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11x-face-detection", filename="model.pt")
    model = YOLO(model_path)

    subject_dirs = list_subject_dirs(root_dir)
    subject_dirs = [d for d in subject_dirs if int(d.name.replace("sub", "")) < 7]

    tasks = []
    for sub_dir in subject_dirs:
        for thera_dir in list_thera_dirs(sub_dir):
            for direction in directions:
                undist_dir = get_undistorted_dir(thera_dir, direction)
                if not undist_dir.exists():
                    continue
                pngs = sorted(undist_dir.glob("*.png"))
                if not pngs:
                    continue
                out_dir = undist_dir.parent / OUT_DIR_NAME
                tasks.append((direction, undist_dir, out_dir, pngs))

    print(f"検出したタスク数: {len(tasks)}")

    done = 0
    skipped = 0

    for direction, undist_dir, out_dir, pngs in tqdm(tasks, desc="全体(フォルダ)", unit="task"):
        if SKIP_IF_OUTPUT_EXISTS and out_dir.exists():
            tqdm.write(f"[SKIP] 出力dirが既に存在: {out_dir}")
            skipped += 1
            continue

        if direction == "fl":
            tqdm.write(f"\nCOPY(fl): {undist_dir} -> {out_dir} (frames={len(pngs)})")
            n = copy_pngs_multithread(undist_dir, out_dir, workers=IO_WORKERS)
            tqdm.write(f"[OK] 完了(COPY): {out_dir} copied={n}")
            done += 1
            continue

        if direction == "fr":
            tqdm.write(f"\nMASK(fr): {undist_dir} -> {out_dir} (frames={len(pngs)})")
            process_fr_with_prefetch(model, pngs, out_dir)
            tqdm.write(f"[OK] 完了(MASK): {out_dir}")
            done += 1
            continue

    print("\n" + "=" * 70)
    print("すべての処理が完了しました。")
    print(f"OK={done}, SKIP={skipped}")
    print("=" * 70)


if __name__ == "__main__":
    main()
