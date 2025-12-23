"""
ViTPose（easy_ViTPose）で OpenPose互換JSONを出力する一括処理スクリプト（1_2の探索フロー対応）

探索:
  root_dir/sub*/thera*/gopro/(fl|fr)/undistorted_facemasked を対象

実行条件:
  どの条件でも undistorted_facemasked のみ推論

出力:
  .../gopro/<direction>/ViTPose/
    ├─ images/
    │   ├─ frame_00000_rendered.jpg
    │   └─ ...
    ├─ json/
    │   ├─ frame_00000_keypoints.json   (OpenPose互換)
    │   └─ ...
    └─ vitpose.mp4

特徴:
  - 推論と保存を並行化（Queue + writer thread）
  - 既に vitpose.mp4 があればスキップ可能
  - ディレクトリ単位で処理時間 / 累計 / 残り予測を表示
"""

import json
import re
import threading
import queue
import time
from pathlib import Path

import cv2
from tqdm import tqdm
from easy_ViTPose import VitInference


# =========================
# 設定（ここだけ調整）
# =========================
root_dir = Path(r"G:\gait_pattern\BR9G_shuron")
directions = ["fl", "fr"]

# 入力ディレクトリ名（固定）
IN_DIR_NAME = "undistorted_facemasked"

# 出力ディレクトリ名（direction直下）
OUT_DIR_NAME = "ViTPose"

# thera0-1 などを除外する
EXCLUDE_THERA0_1 = True

# 既に出力があれば飛ばす（動画ファイルがあればSKIP）
SKIP_IF_OUTPUT_EXISTS = True

# ===== ViTPose/easy_ViTPose =====
MODEL_PATH = Path(r"C:\Users\Tomson\StrokeProject\easy_ViTPose\models\vitpose-l-coco_25.pth")
YOLO_PATH  = Path(r"C:\Users\Tomson\StrokeProject\easy_ViTPose\yolov8x.pt")

MODEL_NAME = "l"
YOLO_SIZE = 640

# 動画fps（元の撮影fps）
FPS = 60.0
EVERY = 1  # 1=全フレーム、2=1/2 など

# 出力サイズ（可視化/動画）
OUT_W, OUT_H = 1920, 1080

# 保存
SAVE_JPG = True
SAVE_VIDEO = True
VIDEO_NAME = "vitpose.mp4"  # mp4vで書く
JPEG_QUALITY = 95

# OpenPose互換JSON
KEYPOINTS_DIR_NAME = "json"
FRAME_DIGITS = 5  # frame_00000_keypoints.json の "00000" の桁数

# 描画
POINT_RADIUS = 3
LINE_THICKNESS = 2
CONF_TH = 0.2

# 並行化（保存側キュー最大：大きすぎるとメモリ増）
QUEUE_MAX = 16
# =========================


# =========================
# OpenPose BODY_25 定義
# =========================
POSE_KPT_COLORS = [
    (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (255, 0, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255),
    (255, 0, 170), (170, 0, 255), (255, 0, 255), (85, 0, 255),
    (0, 0, 255), (0, 0, 255), (0, 0, 255),
    (0, 255, 255), (0, 255, 255), (0, 255, 255),
]

POSE_PAIRS = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
    (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24),
]

# easy_ViTPose coco_25 → OpenPose BODY_25（あなたの流れで使っていた確定マップ）
ORDER_MAP = [
    0, 5, 7, 9, 11, 6, 8, 10, 14, 13, 16, 18, 12, 15, 17, 2, 1, 4, 3, 19, 20, 21, 22, 23, 24
]


# =========================
# 時間表示
# =========================
def sec_to_hms(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# =========================
# Utility（探索まわり）
# =========================
def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def has_images(img_dir: Path) -> bool:
    if not img_dir.exists():
        return False
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        if any(img_dir.glob(ext)):
            return True
    return False


def list_subject_dirs(root: Path):
    return [d for d in root.iterdir() if d.is_dir() and d.name.startswith("sub")]


def list_thera_dirs(subject_dir: Path):
    thera = [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("thera")]

    if EXCLUDE_THERA0_1:
        # thera0-1*, thera0-2_1* を除外
        thera = [d for d in thera if not d.name.startswith("thera0-1")]
        thera = [d for d in thera if not d.name.startswith("thera0-2_1")]
        # thera1-1* ～ thera6-1* を除外
        exclude_patterns = [f"thera{i}-1" for i in range(1, 7)]
        for pattern in exclude_patterns:
            thera = [d for d in thera if not d.name.startswith(pattern)]

    return sorted(thera, key=natural_key)


def out_dir_for_direction(direction_dir: Path) -> Path:
    """
    direction / ViTPose を返す
      .../gopro/fr -> .../gopro/fr/ViTPose
    """
    return direction_dir / OUT_DIR_NAME


# =========================
# Utility（描画/JSON）
# =========================
def blend(c1, c2):
    return ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2, (c1[2] + c2[2]) // 2)


def draw_openpose(img_rgb, kpts_yx, conf_th):
    # links
    for a, b in POSE_PAIRS:
        if kpts_yx[a, 2] > conf_th and kpts_yx[b, 2] > conf_th:
            xa, ya = int(kpts_yx[a, 1]), int(kpts_yx[a, 0])
            xb, yb = int(kpts_yx[b, 1]), int(kpts_yx[b, 0])
            cv2.line(
                img_rgb,
                (xa, ya),
                (xb, yb),
                blend(POSE_KPT_COLORS[a], POSE_KPT_COLORS[b]),
                LINE_THICKNESS,
            )

    # points
    for i in range(25):
        y, x, s = kpts_yx[i]
        if s > conf_th:
            cv2.circle(img_rgb, (int(x), int(y)), POINT_RADIUS, POSE_KPT_COLORS[i], -1)


def to_openpose_person_dict(k_op_yx, person_id=-1):
    """
    k_op_yx: (25,3) [y, x, score] （OpenPose順）
    OpenPoseの frame_xxxxx_keypoints.json の people[i] 形式へ変換
    """
    flat = []
    for y, x, s in k_op_yx:
        flat.extend([float(x), float(y), float(s)])

    return {
        "person_id": [int(person_id)],
        "pose_keypoints_2d": flat,
        "face_keypoints_2d": [],
        "hand_left_keypoints_2d": [],
        "hand_right_keypoints_2d": [],
        "pose_keypoints_3d": [],
        "face_keypoints_3d": [],
        "hand_left_keypoints_3d": [],
        "hand_right_keypoints_3d": [],
    }


# =========================
# writer thread
# =========================
def writer_worker(q: "queue.Queue", vis_dir: Path, keypoints_dir: Path, video_path: Path):
    vw = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(video_path), fourcc, float(FPS), (OUT_W, OUT_H))
        if not vw.isOpened():
            raise RuntimeError(
                "VideoWriter open failed. codec/name を変更してください。"
                "例: VIDEO_NAME='result.avi', fourcc='XVID'"
            )

    try:
        while True:
            item = q.get()
            if item is None:
                break

            frame_idx, vis_rgb, openpose_frame = item

            # 1) vis jpg
            if SAVE_JPG:
                out_img = vis_dir / f"frame_{frame_idx:05d}_rendered.jpg"
                cv2.imwrite(
                    str(out_img),
                    cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, int(JPEG_QUALITY)],
                )

            # 2) video
            if vw is not None:
                vw.write(cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))

            # 3) keypoints json
            out_json = keypoints_dir / f"frame_{frame_idx:0{FRAME_DIGITS}d}_keypoints.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(openpose_frame, f, ensure_ascii=False)

            q.task_done()

    finally:
        if vw is not None:
            vw.release()


# =========================
# 1ディレクトリ分の処理
# =========================
def process_image_dir(img_dir: Path, out_dir: Path, model: VitInference):
    out_dir = ensure_dir(out_dir)
    vis_dir = ensure_dir(out_dir / "images")
    keypoints_dir = ensure_dir(out_dir / KEYPOINTS_DIR_NAME)
    video_path = out_dir / VIDEO_NAME

    # 入力画像列挙
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        imgs.extend(img_dir.glob(ext))
    imgs = sorted(imgs, key=natural_key)

    if not imgs:
        raise RuntimeError(f"No images found in: {img_dir}")

    q = queue.Queue(maxsize=int(QUEUE_MAX))
    th = threading.Thread(target=writer_worker, args=(q, vis_dir, keypoints_dir, video_path), daemon=True)
    th.start()

    try:
        frame_out_idx = 0
        for src_idx, p in enumerate(tqdm(imgs, desc=f"images ({img_dir.parent.name}/{img_dir.name})", leave=False)):
            if src_idx % EVERY != 0:
                continue

            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            sx, sy = OUT_W / w, OUT_H / h

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # 推論（dict: {person_id: (25,3)} 形式）
            kpts_dict = model.inference(img_rgb)

            # 可視化キャンバス（FHD）
            vis = cv2.resize(img_rgb, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

            people = []
            for _pid, kpts in kpts_dict.items():
                # OpenPose順へ並び替え (25,3) [y,x,score]
                k_op_src = kpts[ORDER_MAP].copy()

                # 描画用にFHDへスケーリング
                k_op_fhd = k_op_src.copy()
                k_op_fhd[:, 0] *= sy
                k_op_fhd[:, 1] *= sx

                draw_openpose(vis, k_op_fhd, CONF_TH)

                # JSONは元画像座標（入力画像座標）のまま
                people.append(to_openpose_person_dict(k_op_src, person_id=-1))

            openpose_frame = {"people": people}

            # 保存側へ渡す（キューが詰まったら待つ＝メモリ暴走防止）
            q.put((frame_out_idx, vis, openpose_frame))
            frame_out_idx += 1

    finally:
        q.put(None)
        th.join()

    return video_path


# =========================
# Main
# =========================
def main():
    subject_dirs = sorted(list_subject_dirs(root_dir), key=natural_key)
    print(f"対象sub: {[d.name for d in subject_dirs]}")

    # 実行予定数を先に数える（SKIP適用後）
    total_runs = 0
    for subject_dir in subject_dirs:
        for thera_dir in list_thera_dirs(subject_dir):
            for direction in directions:
                direction_dir = thera_dir / "gopro" / direction
                img_dir = direction_dir / IN_DIR_NAME
                if not has_images(img_dir):
                    continue

                out_dir = out_dir_for_direction(direction_dir)
                out_video = out_dir / VIDEO_NAME
                if SKIP_IF_OUTPUT_EXISTS and out_video.exists():
                    continue
                total_runs += 1

    print(f"ViTPose 実行予定数: {total_runs}")

    # モデルは一度だけロードして使い回す
    model = VitInference(
        str(MODEL_PATH),
        str(YOLO_PATH),
        model_name=MODEL_NAME,
        yolo_size=YOLO_SIZE,
        is_video=False,
        device=None,
        dataset="coco_25",
        det_class="human",
    )

    run_count = 0
    total_elapsed = 0.0
    t_all_start = time.perf_counter()

    try:
        for subject_dir in subject_dirs:
            thera_dirs = list_thera_dirs(subject_dir)
            print(f"\n[{subject_dir.name}] thera: {[d.name for d in thera_dirs]}")

            for thera_dir in thera_dirs:
                for direction in directions:
                    direction_dir = thera_dir / "gopro" / direction
                    img_dir = direction_dir / IN_DIR_NAME
                    if not has_images(img_dir):
                        print(f"[SKIP] 入力が無い: {img_dir}")
                        continue

                    out_dir = out_dir_for_direction(direction_dir)
                    out_video = out_dir / VIDEO_NAME

                    if SKIP_IF_OUTPUT_EXISTS and out_video.exists():
                        print(f"[SKIP] すでに存在: {out_video}")
                        continue

                    run_count += 1
                    print(f"\n[RUN {run_count}/{total_runs}] {img_dir}")
                    print(f"  out: {out_dir}")

                    t0 = time.perf_counter()
                    try:
                        process_image_dir(img_dir, out_dir, model)
                    except Exception as e:
                        print(f"[ERROR] {img_dir}: {e}")
                    dt = time.perf_counter() - t0

                    total_elapsed += dt
                    avg = total_elapsed / max(run_count, 1)
                    remain = max(total_runs - run_count, 0)
                    est_remain = avg * remain

                    print(
                        f"[DONE {run_count}/{total_runs}] "
                        f"今回: {sec_to_hms(dt)} / "
                        f"累計: {sec_to_hms(total_elapsed)} / "
                        f"平均: {sec_to_hms(avg)} / "
                        f"残り予測: {sec_to_hms(est_remain)}"
                    )

    finally:
        try:
            model.reset()
        except Exception:
            pass

    t_all = time.perf_counter() - t_all_start
    print("\n" + "=" * 70)
    print("すべて完了")
    print(f"ViTPose実行回数: {run_count}/{total_runs}")
    print(f"総経過時間(全体): {sec_to_hms(t_all)}  (処理累計: {sec_to_hms(total_elapsed)})")
    print("=" * 70)


if __name__ == "__main__":
    main()
