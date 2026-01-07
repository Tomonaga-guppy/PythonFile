"""
介助歩行用
2人以上検出したらPTをマスクする目的
YOLO セグメンテーションで人物を検出し、赤マスクを重ねるスクリプト
左右のカメラに対応
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# 設定
# =========================
root_dir = Path(r"G:\gait_pattern\2025_shuron_BR9G")

directions = ["fl", "fr"]
OUT_DIR_NAME = "undistorted_seg"            # 出力
INPUT_DIR_NAME = "undistorted_facemasked"   # 入力

# 保存スレッド数（SSDなら 4〜8 推奨）
SAVE_WORKERS = 8

# 書き込みfutureを溜めすぎない（メモリ暴走防止）
WRITE_FUTURES_MAX = 128
WRITE_DRAIN_BATCH = 64

# =========================
# モデル
# =========================
model = YOLO("yolov8x-seg.pt")

# =========================
# ユーティリティ
# =========================
def apply_red_overlay(image, mask, color=(0, 0, 255), alpha=1.0):
    if mask is None or mask.size == 0:
        return image

    overlay = image.copy()
    overlay[mask > 0] = color
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, color, 2)
    return result


def list_subject_dirs(root: Path):
    return [d for d in root.iterdir() if d.is_dir() and d.name.startswith("sub")]


def list_thera_dirs(subject_dir: Path):
    thera = [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("thera")]

    # 介助歩行の条件のみ対象
    target_patterns = [f"thera{i}-0" for i in range(1, 11)]
    thera = [d for d in thera if any(d.name.startswith(pattern) for pattern in target_patterns)]
    return thera


def get_input_dir(thera_dir: Path, direction: str) -> Path:
    return thera_dir / "gopro" / direction / INPUT_DIR_NAME


def write_image(path: Path, img):
    cv2.imwrite(str(path), img)
    return path


# =========================
# メイン処理
# =========================
def main():
    # OpenCV内部スレッドで競合しやすいので抑制（保存スレッドと喧嘩しがち）
    cv2.setNumThreads(0)

    class_names = model.names
    person_class_id = next(
        (cid for cid, name in class_names.items() if str(name).lower() == "person"),
        None
    )
    if person_class_id is None:
        raise RuntimeError("person クラスが見つかりません")

    subject_dirs = list_subject_dirs(root_dir)

    # =========================
    # 全タスクを収集して全体進捗
    # =========================
    tasks = []
    for sub_dir in subject_dirs:
        for thera_dir in list_thera_dirs(sub_dir):
            for direction in directions:
                input_dir = get_input_dir(thera_dir, direction)
                if not input_dir.exists():
                    continue
                image_files = sorted(input_dir.glob("*.png"))
                if not image_files:
                    continue
                output_dir = input_dir.parent / OUT_DIR_NAME
                tasks.append((sub_dir, thera_dir, direction, input_dir, output_dir, image_files))

    print(f"検出したタスク数: {len(tasks)}")

    # 保存はスレッド、推論はメイン（直列）
    with ThreadPoolExecutor(max_workers=SAVE_WORKERS) as save_ex:
        # フォルダ（タスク）単位の全体進捗
        for sub_dir, thera_dir, direction, input_dir, output_dir, image_files in tqdm(
            tasks, desc="全体(フォルダ)", unit="task"
        ):
            # すでに処理済みならスキップ
            if output_dir.exists():
                existing_files = list(output_dir.glob("*.png"))
                if len(existing_files) == len(image_files):
                    tqdm.write(f"SKIP(already processed): {output_dir}")
                    continue
            
            output_dir.mkdir(exist_ok=True)
            tqdm.write(f"\nSEG: {input_dir} -> {output_dir} ({len(image_files)} frames)")

            write_futs = []

            # フレーム単位
            for img_path in tqdm(
                image_files,
                desc=f"{sub_dir.name}/{thera_dir.name}/{direction}",
                unit="frame",
                leave=False
            ):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                h, w = img.shape[:2]
                results = model(img, verbose=False, device=0)

                person_masks = []
                person_boxes = []

                for r in results:
                    if r.masks is None:
                        continue
                    masks = r.masks.data.cpu().numpy()
                    boxes = r.boxes.xyxy.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy().astype(int)

                    for mask, box, cls_id in zip(masks, boxes, classes):
                        if cls_id != person_class_id:
                            continue
                        mask = cv2.resize(mask, (w, h))
                        mask = (mask > 0.5).astype(np.uint8)
                        person_masks.append(mask)
                        person_boxes.append(tuple(map(int, box)))  # (x1,y1,x2,y2)

                # 2人以上検出されたら、PT側をマスク（左右で選択）
                if len(person_boxes) >= 2:
                    if direction == "fr":
                        # 左をPTとしてマスク（あなたのロジックを維持）
                        idx = min(range(len(person_boxes)), key=lambda i: person_boxes[i][2])
                    else:
                        # 右をPTとしてマスク（あなたのロジックを維持）
                        idx = max(range(len(person_boxes)), key=lambda i: person_boxes[i][0])

                    img = apply_red_overlay(img, person_masks[idx], alpha=1.0)

                # ★ 保存だけ別スレッドへ
                out_path = output_dir / img_path.name
                write_futs.append(save_ex.submit(write_image, out_path, img))

                # futureを溜めすぎない（メモリ安定化）
                if len(write_futs) >= WRITE_FUTURES_MAX:
                    for fut in as_completed(write_futs[:WRITE_DRAIN_BATCH]):
                        fut.result()
                    write_futs = write_futs[WRITE_DRAIN_BATCH:]

            # フォルダの残り保存を回収
            for fut in tqdm(
                as_completed(write_futs),
                total=len(write_futs),
                desc="write(flush)",
                unit="file",
                leave=False
            ):
                fut.result()

    print("\nすべて完了")


if __name__ == "__main__":
    main()