"""
easyVitPoseを使って姿勢推定を行うサンプルコード（画像フォルダ入力版)
"""


import json
from pathlib import Path
import cv2
from tqdm import tqdm

from easy_ViTPose import VitInference


# =========================
# ここだけ書き換えればOK
# =========================
IMG_DIR = Path(r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\gopro\fr\undistorted_facemasked")
OUT_DIR = Path(r"outputs_api_img")  # easy_ViTPose直下からの相対でもOK

MODEL_PATH = Path(r"C:\Users\Tomson\StrokeProject\easy_ViTPose\models\vitpose-l-coco_25.pth")
YOLO_PATH  = Path(r"C:\Users\Tomson\StrokeProject\easy_ViTPose\yolov8x.pt")


MODEL_NAME = "l"        # s / b / l / h
YOLO_SIZE = 640         # 640 推奨（重いなら 512/416 など）
FPS = 60.0              # timestamp用
EVERY = 1               # 1=全フレーム、2=1/2間引き
SHOW_YOLO = False       # bboxも描きたいなら True
JPEG_QUALITY = 95
# =========================


def natural_key(p: Path):
    import re
    s = p.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def main():
    root = Path(__file__).resolve().parent

    img_dir = IMG_DIR
    if not img_dir.is_dir():
        raise RuntimeError(f"Not a directory: {img_dir}")

    model_path = (root / MODEL_PATH).resolve() if not MODEL_PATH.is_absolute() else MODEL_PATH
    yolo_path = (root / YOLO_PATH).resolve() if not YOLO_PATH.is_absolute() else YOLO_PATH

    if not model_path.is_file():
        raise FileNotFoundError(f"ViTPose model not found: {model_path}")
    if not yolo_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {yolo_path}")

    out_dir = ensure_dir((root / OUT_DIR).resolve() if not OUT_DIR.is_absolute() else OUT_DIR)
    vis_dir = ensure_dir(out_dir / "vis")
    json_path = out_dir / "keypoints.jsonl"  # 1 line per frame

    # 入力が連続画像なので is_video=False（追跡IDが要らない前提）
    model = VitInference(
        str(model_path),
        str(yolo_path),
        model_name=MODEL_NAME,
        yolo_size=YOLO_SIZE,
        is_video=False,
        device=None,        # cuda -> mps -> cpu 自動
        dataset="coco_25",
        det_class="human",
    )

    # 画像一覧
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        imgs.extend(img_dir.glob(ext))
    imgs = sorted(imgs, key=natural_key)

    if not imgs:
        raise RuntimeError(f"No images found in: {img_dir}")

    fjson = open(json_path, "w", encoding="utf-8")

    try:
        for idx, p in enumerate(tqdm(imgs, desc="images")):
            if idx % EVERY != 0:
                continue

            frame_bgr = cv2.imread(str(p))
            if frame_bgr is None:
                continue

            img_rgb = bgr_to_rgb(frame_bgr)

            # 推論
            kpts_dict = model.inference(img_rgb)  # {person_id: np.ndarray (25,3)}

            # 可視化（RGB）
            vis_rgb = model.draw(show_yolo=SHOW_YOLO)

            # 画像保存（連番）
            out_img = vis_dir / f"{idx:06d}.jpg"
            cv2.imwrite(str(out_img), rgb_to_bgr(vis_rgb), [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])

            # JSONL（フレーム単位）
            people = []
            for pid, kpts in kpts_dict.items():
                people.append({
                    "id": int(pid),
                    "keypoints": kpts.tolist(),
                })

            rec = {
                "frame_index": int(idx),
                "timestamp_sec": float(idx / FPS),
                "image_name": p.name,
                "people": people,
            }
            fjson.write(json.dumps(rec, ensure_ascii=False) + "\n")

    finally:
        try:
            model.reset()
        except Exception:
            pass
        fjson.close()

    print("\n[OK] Done")
    print(f"- input dir: {img_dir}")
    print(f"- vis frames: {vis_dir}")
    print(f"- keypoints jsonl: {json_path}")
    print("\nIf you want MP4 (example 60fps):")
    print(f'ffmpeg -y -framerate {FPS} -i "{vis_dir}\\%06d.jpg" -c:v libx264 -pix_fmt yuv420p "{out_dir}\\result.mp4"')


if __name__ == "__main__":
    main()
