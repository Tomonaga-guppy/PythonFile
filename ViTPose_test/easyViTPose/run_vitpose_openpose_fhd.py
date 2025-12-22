"""
easy_ViTPose を使って OpenPose BODY_25 形式で画像群を処理するサンプルコード
"""

import json
from pathlib import Path
import re

import cv2
from tqdm import tqdm
from easy_ViTPose import VitInference


# =========================
# 設定（ここだけ調整）
# =========================
IMG_DIR = Path(r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\gopro\fr\undistorted_facemasked")
OUT_DIR = Path(r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\gopro\fr\vit_dir")

MODEL_PATH = Path(r"C:\Users\Tomson\StrokeProject\easy_ViTPose\models\vitpose-l-coco_25.pth")
YOLO_PATH  = Path(r"C:\Users\Tomson\StrokeProject\easy_ViTPose\yolov8x.pt")

MODEL_NAME = "l"
YOLO_SIZE = 640

FPS = 60.0
EVERY = 1

# 保存解像度（FHD）
OUT_W, OUT_H = 1920, 1080

# 描画
POINT_RADIUS = 3
LINE_THICKNESS = 2
CONF_TH = 0.2
JPEG_QUALITY = 95
# =========================


# =========================
# OpenPose BODY_25 定義
# =========================
POSE_KPT_COLORS = [
    (255,0,85),(255,0,0),(255,85,0),(255,170,0),(255,255,0),
    (170,255,0),(85,255,0),(0,255,0),(255,0,0),(0,255,85),
    (0,255,170),(0,255,255),(0,170,255),(0,85,255),(0,0,255),
    (255,0,170),(170,0,255),(255,0,255),(85,0,255),
    (0,0,255),(0,0,255),(0,0,255),
    (0,255,255),(0,255,255),(0,255,255),
]

POSE_PAIRS = [
    (1,8),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),
    (8,9),(9,10),(10,11),(8,12),(12,13),(13,14),
    (1,0),(0,15),(15,17),(0,16),(16,18),
    (14,19),(19,20),(14,21),(11,22),(22,23),(11,24),
]
POSE_LINK_COLORS = [
    (255,0,85),(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),
    (0,255,0),(255,0,0),(0,255,85),(0,255,170),(0,255,255),(0,170,255),
    (0,85,255),(255,0,170),(170,0,255),(255,0,255),(85,0,255),
    (0,0,255),(0,0,255),(0,0,255),(0,255,255),(0,255,255),(0,255,255),
]

# easy_ViTPose coco_25 → OpenPose BODY_25
ORDER_MAP = [
    0,   # Nose
    5,   # Neck
    7,   # RShoulder
    9,   # RElbow
    11,  # RWrist
    6,   # LShoulder
    8,   # LElbow
    10,  # LWrist
    14,  # MidHip
    13,  # RHip
    16,  # RKnee
    18,  # RAnkle
    12,  # LHip
    15,  # LKnee
    17,  # LAnkle
    2,   # REye
    1,   # LEye
    4,   # REar
    3,   # LEar
    19,  # LBigToe
    20,  # LSmallToe
    21,  # LHeel
    22,  # RBigToe
    23,  # RSmallToe
    24,  # RHeel
]


# =========================
# Utility
# =========================
def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def blend(c1, c2):
    return ((c1[0]+c2[0])//2, (c1[1]+c2[1])//2, (c1[2]+c2[2])//2)

def draw_openpose(img_rgb, kpts, conf_th):
    # links: 両端のキーポイント色を平均（OpenPoseっぽい）
    for (a, b) in POSE_PAIRS:
        if kpts[a,2] > conf_th and kpts[b,2] > conf_th:
            xa, ya = int(kpts[a,1]), int(kpts[a,0])
            xb, yb = int(kpts[b,1]), int(kpts[b,0])
            cv2.line(img_rgb, (xa,ya), (xb,yb),
                     blend(POSE_KPT_COLORS[a], POSE_KPT_COLORS[b]),
                     LINE_THICKNESS)

    # points: キーポイント色（あなたの配列をそのまま）
    for i in range(25):
        y, x, s = kpts[i]
        if s > conf_th:
            cv2.circle(img_rgb, (int(x),int(y)), POINT_RADIUS, POSE_KPT_COLORS[i], -1)



# =========================
# Main
# =========================
def main():
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

    out_dir = ensure_dir(OUT_DIR)
    vis_dir = ensure_dir(out_dir / "vis")
    json_path = out_dir / "keypoints.jsonl"

    imgs = sorted(
        [p for ext in ("*.jpg","*.png") for p in IMG_DIR.glob(ext)],
        key=natural_key
    )

    fjson = open(json_path, "w", encoding="utf-8")

    try:
        for idx, p in enumerate(tqdm(imgs, desc="images")):
            if idx % EVERY != 0:
                continue

            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            sx, sy = OUT_W / w, OUT_H / h

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            kpts_dict = model.inference(img_rgb)

            vis = cv2.resize(img_rgb, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

            for kpts in kpts_dict.values():
                # 並び替え（OpenPose順）
                kpts_op = kpts[ORDER_MAP].copy()

                # スケール（[y, x, score]）
                kpts_op[:,0] *= sy
                kpts_op[:,1] *= sx

                draw_openpose(vis, kpts_op, CONF_TH)

            out_img = vis_dir / f"{idx:06d}.jpg"
            cv2.imwrite(
                str(out_img),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            fjson.write(json.dumps({
                "frame_index": idx,
                "timestamp_sec": idx / FPS,
                "image_name": p.name,
                "people": [
                    {"id": int(pid), "keypoints": k.tolist()}
                    for pid, k in kpts_dict.items()
                ]
            }, ensure_ascii=False) + "\n")

    finally:
        model.reset()
        fjson.close()

    print("\n[OK] Done")
    print(f"- images (FHD): {vis_dir}")
    print(f"- json: {json_path}")
    print("\nmp4作成例:")
    print(f'ffmpeg -y -framerate {FPS} -i "{vis_dir}\\%06d.jpg" '
          f'-c:v libx264 -pix_fmt yuv420p "{out_dir}\\result.mp4"')


if __name__ == "__main__":
    main()
