"""
キーポイントのindexを確認する用のスクリプト
"""

from pathlib import Path
import re
import cv2
from easy_ViTPose import VitInference


# =========================
# 設定（ここだけ調整）
# =========================
IMG_DIR = Path(r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\gopro\fr\undistorted_facemasked")

# どの画像を使うか（Noneなら自然順の先頭）
TARGET_IMAGE = None  # 例: "000123.jpg" とか

OUT_DIR = Path(r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\gopro\fr\vit_dir_debug")
OUT_NAME = "debug_indices.jpg"  # 出力ファイル名

MODEL_PATH = Path(r"C:\Users\Tomson\StrokeProject\easy_ViTPose\models\vitpose-l-coco_25.pth")
YOLO_PATH  = Path(r"C:\Users\Tomson\StrokeProject\easy_ViTPose\yolov8x.pt")
MODEL_NAME = "l"
YOLO_SIZE = 640

# 保存解像度（FHD）
OUT_W, OUT_H = 1920, 1080

# 描画
CONF_TH = 0.2
POINT_RADIUS = 2
TEXT_SCALE = 0.3
TEXT_THICK_BG = 3   # 白縁
TEXT_THICK_FG = 1   # 黒字
# =========================

"""
easy_ViTPoseのキーポイント位置メモ
0: nose
1: LEye
2: REye
3: LEar
4: REar
5: Neck
6: LShoulder
7: RShoulder
8:LElbow
9:RElbow
10: LWrist
11: RWrist
12: LHip
13: RHip
14: MidHip
15: LKnee
16: RKnee
17: LAnkle
18: RAnkle
19:LBigToe
20:LSmallToe
21:LHeel
22:RBigToe
23:RSmallToe
24:RHeel
"""

def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    if not IMG_DIR.is_dir():
        raise RuntimeError(f"Not a directory: {IMG_DIR}")

    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        imgs.extend(IMG_DIR.glob(ext))
    imgs = sorted(imgs, key=natural_key)

    if not imgs:
        raise RuntimeError(f"No images found in: {IMG_DIR}")

    if TARGET_IMAGE is None:
        img_path = imgs[0]
    else:
        img_path = IMG_DIR / TARGET_IMAGE
        if not img_path.is_file():
            raise FileNotFoundError(f"TARGET_IMAGE not found: {img_path}")

    print("[INFO] input:", img_path)

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

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    h, w = img_bgr.shape[:2]
    sx, sy = OUT_W / w, OUT_H / h
    print(f"[INFO] orig w,h = {w},{h}  -> FHD {OUT_W},{OUT_H}  sx,sy = {sx:.6f},{sy:.6f}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    kpts_dict = model.inference(img_rgb)  # {pid: (25,3) [y,x,score]}

    # 背景画像（FHD）
    vis = cv2.resize(img_rgb, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

    # 人ごとに index 番号を書いていく
    for pid, kpts in kpts_dict.items():
        k = kpts.copy()

        # k is [y, x, score] → FHDスケール
        k[:, 0] *= sy  # y
        k[:, 1] *= sx  # x

        for i in range(k.shape[0]):
            y, x, s = k[i]
            if s < CONF_TH:
                continue

            xi, yi = int(x), int(y)

            # 点
            cv2.circle(vis, (xi, yi), POINT_RADIUS, (0, 255, 0), -1)

            # index文字（白縁→黒字で読みやすく）
            txt = str(i)
            org = (xi + 6, yi - 6)
            cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (255, 255, 255), TEXT_THICK_BG, cv2.LINE_AA)
            cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 0, 0), TEXT_THICK_FG, cv2.LINE_AA)

        # person id も左上に出す（複数人がいる場合の区別）
        cv2.putText(vis, f"person_id={pid}", (20, 40 + 30 * int(pid)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(vis, f"person_id={pid}", (20, 40 + 30 * int(pid)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 1, cv2.LINE_AA)

    out_dir = ensure_dir(OUT_DIR)
    out_path = out_dir / OUT_NAME
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

    try:
        model.reset()
    except Exception:
        pass

    print("[OK] saved:", out_path)


if __name__ == "__main__":
    main()
