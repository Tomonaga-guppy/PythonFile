"""
easy_ViTPose を使って OpenPose BODY_25 形式で画像群を処理
- FHDで可視化画像を書き出し(frame_00000_rendered.jpg)
- cv2.VideoWriter で mp4 も同時に出力
- OpenPose互換の「フレームごとの json (frame_00000_keypoints.json)」を出力
- 推論と保存を並行化（Queue + writer thread）
"""

import json
import re
import threading
import queue
from pathlib import Path

import cv2
from tqdm import tqdm
from easy_ViTPose import VitInference


# =========================
# 設定（ここだけ調整）
# =========================
# IMG_DIR = Path(r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\gopro\fr\undistorted_facemasked")
# OUT_DIR = Path(r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\gopro\fr\vit_dir")
IMG_DIR = Path(r"C:\Users\Tomson\Desktop\vitpose_kasa\pa5_pt1_cali_frames")
OUT_DIR = Path(r"C:\Users\Tomson\Desktop\vitpose_kasa\pa5_pt1_cali_vitpose_results")

MODEL_PATH = Path(r"C:\Users\Tomson\StrokeProject\easy_ViTPose\models\vitpose-l-coco_25.pth")
YOLO_PATH  = Path(r"C:\Users\Tomson\StrokeProject\easy_ViTPose\yolov8x.pt")

MODEL_NAME = "l"
YOLO_SIZE = 640

FPS = 60.0
EVERY = 1

# 出力
OUT_W, OUT_H = 1920, 1080
SAVE_JPG = True
SAVE_VIDEO = True
VIDEO_NAME = "vit_result.mp4"   # mp4vで書く
JPEG_QUALITY = 95

# OpenPose互換JSONの出力先（OUT_DIR直下に "keypoints" を作成）
KEYPOINTS_DIR_NAME = "keypoints"
FRAME_DIGITS = 5  # frame_00000_keypoints.json の "00000" の桁数

# 描画
POINT_RADIUS = 3
LINE_THICKNESS = 2
CONF_TH = 0.2

# 並行化(処理ちょっとだけ早くするため,不要な場合は値を1にするか削除)
QUEUE_MAX = 16
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

# easy_ViTPose coco_25 → OpenPose BODY_25（あなたの確定結果）
ORDER_MAP = [
    0, 5, 7, 9, 11, 6, 8, 10, 14, 13, 16, 18, 12, 15, 17, 2, 1, 4, 3, 19, 20, 21, 22, 23, 24
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

def draw_openpose(img_rgb, kpts_yx, conf_th):
    # links: 両端のキーポイント色を平均（見た目がOpenPose寄り）
    for a, b in POSE_PAIRS:
        if kpts_yx[a,2] > conf_th and kpts_yx[b,2] > conf_th:
            xa, ya = int(kpts_yx[a,1]), int(kpts_yx[a,0])
            xb, yb = int(kpts_yx[b,1]), int(kpts_yx[b,0])
            cv2.line(img_rgb, (xa,ya), (xb,yb), blend(POSE_KPT_COLORS[a], POSE_KPT_COLORS[b]), LINE_THICKNESS)

    # points
    for i in range(25):
        y, x, s = kpts_yx[i]
        if s > conf_th:
            cv2.circle(img_rgb, (int(x), int(y)), POINT_RADIUS, POSE_KPT_COLORS[i], -1)

def to_openpose_person_dict(k_op_yx):
    """
    k_op_yx: (25,3) [y, x, score] （OpenPose順）
    OpenPoseの frame_xxxxx_keypoints.json の people[i] 形式へ変換する
    """
    # OpenPoseは [x, y, score] を 25点ぶんフラットに持つ
    flat = []
    for y, x, s in k_op_yx:
        flat.extend([float(x), float(y), float(s)])

    return {
        "person_id": [-1],               # OpenPoseのtracking無し相当
        "pose_keypoints_2d": flat,       # BODY_25
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
        # OpenH264 dll問題を避けるため、まず mp4v を使う
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(video_path), fourcc, float(FPS), (OUT_W, OUT_H))
        if not vw.isOpened():
            raise RuntimeError("VideoWriter open failed. Try changing codec/name (e.g., VIDEO_NAME='result.avi' and fourcc='XVID').")

    try:
        while True:
            item = q.get()
            if item is None:
                break

            frame_idx, vis_rgb, openpose_frame = item

            # 1) 可視化画像
            if SAVE_JPG:
                out_img = vis_dir / f"frame_{frame_idx:05d}_rendered.jpg"
                cv2.imwrite(
                    str(out_img),
                    cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, int(JPEG_QUALITY)],
                )

            # 2) 動画
            if vw is not None:
                vw.write(cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))

            # 3) OpenPose互換 JSON（フレームごと）
            out_json = keypoints_dir / f"frame_{frame_idx:0{FRAME_DIGITS}d}_keypoints.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(openpose_frame, f, ensure_ascii=False)

            q.task_done()

    finally:
        if vw is not None:
            vw.release()


# =========================
# Main
# =========================
def main():
    out_dir = ensure_dir(OUT_DIR)
    vis_dir = ensure_dir(out_dir / "vis")
    keypoints_dir = ensure_dir(out_dir / KEYPOINTS_DIR_NAME)
    video_path = out_dir / VIDEO_NAME

    imgs = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
        imgs.extend(IMG_DIR.glob(ext))
    imgs = sorted(imgs, key=natural_key)

    if not imgs:
        raise RuntimeError(f"No images found in: {IMG_DIR}")

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

    q = queue.Queue(maxsize=int(QUEUE_MAX))
    th = threading.Thread(target=writer_worker, args=(q, vis_dir, keypoints_dir, video_path), daemon=True)
    th.start()

    try:
        frame_out_idx = 0
        for src_idx, p in enumerate(tqdm(imgs, desc="images")):
            if src_idx % EVERY != 0:
                continue

            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            sx, sy = OUT_W / w, OUT_H / h

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            kpts_dict = model.inference(img_rgb)

            vis = cv2.resize(img_rgb, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

            people = []
            for pid, kpts in kpts_dict.items():
                # OpenPose順へ並び替え (25,3) [y,x,score]
                k_op_4k = kpts[ORDER_MAP].copy()

                # FHDスケール
                k_op_fhd = k_op_4k.copy()
                k_op_fhd[:,0] *= sy
                k_op_fhd[:,1] *= sx

                # 描画
                draw_openpose(vis, k_op_fhd , CONF_TH)

                # OpenPoseの people[i] 形式へ
                people.append(to_openpose_person_dict(k_op_4k))

            openpose_frame = {
                "people": people,
            }

            # 保存側へ渡す（キューが詰まったらここで待つ＝メモリ暴走防止）
            q.put((frame_out_idx, vis, openpose_frame))
            frame_out_idx += 1

    finally:
        try:
            model.reset()
        except Exception:
            pass

        q.put(None)
        th.join()

    print("\n[OK] Done")
    print(f"- vis jpg     : {vis_dir} (SAVE_JPG={SAVE_JPG})")
    print(f"- keypoints   : {keypoints_dir} (OpenPose-style json per frame)")
    if SAVE_VIDEO:
        print(f"- video       : {video_path} (codec=mp4v)")


if __name__ == "__main__":
    main()
