import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForKeypointDetection

from mmdet.apis import init_detector, inference_detector
from mmpose.apis import process_mmdet_results


# =========================
# 設定（ここだけ変える）
# =========================
VIDEO_PATH = r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\gopro\fl"

# MMDetection 人物検出（COCO person=1想定）
DET_CONFIG = r"C:\Users\Tomson\StrokeProject\ViTPose\demo\mmdetection_cfg\faster_rcnn_r50_fpn_coco.py"
DET_CHECKPOINT = r"C:\Users\Tomson\StrokeProject\ViTPose\models\faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

# 出力
OUT_DIR = r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\gopro\fl\vitpose_output"

# ここが本命：ViTPose+（HFのモデルID）
HF_MODEL_ID = "usyd-community/vitpose-plus-large"

# もし「今の wholebody.pth を上書きで使いたい」なら入れる（ダメなら自動で無視してHF重みで続行）
LOCAL_PTH = r"C:\Users\Tomson\StrokeProject\ViTPose\models\wholebody.pth"
TRY_LOAD_LOCAL_PTH = True

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DET_CAT_ID = 1      # COCO person
BBOX_THR = 0.30
KPT_THR = 0.30

# 出力はHD
OUT_W, OUT_H = 1920, 1080

# dataset_index=5 が WholeBody expert
DATASET_INDEX = 5


# =========================
# 便利関数
# =========================
def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2

def draw_kpts(img, kpts, thr=0.3, r=3):
    out = img
    for x, y, s in kpts:
        if s >= thr:
            cv2.circle(out, (int(x), int(y)), r, (0, 255, 0), -1)
    return out

def to_openpose_people(kpts_list, bboxes=None, version=1.3):
    people = []
    for i, kpts in enumerate(kpts_list):
        flat = []
        for x, y, s in kpts:
            flat.extend([float(x), float(y), float(s)])
        person = {
            "person_id": [-1],
            "pose_keypoints_2d": flat,
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": [],
        }
        if bboxes is not None:
            person["bbox"] = [float(v) for v in bboxes[i]]
        people.append(person)
    return {"version": version, "people": people}


# =========================
# メイン
# =========================
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir = out_dir / "json"
    json_dir.mkdir(exist_ok=True)
    vis_dir = out_dir / "vis_frames"
    vis_dir.mkdir(exist_ok=True)

    # 動画
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 60.0  # 取得できない場合の保険
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 出力動画
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_dir / "vitpose_vis.mp4"), fourcc, float(fps), (OUT_W, OUT_H))

    # Detector
    det_model = init_detector(DET_CONFIG, DET_CHECKPOINT, device=DEVICE)

    # ViTPose+ (Transformers)
    processor = AutoProcessor.from_pretrained(HF_MODEL_ID)
    pose_model = AutoModelForKeypointDetection.from_pretrained(HF_MODEL_ID).to(DEVICE).eval()

    # 可能ならローカルpthを上書きロード（失敗したらHF重みで続行）
    if TRY_LOAD_LOCAL_PTH and Path(LOCAL_PTH).exists():
        try:
            ckpt = torch.load(LOCAL_PTH, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt)
            # そのままだと key が一致しないこともあるので strict=False
            missing, unexpected = pose_model.load_state_dict(sd, strict=False)
            print("[INFO] Loaded LOCAL_PTH into HF model (strict=False)")
            print("  missing:", len(missing), " unexpected:", len(unexpected))
        except Exception as e:
            print("[WARN] Failed to load LOCAL_PTH; continue with HF weights.")
            print("  ", repr(e))

    dataset_index = torch.tensor([DATASET_INDEX], device=DEVICE)

    pbar = tqdm(total=total if total > 0 else None, desc="ViTPose+ (WholeBody) inference", unit="frame")

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]

        # 人物検出
        det_results = inference_detector(det_model, frame)
        persons = process_mmdet_results(det_results, DET_CAT_ID)

        all_kpts = []
        all_bboxes = []
        vis = frame.copy()

        # 各person bboxでcrop→pose
        for p in persons:
            bbox = p["bbox"]
            x1, y1, x2, y2 = bbox[:4]
            score = float(bbox[4]) if len(bbox) >= 5 else 1.0
            if score < BBOX_THR:
                continue

            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            inputs = processor(images=crop, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                # ここが重要：dataset_index を渡す
                out = pose_model(**inputs, dataset_index=dataset_index)

            # out.keypoints: (1, K, 2), out.scores: (1, K)
            kpts_xy = out.keypoints[0].detach().cpu().numpy()
            kpts_sc = out.scores[0].detach().cpu().numpy()

            kpts = np.zeros((kpts_xy.shape[0], 3), dtype=np.float32)
            kpts[:, 0] = kpts_xy[:, 0] + x1
            kpts[:, 1] = kpts_xy[:, 1] + y1
            kpts[:, 2] = kpts_sc.astype(np.float32)

            all_kpts.append(kpts)
            all_bboxes.append([x1, y1, x2, y2, score])

            # 可視化（点のみ）
            vis = draw_kpts(vis, kpts, thr=KPT_THR, r=3)

        # JSON保存（openpose風）
        out_json = to_openpose_people(all_kpts, all_bboxes, version=1.3)
        with open(json_dir / f"frame_{fi:05d}_keypoints.json", "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False)

        # HD書き出し
        vis_hd = cv2.resize(vis, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(vis_dir / f"frame_{fi:05d}.jpg"), vis_hd, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        vw.write(vis_hd)

        fi += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    vw.release()

    print("Done.")
    print("Video :", str(out_dir / "vitpose_vis.mp4"))
    print("JSON  :", str(json_dir))
    print("Frames:", str(vis_dir))


if __name__ == "__main__":
    main()
