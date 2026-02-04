"""
ViTPose（easy_ViTPose）で1枚画像を推論し、
OpenPose BODY_25 っぽく（スケルトン + keypoint id）を描画して保存する。

使い方例:
  python 2_run_vitpose_asOP_single_image_draw.py ^
    --img "G:\...\frame_00000.jpg" ^
    --out "G:\...\frame_00000_openpose_like.jpg"

注意:
- dataset="coco_25" を想定（vitpose-l-coco_25.pth）
- ORDER_MAPで「easy_ViTPose coco_25 -> OpenPose BODY_25」へ並べ替え
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from easy_ViTPose import VitInference


# =========================
# 設定（必要なら調整）
# =========================
MODEL_PATH = Path(r"C:\Users\Tomson\StrokeProject\easy_ViTPose\models\vitpose-l-coco_25.pth")
YOLO_PATH  = Path(r"C:\Users\Tomson\StrokeProject\easy_ViTPose\yolov8x.pt")

MODEL_NAME = "l"
YOLO_SIZE = 640

# 描画
POINT_RADIUS = 4
LINE_THICKNESS = 2
CONF_TH = 0.2

# 文字
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_THICKNESS = 1

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

# easy_ViTPose coco_25 → OpenPose BODY_25 変換（元スクリプトと同じ）
ORDER_MAP = [
    0, 5, 7, 9, 11, 6, 8, 10, 14, 13, 16, 18, 12, 15, 17, 2, 1, 4, 3, 19, 20, 21, 22, 23, 24
]


def blend(c1, c2):
    return ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2, (c1[2] + c2[2]) // 2)


def draw_openpose_like(img_rgb: np.ndarray, kpts_yx: np.ndarray, conf_th: float) -> None:
    """
    img_rgb: RGB画像（描画先）
    kpts_yx: (25,3) [y,x,score]（OpenPose順）
    """
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

    # points + id
    for i in range(25):
        y, x, s = kpts_yx[i]
        if s > conf_th:
            xi, yi = int(x), int(y)
            cv2.circle(img_rgb, (xi, yi), POINT_RADIUS, POSE_KPT_COLORS[i], -1)

            # IDを少し右上に描画（見やすさ優先で黒縁取り）
            text = str(i)
            org = (xi + 10, yi - 10)
            
            if i == 0: # Nose
                org = (xi + 10, yi + 10)
            elif i == 17: #right ear
                org = (xi - 50, yi + 20)
            elif i == 15:  #right eye
                org = (xi - 50, yi - 10)
            elif i == 19:  #left big toe
                org = (xi - 10, yi - 10)
            elif i == 22:  #right big toe
                org = (xi + 10, yi + 20)
            elif i==21:  #left heel
                org = (xi + 10, yi + 10)
            elif i==24:  #right heel
                org = (xi + 10, yi)
            
            # cv2.putText(img_rgb, text, org, FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS + 2, cv2.LINE_AA)
            # cv2.putText(img_rgb, text, org, FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
            cv2.putText(img_rgb, text, org, FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="入力画像パス")
    ap.add_argument("--out", required=True, help="出力画像パス（.jpg/.png）")
    ap.add_argument("--person", type=int, default=0, help="複数人検出時に描画するperson index（0始まり）")
    ap.add_argument("--conf_th", type=float, default=CONF_TH, help="描画する信頼度閾値")
    args = ap.parse_args()

    img_path = Path(args.img)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"画像が読めません: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # モデル（1回だけロード）
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

    try:
        # 推論（{person_id: (25,3)[y,x,score]}）
        kpts_dict = model.inference(img_rgb)

        if not kpts_dict:
            # 何も検出されない場合も出力は作る
            # vis_rgb = img_rgb.copy()
            vis_rgb = np.full_like(img_rgb, 0) # 真っ黒背景
            cv2.putText(
                vis_rgb,
                "No person detected",
                (20, 40),
                FONT,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            # person indexで選ぶ（dictの順序依存を避けてキーでソート）
            pids = sorted(list(kpts_dict.keys()))
            person_idx = int(np.clip(args.person, 0, len(pids) - 1))
            pid = pids[person_idx]
            kpts = kpts_dict[pid]  # (25,3) [y,x,score] easy_ViTPose coco_25順

            # OpenPose順へ並べ替え
            k_op = kpts[ORDER_MAP].copy()

            # vis_rgb = img_rgb.copy()
            vis_rgb = np.full_like(img_rgb, 0) # 真っ黒背景
            draw_openpose_like(vis_rgb, k_op, float(args.conf_th))

            # ついでに選んだperson情報を左上に表示
            info = f"person={person_idx} (pid={pid})"
            cv2.putText(vis_rgb, info, (20, 40), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(vis_rgb, info, (20, 40), FONT, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            # cv2.putText(vis_rgb, info, (20, 40), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # 保存（BGRで書く）
        out_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(out_path), out_bgr)
        if not ok:
            raise RuntimeError(f"保存に失敗: {out_path}")

        print(f"[OK] saved: {out_path}")

    finally:
        try:
            model.reset()
        except Exception:
            pass


if __name__ == "__main__":
    main()
