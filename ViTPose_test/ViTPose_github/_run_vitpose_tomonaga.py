import os
import json
import glob
import warnings
from pathlib import Path

import cv2
from tqdm import tqdm

from mmpose.apis import (
    init_pose_model,
    inference_top_down_pose_model,
    process_mmdet_results,
    vis_pose_result,
)
from mmpose.datasets import DatasetInfo

from mmdet.apis import init_detector, inference_detector


def natural_sort_key(p: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(p).name)]


def pose_results_to_openpose_like(pose_results, version=1.3):
    people = []
    for pr in pose_results:
        kpts = pr.get("keypoints", None)
        if kpts is None:
            continue

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

        if "bbox" in pr:
            person["bbox"] = [float(v) for v in pr["bbox"]]

        people.append(person)

    return {"version": version, "people": people}


def run(
    img_dir: str,
    det_config: str,
    det_checkpoint: str,
    pose_config: str,
    pose_checkpoint: str,
    out_dir: str,
    out_video_name: str = "vis.mp4",
    fps: float = 30.0,
    device: str = "cuda:0",
    det_cat_id: int = 1,
    bbox_thr: float = 0.3,
    kpt_thr: float = 0.3,
    radius: int = 4,
    thickness: int = 1,
    save_vis_frames: bool = True,
    vis_ext: str = ".jpg",   # ".png" でもOK
    vis_jpg_quality: int = 95,
):
    img_dir = str(Path(img_dir))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = out_dir / "vis_frames"
    if save_vis_frames:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # 画像一覧
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    img_paths = []
    for e in exts:
        img_paths += glob.glob(os.path.join(img_dir, e))
    img_paths = sorted(img_paths, key=natural_sort_key)
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No images found in: {img_dir}")

    # モデル初期化
    det_model = init_detector(det_config, det_checkpoint, device=device.lower())
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device.lower())

    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info_cfg = pose_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info_cfg is None:
        warnings.warn(
            "pose_config に dataset_info がありません。骨格のリンク描画などが不完全になる可能性があります。",
            DeprecationWarning,
        )
        dataset_info = None
    else:
        dataset_info = DatasetInfo(dataset_info_cfg)

    # 動画Writer準備（最初の画像サイズに合わせる）
    first = cv2.imread(img_paths[0])
    if first is None:
        raise RuntimeError(f"Failed to read: {img_paths[0]}")
    H, W = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path = str(out_dir / out_video_name)
    vw = cv2.VideoWriter(out_video_path, fourcc, fps, (W, H))

    # 推論（tqdm）
    for fi, p in enumerate(tqdm(img_paths, desc="ViTPose inference", unit="frame")):
        img = cv2.imread(p)
        if img is None:
            tqdm.write(f"[WARN] skip unreadable: {p}")
            continue

        # mmdet 推論 → person bbox抽出
        mmdet_results = inference_detector(det_model, img)
        person_results = process_mmdet_results(mmdet_results, det_cat_id)

        # mmpose top-down 推論
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=bbox_thr,
            format="xyxy",
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None,
        )

        # json保存（OpenPose風）
        out_json = pose_results_to_openpose_like(pose_results, version=1.3)
        json_name = f"frame_{fi:05d}_keypoints.json"
        with open(json_dir / json_name, "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False)

        # 可視化
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=kpt_thr,
            radius=radius,
            thickness=thickness,
            show=False,
        )

        # 動画に書き込み
        vw.write(vis_img)

        # フレーム画像として保存
        if save_vis_frames:
            vis_name = f"frame_{fi:05d}{vis_ext}"
            vis_path = str(vis_dir / vis_name)

            if vis_ext.lower() in [".jpg", ".jpeg"]:
                cv2.imwrite(vis_path, vis_img, [int(cv2.IMWRITE_JPEG_QUALITY), int(vis_jpg_quality)])
            else:
                cv2.imwrite(vis_path, vis_img)

    vw.release()
    print("Done.")
    print(f"Video      : {out_video_path}")
    print(f"JSON dir   : {json_dir}")
    if save_vis_frames:
        print(f"Vis frames : {vis_dir}")


if __name__ == "__main__":
    run(
        img_dir=r"G:\gait_pattern\BR9G_shuron\sub1\thera1-1\fl\undistorted",
        det_config=r"demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
        det_checkpoint=r"https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        pose_config=r"C:\Users\Tomson\StrokeProject\ViTPose\configs\body\2d_kpt_sview_rgb_img\topdown_heatmap\coco\vitPose+_large_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py",
        pose_checkpoint=r"C:\Users\Tomson\StrokeProject\ViTPose\models\wholebody.pth",
        out_dir=r"G:\gait_pattern\BR9G_shuron\sub1\thera1-1\fl\vitpose_output",
        out_video_name="vitpose_vis.mp4",
        fps=60.0,
        device="cuda:0",
        bbox_thr=0.3,
        kpt_thr=0.3,
        save_vis_frames=True,
        vis_ext=".jpg",
        vis_jpg_quality=95,
    )

"""
+Largemodelのconfig?
"C:\Users\Tomson\StrokeProject\ViTPose\configs\body\2d_kpt_sview_rgb_img\topdown_heatmap\coco\vitPose+_large_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py"

これはなんか違うみたい
"C:\Users\Tomson\StrokeProject\ViTPose\configs\wholebody\2d_kpt_sview_rgb_img\topdown_heatmap\coco-wholebody\ViTPose_large_wholebody_256x192.py"
"""