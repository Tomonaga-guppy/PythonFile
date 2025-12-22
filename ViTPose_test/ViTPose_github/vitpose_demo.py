import os
import json
import glob
import warnings
from pathlib import Path
from queue import Queue
from threading import Thread

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


def io_worker(q, vw, json_dir, vis_dir, save_vis_frames, vis_ext, vis_jpg_quality):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break

        fi, out_json, vis_img = item

        # json
        with open(json_dir / f"frame_{fi:05d}_keypoints.json", "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False)

        # video
        vw.write(vis_img)

        # frame image
        if save_vis_frames:
            path = vis_dir / f"frame_{fi:05d}{vis_ext}"
            if vis_ext.lower() in [".jpg", ".jpeg"]:
                cv2.imwrite(
                    str(path),
                    vis_img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(vis_jpg_quality)],
                )
            else:
                cv2.imwrite(str(path), vis_img)

        q.task_done()


def run(
    img_dir,
    det_config,
    det_checkpoint,
    pose_config,
    pose_checkpoint,
    out_dir,
    out_video_name="vitpose_vis.mp4",
    device="cuda:0",
    det_cat_id=1,
    bbox_thr=0.3,
    kpt_thr=0.3,
    radius=4,
    thickness=1,
    save_vis_frames=True,
    vis_ext=".jpg",
    vis_jpg_quality=85,
    io_queue_size=64,
    # ===== 出力設定 =====
    output_width=1920,
    output_height=1080,
    fps=60.0,
):
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_dir = out_dir / "json"
    json_dir.mkdir(exist_ok=True)

    vis_dir = out_dir / "vis_frames"
    if save_vis_frames:
        vis_dir.mkdir(exist_ok=True)

    # image list
    img_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg", "*.bmp", "*.tif"):
        img_paths += glob.glob(str(img_dir / ext))
    img_paths = sorted(img_paths, key=natural_sort_key)

    # models
    det_model = init_detector(det_config, det_checkpoint, device=device)
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
    
    
    print("loaded cfg:", pose_model.cfg.model.type if hasattr(pose_model.cfg, "model") else "unknown")


    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info_cfg = pose_model.cfg.data["test"].get("dataset_info", None)
    dataset_info = DatasetInfo(dataset_info_cfg) if dataset_info_cfg else None

    # VideoWriter (HD)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(
        str(out_dir / out_video_name),
        fourcc,
        fps,
        (output_width, output_height),
    )

    # IO thread
    q = Queue(maxsize=io_queue_size)
    t = Thread(
        target=io_worker,
        args=(q, vw, json_dir, vis_dir, save_vis_frames, vis_ext, vis_jpg_quality),
        daemon=True,
    )
    t.start()

    for fi, img_path in enumerate(tqdm(img_paths, desc="ViTPose inference", unit="frame")):
        img = cv2.imread(img_path)
        if img is None:
            continue

        # detection
        det_results = inference_detector(det_model, img)
        persons = process_mmdet_results(det_results, det_cat_id)

        # pose
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            img,
            persons,
            bbox_thr=bbox_thr,
            format="xyxy",
            dataset=dataset,
            dataset_info=dataset_info,
        )

        out_json = pose_results_to_openpose_like(pose_results)

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

        # ★ HDへリサイズ（出力のみ）
        vis_img = cv2.resize(
            vis_img,
            (output_width, output_height),
            interpolation=cv2.INTER_AREA,
        )

        q.put((fi, out_json, vis_img))

    q.put(None)
    q.join()
    t.join()
    vw.release()

    print("Done.")
    print(f"Video : {out_dir / out_video_name}")
    print(f"JSON  : {json_dir}")
    if save_vis_frames:
        print(f"Frames: {vis_dir}")


if __name__ == "__main__":
    run(
        img_dir=r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\gopro\fl\undistorted_facemasked",
        det_config=r"demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
        det_checkpoint=r"https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        pose_config=r"C:\Users\Tomson\StrokeProject\ViTPose\configs\wholebody\2d_kpt_sview_rgb_img\topdown_heatmap\coco-wholebody\ViTPose_large_wholebody_256x192.py",
        pose_checkpoint=r"C:\Users\Tomson\StrokeProject\ViTPose\models\wholebody.pth",
        out_dir=r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\gopro\fl\vitpose_output",
        fps=60.0,
    )
    
    # """
    # "C:\Users\Tomson\StrokeProject\ViTPose\configs\body\2d_kpt_sview_rgb_img\topdown_heatmap\coco\vitPose+_large_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py"
    # """
