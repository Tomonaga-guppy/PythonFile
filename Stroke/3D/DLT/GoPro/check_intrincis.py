import pickle
from pathlib import Path
import cv2
import numpy as np
import os
from tqdm.auto import tqdm

root_dir = Path(r"G:\gait_pattern\20241112\gopro")
intrinsics_ch_paths = list(Path(root_dir).glob("*Intrinsic*front_l.pickle"))

print(f"intrinsics_ch_paths: {intrinsics_ch_paths}")

for intrinsics_ch_path in intrinsics_ch_paths:
    id = intrinsics_ch_path.stem.split("Intrinsics_")[-1]
    with open(intrinsics_ch_path, "rb") as f:
        CameraParams = pickle.load(f)

    img_paths = list((root_dir/id/"Intrinsic_ori").glob("*.png"))
    save_folder = root_dir/id/"Intrinsic_undistorted"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for img_path in tqdm(img_paths):
        img = cv2.imread(str(img_path))
        undistort_img = cv2.undistort(img, CameraParams['intrinsicMat'], CameraParams['distortion'])
        cv2.imwrite(os.path.join(save_folder, img_path.name), undistort_img)
    # for img_path in img_paths:
    #     img = cv2.imread(str(img_path))
    #     undistort_img = cv2.undistort(img, CameraParams['intrinsicMat'], CameraParams['distortion'])
    #     cv2.imwrite(os.path.join(save_folder, img_path.name), undistort_img)
