import pickle
from pathlib import Path
import cv2
import os
from tqdm.auto import tqdm

root_dir = Path(r"G:\gait_pattern\int_cali\ota")
intrinsics_ch_paths = list(root_dir.glob("*Intrinsic_fl*.pickle"))

print(f"intrinsics_ch_paths: {intrinsics_ch_paths}")

for intrinsics_ch_path in intrinsics_ch_paths:
    target = intrinsics_ch_path.stem
    with open(intrinsics_ch_path, "rb") as f:
        CameraParams = pickle.load(f)
    print(f"intrinsics_ch_path: {intrinsics_ch_path}")
    print(f"CameraParams: {CameraParams}")

    img_paths = list((root_dir/target).glob("*.png"))
    save_folder = root_dir/f"{target}_undistorted"
    if not save_folder.exists():
        save_folder.mkdir()

    for img_path in tqdm(img_paths):
        img = cv2.imread(str(img_path))
        undistort_img = cv2.undistort(img, CameraParams['intrinsicMat'], CameraParams['distortion'])
        cv2.imwrite(os.path.join(save_folder, img_path.name), undistort_img)
