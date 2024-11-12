import pickle
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

root_dir = Path(r"G:\gait_pattern\20241111\gopro")
intrinsics_ch_paths = list(Path(root_dir).glob("*Intrinsic*.pickle"))

print(f"intrinsics_ch_paths: {intrinsics_ch_paths}")

for intrinsics_ch_path in intrinsics_ch_paths:
    id = intrinsics_ch_path.stem.split("Intrinsics_")[-1]
    with open(intrinsics_ch_path, "rb") as f:
        CameraParams = pickle.load(f)

    img_path = root_dir/id/"original"/"243.png"
    img = cv2.imread(str(img_path))
    undistort_img = cv2.undistort(img, CameraParams['intrinsicMat'], CameraParams['distortion'])

    img_resize = cv2.resize(img, (640, 360))
    undistort_img_resize = cv2.resize(undistort_img, (640, 360))

    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    undistort_img_resize = cv2.cvtColor(undistort_img_resize, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img_resize)
    ax1.set_title("Original")
    ax2.imshow(undistort_img_resize)
    ax2.set_title("Undistorted")

    cv2.imwrite(str(root_dir/id/"0.png"), img)
    cv2.imwrite(str(root_dir/id/"0_undistorted.png"), undistort_img)
    plt.savefig(root_dir/id/"0_ref.png")

    plt.show()
