import matplotlib.pyplot as plt
import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

root_dir = r"F:\Tomson\gait_pattern\20240808"

target_dirs = glob.glob(os.path.join(root_dir, "[0-2]*"))


for i, target_dir in enumerate(target_dirs):
    print(f"target_dir = {target_dir}")
    depth_image_folder = os.path.join(target_dir, "filled_depth_image")
    depth_image_files = glob.glob(os.path.join(depth_image_folder, "*.png"))

    for j, depth_image_file in enumerate(depth_image_files):
        depth_image = cv2.imread(depth_image_file, cv2.IMREAD_UNCHANGED)

        # 最小値と最大値を取得（16ビットの範囲を確認）
        min_val, max_val = np.min(depth_image), np.max(depth_image)
        print(f"Min depth: {min_val}, Max depth: {max_val}")
        # 16ビットのデータを0〜1の範囲に正規化
        normalized_depth_image = (depth_image - min_val) / (max_val - min_val)

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        im1 = axes[0].imshow(depth_image)
        axes[0].set_title("Original Depth Image")

        im2 = axes[1].imshow(normalized_depth_image, cmap="gray")
        axes[1].set_title("normalized Depth Image")

        im3 = axes[2].imshow(normalized_depth_image, cmap="jet")
        axes[2].set_title("Colormap Depth Image")

        fig.colorbar(im1, ax=axes[0])
        fig.colorbar(im2, ax=axes[1])
        fig.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        plt.show()

