import os
import glob
import shutil

root_dir = r"F:\Tomson\gait_pattern\20240822"
original_dirs = glob.glob(os.path.join(root_dir, "calibration_sub3_0_dev2" ,"original.mp4"))

for i, original_mov in enumerate(original_dirs):
    after_path = os.path.join(root_dir, "movie", os.path.basename(os.path.dirname(original_mov))+".mp4")
    if not os.path.exists(os.path.join(root_dir, "movie")):
        os.mkdir(os.path.join(root_dir, "movie"))
    shutil.copyfile(original_mov, after_path)
    print(f"{i+1}/{len(original_dirs)}: {after_path}")