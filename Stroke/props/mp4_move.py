import os
import glob
import shutil

root_dir = r"F:\Tomson\gait_pattern\20240808"
original_dirs = glob.glob(os.path.join(root_dir, "*" ,"original.mp4"))

for i, original_mov in enumerate(original_dirs):
    after_path = os.path.join(root_dir, os.path.basename(os.path.dirname(original_mov))+".mp4")
    shutil.copyfile(original_mov, after_path)
    print(f"{i+1}/{len(original_dirs)}: {after_path}")