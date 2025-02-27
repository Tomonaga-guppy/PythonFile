import os
import glob
import shutil

root_dir = r"G:\gait_pattern\20241016"
original_dirs = glob.glob(os.path.join(root_dir, "*dev0*" ,"original.mp4"))

for i, original_mov in enumerate(original_dirs):
    mov_name = os.path.basename(os.path.dirname(original_mov)).split("_dev")[0]
    print(f"mov_name = {mov_name}")
    after_path = os.path.join(root_dir, "movie", mov_name+".mp4")
    if not os.path.exists(os.path.join(root_dir, "movie")):
        os.mkdir(os.path.join(root_dir, "movie"))
    shutil.copyfile(original_mov, after_path)
    print(f"{i+1}/{len(original_dirs)}: {after_path}")