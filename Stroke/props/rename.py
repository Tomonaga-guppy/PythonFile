import glob
import os
import re

root_dir = r"C:\Users\Tomson\BRLAB\gait_pattern\20240712"
keyward = "*"
ori_mov_paths = glob.glob(os.path.join(root_dir, ("*" + keyward + "*"), '*original.mp4'))

for ori_mov_path in ori_mov_paths:
    directory, filename = os.path.split(ori_mov_path)
    new_name = 'original.mp4'
    new_path = os.path.join(directory, new_name)

    # 同じ名前のファイルが存在する場合は、重複を避けるために番号を付加
    counter = 1
    while os.path.exists(new_path):
        name, ext = os.path.splitext(new_name)
        new_path = os.path.join(directory, f"{name}_{counter}{ext}")
        counter += 1

    os.rename(ori_mov_path, new_path)
    print(f"Renamed {ori_mov_path} to {new_path}")
