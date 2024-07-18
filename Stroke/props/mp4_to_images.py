import pandas as pd
import glob
import os
import numpy as np
import cv2

root_dir = r"F:\Tomson\gait_pattern\20240712"
estimated_dirs = glob.glob(os.path.join(root_dir, "*" ,"estimated.avi"))
print(f'os.path.join(root_dir, "*" ,"estimated.avi") = {os.path.join(root_dir, "*output*" ,"estimated.avi")}')
print(f"estimated_dirs: {estimated_dirs}")

for i, estimated_mov in enumerate(estimated_dirs):
    print(f"{i+1}/{len(estimated_dirs)}: {estimated_mov}")
    cap = cv2.VideoCapture(estimated_mov)

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not os.path.exists(f"{os.path.dirname(estimated_mov)}/images"):
            os.makedirs(f"{os.path.dirname(estimated_mov)}/images")

        cv2.imwrite(f"{os.path.dirname(estimated_mov)}/images/{frame_num}.jpg", frame)

        frame_num += 1