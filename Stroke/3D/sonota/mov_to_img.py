import os
import glob
import cv2
from pathlib import Path

root_dir = r"G:\gait_pattern\20241106"

estimateds = list(Path(root_dir).glob('*/original.mp4'))
# estimateds = list(Path(root_dir).glob('*front_30*/estimated.avi'))

for i, estimated in enumerate(estimateds):
    print(f"{i+1}/{len(estimateds)} {estimated}")
    cap = cv2.VideoCapture(str(estimated))
    if not cap.isOpened():
        print("Error")
        continue

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print(f"{i+1}/{len(estimateds)} {estimated}: {frame_count}")

        # estimated_folder = os.path.dirname(estimated) + '/estimated'
        estimated_folder = os.path.dirname(estimated) + '/original'
        if not os.path.exists(estimated_folder):
            os.makedirs(estimated_folder)
        cv2.imwrite(os.path.join(estimated_folder, f"{frame_count}.png"), frame)
        frame_count += 1
    cap.release()