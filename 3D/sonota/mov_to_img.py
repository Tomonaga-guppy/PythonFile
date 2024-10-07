import os
import glob
import cv2

root_dir = r"F:\Tomson\gait_pattern\20240912"

estimateds  = glob.glob(os.path.join(root_dir, '*tpose*/original.mp4'))
# estimateds  = glob.glob(os.path.join(root_dir, '*/estimated.avi'))

for i, estimated in enumerate(estimateds):
    cap = cv2.VideoCapture(estimated)
    if not cap.isOpened():
        print("Error")
        continue

    frame_count = 1
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