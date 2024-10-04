import os
import glob
import cv2

root_dir = r"F:\Tomson\gait_pattern\20240808"

estimateds  = glob.glob(os.path.join(root_dir, '*/estimated.avi'))

for estimated in estimateds:
    print(estimated)
    cap = cv2.VideoCapture(estimated)
    if not cap.isOpened():
        print("Error")
        continue

    frame_count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print(f"{frame_count}")

        estimated_folder = os.path.dirname(estimated) + '/estimated'
        if not os.path.exists(estimated_folder):
            os.makedirs(estimated_folder)
        cv2.imwrite(os.path.join(estimated_folder, f"{frame_count}.png"), frame)
        frame_count += 1
    cap.release()