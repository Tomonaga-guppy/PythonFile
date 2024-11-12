import os
import glob
import cv2
from pathlib import Path

root_dir = r"G:\gait_pattern\20241112\gopro"

movies = list(Path(root_dir).glob('checker_test/*.MP4'))
# movies = list(Path(root_dir).glob('*/intrinsic.MP4'))
print(movies)

for i, movie in enumerate(movies):
    print(f"{i+1}/{len(movies)} {movie}")
    cap = cv2.VideoCapture(str(movie))
    if not cap.isOpened():
        print("Error")
        continue

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 10 != 0:
            frame_count += 1
            continue
        # print(f"{i+1}/{len(movies)} {movie}: {frame_count}")

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        movie_folder = os.path.dirname(movie) + '/' + os.path.basename(movie).split(".")[0]
        # movie_folder = os.path.dirname(movie) + '/Intrinsic_ori'
        if not os.path.exists(movie_folder):
            os.makedirs(movie_folder)
        cv2.imwrite(os.path.join(movie_folder, f"{frame_count}.png"), frame)
        print(f"{os.path.join(movie_folder, f'{frame_count}.png')}")
        frame_count += 1
    cap.release()