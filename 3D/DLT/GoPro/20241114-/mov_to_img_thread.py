import os
import cv2
from pathlib import Path
from multiprocessing import Pool

root_dir = r"G:\gait_pattern\20241112(1)\gopro"

# ビデオファイルのリストを取得
movies = list(Path(root_dir).glob('*/estimated.avi'))
print(movies)

def process_video(args):
    movie, index, total = args
    print(f"{index+1}/{total} {movie}")
    cap = cv2.VideoCapture(str(movie))
    if not cap.isOpened():
        print("Error opening video file:", movie)
        return

    frame_count = 0
    movie_folder = os.path.join(os.path.dirname(movie), 'original')
    if not os.path.exists(movie_folder):
        os.makedirs(movie_folder)

    while True:
        if frame_count % 10 != 0:
            continue

        ret, frame = cap.read()
        if not ret:
            break
        print(f"{index+1}/{total} {movie}: {frame_count}")
        cv2.imwrite(os.path.join(movie_folder, f"{frame_count}.png"), frame)
        frame_count += 1
    cap.release()

# マルチプロセスで処理を実行
if __name__ == "__main__":
    with Pool() as pool:
        pool.map(process_video, [(movie, i, len(movies)) for i, movie in enumerate(movies)])
