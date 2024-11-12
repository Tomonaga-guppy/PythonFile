#動画を読み込む
import cv2
from pathlib import Path

root_dir = Path(r"C:\gait_pattern\20241111\gopro")
device_folder = [device_folder for device_folder in root_dir.iterdir() if device_folder.is_dir()]

for device in device_folder:
    video_files = [video_file for video_file in device.glob("*tstmp*.MP4")]
    for video_file in video_files:
        print(f"video_file: {video_file}")
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print("Error: cannot open video.")
            continue
        # 動画の情報を取得
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"fps: {fps}, frame_count: {frame_count}")

        # # 最初のフレームを取得して保存
        # ret, first_frame = cap.read()
        # if ret:
        #     first_frame_filename = f"{video_file.stem}_first_frame.jpg"
        #     first_frame_path = device / first_frame_filename
        #     cv2.imwrite(str(first_frame_path), first_frame)
        #     print(f"Saved first frame as {first_frame_path}")

        # # 最後のフレームを取得して保存
        # if frame_count > 1:
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)  # 最後のフレームの位置に移動
        #     ret, last_frame = cap.read()
        #     if ret:
        #         last_frame_filename = f"{video_file.stem}_last_frame.jpg"
        #         last_frame_path = device / last_frame_filename
        #         cv2.imwrite(str(last_frame_path), last_frame)
        #         print(f"Saved last frame as {last_frame_path}")

        cap.release()