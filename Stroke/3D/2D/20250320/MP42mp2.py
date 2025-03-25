from pathlib import Path
import cv2

#goproの高画質な動画の容量を抑えるため、画質を下げてmp4に変換するプログラム
MP4_dir  = Path(r"G:\gait_pattern\20241114_ota_test\gopro\sagi")
MP4_path = list(MP4_dir.glob("*sub0_abngait*.MP4"))[0]
print(f"MP4_path:{MP4_path}")

cap = cv2.VideoCapture(str(MP4_path))
if not cap.isOpened():
    print(f"Error: Could not open video file: {MP4_path}")
    exit()  # プログラムを終了

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 4
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 4
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

print(f"width:{width}, height:{height}, fps:{fps}")

writer = cv2.VideoWriter(str(MP4_path.with_name("Check_Full.mp4")), fourcc, fps, (width, height))
frame_count = 0
while True:
    print(f"frame_count:{frame_count}")
    ret, frame = cap.read()
    if not ret:
        print("終了")
        break

    frame_resized = cv2.resize(frame, (width, height))
    cv2.imshow("frame", frame_resized)
    writer.write(frame_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

    if frame_count >= fps * 10:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()


