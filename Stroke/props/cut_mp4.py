import cv2

input_path = r"G:\gait_pattern\20241102\front_30m_.mp4"
output_path = r"G:\gait_pattern\20241102\front_30m.mp4"

#挿画を読み込んではじめと終わりの30フレームを削除する
def cut_frames(input_path, output_path):
    frames_to_cut = 30

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {input_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    current_frame = 0

    start_frame = fps * 5
    end_frame = fps * 13

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # if frames_to_cut <= current_frame < total_frames - frames_to_cut:
        if start_frame <= current_frame < end_frame:
            print(f"{current_frame}")
            out.write(frame)

        if current_frame >= end_frame:
            break

        current_frame += 1

    cap.release()
    out.release()
    print("Finished cutting frames.")

if __name__ == "__main__":
    cut_frames(input_path, output_path)
