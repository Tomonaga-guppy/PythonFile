import cv2

# 入力ファイルのパス
input_file = r"G:\gait_pattern\20240912\sub3_normalgait_f_1_480_dev0\original.mp4"

# 出力ファイルのパス
# output_file_1280x720 = r"G:\gait_pattern\20240912\sub3_normalgait_f_1_480_dev0\output_1280x720.mp4"
output_file_720x405 = r"G:\gait_pattern\20240912\sub3_normalgait_f_1_480_dev0\output_720x405.mp4"

# 動画の読み込み
cap = cv2.VideoCapture(input_file)

# 動画のプロパティを取得
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 出力動画のコーデック
fps = int(cap.get(cv2.CAP_PROP_FPS))      # フレームレート
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 出力動画の設定
# out_1280x720 = cv2.VideoWriter(output_file_1280x720, fourcc, fps, (1280, 720))
out_720x405 = cv2.VideoWriter(output_file_720x405, fourcc, fps, (720, 405))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1280x720にリサイズ
    resized_frame_1280x720 = cv2.resize(frame, (1280, 720))
    # out_1280x720.write(resized_frame_1280x720)

    # 740x520にリサイズ
    resized_frame_720x405 = cv2.resize(frame, (720, 405))
    out_720x405.write(resized_frame_720x405)

# リソースの解放
cap.release()
# out_1280x720.release()
out_720x405.release()
cv2.destroyAllWindows()
