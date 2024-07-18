import cv2

def get_images_from_mp4(mp4_path):
    # ビデオキャプチャオブジェクトを作成
    cap = cv2.VideoCapture(mp4_path)

    # フレーム数を取得
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # フレームを格納するリストを作成
    frames = []

    # フレームを1枚ずつ取得
    for i in range(frame_count):
        # フレームを取得
        ret, frame = cap.read()

        # フレームが取得できなかった場合は終了
        if not ret:
            break

        # フレームをリストに追加
        frames.append(frame)

    # キャプチャを解放
    cap.release()

    return frames

def get_image_bit_depth(image):
    # 画像の深度（ビット数）を取得
    depth = image.dtype

    # データ型に応じたビット深度を決定
    depth_map = {
        'uint8': 8,
        'uint16': 16,
        'float32': 32,
        'int32': 32
    }

    # 画像のチャンネル数を取得
    channels = image.shape[2] if len(image.shape) == 3 else 1

    # ビット深度を計算
    bit_depth = depth_map.get(depth.name, 'Unknown depth') * channels

    return bit_depth


# 例として画像のパスを指定
mp4_path = r"F:\Tomson\gait_pattern\first_test\recorded_data\realsense\two_dev\output_device1_test_7\original_depth.mp4"
image_path = get_images_from_mp4(mp4_path)
print(image_path[0])
bit_depth = get_image_bit_depth(image_path[0])
print(f'The bit depth of the image is: {bit_depth} bits')
