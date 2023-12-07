import cv2

def video_to_images(video_path, output_folder):
    # 動画ファイルを読み込む
    video = cv2.VideoCapture(video_path)

    # フレーム数を取得
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # フレームごとに画像を保存
    for i in range(frame_count):
        # フレームを読み込む
        ret, frame = video.read()

        # 画像を保存するパス
        image_path = f"{output_folder}/{str(i).zfill(4)}.png"

        # 画像を保存
        cv2.imwrite(image_path, frame)

    # 動画ファイルを解放
    video.release()

# 動画ファイルのパスと出力フォルダのパスを指定
video_path = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\2023_12_demo\1\SealDetection.mp4"
output_folder = r"C:\Users\zutom\.vscode\PythonFile\test\GIF\RGB"

# 動画を連続した画像に変換
video_to_images(video_path, output_folder)
