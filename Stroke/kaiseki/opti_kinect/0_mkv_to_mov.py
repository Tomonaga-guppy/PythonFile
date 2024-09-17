from pyk4a import PyK4APlayback
import cv2
import glob
import os
import sys

helpers_dir = r"C:\Users\pyk4a\example"
# helpers_dir = r"C:\Users\tus\pyk4a\example"
os.chdir(helpers_dir)
sys.path.append(helpers_dir)
from helpers import convert_to_bgra_if_required

def main():
    # mkv_folder = r"F:\Tomson\gait_pattern\20240808"
    mkv_folder = r"f:\Tomson\gait_pattern\20240808"
    mkv_files = glob.glob(os.path.join(mkv_folder, '[0-9]*.mkv'))
    print(f"mkv_files = {mkv_files}")

    for i, mkv_file_path in enumerate(mkv_files):

        folder_path = os.path.dirname(mkv_file_path) + '/' + os.path.basename(mkv_file_path).split('.')[0]
        if os.path.exists(folder_path) == False:
            os.mkdir(folder_path)
        else:
            continue

        # MKVファイルの再生
        playback = PyK4APlayback(mkv_file_path)
        playback.open()
        calibration = playback.calibration

        frame_count = 1

        mp4file = folder_path + "/original.mp4"
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
        fps = 30.0
        size = (1920,1080)
        writer = cv2.VideoWriter(mp4file, fmt, fps, size) # ライター作成

        while True:  #100フレーム分正常にデータを取得したら終了
            try:
                # 画像をキャプチャ
                capture = playback.get_next_capture()
            except:
                print("再生を終了します")
                break

            # キャプチャが有効でない場合（ファイルの終わり）ループを抜ける
            if capture is None:
                break

            if capture.color is None:
                print(f"Frame {frame_count} has no RGB image data.")
                continue

            # RGB画像を取得
            rgb_image = convert_to_bgra_if_required(playback.configuration["color_format"], capture.color)
            depth_image = capture.transformed_depth

            rgb_image_mini = cv2.resize(rgb_image, (1080,720))
            cv2.imshow("RGB Image", rgb_image_mini)

            # キーが押されるまで待機
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(f"frame_count = {frame_count}")
            writer.write(rgb_image)
            frame_count += 1

        # クリーンアップ
        playback.close()
        cv2.destroyAllWindows()
        writer.release()

if __name__ == '__main__':
    main()