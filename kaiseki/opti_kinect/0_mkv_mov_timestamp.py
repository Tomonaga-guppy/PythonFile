from pyk4a import PyK4APlayback
import cv2
import glob
import os
import sys
import numpy as np

helpers_dir = r"C:\Users\pyk4a\example"
os.chdir(helpers_dir)
sys.path.append(helpers_dir)
from helpers import convert_to_bgra_if_required

def main():
    mkv_folder = r"F:\Tomson\gait_pattern\20240808"
    # mkv_folder = r"F:\Tomson\gait_pattern\20240712"
    mkv_files = glob.glob(os.path.join(mkv_folder, '[0-9]*.mkv'))

    for i, mkv_file_path in enumerate(mkv_files):


        folder_path = os.path.dirname(mkv_file_path) + '/' + os.path.basename(mkv_file_path).split('.')[0]
        if os.path.exists(folder_path) == False:
            os.mkdir(folder_path)

        # MKVファイルの再生
        playback = PyK4APlayback(mkv_file_path)
        playback.open()
        calibration = playback.calibration

        frame_count = 1

        mp4file = folder_path + "/original.mp4"
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
        fps = 30.0
        size = (1920,1080)
        # writer = cv2.VideoWriter(mp4file, fmt, fps, size) # ライター作成

        timestampfolder = folder_path + "/timestamp"
        if os.path.exists(timestampfolder) == False:
            os.mkdir(timestampfolder)

        while True:  #100フレーム分正常にデータを取得したら終了
            print(f"{i+1}/{len(mkv_files)} mkv_file = {mkv_file_path}:frame_count = {frame_count}")

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
            rgb_image_mini = cv2.resize(rgb_image, (720, 480))
            color_timestamp = capture._color_timestamp_usec
            cv2.putText(rgb_image_mini, f"frame_count:{str(frame_count)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(rgb_image_mini, f"timestamp:{str(color_timestamp)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            depth_image = capture.transformed_depth
            if depth_image is None:
                depth_image_mini = np.zeros((480, 720, 3), dtype=np.uint8)
            else:
                depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_image_mini = cv2.resize(depth_image, (720, 480))
                depth_image_mini_3ch = cv2.cvtColor(depth_image_mini, cv2.COLOR_GRAY2BGR)
                depth_timestamp = capture._depth_timestamp_usec
                cv2.putText(depth_image_mini_3ch, f"frame_count:{str(frame_count)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(depth_image_mini_3ch, f"timestamp:{str(depth_timestamp)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            timestamp_image = cv2.vconcat([rgb_image_mini, depth_image_mini_3ch])
            cv2.imwrite(timestampfolder + f"/frame_{str(frame_count)}.png", timestamp_image)

            # キーが押されるまで待機
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # writer.write(rgb_image)
            frame_count += 1

        # クリーンアップ
        playback.close()
        cv2.destroyAllWindows()
        # writer.release()

if __name__ == '__main__':
    main()