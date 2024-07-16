import pyrealsense2 as rs
import numpy as np
import cv2
import os
import glob

# root_dir = r"C:\Users\zutom\BRLAB\gait pattern\sync\test"
root_dir = r"F:\Tomson\gait_pattern\20240607"

# #input関数でターミナルに入力された番号を取得
# blinking_interval = input("点灯間隔を入力してください: ")

bag_file_path_list = glob.glob(os.path.join(root_dir, f'*.bag'))

for bag_file_path in bag_file_path_list:

    # # .bagファイルのパスa
    # bag_file_path = os.path.join(root_dir, f'output_solo_{blinking_interval}.bag')

    # 出力ディレクトリの作成
    print(f"Processing {bag_file_path}")
    output_dir = os.path.join(root_dir, f'{os.path.basename(bag_file_path)[:-4]}')
    os.makedirs(output_dir, exist_ok=True)

    # パイプライン設定
    pipeline = rs.pipeline()
    config = rs.config()

    # .bagファイルからのデータの読み込みを設定
    config.enable_device_from_file(bag_file_path, repeat_playback=False)

    try:
        # パイプライン開始
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)  # リアルタイム再生を無効化

        # フレーム取得用のアライメントオブジェクト
        align_to = rs.stream.color
        align = rs.align(align_to)

        frame_count = 1  # フレーム番号を初期化

        while True:
            # フレームセットを待つ
            frames = pipeline.wait_for_frames()
            if not frames:
                break  # フレームがもうない場合はループを抜ける

            # アライメント処理
            aligned_frames = align.process(frames)

            # RGBフレームの取得
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                break

            # Numpy配列への変換
            color_image = np.asanyarray(color_frame.get_data())

            # ファイル名を連番で生成し、画像を保存
            frame_id = f'{frame_count:04d}'  # 4桁のファイル番号
            output_filename = os.path.join(output_dir, f'image_{frame_id}.jpg')
            cv2.imwrite(output_filename, color_image)
            frame_count += 1

    finally:
        # パイプライン停止
        pipeline.stop()
        print(f"Saved {frame_count - 1} images to {output_dir}")
