import pyrealsense2 as rs
import numpy as np
import cv2
import os
import glob

root_dir = r"F:\Tomson\gait_pattern\20240607"

bag_file_path_list = glob.glob(os.path.join(root_dir, f'*.bag'))

for i, bag_file_path in enumerate(bag_file_path_list):

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

        for stream in profile.get_streams():
            vprof = stream.as_video_stream_profile()
            if  vprof.format() == rs.format.rgb8:
                frame_rate = vprof.fps()
                size = (vprof.width(), vprof.height())

        mp4file = output_dir + "/original.mp4"
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
        writer = cv2.VideoWriter(mp4file, fmt, frame_rate, size) # ライター作成

        while True:
            try:
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

                # Numpy配列への変換 OpenCVはBGR形式のため変換
                color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_RGB2BGR)

                # # ファイル名を連番で生成し、画像を保存
                # frame_id = f'{frame_count:04d}'  # 4桁のファイル番号
                # output_filename = os.path.join(output_dir, f'image_{frame_id}.jpg')
                # cv2.imwrite(output_filename, color_image)

                # cv2.imshow("RGB Image", color_image)
                # cv2.waitKey(1)

                writer.write(color_image)
                print(f"進捗：{i+1}/{len(bag_file_path_list)} フレーム番号：{frame_count}")
                frame_count += 1
            except RuntimeError:
                break

    finally:
        # パイプライン停止
        pipeline.stop()
        writer.release()
        print(f"Saved {frame_count - 1} images to {output_dir}")
