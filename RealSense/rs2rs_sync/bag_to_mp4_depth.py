import pyrealsense2 as rs
import numpy as np
import cv2
import os
import glob

root_dir = r"F:\Tomson\gait_pattern\first_test\recorded_data\realsense\two_dev"

bag_file_path_list = glob.glob(os.path.join(root_dir, f'*test_[7-8].bag'))

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

        # hole_filling_filterのパラメータ
        hole_filling = rs.hole_filling_filter(2)

        frame_count = 1  # フレーム番号を初期化

        for stream in profile.get_streams():
            vprof = stream.as_video_stream_profile()
            if  vprof.format() == rs.format.rgb8:
                frame_rate = vprof.fps()
                size = (vprof.width(), vprof.height())

        mp4file = output_dir + "/original_depth.mp4"
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

                # depthフレームの取得
                depth_frame = aligned_frames.get_depth_frame()
                if not depth_frame:
                    break
                filter_frame = hole_filling.process(depth_frame)
                result_frame = filter_frame.as_depth_frame()


                # Numpy配列への変換 OpenCVはBGR形式のため変換
                depth_image = np.asanyarray(result_frame.get_data())

                depth_dir = output_dir + '/depth_image'
                if not os.path.exists(depth_dir):
                    os.makedirs(depth_dir)

                # ファイル名を連番で生成し、画像を保存
                depth_image_path = os.path.join(depth_dir, f"{str(frame_count).zfill(4)}.png")
                cv2.imwrite(depth_image_path, depth_image)

                cv2.imshow("depth Image", depth_image)
                cv2.waitKey(1)

                # depth_imageをカラー画像に変換
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                writer.write(depth_colormap)
                print(f"進捗：{i+1}/{len(bag_file_path_list)} フレーム番号：{frame_count}")
                frame_count += 1
            except RuntimeError:
                break

    finally:
        # パイプライン停止
        pipeline.stop()
        writer.release()
        print(f"Saved {frame_count - 1} images to {output_dir}")
