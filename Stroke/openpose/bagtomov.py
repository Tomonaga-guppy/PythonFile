import pyrealsense2 as rs
import numpy as np
import cv2
import glob
import os

#RealSenseで撮影したbagファイルを再生してRGB画像，動画を保存する Depth画像も保存する場合も同様の流れ
bagsfolder = r"C:\Users\Tomson\BRLAB\gait_pattern\first_test\recorded_data\realsense\two_dev"
bagfiles = glob.glob(os.path.join(bagsfolder, '*6*.bag'),recursive=True)


for progress, bagfile in enumerate(bagfiles):
    path = os.path.dirname(bagfile) + '/' + os.path.basename(bagfile).split('.')[0]
    # print(path)
    if not os.path.exists(path):
        os.mkdir(path)

    RGB_path = path + '/RGB_image'
    if not os.path.exists(RGB_path):
        os.mkdir(RGB_path)

    mp4file = path + '/original.mp4'

    config = rs.config()
    config.enable_device_from_file(bagfile)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # create Align Object
    align_to = rs.stream.color
    align = rs.align(align_to)

    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)  #リアルタイム再生をオフにするとフレームごとに処理が終わるまで待機してくれる

    for stream in profile.get_streams():
        vprof = stream.as_video_stream_profile()
        if  vprof.format() == rs.format.rgb8:
            frame_rate = vprof.fps()
            size = (vprof.width(), vprof.height())

    print(f"frame_rate = {frame_rate}, size = {size}")
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter(mp4file, fmt, frame_rate, size) # ライター作成

    print('{}/{}: '.format(progress+1, len(bagfiles)),bagfile, "size =",size, "frame_rate =",frame_rate)
    fps_count = 1

    try:
        pre_time = 0
        while True:
            frames = pipeline.wait_for_frames()

            cur_time = playback.get_position()  #再生時間の取得 単位はナノ秒

            #前フレームより再生時間が進んでいない or 想定フレーム以上になったら終了
            if cur_time < pre_time:# or fps_count > 300:
                break

            if cur_time - pre_time <= 30000000:
                continue

            pre_time = cur_time

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()

            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            cv2.imwrite('{}/{}.png'.format(RGB_path, str(fps_count).zfill(4)), color_image)
            writer.write(color_image)

            print("RGB fps_count", fps_count)
            fps_count += 1

    finally:
        print(f'RGB_fps_count = {fps_count - 1}')
        pipeline.stop()
        writer.release()
        cv2.destroyAllWindows()