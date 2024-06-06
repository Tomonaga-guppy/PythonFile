import pyrealsense2 as rs
import numpy as np
import cv2
import glob
import os

#RealSenseで撮影したbagファイルを再生してRGB画像，動画を保存する Depth画像も保存する場合も同様の流れ
bagsfolder = r"C:\Users\Tomson\BRLAB\gait_pattern\first_test\recorded_data\realsense\two_dev"
bagfiles = glob.glob(os.path.join(bagsfolder, '*.bag'),recursive=True)


for progress, bagfile in enumerate(bagfiles):
    path = os.path.dirname(bagfile) + '/' + os.path.basename(bagfile).split('.')[0]
    # print(path)
    if not os.path.exists(path):
        os.mkdir(path)

    RGB_mp4_path = path + '/original.mp4'

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
    writer = cv2.VideoWriter(RGB_mp4_path, fmt, frame_rate, size) # ライター作成

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

            writer.write(color_image)

            print(f"{progress+1}/{len(bagfiles)} RGB fps_count {fps_count}")
            fps_count += 1

    finally:
        print(f"{progress+1}/{len(bagfiles)} RGB_fps_count = {fps_count - 1}")
        pipeline.stop()
        writer.release()
        cv2.destroyAllWindows()





    #Save Depth image
    Depth_path = path + '/Depth_image'
    if not os.path.exists(Depth_path):
        os.mkdir(Depth_path)

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

    #depthデータに欠損があるのでholefillingをかける https://qiita.com/keoitate/items/efe4212b0074e10378ec
    """
    # decimarion_filterのパラメータ
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 1)
    # spatial_filterのパラメータ
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 1)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    """
    # hole_filling_filterのパラメータ
    hole_filling = rs.hole_filling_filter(2)
    """
    # disparity
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    """


    try:
        fps_count = 1
        pre_time = 0
        while True:
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)
            # color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            #depthデータの後処理(フィルター)
            #filter_frame = decimate.process(depth_frame)
            #filter_frame = depth_to_disparity.process(filter_frame)
            #filter_frame = spatial.process(filter_frame)
            #filter_frame = disparity_to_depth.process(filter_frame)

            filter_frame = hole_filling.process(depth_frame)
            result_frame = filter_frame.as_depth_frame()

            cur_time = playback.get_position()  #再生時間の取得 単位はナノ秒

            if cur_time < pre_time:# or fps_count > 300:
                break

            if cur_time - pre_time <= 30000000:
                continue

            pre_time = cur_time


            # Make depth image
            depth_image = np.asanyarray(result_frame.get_data())
            cv2.imwrite('{}/{}.png'.format(Depth_path, str(fps_count).zfill(4)), depth_image)


            print(f"{progress+1}/{len(bagfiles)} Depth fps_count {fps_count}")
            fps_count += 1

    finally:
        print(f"{progress+1}/{len(bagfiles)} Depth_fps_count = {fps_count - 1}")
        pipeline.stop()