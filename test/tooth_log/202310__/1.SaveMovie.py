# sample code: https://teratail.com/questions/218884

import pyrealsense2 as rs
import numpy as np
import cv2
import glob
import os
import sys

#RGBとDepthで撮影タイミングがごく僅かに違う

root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_09_000"

# if len(sys.argv) > 1:
#     root_dir = sys.argv[1]
# else:
#     print("bagファイルのあるディレクトリパスが指定されていません。")
#     sys.exit()

pattern = os.path.join(root_dir, '*A1*.bag')
bag_files = glob.glob(pattern, recursive=True)

# print(bag_files)
num_bags = len(bag_files)

for progress, bagfile in enumerate(bag_files):

    # Save RGB image

    path = os.path.dirname(bagfile) + '/' + os.path.basename(bagfile).split('.')[0]
    # print(path)
    if not os.path.exists(path):
        os.mkdir(path)

    mp4file = path + '/' + os.path.basename(bagfile).split('.')[0] + '_original.mp4'
    # if os.path.isfile(mp4file):
    #     continue

    config = rs.config()
    config.enable_device_from_file(bagfile)

    pipeline = rs.pipeline()
    try:
        profile = pipeline.start(config)
    except RuntimeError:
        print(f" Error! {bagfile}が読み込めませんでした。ファイルが破損している可能性があります")
        break

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

    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter(mp4file, fmt, frame_rate, size) # ライター作成

    print('{}/{}: '.format(progress+1, num_bags),bagfile, size, frame_rate)
    fps_count = 1

    try:
        pre_time = 0
        while True:
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()

            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            writer.write(color_image)

            cur_time = playback.get_position()  #再生時間の取得 単位はナノ秒
            if cur_time < pre_time:
                break
            pre_time = cur_time

            # print(fps_count)
            fps_count += 1

    finally:
        print(f'RGB_fps_count = {fps_count - 1}')
        pipeline.stop()
        writer.release()
        cv2.destroyAllWindows()


    #Save Depth image
    path = os.path.dirname(bagfile) + '/' + os.path.basename(bagfile).split('.')[0] + '/'

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

    #depthデータに欠損があるのでフィルターをかける https://qiita.com/keoitate/items/efe4212b0074e10378ec
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for stream in profile.get_streams():
        vprof = stream.as_video_stream_profile()
        if  vprof.format() == rs.format.rgb8:
            frame_rate = vprof.fps()
            size = (vprof.width(), vprof.height())

    # print('{}/{}: '.format(progress+1, num_bags),bagfile, size, frame_rate)

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

            #depth_scale = device.first_depth_sensor().get_depth_scale()
            depth_scale = depth_frame.get_units()
            # print(fps_count, "depth_data *",depth_scale,"= meter")

            # Make depth image
            depth_image = np.asanyarray(result_frame.get_data())
            cv2.imwrite('{}{}.png'.format(path, str(fps_count).zfill(4)), depth_image)

            cur_time = playback.get_position()  #再生時間の取得 単位はナノ秒
            if cur_time < pre_time:
                break
            pre_time = cur_time

            # print(fps_count)
            fps_count += 1

    finally:
        print(f'Depth_fps_count = {fps_count - 1}')
        pipeline.stop()

