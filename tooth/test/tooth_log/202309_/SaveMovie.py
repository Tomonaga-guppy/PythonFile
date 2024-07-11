import pyrealsense2 as rs
import numpy as np
import cv2
import glob
import os

root_dir = 'C:/Users/Tomson/BRLAB/tooth/Temporomandibular_movement/movie/2023_06_05'
pattern = os.path.join(root_dir, '*.bag')
bag_files = glob.glob(pattern, recursive=True)

print(bag_files)
num_bags = len(bag_files)

# Save RGB image
#dirnameでファイル名、basenameでフォルダ名を取得
for progress, bagfile in enumerate(bag_files):
    mp4file = os.path.dirname(bagfile) + '/' + os.path.basename(bagfile).split('.')[0] + '_original.mp4'
    path = os.path.dirname(bagfile) + '/' + os.path.basename(bagfile).split('.')[0]

    if os.path.isfile(mp4file):
        continue

    config = rs.config()
    config.enable_device_from_file(bagfile)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # create Align Object
    align_to = rs.stream.color
    align = rs.align(align_to)

    device = profile.get_device()
    playback = device.as_playback()

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
        cur = -1
        while True:
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            writer.write(color_image)

            fps_count += 1

            next = playback.get_position()
            if next < cur:
                break
            cur = next

    finally:
        pipeline.stop()
        writer.release()
        cv2.destroyAllWindows()

# sample code: https://teratail.com/questions/218884

# Save depth image
for progress, bagfile in enumerate(bag_files):
    path = os.path.dirname(bagfile) + '/'
    print(path)

    config = rs.config()
    config.enable_device_from_file(bagfile)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # create Align Object
    align_to = rs.stream.color
    align = rs.align(align_to)

    device = profile.get_device()
    playback = device.as_playback()

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
    #writer_depth = cv2.VideoWriter('{}depth_color.mp4'.format(path), fourcc, 30, (1280, 720), True)

    for stream in profile.get_streams():
        vprof = stream.as_video_stream_profile()
        if  vprof.format() == rs.format.rgb8:
            frame_rate = vprof.fps()
            size = (vprof.width(), vprof.height())

    print('{}/{}: '.format(progress+1, num_bags),bagfile, size, frame_rate)
    fps_count = 1

    try:
        cur = -1
        while True:
            frames = pipeline.wait_for_frames()


            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
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
            print(fps_count, "depth_data *",depth_scale,"= meter")

            """
            # Make depth color map movie
            range_min=0.2
            range_max=0.8
            depth_image = np.asanyarray(result_frame.get_data())
            depth = depth_image.astype(np.float64) * depth_scale
            depth = np.where((depth >= range_min) & (depth < range_max), depth - range_min, 0)
            depth = depth * ( 255 / (range_max - range_min) )
            depth = depth.astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

            writer_depth.write(depth_colormap)
            """

            # Make depth image
            depth_image = np.asanyarray(result_frame.get_data())
            cv2.imwrite('{}{}.png'.format(path, str(fps_count).zfill(4)), depth_image)
            fps_count += 1

            next = playback.get_position()
            if next < cur:
                break
            cur = next

    finally:
        pipeline.stop()
        #writer_depth.release