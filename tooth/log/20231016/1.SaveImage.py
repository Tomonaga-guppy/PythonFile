# sample code: https://teratail.com/questions/218884

import pyrealsense2 as rs
import numpy as np
import cv2
import glob
import os
import sys
import csv

root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_09_000"

# if len(sys.argv) > 1:
#     root_dir = sys.argv[1]
# else:
#     print("bagファイルのあるディレクトリパスが指定されていません。")
#     sys.exit()

pattern = os.path.join(root_dir, '*A2*.bag')
bag_files = glob.glob(pattern, recursive=True)

error_bagfiles = []  #読み取れなかったbagファイル記録用

#RGBとDepthで撮影タイミングがごく僅かに違うため別々で保存
for progress, bagfile in enumerate(bag_files):
    timestamp_RBG = []  #IMUデータの取得時間調整用

    # Save RGB image
    path = os.path.dirname(bagfile) + '/' + os.path.basename(bagfile).split('.')[0]
    # print(path)
    if not os.path.exists(path):
        os.mkdir(path)

    mp4file = path + '/' + os.path.basename(bagfile).split('.')[0] + '_original.mp4'
    # if os.path.isfile(mp4file):
    #     continue

    RGB_path = path + '/RGB_image'
    if not os.path.exists(RGB_path):
        os.mkdir(RGB_path)

    config = rs.config()
    config.enable_device_from_file(bagfile)

    pipeline = rs.pipeline()
    try:
        profile = pipeline.start(config)
    except RuntimeError:
        error_bagfiles.append(bagfile)
        continue

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

    print('{}/{}: '.format(progress+1, len(bag_files)),bagfile, "size =",size, "frame_rate =",frame_rate)
    fps_count = 1

    try:
        pre_time = 0
        while True:
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()

            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            cur_time = playback.get_position()  #再生時間の取得 単位はナノ秒

            if cur_time < pre_time:  #前フレームより再生時間が進んでいなかったら終了
                break

            cv2.imwrite('{}/{}.png'.format(RGB_path, str(fps_count).zfill(4)), color_image)
            writer.write(color_image)

            timestamp_RBG.append(pre_time)
            pre_time = cur_time

            print(fps_count)
            fps_count += 1

    finally:
        print(f'RGB_fps_count = {fps_count - 1}')
        pipeline.stop()
        writer.release()
        cv2.destroyAllWindows()




    # #Save Depth image
    # Depth_path = os.path.dirname(bagfile)  + '/' + os.path.basename(bagfile).split('.')[0]+ '/Depth_image'
    # if not os.path.exists(Depth_path):
    #     os.mkdir(Depth_path)


    # config = rs.config()
    # config.enable_device_from_file(bagfile)

    # pipeline = rs.pipeline()
    # profile = pipeline.start(config)

    # # create Align Object
    # align_to = rs.stream.color
    # align = rs.align(align_to)

    # device = profile.get_device()
    # playback = device.as_playback()
    # playback.set_real_time(False)  #リアルタイム再生をオフにするとフレームごとに処理が終わるまで待機してくれる

    # #depthデータに欠損があるのでフィルターをかける https://qiita.com/keoitate/items/efe4212b0074e10378ec
    # """
    # # decimarion_filterのパラメータ
    # decimate = rs.decimation_filter()
    # decimate.set_option(rs.option.filter_magnitude, 1)
    # # spatial_filterのパラメータ
    # spatial = rs.spatial_filter()
    # spatial.set_option(rs.option.filter_magnitude, 1)
    # spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
    # spatial.set_option(rs.option.filter_smooth_delta, 50)
    # """
    # # hole_filling_filterのパラメータ
    # hole_filling = rs.hole_filling_filter(2)
    # """
    # # disparity
    # depth_to_disparity = rs.disparity_transform(True)
    # disparity_to_depth = rs.disparity_transform(False)
    # """

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # for stream in profile.get_streams():
    #     vprof = stream.as_video_stream_profile()
    #     if  vprof.format() == rs.format.rgb8:
    #         frame_rate = vprof.fps()
    #         size = (vprof.width(), vprof.height())

    # # print('{}/{}: '.format(progress+1, len(bag_files)),bagfile, size, frame_rate)

    # try:
    #     fps_count = 1
    #     pre_time = 0
    #     while True:
    #         frames = pipeline.wait_for_frames()

    #         aligned_frames = align.process(frames)
    #         # color_frame = aligned_frames.get_color_frame()
    #         depth_frame = aligned_frames.get_depth_frame()

    #         #depthデータの後処理(フィルター)
    #         #filter_frame = decimate.process(depth_frame)
    #         #filter_frame = depth_to_disparity.process(filter_frame)
    #         #filter_frame = spatial.process(filter_frame)
    #         #filter_frame = disparity_to_depth.process(filter_frame)

    #         filter_frame = hole_filling.process(depth_frame)
    #         result_frame = filter_frame.as_depth_frame()

    #         cur_time = playback.get_position()  #再生時間の取得 単位はナノ秒
    #         if cur_time < pre_time:
    #             break

    #         #depth_scale = device.first_depth_sensor().get_depth_scale()
    #         # depth_scale = depth_frame.get_units()
    #         # print(fps_count, "depth_data *",depth_scale,"= meter")

    #         # Make depth image
    #         depth_image = np.asanyarray(result_frame.get_data())
    #         cv2.imwrite('{}/{}.png'.format(Depth_path, str(fps_count).zfill(4)), depth_image)
    #         # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
    #         # cv2.imwrite('{}{}.png'.format(Depth_path + "/", str(fps_count).zfill(4)), depth_colormap)

    #         pre_time = cur_time

    #         print(fps_count)
    #         fps_count += 1

    # finally:
    #     print(f'Depth_fps_count = {fps_count - 1}')
    #     pipeline.stop()





    #IMUデータを保存する
    npy_filename = os.path.dirname(bagfile) + '/' + os.path.basename(bagfile).split('.')[0] + '/accel_data.npy'
    accel_data = []
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bagfile)
    config.enable_stream(rs.stream.accel)
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)  #リアルタイム再生をオフにするとフレームごとに処理が終わるまで待機してくれる

try:
    fps_count = 1
    pre_time = 0

    while True:
        frames = pipeline.wait_for_frames()
        cur_time = playback.get_position()  #再生時間の取得 単位はナノ秒
        accel_info = frames[0].as_motion_frame().get_motion_data()  #加速度データの取得 （frames[1]はジャイロ）
        accel = np.array([accel_info.x, accel_info.y, accel_info.z])
        # if len(timestamp_RBG) == fps_count-1 or cur_time < pre_time:  #frame数が合わなくなるか再生時間が進まなくなったら終了
        if len(timestamp_RBG) == fps_count or cur_time < pre_time:  #frame数が合わなくなるか再生時間が進まなくなったら終了
            accel_data.append(accel)
            # print(f"len(timestamp_RBG) = {len(timestamp_RBG)}")
            # print(f"fps_count = {fps_count}")
            # print(f"accel = {len(accel_data)}")
            break

        if cur_time >= timestamp_RBG[fps_count-1]:  #加速度センサーのfpsが約62.5のためカメラ(fps30)と合わせる
            accel_data.append(accel)
            fps_count += 1
            pass

            pre_time = cur_time
finally:
    accel_data = np.array(accel_data)
    np.save(npy_filename, accel_data)
    print(f"accel_data is saved in {npy_filename}")
    pipeline.stop()

if not len(error_bagfiles) == 0:
    print(f"以下のファイルが読み取れませんでした。破損している可能性があります")
    for i in range(len(error_bagfiles)):
        print(f"・{error_bagfiles[i]}")