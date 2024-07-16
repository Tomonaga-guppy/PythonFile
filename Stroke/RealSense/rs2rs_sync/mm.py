import glob
import pyrealsense2 as rs
import numpy as np
import cv2

root_dir = r"C:\Users\zutom\BRLAB\gait pattern\sync\test"

blinking_interval = input("点灯間隔を入力してください: ")

bag_file_path = root_dir+f'/output_solo_{blinking_interval}.bag'

# パイプライン設定
pipeline = rs.pipeline()
config = rs.config()

# .bagファイルからのデータの読み込みを設定
config.enable_device_from_file(bag_file_path, repeat_playback=False)

#bagファイルからデータを読み込み指定ピクセルの位置[mm]を取得
def get_mm_value(profile, aligned_frames, x, y):
    depth_frame = hole_filling.process(aligned_frames.get_depth_frame()).as_depth_frame()
    color_intr = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()
    depth_pixel = [x, y]
    depth_point = rs.rs2_deproject_pixel_to_point(color_intr, depth_pixel, depth_frame.get_distance(x, y))
    depth_point=np.array(depth_point)*1000
    print(type(depth_point))
    return depth_point

try:
    # パイプライン開始
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)  # リアルタイム再生を無効化

    # フレーム取得用のアライメントオブジェクト
    align_to = rs.stream.color
    align = rs.align(align_to)

    hole_filling = rs.hole_filling_filter(2)

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

        #depthフレームの取得
        depth_frame = aligned_frames.get_depth_frame()
        filter_frame = hole_filling.process(depth_frame)
        depth_frame = filter_frame.as_depth_frame()

        # Numpy配列への変換
        color_image = np.asanyarray(color_frame.get_data())

        # 指定ピクセルの位置を取得
        x_pixel = 1000
        y_pixel = 264
        point_position = get_mm_value(profile, aligned_frames, x_pixel, y_pixel)
        print(f"{frame_count} Point position: {point_position}")

        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)
        frame_count += 1

finally:
    # パイプライン停止
    pipeline.stop()
