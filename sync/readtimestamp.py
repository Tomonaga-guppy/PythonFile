import pyrealsense2 as rs

# .bagファイルのパスを指定
bag_file_path = r"D:\Duser\Dbrlab\Desktop\tomonaga\sync_test\rs-mocap\output_master_30000.bag "

# パイプラインの設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file_path, repeat_playback=False)

# ストリームの設定
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)

# パイプラインの開始
pipeline.start(config)

# フレームの取得とタイムスタンプの表示
try:
    while True:
        frames = pipeline.wait_for_frames()

        # 深度フレームのタイムスタンプ
        depth_frame = frames.get_depth_frame()
        if depth_frame:
            depth_timestamp = depth_frame.get_timestamp()
            print(f'Depth Frame Timestamp: {depth_timestamp} ms')

        # カラーフレームのタイムスタンプ
        color_frame = frames.get_color_frame()
        if color_frame:
            color_timestamp = color_frame.get_timestamp()
            print(f'Color Frame Timestamp: {color_timestamp} ms')

except Exception as e:
    print(e)

finally:
    # パイプラインの停止
    pipeline.stop()
