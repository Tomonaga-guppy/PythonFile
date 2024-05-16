import pyrealsense2 as rs
import time

# RealSenseパイプラインを作成
pipeline = rs.pipeline()

# コンフィギュレーションを設定
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# スタート時間を記録
start_time = time.time()

# パイプラインを開始
pipeline.start(config)

# 最初のフレームを取得するまでの時間を計測
frames = pipeline.wait_for_frames()

# 初めのフレーム取得時間を記録
first_frame_time = time.time()
time_to_first_frame = first_frame_time - start_time

# 次のフレームを取得するまでの時間を計測
next_frames = pipeline.wait_for_frames()

# 次のフレーム取得時間を記録
second_frame_time = time.time()
time_to_second_frame = second_frame_time - first_frame_time

print(f"Time to first frame: {time_to_first_frame:.4f} seconds")
print(f"Time to next frame: {time_to_second_frame:.4f} seconds")

# パイプラインを停止
pipeline.stop()
