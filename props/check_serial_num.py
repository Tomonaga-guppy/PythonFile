#RealSenseを接続して、シリアル番号を取得するプログラムです

import pyrealsense2 as rs
import cv2
import numpy as np

# RealSenseデバイスのコンテキストを取得
ctx = rs.context()

device_num = 1
serial_number_list = []
# デバイス数だけパイプラインを作成
pipelines = [rs.pipeline() for _ in ctx.devices]

# 各デバイスに対してストリームを設定し、パイプラインを開始
for pipeline, device in zip(pipelines, ctx.devices):
    serial_number = device.get_info(rs.camera_info.serial_number)
    print(f"Serial Number {device_num}: {serial_number}")
    serial_number_list.append(serial_number)
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
    pipeline.start(config)
    device_num += 1

try:
    while True:
        # 各パイプラインからフレームを取得し、ウィンドウに表示
        for i, pipeline in enumerate(pipelines):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            cv2.imshow(f"RealSense Camera {serial_number_list[i]}", image)
        
        # ' 'キーが押されたらループを抜ける
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
except Exception as e:
    print(f"Error: {e}")
finally:
    # 各パイプラインを停止
    for pipeline in pipelines:
        pipeline.stop()
    cv2.destroyAllWindows()

# 自分用メモ
# SER_NUM_1 = 231522070603 (master)
# SER_NUM_2 = 233722072880 (slave)