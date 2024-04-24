import cv2
import pyrealsense2 as rs
import numpy as np

# 各カメラの設定を行う
pipeline1 = rs.pipeline()
pipeline2 = rs.pipeline()
config1 = rs.config()
config2 = rs.config()

# 使用するデバイスのシリアル番号を指定
config1.enable_device('947522071129')  # マスターカメラ
config2.enable_device('947522072616')  # スレーブカメラ

# 各カメラからの映像を取得する設定（深度ストリームに変更）
config1.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config2.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# デバイスの取得と同期モードの設定
ctx = rs.context()
devices = ctx.query_devices()
master_device = devices[0]
slave_device = devices[1]

# 同期モードを設定する前に、適切なデバイスを選択することが重要です
for dev in devices:
    if dev.get_info(rs.camera_info.serial_number) == '947522071129':
        master_device = dev
    elif dev.get_info(rs.camera_info.serial_number) == '947522072616':
        slave_device = dev

master_sensor = master_device.first_depth_sensor()
slave_sensor = slave_device.first_depth_sensor()

master_sensor.set_option(rs.option.inter_cam_sync_mode, 1)  # マスターとして設定
slave_sensor.set_option(rs.option.inter_cam_sync_mode, 2)   # スレーブとして設定

# カメラの起動
pipeline1.start(config1)
pipeline2.start(config2)

try:
    while True:
        # 1台目のカメラからフレームを取得
        frames1 = pipeline1.wait_for_frames()
        depth_frame1 = frames1.get_depth_frame()
        if not depth_frame1:
            continue
        depth_image1 = np.asanyarray(depth_frame1.get_data())

        # 2台目のカメラからフレームを取得
        frames2 = pipeline2.wait_for_frames()
        depth_frame2 = frames2.get_depth_frame()
        if not depth_frame2:
            continue
        depth_image2 = np.asanyarray(depth_frame2.get_data())

        # 画像をウィンドウに表示
        cv2.imshow('Master Depth', depth_image1)
        cv2.imshow('Slave Depth', depth_image2)

        # スペースキーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

finally:
    # 停止処理
    pipeline1.stop()
    pipeline2.stop()
    cv2.destroyAllWindows()
