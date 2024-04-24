import cv2
import pyrealsense2 as rs
import numpy as np

# 各カメラの設定を行う
pipeline1 = rs.pipeline()
pipeline2 = rs.pipeline()
config1 = rs.config()
config2 = rs.config()

# 使用するデバイスのシリアル番号を指定（各自の環境に合わせて変更してください）
config1.enable_device('947522071129')
config2.enable_device('947522072616')

# 各カメラからの映像を取得する設定
config1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# カメラの起動
pipeline1.start(config1)
pipeline2.start(config2)

try:
    while True:
        # 1台目のカメラからフレームを取得
        frames1 = pipeline1.wait_for_frames()
        color_frame1 = frames1.get_color_frame()
        if not color_frame1:
            continue
        image1 = np.asanyarray(color_frame1.get_data())

        # 2台目のカメラからフレームを取得
        frames2 = pipeline2.wait_for_frames()
        color_frame2 = frames2.get_color_frame()
        if not color_frame2:
            continue
        image2 = np.asanyarray(color_frame2.get_data())

        # 画像をウィンドウに表示
        cv2.imshow('RealSense 1', image1)
        cv2.imshow('RealSense 2', image2)

        # スペースキーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

finally:
    # 停止処理
    pipeline1.stop()
    pipeline2.stop()
    cv2.destroyAllWindows()

