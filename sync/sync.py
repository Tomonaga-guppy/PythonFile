import cv2
import pyrealsense2 as rs
import numpy as np
from threading import Thread
import os
import serial

base_path = r"C:\Users\Tomson\BRLAB\Stroke\pretest"

ser = serial.Serial('COM3', 9600)  # Windowsの場合

if not os.path.exists(base_path + "/master_color"):
    os.makedirs(base_path + "/master_color")
if not os.path.exists(base_path + "/master_depth"):
    os.makedirs(base_path + "/master_depth")
if not os.path.exists(base_path + "/slave_color"):
    os.makedirs(base_path + "/slave_color")
if not os.path.exists(base_path + "/slave_depth"):
    os.makedirs(base_path + "/slave_depth")

def save_image(image, path):
    cv2.imwrite(path, image)

# 各カメラのパイプラインを初期化
master_pipeline = rs.pipeline()
slave_pipeline = rs.pipeline()
master_config = rs.config()
slave_config = rs.config()

# 使用するデバイスのシリアル番号を指定
SERIAL_MASTER = '947522071129'  # マスターカメラ
SERIAL_SLAVE = '947522072616'   # スレーブカメラ
master_config.enable_device(SERIAL_MASTER)
slave_config.enable_device(SERIAL_SLAVE)

# 深度ストリームの設定
master_config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
slave_config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# カラーストリームの設定
master_config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
slave_config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

# デバイスの取得と同期モードの設定
ctx = rs.context()
devices = ctx.query_devices()
master_device = None
slave_device = None

# 各デバイスを識別
for dev in devices:
    if dev.get_info(rs.camera_info.serial_number) == SERIAL_MASTER:
        master_device = dev
    elif dev.get_info(rs.camera_info.serial_number) == SERIAL_SLAVE:
        slave_device = dev

if not master_device or not slave_device:
    raise Exception("指定されたシリアル番号のデバイスが見つかりませんでした。")

# 同期モードの設定
master_sensor = master_device.first_depth_sensor()
slave_sensor = slave_device.first_depth_sensor()
master_sensor.set_option(rs.option.inter_cam_sync_mode, 1)  # マスターとして設定
slave_sensor.set_option(rs.option.inter_cam_sync_mode, 2)   # スレーブとして設定

# カメラの起動
master_pipeline.start(master_config)
slave_pipeline.start(slave_config)

ser.write('1'.encode())  # データをエンコードして送信 string -> bytes
print(f"motive recording start")

try:
    frame_counter = 0
    master_depth_timestamp_before = 0
    slave_depth_timestamp_before = 0
    master_color_timestamp_before = 0
    slave_color_timestamp_before = 0
    while True:
        # フレームを取得
        master_frames = master_pipeline.wait_for_frames()
        slave_frames = slave_pipeline.wait_for_frames()
        master_depth_frame = master_frames.get_depth_frame()
        slave_depth_frame = slave_frames.get_depth_frame()
        master_color_frame = master_frames.get_color_frame()
        slave_color_frame = slave_frames.get_color_frame()

        if not master_depth_frame or not slave_depth_frame or not master_color_frame or not slave_color_frame:
            continue

        #撮影開始からの経過時間を取得 (単位はms)
        master_depth_timestamp = master_depth_frame.get_timestamp()
        slave_depth_timestamp = slave_depth_frame.get_timestamp()
        master_color_timestamp = master_color_frame.get_timestamp()
        slave_color_timestamp = slave_color_frame.get_timestamp()

        # タイムスタンプの差異を計算
        timestamp_color_difference = abs(master_color_timestamp - slave_color_timestamp)
        timestamp_depth_difference = abs(master_depth_timestamp - slave_depth_timestamp)
        print(f"Depth Difference master-slave: {timestamp_depth_difference} ms")
        # print(f"Color Difference master-slave: {timestamp_color_difference} ms")

        # # タイムスタンプの差異を計算
        depth_timestamp_difference_master = abs(master_depth_timestamp - master_depth_timestamp_before)
        depth_timestamp_difference_slave = abs(slave_depth_timestamp - slave_depth_timestamp_before)
        color_timestamp_difference_master = abs(master_color_timestamp - master_color_timestamp_before)
        color_timestamp_difference_slave = abs(slave_color_timestamp - slave_color_timestamp_before)

        print(f"Master Depth Difference: {depth_timestamp_difference_master} ms   Slave Depth Difference: {depth_timestamp_difference_slave} ms   Master Color Difference: {color_timestamp_difference_master} ms   Slave Color Difference: {color_timestamp_difference_slave} ms")

        master_depth_timestamp_before = master_depth_timestamp
        slave_depth_timestamp_before = slave_depth_timestamp
        master_color_timestamp_before = master_color_timestamp
        slave_color_timestamp_before = slave_color_timestamp

        master_depth_image = np.asanyarray(master_depth_frame.get_data())
        slave_depth_image = np.asanyarray(slave_depth_frame.get_data())
        master_color_image = np.asanyarray(master_color_frame.get_data())
        slave_color_image = np.asanyarray(slave_color_frame.get_data())

        #画像をRGBに変換
        master_color_image = cv2.cvtColor(master_color_image, cv2.COLOR_RGB2BGR)
        slave_color_image = cv2.cvtColor(slave_color_image, cv2.COLOR_RGB2BGR)

        #4枚の画像を結合してサイズを1/4にして表示
        master_color_image = cv2.resize(master_color_image, (640, 360))
        slave_color_image = cv2.resize(slave_color_image, (640, 360))
        master_depth_image = cv2.resize(master_depth_image, (640, 360))
        slave_depth_image = cv2.resize(slave_depth_image, (640, 360))
        #深度をカラー画像に変換
        master_depth_image = cv2.applyColorMap(cv2.convertScaleAbs(master_depth_image, alpha=0.03), cv2.COLORMAP_JET)
        slave_depth_image = cv2.applyColorMap(cv2.convertScaleAbs(slave_depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #画像を2*2に結合
        images = np.hstack((np.vstack((master_color_image, master_depth_image)), np.vstack((slave_color_image, slave_depth_image))))
        cv2.imshow("Images", images)

        # スレッドを使用して画像を保存
        Thread(target=save_image, args=(master_depth_image, f"{base_path}/master_depth/{frame_counter}.png")).start()
        Thread(target=save_image, args=(slave_depth_image, f"{base_path}/slave_depth/{frame_counter}.png")).start()
        Thread(target=save_image, args=(master_color_image, f"{base_path}/master_color/{frame_counter}.png")).start()
        Thread(target=save_image, args=(slave_color_image, f"{base_path}/slave_color/{frame_counter}.png")).start()

        frame_counter += 1

        # スペースキーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

finally:
    # 停止処理
    master_pipeline.stop()
    slave_pipeline.stop()
    cv2.destroyAllWindows()
    #Arduinoに停止を通知してシリアルポートを閉じる
    ser.write('1'.encode())  # データをエンコードして送信 string -> bytes
    ser.close()
    print(f"motive recording stop")
