import cv2
import pyrealsense2 as rs
import os
import numpy as np
import serial
import time

root_dir = r"C:\Users\Tomson\BRLAB\Stroke\pretest\RealSense"
interval = input("Arduinoの点灯間隔を入力してください (ms):")
# interval = int(30)

SERIAL_MASTER = '233722072880'
SERIAL_SLAVE = '231522070603'

# ser = serial.Serial('COM3', 9600)  # Arduinoのポートを指定

def setup_camera(serial, file_name, sync_mode):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    file_path = os.path.join(root_dir, file_name)
    config.enable_record_to_file(file_path)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    ctx = rs.context()
    devices = ctx.query_devices()
    device_found = False
    for device in devices:
        if device.get_info(rs.camera_info.serial_number) == serial:
            device_found = True
            sensor = device.first_depth_sensor()
            sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)
            break
    if not device_found:
        raise Exception(f"Device with serial number {serial} not found.")
    return pipeline, config

master_pipeline, master_config = setup_camera(SERIAL_MASTER, f'output_master_{interval}.bag', 1)
slave_pipeline, slave_config = setup_camera(SERIAL_SLAVE, f'output_slave_{interval}.bag', 2)

#撮影開始
master_pipeline.start(master_config)
slave_pipeline.start(slave_config)
# ser.write('1'.encode())  # データをエンコードして送信 string -> bytes
start_time = time.time()

try:
    while True:
        master_frames = master_pipeline.wait_for_frames()
        master_color_frame = master_frames.get_color_frame()
        if not master_color_frame:
            continue
        master_image = np.asanyarray(master_color_frame.get_data())
        #RGB順をBGR順に変換
        master_image = cv2.cvtColor(master_image, cv2.COLOR_RGB2BGR)
        resized_master_image = cv2.resize(master_image, (640, 360))

        slave_frames = slave_pipeline.wait_for_frames()
        slave_color_frame = slave_frames.get_color_frame()
        if not slave_color_frame:
            continue
        slave_image = np.asanyarray(slave_color_frame.get_data())
        slave_image = cv2.cvtColor(slave_image, cv2.COLOR_RGB2BGR)
        resized_slave_image = cv2.resize(slave_image, (640, 360))

        #結合
        combined_image = np.hstack((resized_master_image, resized_slave_image))
        cv2.imshow('camera_img', combined_image)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

        #10秒経過したら終了
        if time.time() - start_time > 10:
            break

finally:
    master_pipeline.stop()
    slave_pipeline.stop()
    cv2.destroyAllWindows()
    # ser.write('2'.encode())  # データをエンコードして送信 string -> bytes
    print(f"record finished.")

