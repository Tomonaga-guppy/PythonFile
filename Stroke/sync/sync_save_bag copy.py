import cv2
import pyrealsense2 as rs
import os
import numpy as np
import serial
import threading
import queue
import time

# root_dir = r"C:\Users\Tomson\BRLAB\Stroke\pretest\RealSense"
root_dir = r"D:\Duser\Dbrlab\Desktop\tomonaga\sync_test\rs-mocap"

# interval = input("Arduinoの点灯間隔を入力してください (ms):")
interval = int(30000)

SERIAL_MASTER = '231522070603'
SERIAL_SLAVE = '233722072880'

ser = serial.Serial('COM3', 115200)  # Arduinoのポートを指定

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

def camera_thread(pipeline, config, image_queue):
    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            # RGB順をBGR順に変換
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            resized_image = cv2.resize(image, (640, 360))
            image_queue.put(resized_image)
    finally:
        pipeline.stop()

master_pipeline, master_config = setup_camera(SERIAL_MASTER, f'output_master_{interval}.bag', 1)
slave_pipeline, slave_config = setup_camera(SERIAL_SLAVE, f'output_slave_{interval}.bag', 2)

master_queue = queue.Queue()
slave_queue = queue.Queue()

master_thread = threading.Thread(target=camera_thread, args=(master_pipeline, master_config, master_queue))
slave_thread = threading.Thread(target=camera_thread, args=(slave_pipeline, slave_config, slave_queue))

# スレッド開始
master_thread.start()
slave_thread.start()
start_time = time.perf_counter_ns()  #3撮影開始の時間を計測
ser.write(b'\x01')  # バイナリデータを送信

try:
    while True:
        if not master_queue.empty() and not slave_queue.empty():
            master_image = master_queue.get()
            slave_image = slave_queue.get()
            # 結合
            combined_image = np.hstack((master_image, slave_image))
            cv2.imshow('camera_img', combined_image)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
finally:
    ser.write(b'\x00')  # バイナリデータを送信
    cv2.destroyAllWindows()
    master_thread.join()
    slave_thread.join()
