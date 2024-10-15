import cv2
import pyrealsense2 as rs
import os
import numpy as np
import threading
import queue
import time
import glob
import serial
import concurrent.futures

# root_dir = r"C:\Users\Tomson\BRLAB\gait_pattern\sync_test\rs-mocap"
root_dir = r"c:\Users\tus\Desktop\record\recorded_data\realsense\one_dev"

input_name = 'test'
ser = serial.Serial('COM3', 115200)  # Arduinoのポートを指定

def setup_camera(device, file_name, sync_mode):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device.get_info(rs.camera_info.serial_number))
    file_path = os.path.join(root_dir, file_name)
    config.enable_record_to_file(file_path)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    sensor = device.first_depth_sensor()
    sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)
    return pipeline, config

def camera_thread(pipeline, config, image_queue, start_time_list, first_frame_time_list, thread_name, stop_event, start_event):
    start_event.wait()  # スタートイベントを待つ
    start_time = time.perf_counter_ns()  # 開始時間を記録
    start_time_list.append((thread_name, start_time))  # 開始時間をリストに追加
    pipeline.start(config)
    first_frame_captured = False  # 最初のフレームがキャプチャされたかどうかのフラグ
    try:
        while not stop_event.is_set():  # 停止イベントがセットされていないか確認
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            if not first_frame_captured:
                first_frame_time = time.perf_counter_ns()  # 最初のフレームの時間を記録
                first_frame_time_list.append((thread_name, first_frame_time))
                first_frame_captured = True
            image = np.asanyarray(color_frame.get_data())
            # RGB順をBGR順に変換
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            resized_image = cv2.resize(image, (640, 360))
            image_queue.put(resized_image)
    finally:
        pipeline.stop()

def serial_write_thread(start_event, start_time_list):
    start_event.wait()  # Wait for the start event to be set
    start_time = time.perf_counter_ns()  # Record the start time
    start_time_list.append(('serial', start_time))  # Append the start time to the list
    ser.write(b'\x01')  # バイナリデータを送信

ctx = rs.context()
devices = ctx.query_devices()
if len(devices) < 1:
    raise Exception("少なくとも1台のRealSenseデバイスが必要です。")

device1 = devices[0]

bagfile_name = os.path.join(root_dir, f'output_{input_name}.bag')

if os.path.exists(bagfile_name):
    bagfile_num = len(glob.glob(os.path.join(root_dir, f'output_{input_name}*.bag')))
    bagfile_name = os.path.join(root_dir, f'output_{input_name}_{bagfile_num+1}.bag')

pipeline, config = setup_camera(device1, bagfile_name, 1)

image_queue = queue.Queue()

start_times = []  # 開始時間を記録するリスト
first_frame_times = []  # 最初のフレームの時間を記録するリスト
stop_event = threading.Event()  # スレッド停止用のイベント
start_event = threading.Event()  # スレッド開始用のイベント

thread = threading.Thread(target=camera_thread, args=(pipeline, config, image_queue, start_times, first_frame_times, 'camera', stop_event, start_event))
thread.start()

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(camera_thread, pipeline, config, image_queue, start_times, first_frame_times, 'master', stop_event, start_event),
        executor.submit(serial_write_thread, start_event, start_times)
    ]

    # Optionally wait for threads to start
    time.sleep(1)  # Small delay to ensure threads have started

    start_event.set()  # スレッド開始イベントをセット

    start_time = time.perf_counter_ns()  # 撮影開始の時間を計測

    try:
        while True:
            if not image_queue.empty():
                rs_image = image_queue.get()
                cv2.imshow('camera_img', rs_image)

            if cv2.waitKey(1) & 0xFF == ord(' '):  # Spaceキーで停止
                break
    except KeyboardInterrupt:  # Ctrl+Cで停止
        pass
    finally:
        ser.write(b'\x00')  # バイナリデータを送信
        stop_event.set()  # スレッド停止イベントをセット

        # Ensure all threads are joined
        concurrent.futures.wait(futures)
        cv2.destroyAllWindows()

# # 開始時間の差を計算して表示
# camera_start_time = next(start_time for thread_name, start_time in start_times if thread_name == 'camera')
# print(f'カメラの開始時間: {camera_start_time} ns')

# # 最初のフレームの差を計算して表示
# camera_first_frame_time = next(first_frame_time for thread_name, first_frame_time in first_frame_times if thread_name == 'camera')
# print(f"カメラの最初のフレームの時間: {(camera_first_frame_time - camera_start_time) * 1e-6} ms")


