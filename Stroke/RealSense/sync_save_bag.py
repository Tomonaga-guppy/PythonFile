import cv2
import pyrealsense2 as rs
import os
import numpy as np
import serial
import threading
import queue
import time
import concurrent.futures
import glob

root_dir = r"C:\Users\Tomson\BRLAB\gait_pattern\sync_test\rs-mocap"
# root_dir = r"D:\Duser\Dbrlab\Desktop\tomonaga\sync_test\rs-mocap"
# root_dir = r"c:\Users\tus\Desktop\record\recorded_data\realsense"


input_name = 'test'
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

def camera_thread(pipeline, config, image_queue, start_time_list, first_frame_time_list, thread_name, stop_event, start_event):
    start_event.wait()  # Wait for the start event to be set
    start_time = time.perf_counter_ns()  # Record the start time
    start_time_list.append((thread_name, start_time))  # Append the start time to the list
    pipeline.start(config)
    first_frame_captured = False  # Flag to indicate if the first frame has been captured
    try:
        while not stop_event.is_set():  # Stop eventがセットされているか確認
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            if not first_frame_captured:
                first_frame_time = time.perf_counter_ns()  # Record the time of the first frame
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

bagfile1_name = os.path.join(root_dir, f'output1_{input_name}.bag')
bagfile2_name = os.path.join(root_dir, f'output2_{input_name}.bag')

if os.path.exists(bagfile1_name) and os.path.exists(bagfile2_name):
    bagfile_num = len(glob.glob(os.path.join(root_dir, f'output1_{input_name}*.bag')))
    bagfile2_num = len(glob.glob(os.path.join(root_dir, f'output2_{input_name}*.bag')))
    bagfile1_name = os.path.join(root_dir, f'output1_{input_name}({bagfile_num}).bag')
    bagfile2_name = os.path.join(root_dir, f'output2_{input_name}({bagfile2_num}).bag')

master_pipeline, master_config = setup_camera(SERIAL_MASTER, bagfile1_name, 1)
slave_pipeline, slave_config = setup_camera(SERIAL_SLAVE, bagfile2_name, 2)

master_queue = queue.Queue()
slave_queue = queue.Queue()

start_times = []  # List to store start times
first_frame_times = []  # List to store first frame times
stop_event = threading.Event()  # スレッド停止用のイベント
start_event = threading.Event()  # スレッド開始用のイベント

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(camera_thread, master_pipeline, master_config, master_queue, start_times, first_frame_times, 'master', stop_event, start_event),
        executor.submit(camera_thread, slave_pipeline, slave_config, slave_queue, start_times, first_frame_times, 'slave', stop_event, start_event),
        executor.submit(serial_write_thread, start_event, start_times)
    ]

    # Optionally wait for threads to start
    time.sleep(1)  # Small delay to ensure threads have started

    start_event.set()  # スレッド開始イベントをセット

    start_time = time.perf_counter_ns()  # 撮影開始の時間を計測

    try:
        while True:
            if not master_queue.empty() and not slave_queue.empty():
                master_image = master_queue.get()
                slave_image = slave_queue.get()
                # 結合
                combined_image = np.hstack((master_image, slave_image))
                cv2.imshow('camera_img', combined_image)

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

# # Print start times of each thread
# for thread_name, start_time in start_times:
#     print(f'{thread_name} thread start time: {start_time} ns')

# # Print first frame times of each camera
# for thread_name, first_frame_time in first_frame_times:
#     print(f'{thread_name} first frame time: {first_frame_time} ns')

# Calculate and print start_diff between all threads
master_start_time = next(start_time for thread_name, start_time in start_times if thread_name == 'master')
slave_start_time = next(start_time for thread_name, start_time in start_times if thread_name == 'slave')
serial_start_time = next(start_time for thread_name, start_time in start_times if thread_name == 'serial')

print(f'start_diff (master vs slave) = {(slave_start_time - master_start_time) * 1e-6} ms')
print(f'start_diff (master vs serial) = {(serial_start_time - master_start_time) * 1e-6} ms')

# Calculate and print first_frame_diff between master and slave cameras
master_first_frame_time = next(first_frame_time for thread_name, first_frame_time in first_frame_times if thread_name == 'master')
slave_first_frame_time = next(first_frame_time for thread_name, first_frame_time in first_frame_times if thread_name == 'slave')

print(f"RS1 First Frame Time: {(master_first_frame_time - master_start_time) * 1e-6} ms")