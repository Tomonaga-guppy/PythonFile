import cv2
import pyrealsense2 as rs
import os

base_path = r"C:\Users\Tomson\BRLAB\Stroke\pretest\RealSense"
base_path  = r"C:\Users\zutom\BRLAB\gait pattern\sync\test"

SERIAL_MASTER = '233722072880'
# SERIAL_SLAVE = '231522070603'

def setup_camera(serial, file_name, sync_mode):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    file_path = os.path.join(base_path, file_name)
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
            pipeline.start(config)
            break
    if not device_found:
        raise Exception(f"Device with serial number {serial} not found.")
    return pipeline

master_pipeline = setup_camera(SERIAL_MASTER, 'output_master.bag', 1)
# slave_pipeline = setup_camera(SERIAL_SLAVE, 'output_slave.bag', 2)

try:
    while True:
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
finally:
    master_pipeline.stop()
    # slave_pipeline.stop()
