from argparse import ArgumentParser
from pyk4a import Config, ImageFormat, PyK4A, PyK4ARecord, FPS, DepthMode, ColorResolution, WiredSyncMode
import os
import time

name = input("保存ファイル名をいれてください ")
save_dir = r"C:\Users\tus\Desktop\record\recorded_data\kinect\2024_0807pyk4a"
# save_dir = r"C:\Users\tus\Desktop\record\recorded_data\kinect\2024_0807pyk4a\test.mkv"

# config0 = Config(color_format=ImageFormat.COLOR_MJPG, camera_fps=FPS.FPS_30, depth_mode=DepthMode.NFOV_UNBINNED, color_resolution=ColorResolution.RES_1080P, wired_sync_mode=WiredSyncMode.MASTER)
config0 = Config(color_format=ImageFormat.COLOR_MJPG, camera_fps=FPS.FPS_30, depth_mode=DepthMode.NFOV_UNBINNED, color_resolution=ColorResolution.RES_1080P, wired_sync_mode=WiredSyncMode.SUBORDINATE)
config1 = Config(color_format=ImageFormat.COLOR_MJPG, camera_fps=FPS.FPS_30, depth_mode=DepthMode.NFOV_UNBINNED, color_resolution=ColorResolution.RES_1080P, wired_sync_mode=WiredSyncMode.SUBORDINATE)
config2 = Config(color_format=ImageFormat.COLOR_MJPG, camera_fps=FPS.FPS_30, depth_mode=DepthMode.NFOV_UNBINNED, color_resolution=ColorResolution.RES_1080P, wired_sync_mode=WiredSyncMode.SUBORDINATE)

device0 = PyK4A(config=config0, device_id=0)
device1 = PyK4A(config=config1, device_id=1)
device2 = PyK4A(config=config2, device_id=2)
device1.start()
device2.start()
time.sleep(1)
device0.start()

device0.exposure = 10000
device1.exposure = 10000
device2.exposure = 10000

record1 = PyK4ARecord(device=device1, config=config1, path=os.path.join(save_dir,f"{name}_1.mkv"))
record2 = PyK4ARecord(device=device2, config=config2, path=os.path.join(save_dir,f"{name}_2.mkv"))
record0 = PyK4ARecord(device=device0, config=config0, path=os.path.join(save_dir,f"{name}_0.mkv"))
# record = PyK4ARecord(device=device0, config=config, path=save_dir)

record1.create()
record2.create()
record0.create()

try:
    print("撮影中です... CTRL-Cで撮影終了します")
    while True:
        capture0 = device0.get_capture()
        capture1 = device1.get_capture()
        capture2 = device2.get_capture()
        record0.write_capture(capture0)
        record1.write_capture(capture1)
        record2.write_capture(capture2)
except KeyboardInterrupt:
    print("撮影を終了します")


record0.flush()
record0.close()
record1.flush()
record1.close()
record2.flush()
record2.close()
print(f"device0の総フレーム数:{record0.captures_count}")
print(f"device1の総フレーム数:{record1.captures_count}")
print(f"device2の総フレーム数:{record2.captures_count}")