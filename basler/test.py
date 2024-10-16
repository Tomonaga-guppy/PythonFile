from pypylon import pylon
import numpy as np
import cv2
import time
import datetime

num_frames = []
image = pylon.PylonImage()

tlFactory = pylon.TlFactory.GetInstance()
devices = tlFactory.EnumerateDevices()
num_cameras = len(devices)

if num_cameras < 1:
    raise pylon.RUNTIME_EXCEPTION("At least one camera is required.")

fps = 60
cameras = pylon.InstantCameraArray(num_cameras)

video_writers = []
wh_list = []

# 各カメラごとの初期化
for i, camera in enumerate(cameras):
    camera.Attach(tlFactory.CreateDevice(devices[i]))
    camera.Open()

    if camera.IsGigE():
        # Set frame rate and exposure time for GigE cameras
        camera.AcquisitionFrameRateEnable.SetValue(True)
        camera.AcquisitionFrameRateAbs.SetValue(fps)  # Set frame rate to 60 Hz
        camera.ExposureTimeAbs.SetValue(8000)  # Exposure time in microseconds
    elif camera.IsUsb():
        # Set frame rate and exposure time for USB cameras
        camera.AcquisitionFrameRateEnable.SetValue(True)
        camera.AcquisitionFrameRate.SetValue(fps)  # Set frame rate to 60 Hz
        camera.ExposureTime.SetValue(8000)  # Exposure time in microseconds

    height = camera.Height.GetValue()
    width = camera.Width.GetValue()
    wh_list.append((width, height))
    num_frames.append(0)  # 各カメラごとのフレームカウントをリストで保持

    # Create VideoWriters for each camera
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(f'output{i+1}.mp4', fourcc, fps, (width, height), isColor=False)

    # 動的に作成したVideoWriterをリストに追加
    video_writers.append(video_writer)

# Start grabbing for all cameras
cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByUser)

start_time = time.time()

# Create OpenCV windows for each camera stream
for i in range(num_cameras):
    cv2.namedWindow(f'Camera {i+1} Stream', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'Camera {i+1} Stream', 480,270)  # Adjust the size as needed

try:
    while cameras.IsGrabbing():
        grabResult = cameras.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            cameraID = grabResult.GetCameraContext()
            img = grabResult.GetArray()

            # 表示と保存をカメラごとに行う
            num_frames[cameraID] += 1
            cv2.imshow(f'Camera {cameraID+1} Stream', img)
            video_writers[cameraID].write(img)

            grabResult.Release()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Keyboard interrupt detected. Exiting...")
            break

except KeyboardInterrupt:
    print("\nCTRL-C detected. Exiting gracefully...")

finally:
    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Release all video writers
    for video_writer in video_writers:
        video_writer.release()
    cv2.destroyAllWindows()

    # Print out frame counts and frame rates for each camera
    for i in range(num_cameras):
        frame_rate = num_frames[i] / elapsed_time
        print(f"Number of frames acquired (Camera {i+1}): {num_frames[i]}")
        print(f"Frame rate (Camera {i+1}) is: {frame_rate:.2f} FPS")

