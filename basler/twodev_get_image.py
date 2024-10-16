from pypylon import pylon
import numpy as np
import cv2
import time

# Create smaller OpenCV windows for both camera streams
cv2.namedWindow('Camera 1 Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera 1 Stream', 640, 512)  # Adjust the size as needed

cv2.namedWindow('Camera 2 Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera 2 Stream', 640, 512)  # Adjust the size as needed

num_frames1 = 0
num_frames2 = 0

tlFactory = pylon.TlFactory.GetInstance()
devices = tlFactory.EnumerateDevices()
if len(devices) < 2:
    raise pylon.RUNTIME_EXCEPTION("At least two cameras are required.")

cameras = pylon.InstantCameraArray(2)

for i, camera in enumerate(cameras):
    camera.Attach(tlFactory.CreateDevice(devices[i]))
    camera.Open()

    if camera.IsGigE():
        # Set frame rate and exposure time
        camera.AcquisitionFrameRateEnable.SetValue(True)
        camera.AcquisitionFrameRateAbs.SetValue(30)  # Set frame rate to 30 Hz
        camera.ExposureTimeAbs.SetValue(10000)  # Exposure time in microseconds
    elif camera.IsUsb():
        camera.AcquisitionFrameRateEnable.SetValue(True)
        camera.AcquisitionFrameRate.SetValue(30)  # Set frame rate to 30 Hz
        camera.ExposureTime.SetValue(10000)  # Exposure time in microseconds

    camera.LineSelector.SetValue("Line2")
    camera.LineMode.SetValue("Output")
    camera.LineSource.SetValue("ExposureActive")

# Create VideoWriters for both cameras
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out1 = cv2.VideoWriter('D:\\videoCaptures\\output1.avi', fourcc, 30.0, (1280,1024), isColor=False)
out2 = cv2.VideoWriter('D:\\videoCaptures\\output2.avi', fourcc, 30.0, (1280,1024), isColor=False)

# Starts grabbing for all cameras
cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly,
                      pylon.GrabLoop_ProvidedByUser)

start_time = time.time()

while cameras.IsGrabbing():
    try:
        grabResult = cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():

            cameraID = grabResult.GetCameraContext()
            img = grabResult.GetArray()

            if cameraID == 0:

                num_frames1 += 1
                cv2.imshow('Camera 1 Stream', img)
                out1.write(img)

            if cameraID == 1:
                num_frames2 += 1
                cv2.imshow('Camera 2 Stream', img)
                out2.write(img)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
        break

# Calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
frame_rate = num_frames1/elapsed_time

out1.release()
out2.release()
cv2.destroyAllWindows()

print(f"Recording finished. Elapsed time: {elapsed_time:.2f} seconds")
print(f"Number of frames acquired (Camera 1): {num_frames1}")
print(f"Number of frames acquired (Camera 2): {num_frames2}")
print(f"Frame rate (camera1) is: {frame_rate}")