from pypylon import pylon
import numpy as np
import cv2
import time
import threading

# カメラ処理を行う関数
def process_camera(camera, cameraID, video_writer, num_frames, stop_event):

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByUser)

    cv2.namedWindow(f'Camera {cameraID+1} Stream', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'Camera {cameraID+1} Stream', 480,270)  # Adjust the size as needed

    while camera.IsGrabbing() and not stop_event.is_set():
        try:
            grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                img = grabResult.GetArray()

                # 画像を表示し、保存
                num_frames[cameraID] += 1
                cv2.imshow(f'Camera {cameraID+1} Stream', img)
                video_writer.write(img)

                grabResult.Release()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Keyboard interrupt detected. Exiting...")
                break
        except Exception as e:
            print(f"Error in camera {cameraID+1}: {e}")
            stop_event.set()
            break

# メイン処理
def main():
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
    stop_event = threading.Event()

    # 各カメラごとの初期化
    for i, camera in enumerate(cameras):
        camera.Attach(tlFactory.CreateDevice(devices[i]))
        camera.Open()

        print(f"camera {i+1} is {camera.GetDeviceInfo().GetModelName()}")

        if camera.IsGigE():
            camera.AcquisitionFrameRateEnable.SetValue(True)
            camera.AcquisitionFrameRateAbs.SetValue(fps)  # Set frame rate to 60 Hz
            camera.ExposureTimeAbs.SetValue(8000)  # Exposure time in microseconds
        elif camera.IsUsb():
            camera.AcquisitionFrameRateEnable.SetValue(True)
            camera.AcquisitionFrameRate.SetValue(fps)  # Set frame rate to 60 Hz
            camera.ExposureTime.SetValue(8000)  # Exposure time in microseconds

        height = camera.Height.GetValue()
        width = camera.Width.GetValue()
        num_frames.append(0)  # 各カメラごとのフレームカウントをリストで保持

        # Create VideoWriters for each camera
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(f'output{i+1}.mp4', fourcc, fps, (width, height), isColor=False)

        # 動的に作成したVideoWriterをリストに追加
        video_writers.append(video_writer)

    # スレッドのリスト
    threads = []

    start_time = time.time()

    # 各カメラごとにスレッドを作成して実行
    try:
        for i, camera in enumerate(cameras):
            print(f"Starting camera {i+1} thread...")
            thread = threading.Thread(target=process_camera, args=(camera, i, video_writers[i], num_frames, stop_event))
            threads.append(thread)
            thread.start()

        # CTRL-C (KeyboardInterrupt) をキャッチ
        while any(thread.is_alive() for thread in threads):
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nCTRL-C detected. Exiting gracefully...")
        stop_event.set()  # 全スレッドの停止を要求

    finally:
        # 全てのスレッドが終了するのを待機
        for thread in threads:
            thread.join()

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

if __name__ == "__main__":
    main()
