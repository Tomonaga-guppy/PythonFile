from pypylon import pylon
import numpy as np
import cv2
import time
import threading
import os
import datetime

name = input("保存するファイル名を入力してください: ")
current_date = datetime.datetime.now().strftime('%Y%m%d')
# 保存するディレクトリのパスを設定
root_dir = fr"f:\Tomson\gait_pattern\{current_date}\{name}"

if not os.path.exists(root_dir):
    os.makedirs(root_dir)

# カメラ処理を行う関数
def process_camera(camera, cameraID, num_frames, stop_event,savedir):

    # CImageFormatConverterクラスのインスタンス作成
    converter = pylon.ImageFormatConverter()
    # 出力形式をRGB8に設定
    converter.OutputPixelFormat = pylon.PixelType_RGB8packed
    converter.OutputBitAlignment = "MsbAligned"

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByUser)

    cv2.namedWindow(f'Camera {cameraID+1} Stream', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'Camera {cameraID+1} Stream', 480,270)  # Adjust the size as needed

    while camera.IsGrabbing() and not stop_event.is_set():
        try:
            grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                img = pylon.PylonImage()
                img.AttachGrabResultBuffer(grabResult)

                ipo = pylon.ImagePersistenceOptions()
                ipo.SetQuality(100)
                filename = f"{savedir}\\saved_pypylon_img{cameraID}_{num_frames}.jpeg"
                print(f"Saving image to {filename}")
                img.Save(pylon.ImageFileFormat_Jpeg, filename, ipo)

                # 画像を表示し、保存
                num_frames[cameraID] += 1
                grabResult.Release()

                img = img.GetArray()
                cv2.imshow(f'Camera {cameraID+1} Stream', img)

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

    stop_event = threading.Event()

    save_dir_cameras = []

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

        save_dir_camera = os.path.join(root_dir, f"camera{i+1}")
        if not os.path.exists(save_dir_camera):
            os.makedirs(save_dir_camera)
        save_dir_cameras.append(save_dir_camera)

    # スレッドのリスト
    threads = []

    start_time = time.time()

    # 各カメラごとにスレッドを作成して実行
    try:
        for i, camera in enumerate(cameras):
            print(f"Starting camera {i+1} thread...")
            thread = threading.Thread(target=process_camera, args=(camera, i, num_frames, stop_event, save_dir_cameras[i]))
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

        cv2.destroyAllWindows()

        # Print out frame counts and frame rates for each camera
        for i in range(num_cameras):
            frame_rate = num_frames[i] / elapsed_time
            print(f"Number of frames acquired (Camera {i+1}): {num_frames[i]}")
            print(f"Frame rate (Camera {i+1}) is: {frame_rate:.2f} FPS")

if __name__ == "__main__":
    main()
