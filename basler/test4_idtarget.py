from pypylon import pylon
import numpy as np
import cv2
import time
import threading
import os
import datetime

name = input("保存するファイル名を入力してください: ")
current_date = datetime.datetime.now().strftime('%Y%m%d')
root_dir = fr"f:\Tomson\gait_pattern\{current_date}\{name}"

if not os.path.exists(root_dir):
    os.makedirs(root_dir)

# カメラ処理を行う関数
def process_camera(camera, cameraID, num_frames, stop_event, savedir, start_event):
    print(f"akjvhi")
    if stop_event.is_set():
        print("return")
        return
    print(f"faq")
    start_event.wait()  # トリガー信号を受け取るまで待機
    print(f"baivghfu")

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByUser)

    cv2.namedWindow(f'Camera {cameraID+1} Stream', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'Camera {cameraID+1} Stream', 480, 270)

    while camera.IsGrabbing():
        try:
            grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                img = pylon.PylonImage()
                img.AttachGrabResultBuffer(grabResult)

                ipo = pylon.ImagePersistenceOptions()
                ipo.SetQuality(100)
                filename = f"{savedir}\\saved_pypylon_img{cameraID}_{num_frames[cameraID]}.jpeg"
                print(f"Saving image to {filename}")
                img.Save(pylon.ImageFileFormat_Jpeg, filename, ipo)

                num_frames[cameraID] += 1
                grabResult.Release()

                img = img.GetArray()
                cv2.imshow(f'Camera {cameraID+1} Stream', img)

            if cv2.waitKey(1) & 0xFF == ord('q') or stop_event.is_set():
                print("Keyboard interrupt detected. Exiting...")
                break
        except Exception as e:
            print(f"Error in camera {cameraID+1}: {e}")
            stop_event.set()
            break
    # Grabbingの停止
    camera.StopGrabbing()
    cv2.destroyWindow(f'Camera {cameraID+1} Stream')

# 特定のデバイスがパルス信号を受け取ったら計測を開始し、その後は60Hzで処理を継続
def wait_for_trigger_and_start(camera, cameraID, num_frames, stop_event, savedir, start_event):
    print(f"Waiting for trigger on camera {cameraID+1}...")

    if camera.GetDeviceInfo().GetSerialNumber() == "40121811":
        # パルス信号 (外部トリガー) を待機
        camera.TriggerSelector.Value = "FrameStart"
        camera.TriggerMode.Value = "On"
        camera.TriggerActivation.Value = "FallingEdge"
        camera.TriggerSource.Value = "Line1"  # Line1 からトリガー信号を受信

        while not stop_event.is_set():
            print(f"camera.WaitForFrameTriggerReady(1, pylon.TimeoutHandling_Return) = {camera.WaitForFrameTriggerReady(1, pylon.TimeoutHandling_Return)}")
            print(f"camera.IsGrabbing() = {camera.IsGrabbing()}")
            try:
                # if camera.WaitForFrameTriggerReady(1, pylon.TimeoutHandling_Return):
                if camera.IsGrabbing():
                    print("Trigger signal received. Starting 60Hz measurement...")
                    start_event.set()  # すべてのカメラを開始させるイベントをセット
                    break
            except Exception as e:
                pass
                # print(f"Error while waiting for trigger on camera {cameraID+1}: {e}")

    print(f"っここ")
    # トリガー信号を受け取ったら60Hzで画像取得を開始
    process_camera(camera, cameraID, num_frames, stop_event, savedir, start_event)
    print(f"fawjeovaiub")
    return

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
    start_event = threading.Event()
    save_dir_cameras = []

    for i, camera in enumerate(cameras):
        camera.Attach(tlFactory.CreateDevice(devices[i]))
        camera.Open()

        print(f"camera {i+1} is {camera.GetDeviceInfo().GetModelName()}")

        if camera.IsGigE():
            camera.AcquisitionFrameRateEnable.SetValue(True)
            camera.AcquisitionFrameRateAbs.SetValue(fps)
            camera.ExposureTimeAbs.SetValue(8000)
        elif camera.IsUsb():
            camera.AcquisitionFrameRateEnable.SetValue(True)
            camera.AcquisitionFrameRate.SetValue(fps)
            camera.ExposureTime.SetValue(8000)

        height = camera.Height.GetValue()
        width = camera.Width.GetValue()
        num_frames.append(0)

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
            thread = threading.Thread(target=wait_for_trigger_and_start, args=(camera, i, num_frames, stop_event, save_dir_cameras[i], start_event))
            threads.append(thread)
            thread.start()

        # CTRL-C (KeyboardInterrupt) をキャッチ
        while any(thread.is_alive() for thread in threads):
            time.sleep(0.0001)

    except KeyboardInterrupt:
        print("\nCTRL-C detected. Exiting gracefully...")
        stop_event.set()

    finally:
        print("fainally")
        for thread in threads:
            print(f"threds = {thread}")
            thread.join()

        end_time = time.time()
        elapsed_time = end_time - start_time
        cv2.destroyAllWindows()

        for i in range(num_cameras):
            frame_rate = num_frames[i] / elapsed_time
            print(f"Number of frames acquired (Camera {i+1}): {num_frames[i]}")
            print(f"Frame rate (Camera {i+1}) is: {frame_rate:.2f} FPS")

if __name__ == "__main__":
    main()
