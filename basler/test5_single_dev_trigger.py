from pypylon import pylon
import cv2
import time
import os
import datetime
import threading

# Set the save directory based on user input
name = input("保存するファイル名を入力してください: ")
current_date = datetime.datetime.now().strftime('%Y%m%d')
root_dir = fr"f:\Tomson\gait_pattern\{current_date}\{name}"

if not os.path.exists(root_dir):
    os.makedirs(root_dir)

# Main function to handle camera capture
def process_camera(camera, num_frames, stop_event):
    # Set up image format conversion
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_RGB8packed
    converter.OutputBitAlignment = "MsbAligned"

    # Start grabbing images at 60Hz
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByUser)

    # Create a window to display the camera stream
    cv2.namedWindow('Camera Stream', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Stream', 640, 480)

    while camera.IsGrabbing() and not stop_event.is_set():
        try:
            # Retrieve a single image
            grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                # Convert the image to RGB format
                img = converter.Convert(grabResult)
                img_array = img.GetArray()

                # Display the image
                cv2.imshow('Camera Stream', img_array)

                # Save the image
                filename = os.path.join(root_dir, f"saved_img_{num_frames}.jpeg")
                print(f"Saving image to {filename}")
                cv2.imwrite(filename, img_array)
                num_frames += 1

            grabResult.Release()

            # Handle 'q' to quit the process
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Keyboard interrupt detected. Exiting...")
                stop_event.set()
                break

        except Exception as e:
            print(f"Error while capturing: {e}")
            stop_event.set()
            break

    # Clean up
    camera.StopGrabbing()
    cv2.destroyAllWindows()

# Function to wait for the external trigger signal
def wait_for_trigger_and_start(camera, stop_event):
    print("Waiting for trigger signal...")

    # Configure the camera for external triggering
    camera.TriggerSelector.Value = "FrameStart"
    camera.TriggerMode.Value = "On"
    camera.TriggerSource.Value = "Line1"  # External trigger from Line1
    camera.TriggerActivation.Value = "FallingEdge"  # Use FallingEdge for trigger activation

    # Wait for the trigger signal
    while not stop_event.is_set():
        print(f"Waiting for trigger signal")
        if camera.WaitForFrameTriggerReady(1, pylon.TimeoutHandling_Return):
            print("Trigger signal received. Starting capture...")
            process_camera(camera, 0, stop_event)
            break

# Main program entry point
def main():
    stop_event = threading.Event()

    # Set up the camera
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()

    if len(devices) < 1:
        raise pylon.RUNTIME_EXCEPTION("No camera detected.")

    # Attach and open the camera
    camera = pylon.InstantCamera(tlFactory.CreateDevice(devices[0]))
    camera.Open()

    if camera.GetDeviceInfo().GetSerialNumber() == "40121811":
        print(f"Camera {camera.GetDeviceInfo().GetModelName()} detected. Ready to capture.")

        try:
            wait_for_trigger_and_start(camera, stop_event)
        except KeyboardInterrupt:
            print("\nCTRL-C detected. Exiting gracefully...")
            stop_event.set()
        finally:
            camera.Close()

if __name__ == "__main__":
    main()
