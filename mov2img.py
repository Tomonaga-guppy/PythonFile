import cv2
import os

def save_frames(video_path, output_folder):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not video.isOpened():
        print("Error opening video file")
        return

    # Read and save each frame
    frame_count = 0
    while True:
        # Read the next frame
        ret, frame = video.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        # Save the frame as an image
        output_path = f"{output_folder}/frame_{frame_count}.jpg"
        cv2.imwrite(output_path, frame)

        # Increment the frame count
        frame_count += 1

    # Release the video file
    video.release()

# Example usage
video_path = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\scale\20231218_d\OpenFace.avi"
output_folder = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\scale\20231218_d\OpenFace"

if os.path.exists(output_folder) == False:
    os.mkdir(output_folder)

save_frames(video_path, output_folder)
