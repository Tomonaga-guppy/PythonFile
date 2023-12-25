import cv2
import os
import glob

# root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_12_demo"
root_dir = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\2023_12_20"

target = "*/"
# flag = "OpenFace"
flag = "SealDetection"

def save_frames(video_path, output_folder):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error opening video file")
        return
    frame_count = 1
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # print(frame_count)
        output_path = f"{output_folder}/{str(frame_count).zfill(4)}.jpg"
        cv2.putText(frame, str(frame_count), (100, 100), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(output_path, frame)
        frame_count += 1
    video.release()

if flag == "OpenFace":  video_paths = glob.glob(os.path.join(root_dir,(target + "OpenFace.avi")))
elif flag == "SealDetection":  video_paths = glob.glob(os.path.join(root_dir, (target + "SealDetection.mp4")))

for i,video_path in enumerate(video_paths):
    print(f"{i+1}/{len(video_paths)}")
    dir_name = os.path.dirname(video_path)
    if flag == "OpenFace":  output_folder = os.path.join(dir_name, "OpenFace")
    elif flag == "SealDetection":  output_folder = os.path.join(dir_name, "SealDetection")

    if os.path.exists(output_folder) == False:
        os.mkdir(output_folder)

    save_frames(video_path, output_folder)
