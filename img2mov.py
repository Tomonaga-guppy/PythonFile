import cv2
import os
import glob

root_dir = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\scale"

def images_to_video(image_folder, video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
    frame_count = 1
    for image in images:
        # #frame数を動画に表示
        frame = cv2.imread(os.path.join(image_folder, image))
        cv2.putText(frame, str(frame_count), (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
        video.write(frame)
        frame_count += 1

    cv2.destroyAllWindows()
    video.release()

# 画像フォルダと出力する動画の名前を指定して呼び出す
input_dir = glob.glob(os.path.join(root_dir,"*d/OpenFace"))[0]
dir_name = os.path.dirname(input_dir)
save_path = os.path.join(dir_name,"OpenFaceFrame.mp4")
images_to_video(input_dir, save_path)
