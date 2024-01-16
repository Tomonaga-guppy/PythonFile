import cv2
import matplotlib.pyplot as plt
import os
import glob

root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_12_20"
dir_path = glob.glob(os.path.join(root_dir,"*b"))[0]
print(f"dir_path = {dir_path}")

def save_frames(video_path, output_dir):
    # 動画を読み込む
    video = cv2.VideoCapture(video_path)

    # フレーム数を取得する
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # フレームごとに保存する
    for i in range(1,frame_count):
            # フレームを読み込む
            ret, frame = video.read()

            # フレームが正常に読み込まれた場合は保存する
            if ret:
                print(i)
                # if i == 0 or i == 410:  # 1フレーム目と410フレーム目を保存
                # 画像を保存する範囲を指定
                # frame = frame[600:1080,500:980]  # 1920*1080 →480*480に切り取り  fの場合
                frame = frame[700:1180,600:1080]  # 1920*1080 →480*480に切り取り  bの場合
                frame_path = f"{output_dir}/"+str(i).zfill(4)+ ".jpg"
                cv2.imwrite(frame_path, frame)

    # 動画を解放する
    video.release()
# 動画のパスと保存先ディレクトリを指定する
video_path = dir_path + "/b.mp4"
# video_path = dir_path + "/f.mp4"
output_dir = dir_path + "/yoko"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    # フレームを保存する
save_frames(video_path, output_dir)


image1_path = output_dir + "/0001.jpg"  #b
image2_path = output_dir + "/0405.jpg"  #
# image1_path = root_dir + "/yoko/0001.jpg"  #f
# image2_path = root_dir + "/yoko/0542.jpg"
# 画像の読み込み
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# 画像の結合
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
ax1.set_xlabel('Z-axis', fontsize=15)
ax1.set_ylabel('Y-axis', fontsize=15)
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
ax2.set_xlabel('Z-axis', fontsize=15)
ax2.set_ylabel('Y-axis', fontsize=15)
plt.savefig(root_dir + "/yoko_pix.png")
plt.show()
