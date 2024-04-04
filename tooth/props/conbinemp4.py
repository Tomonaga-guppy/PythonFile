
import moviepy.editor as mp
import glob
import os

root_dir = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\scale"

# 動画ファイルのパス
video1_path = glob.glob(root_dir + "\*B2\*original.mp4")[0]
video2_path = glob.glob(root_dir + "\*B2\*OpenFace.avi")[0]

# 動画の読み込み
video1 = mp.VideoFileClip(video1_path)
video2 = mp.VideoFileClip(video2_path)

# 動画の同時再生
combined_video = mp.clips_array([[video1, video2]])

# 同時再生した動画の保存
save_path = os.path.join(os.path.dirname(video1_path),"combined_video.mp4")
combined_video.write_videofile(save_path)
