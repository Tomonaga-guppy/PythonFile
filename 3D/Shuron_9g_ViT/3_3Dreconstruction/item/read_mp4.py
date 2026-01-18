"""
動画の指定フレームをmatplotlibで表示する
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# 動画ファイルのパスとフレーム番号を指定
video_path = Path(r"G:\gait_pattern\2025_shuron_BR9G\sub6\thera6-0\gopro\sagi\ViTPose\ViTPose_overlay_HD_raw.mp4")
frame_number = 4  # 読み込みたいフレーム番号(0から始まる)

# 動画を開く
cap = cv2.VideoCapture(str(video_path))

# 指定フレームに移動
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# フレームを読み込む
ret, img = cap.read()

if ret:
    height, width = img.shape[:2]
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Frame {frame_number}")
    plt.axis("off")
    plt.show()
else:
    print(f"フレーム {frame_number} の読み込みに失敗しました")

# 動画ファイルを閉じる
cap.release()