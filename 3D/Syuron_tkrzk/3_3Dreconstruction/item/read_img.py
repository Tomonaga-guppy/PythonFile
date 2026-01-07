"""
imageをmatplotlibで表示する（キーポイント検出領域によるフィルタ）
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt

img_path = Path(r"G:\gait_pattern\BR9G_shuron\sub5\thera5-0\gopro\fr\undistorted_facemasked\frame_00000.png")
img = cv2.imread(str(img_path))
height, width = img.shape[:2]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
