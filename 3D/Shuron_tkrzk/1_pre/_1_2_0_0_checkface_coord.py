from pathlib import Path
import cv2
import matplotlib.pyplot as plt

imgdir = Path(r"G:\gait_pattern\BR9G_shuron\sub1\thera0-0_0\gopro\fr\undistorted")
imgfiles = sorted(imgdir.glob("*.png"))

for imgfile in imgfiles:
    img = cv2.imread(str(imgfile))
    height, width = img.shape[:2]
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()