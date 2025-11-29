from pathlib import Path
import cv2
import matplotlib.pyplot as plt

imgdir = Path("g:/gait_pattern/20250811_br/sub1/thera1-0/fl_yolo/undistorted_ori")
imgfiles = sorted(imgdir.glob("*.png"))

for imgfile in imgfiles:
    img = cv2.imread(str(imgfile))
    height, width = img.shape[:2]
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()