# 一部隠れたArucoマーカーを補完するプログラム

import cv2
import glob
import os
import numpy as np

root_dir = r"F:\Tomson\gait_pattern\20240912"
hidden_aruco_video =  glob.glob(os.path.join(root_dir, '*hidden*.mp4'))[0]




