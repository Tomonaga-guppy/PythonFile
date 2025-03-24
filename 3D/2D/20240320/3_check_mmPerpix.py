from pathlib import Path
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle

# 画像から1ピクセルあたりの長さを求める
root_dir = Path(r"G:\gait_pattern\20250228_ota\data\20250221\sub0")
# mov_path = Path(r"G:\gait_pattern\20241114_ota_test\gopro\sagi\sub0_asgait_2_udCropped.MP4")
# target = "ota"
# target = "ota_20250228"
target = "ota_20250228_custom"
img_dir_path = root_dir / "thera0-3" / "sagi" / "Undistort"
img_path = list(img_dir_path.glob("*.png"))[0]

img = cv2.imread(str(img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# frame = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()

"""
図から目視で-1,0,1 m地点のテープの中心座標を記録（m1, m0, p1）
"""
# m1 = np.array([2499.30, 1647.14])
# m0 = np.array([1885.82, 1647.17])
# p1 = np.array([1271.43, 1620.50])

if target == "ota":
    m = np.array([2512.5, 1638.3])
    o = np.array([1897.5, 1640.8])
    p = np.array([1280.6, 1614.2])
elif target == "ota_20250228":
    m = np.array([2510.1, 1633.9])
    o = np.array([1897.6, 1637.3])
    p = np.array([1283.0, 1609.9])
elif target == "ota_20250228_custom": # -2,0,2mのテープの中心座標
    m = np.array([3151.5, 1654.3])
    o = np.array([1897.2, 1647.4])
    p = np.array([626.7, 1612.1])
mo = np.linalg.norm(o - m)
op = np.linalg.norm(p - o)
if target == "ota_20250228_custom":
    mmperpix = 4000 / (mo + op)
else:
    mmperpix = 2000 / (mo + op)
mesure_param_dict = {"m": m, "o": o, "p": p, "mmperpix": mmperpix}
print(f"mmperpix: {mmperpix}")
pickle_path = root_dir / f"sagi_mesure_params_{target}.pickle"
print(f"pickle_path: {pickle_path}")
with open(pickle_path, "wb") as f:
    pickle.dump(mesure_param_dict, f)


