from pathlib import Path
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle

# 画像から1ピクセルあたりの長さを求める
root_dir = Path(r"G:\gait_pattern\20250228_ota\data\20250221\sub0")
# mov_path = Path(r"G:\gait_pattern\20241114_ota_test\gopro\sagi\sub0_asgait_2_udCropped.MP4")
target = "ota"
# target = "ota_20250228"
mov_path = root_dir / "thera0-3" / "sagi" /  f"UD_{target}.MP4"

cap = cv2.VideoCapture(str(mov_path))
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame = cap.read()
if not ret:
    print(f"開始のフレームが取得できませんでした")
    sys.exit()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame)
# plt.show()

"""
図から目視で-1,0,1 m地点のテープの中心座標を記録（m1, m0, p1）
"""
# m1 = np.array([2499.30, 1647.14])
# m0 = np.array([1885.82, 1647.17])
# p1 = np.array([1271.43, 1620.50])

if target == "ota":
    m1 = np.array([2512.5, 1638.3])
    m0 = np.array([1897.5, 1640.8])
    p1 = np.array([1280.6, 1614.2])
elif target == "ota_20250228":
    m1 = np.array([2510.1, 1633.9])
    m0 = np.array([1897.6, 1637.3])
    p1 = np.array([1283.0, 1609.9])

m1m0 = np.linalg.norm(m1 - m0)
p1m0 = np.linalg.norm(p1 - m0)
mmperpix = 2000 / (m1m0 + p1m0)
print(f"mmperpix: {mmperpix}")

mesure_param_dict = {"m1": m1, "m0": m0, "p1": p1, "mmperpix": mmperpix}
pickle_path = root_dir / f"sagi_mesure_params_{target}.pickle"
print(f"pickle_path: {pickle_path}")
with open(pickle_path, "wb") as f:
    pickle.dump(mesure_param_dict, f)


