import pickle
from pathlib import Path
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt

root_dir = Path(r"G:\gait_pattern\int_cali\ota")
pickle_path = root_dir / "Intrinsic_sg.pickle"
Cheker_imgs_folder_path = root_dir / "Checkerboards_Origin_sg"

with open(pickle_path, "rb") as f:
    CameraParams = pickle.load(f)
print(f"歪み係数[k1 k2 p1 p2 k3]: {CameraParams['distortion']}")

#追加の歪み係数
k1_add = -0.4
k2_add = 0.2
p1_add = 0.0
p2_add = 0.0
k3_add = 0.45

add_dist_coef = np.array([k1_add, k2_add, p1_add, p2_add, k3_add])
distParams_af = CameraParams['distortion'] + add_dist_coef
print(f"調整後の歪み係数[k1 k2 p1 p2 k3]: {distParams_af}")

custom_folder_path = root_dir / "CustomDistortion"
if not custom_folder_path.exists():
    custom_folder_path.mkdir()

cheker_ori_img_path = list(Cheker_imgs_folder_path.glob("*.jpg"))[0]
ori_img_path = custom_folder_path / "Ori.jpg"
if not ori_img_path.exists():
    ori_img_path = shutil.copy(str(cheker_ori_img_path), str(ori_img_path))

undistrot_def_img_path = custom_folder_path / "Undist_def.jpg"
ori_img = cv2.imread(str(ori_img_path))
undist_img = cv2.undistort(ori_img, CameraParams['intrinsicMat'], CameraParams['distortion'])
undist_img_af = cv2.undistort(ori_img, CameraParams['intrinsicMat'], distParams_af)

cv2.imwrite(str(undistrot_def_img_path), undist_img)
undist_af_img_path = custom_folder_path / f"Undist_{k1_add}_{k2_add}_{p1_add}_{p2_add}_{k3_add}.jpg"
cv2.imwrite(str(undist_af_img_path), undist_img_af)

Custom_CamParams = {
    "distortion": distParams_af,
    "intrinsicMat": CameraParams['intrinsicMat'],
    "imageSize": CameraParams['imageSize'],
    "rms": CameraParams['rms'],
    "add_dist_coef": add_dist_coef
}

custom_pickle_path = root_dir / "Intrinsic_sg_custom.pickle"
with open(custom_pickle_path, "wb") as f:
    pickle.dump(Custom_CamParams, f)

# print(f"CameraParams: {CameraParams}")
# print(f"Custom_CamParams: {Custom_CamParams}")

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB))
# plt.title("Original")
# plt.axis("off")
# plt.subplot(1, 3, 2)
# plt.imshow(cv2.cvtColor(undist_img, cv2.COLOR_BGR2RGB))
# plt.title("Undistorted")
# plt.axis("off")
# plt.subplot(1, 3, 3)
# plt.imshow(cv2.cvtColor(undist_img_af, cv2.COLOR_BGR2RGB))
# plt.title(f"Undistorted{k1_add}_{k2_add}_{p1_add}_{p2_add}_{k3_add}")
# plt.axis("off")
# plt.tight_layout()
# plt.show()


