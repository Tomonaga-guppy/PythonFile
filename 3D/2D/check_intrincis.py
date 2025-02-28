import pickle
from pathlib import Path
import cv2
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np

# 複数画像から内部パラメータを求める
root_dir = Path(r"G:\gait_pattern")

target_facility = "ota"  #解析対象施設：tkrzk_9g か ota

int_cali_dir = root_dir / "int_cali" / target_facility  #内部キャリブレーション結果を保存するフォルダ
pickle_path = int_cali_dir/f"Intrinsic_sg.pickle"

target = pickle_path.stem
with open(pickle_path, "rb") as f:
    CameraParams = pickle.load(f)
print(f"pickle_path: {pickle_path}")
print(f"焦点距離(fx, fy): {CameraParams['intrinsicMat'][0,0],CameraParams['intrinsicMat'][1,1]}")
print(f"光学中心(cx, cy): {CameraParams['intrinsicMat'][0,2],CameraParams['intrinsicMat'][1,2]}")
print(f"歪み係数: {CameraParams['distortion']}")

img_paths = list((int_cali_dir/f"Checkerboards_Used_sg").glob("*.jpg"))
save_folder = int_cali_dir/(img_paths[0].parent.stem+"_Undistorted")
# save_folder = int_cali_dir/f"Undistorted_{target_camrra}"
if not save_folder.exists():
    save_folder.mkdir()

yogami = [[-0.02989354 -0.15617849 -0.00240837 -0.0015473  -0.27083304]]
for img_path in tqdm(img_paths):
    img = cv2.imread(str(img_path))
    # undistort_img = cv2.undistort(img, CameraParams['intrinsicMat'], np.array([[-0.22989354, -0.15617849, -0.00240837, -0.0015473, -0.27083304]]) )
    undistort_img = cv2.undistort(img, CameraParams['intrinsicMat'], CameraParams['distortion'])
    cv2.imwrite((save_folder/(img_path.stem+"_ud.jpg")).as_posix(), undistort_img)
