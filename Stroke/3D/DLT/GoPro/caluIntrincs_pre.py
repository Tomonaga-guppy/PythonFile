import os
import cv2
import numpy as np
import glob
from multiprocessing import Pool

root_dir = r"G:\gait_pattern\20241112\gopro"

def process_calibration_images(args):
    cali_intrinsic_dir, checker_pattern = args
    imageFiles = glob.glob(os.path.join(os.path.dirname(cali_intrinsic_dir), "Wide", "*.png"))
    # imageFiles = glob.glob(os.path.join(os.path.dirname(cali_intrinsic_dir), "Intrinsic_ori", "*.png"))
    # print(f"imageFiles: {imageFiles}")

    for i, pathName in enumerate(imageFiles):
        # if i % 10 != 0:  #量が多いので10枚に1枚だけ処理
        #     continue
        iImage = os.path.basename(pathName).split(".")[0]
        image = cv2.imread(pathName)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners, meta = cv2.findChessboardCornersSBWithMeta(
            grayColor, checker_pattern,
            cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_LARGER
        )

        if ret:
            image_check = cv2.drawChessboardCorners(image, checker_pattern, corners, ret)
            print(f"{i+1}/{len(imageFiles)} {pathName} Found checkerboard")
            cv2.imwrite(os.path.join(cali_intrinsic_dir, str(iImage) + '.jpg'), image_check)
        else:
            print(f"{i+1}/{len(imageFiles)} {pathName} Not found checkerboard")

# マルチプロセスで処理を実行
if __name__ == "__main__":
    # キャリブレーション画像の処理
    pre_dirs = glob.glob(os.path.join(root_dir, "*checker_test*"))
    pre_dirs = [pre_dir for pre_dir in pre_dirs if os.path.isdir(pre_dir)]

    cali_intrinsic_dirs = []
    for pre_dir in pre_dirs:
        cali_intrinsic_dir = os.path.join(pre_dir, "Intrinsic_check")
        cali_intrinsic_dirs.append(cali_intrinsic_dir)
        if not os.path.exists(cali_intrinsic_dir):
            os.makedirs(cali_intrinsic_dir)
    print(cali_intrinsic_dirs)

    checker_pattern = (5, 4)
    with Pool() as pool:
        pool.map(process_calibration_images, [(cali_intrinsic_dir, checker_pattern) for cali_intrinsic_dir in cali_intrinsic_dirs])
