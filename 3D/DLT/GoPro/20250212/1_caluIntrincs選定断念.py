import cv2
import numpy as np
import glob
import os
import re
import pickle
from pathlib import Path
import pandas as pd

# 複数画像から内部パラメータを求める

# root_dir = r"G:\gait_pattern"
# int_cali_dir = os.path.join(root_dir, "int_cali", "ota")  #内部キャリブレーション結果を保存するフォルダ
# mov_dir = os.path.join(root_dir, "20241114_ota_test", "gopro", "fl")  #キャリブレーション動画や歩行動画が入っているフォルダ

root_dir = Path(r"G:\gait_pattern")
int_cali_dir = root_dir / "int_cali" / "tkrzk_9g"  #内部キャリブレーション結果を保存するフォルダ
mov_dir = root_dir / "20241126_br9g" / "gopro" / "fl"  #キャリブレーション動画や歩行動画が入っているフォルダ

def generate3Dgrid(checker_pattern, squareSize):
    #  3D points real world coordinates. Assuming z=0
    objectp3d = np.zeros((1, checker_pattern[0]
                          * checker_pattern[1],
                          3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:checker_pattern[0],
                                    0:checker_pattern[1]].T.reshape(-1, 2)

    objectp3d = objectp3d * squareSize

    return objectp3d

def saveCameraParameters(filename,CameraParams):
    open_file = open(filename, "wb")
    pickle.dump(CameraParams, open_file)
    open_file.close()

    return True

def main():
    cali_mov = mov_dir / "int_cali.MP4"
    cap = cv2.VideoCapture(str(cali_mov))
    all_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #総フレーム数
    block_num = 5 #何ブロックに分けるか(-2m~2mまでの範囲を全体的に使用するために分割)
    all_frame_array = np.array_split(np.arange(all_frame_count), block_num) #フレーム数をブロック数に分割
    print(f"frame_list: {all_frame_array}")

    score_df = pd.DataFrame(columns=["block", "frame", "score"])

        # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Vector for 3D points
    threedpoints = []

    # Vector for 2D points
    twodpoints = []

    checker_pattern = (5, 4)
    squareSize = 35  # mm

    imageScaleFactor = 1  # ここを変更すると小さく映っているチェッカーパターンの検出率が向上するみたい？

    for block_num, frame_count_array in enumerate(all_frame_array): #各ブロックごとに評価
        print(f"{block_num+1}/{len(all_frame_array)}: {frame_count_array}")
        indices = np.linspace(0, len(frame_count_array)-1, 40, dtype=int)
        block_frame_array = frame_count_array[indices]  #各ブロックから40枚を抽出

        for image_num, frame_num in enumerate(block_frame_array): #各ブロックで40枚評価
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"frame_num: {frame_num} is not read")
                continue
            image = cv2.imread(frame)
            imageSize = np.reshape(np.asarray(np.shape(image)[0:2]).astype(np.float64),(2,1)) # This all to be able to copy camera param dictionary

            grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(f"{block_num} : {image_num} : {frame_num}  used for intrinsics calibration.")

            ret,corners,meta = cv2.findChessboardCornersSBWithMeta(	grayColor, checker_pattern,
                                                            cv2.CALIB_CB_EXHAUSTIVE +
                                                            cv2.CALIB_CB_ACCURACY +
                                                            cv2.CALIB_CB_LARGER)

            # If desired number of corners can be detected then,
            # refine the pixel coordinates and display
            # them on the images of checker board
            if ret == True:
                # # 3D points real world coordinates
                # checker_pattern = meta.shape[::-1] # reverses order so width is first
                # # print(f"    Checkerboard pattern: {checker_pattern}")
                # objectp3d = generate3Dgrid(checker_pattern, squareSize)

                # threedpoints.append(objectp3d)

                # corners2 = corners/imageScaleFactor # Don't need subpixel refinement with findChessboardCornersSBWithMeta（戻り値がサブピクセル精度）
                # twodpoints.append(corners2)

                #評価用に3次元座標を取得
                checker_pattern = meta.shape[::-1] # reverses order so width is first
                objectp3d = generate3Dgrid(checker_pattern, squareSize)
                corners2 = corners/imageScaleFactor # Don't need subpixel refinement with findChessboardCornersSBWithMeta（戻り値がサブピクセル精度）





            if ret == False:
                print("Couldn't find checkerboard in " + frame_num)

            pass


    # 評価結果から各ブロック上位8枚（合計40枚）を取得
    # Draw and display the corners
    image = cv2.drawChessboardCorners(image,
                                        meta.shape[::-1],
                                        corners2, ret)

    #findAspectRatio
    ar = imageSize[1]/imageSize[0]
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.resize(image,(int(600*ar),600))

    # Save intrinsic images
    imageSaveDir = os.path.join(os.path.dirname(int_cali_dir),f'{folder_name}_Checkerboards')
    if not os.path.exists(imageSaveDir):
        os.mkdir(imageSaveDir)
    cv2.imwrite(os.path.join(imageSaveDir, str(frame_num) + '.jpg'), image)




if __name__ == '__main__':
    main()