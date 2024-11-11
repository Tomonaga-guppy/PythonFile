import cv2
import numpy as np
import glob
import os
import re
import pickle

root_dir = r"G:\gait_pattern\20241106"
cali_intrinsic_dirs = glob.glob(os.path.join(root_dir, "*cali_ch*1080*"))
cali_intrinsic_dirs = [cali_intrinsic_dir for cali_intrinsic_dir in cali_intrinsic_dirs if os.path.isdir(cali_intrinsic_dir)]
visualize = False

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
    load_num = 50

    # Load images in for calibration
    for cali_intrinsic_dir in cali_intrinsic_dirs:
        imageFiles = glob.glob(os.path.join(cali_intrinsic_dir, "original", "*.png"))

        # 自然順で並べ替える関数
        def natural_sort_key(file_path):
            # 数字でソートできるように、ファイル名から数字を抽出する
            base_name = os.path.basename(file_path)
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', base_name)]

        imageFiles = sorted(imageFiles, key=natural_sort_key)

        if len(imageFiles) > load_num:
            indeces = np.linspace(0, len(imageFiles)-1, load_num, dtype=int)
            imageFiles = [imageFiles[i] for i in indeces]
        else:
            imageFiles = imageFiles

        # print(f"imageFiles: {imageFiles}")
        for i, pathName in enumerate(imageFiles):

            iImage = os.path.basename(pathName).split(".")[0]

            image = cv2.imread(pathName)
            # if imageScaleFactor != 1:
            #     dim = (int(imageScaleFactor*image.shape[1]),int(imageScaleFactor*image.shape[0]))
            #     image = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
            imageSize = np.reshape(np.asarray(np.shape(image)[0:2]).astype(np.float64),(2,1)) # This all to be able to copy camera param dictionary

            grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(f"{iImage} : {pathName}  used for intrinsics calibration.")

            ret,corners,meta = cv2.findChessboardCornersSBWithMeta(	grayColor, checker_pattern,
                                                            cv2.CALIB_CB_EXHAUSTIVE +
                                                            cv2.CALIB_CB_ACCURACY +
                                                            cv2.CALIB_CB_LARGER)

            # If desired number of corners can be detected then,
            # refine the pixel coordinates and display
            # them on the images of checker board
            if ret == True:

                image_check = cv2.drawChessboardCorners(image, checker_pattern, corners, ret)
                image_check = cv2.resize(image_check,(int(600*imageSize[1]/imageSize[0]),600))
                cv2.imshow('if detection id failed click "s" key ', image_check)
                key = cv2.waitKey(0)

                if key == ord("s"):
                    print("検出に失敗と判断したためスキップします")
                    continue

                # 3D points real world coordinates
                checker_pattern = meta.shape[::-1] # reverses order so width is first
                print(f"    Checkerboard pattern: {checker_pattern}")
                objectp3d = generate3Dgrid(checker_pattern, squareSize)

                threedpoints.append(objectp3d)

                # Refining pixel coordinates
                # for given 2d points.
                # corners2 = cv2.cornerSubPix(
                #     grayColor, corners, (11, 11), (-1, -1), criteria)

                corners2 = corners/imageScaleFactor # Don't need subpixel refinement with findChessboardCornersSBWithMeta
                twodpoints.append(corners2)

                # Draw and display the corners
                image = cv2.drawChessboardCorners(image,
                                                    meta.shape[::-1],
                                                    corners2, ret)

                #findAspectRatio
                ar = imageSize[1]/imageSize[0]
                # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                cv2.resize(image,(int(600*ar),600))

                # Save intrinsic images
                imageSaveDir = os.path.join(cali_intrinsic_dir,'IntrinsicCheckerboards')
                if not os.path.exists(imageSaveDir):
                    os.mkdir(imageSaveDir)
                cv2.imwrite(os.path.join(imageSaveDir,'intrinsicCheckerboard' + str(iImage) + '.jpg'), image)

                if visualize:
                    print('Press enter or close image to continue')
                    cv2.imshow('img', image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            if ret == False:
                print("Couldn't find checkerboard in " + pathName)

        # Perform camera calibration by
        # passing the value of above found out 3D points (threedpoints)
        # and its corresponding pixel coordinates of the
        # detected corners (twodpoints)

        print(f"\nCalculating camera parameters for {cali_intrinsic_dir}")

        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
            threedpoints, twodpoints, grayColor.shape[::-1], None, None)

        CamParams = {'distortion':distortion,'intrinsicMat':matrix,'imageSize':imageSize}

        saveFileName = os.path.join(root_dir, f'Intrinsics_{cali_intrinsic_dir.split("ch_")[-1]}.pickle')
        saveCameraParameters(saveFileName,CamParams)

        print(f"Camera parameters saved to {saveFileName} !\n")


if __name__ == '__main__':
    main()