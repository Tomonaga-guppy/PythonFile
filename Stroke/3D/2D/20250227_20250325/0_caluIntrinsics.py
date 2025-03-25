import cv2
import numpy as np
import pickle
from pathlib import Path

# 複数画像から内部パラメータを求める
root_dir = Path(r"G:\gait_pattern")

target_facility = "ota"
mov = "20241114_ota_test"

int_cali_dir = root_dir / "int_cali" / target_facility  #内部キャリブレーション結果を保存するフォルダ
mov_dir = root_dir / mov / "gopro" / "sagi"  #キャリブレーション動画や歩行動画が入っているフォルダ

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
    print(f"Total frame count: {all_frame_count}")

    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Vector for 3D points
    threedpoints = []

    # Vector for 2D points
    twodpoints = []

    checker_pattern = (4, 5)
    squareSize = 35  # mm

    imageScaleFactor = 1  # ここを変更すると小さく映っているチェッカーパターンの検出率が向上するみたい？

    check_frame_range = [[0, 1172], [2953, 4215], [5894, 7245]]  #カメラにおいておおよそ-2m, 0m, 2mの位置にいる画像のフレーム範囲
    check_frame_list = [np.linspace(check_frame_range[0], check_frame_range[1], 20, dtype=int) for check_frame_range in check_frame_range]
    # print(f"check_frame_list: {check_frame_list}")

    check_num = 40  #検出を行う画像数
    check_frame_nums = np.linspace(0, all_frame_count, check_num, dtype=int)  #検出を行う画像のフレーム番号
    used_frame_nums = 0  #パラメータ算出に使用した画像の枚数記録用

    for i, frame_num in enumerate(check_frame_nums): #各フレーム数ごとに検出
        print(f"{i+1}/{len(check_frame_nums)} {frame_num}frame  used for intrinsics calibration.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"    frame_num: {frame_num} is not read")
            continue
        # image = cv2.imread(frame)
        image = frame
        imageSaveDir0 = int_cali_dir / f"Checkerboards_Origin_sg"
        imageSaveDir0.mkdir(exist_ok=True)  #ない場合は作成
        cv2.imwrite(str(imageSaveDir0 / f"{frame_num}.jpg"), image)  #元画像を保存

        imageSize = np.reshape(np.asarray(np.shape(image)[0:2]).astype(np.float64),(2,1)) # This all to be able to copy camera param dictionary
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret,corners,meta = cv2.findChessboardCornersSBWithMeta(	grayColor, checker_pattern,
                                                        cv2.CALIB_CB_EXHAUSTIVE +
                                                        cv2.CALIB_CB_ACCURACY +
                                                        cv2.CALIB_CB_LARGER)

        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        if ret == True:  #チェッカーパターンが検出された場合
            # 3次元座標と2次元座標を保存
            # 3D points real world coordinates
            checker_pattern = meta.shape[::-1] # reverses order so width is first
            objectp3d = generate3Dgrid(checker_pattern, squareSize)
            threedpoints.append(objectp3d)
            corners2 = corners/imageScaleFactor # Don't need subpixel refinement with findChessboardCornersSBWithMeta（戻り値がサブピクセル精度）
            twodpoints.append(corners2)

            # 検出したパターンを描画
            image = cv2.drawChessboardCorners(image,
                                                meta.shape[::-1],
                                                corners2, ret)

            # 検出した画像を保存
            imageSaveDir = int_cali_dir / f"Checkerboards_Used_sg"
            imageSaveDir.mkdir(exist_ok=True)  #ない場合は作成
            cv2.imwrite(str(imageSaveDir / f"{frame_num}.jpg"), image)

            used_frame_nums += 1

        if ret == False:
            print("    Couldn't find checkerboard in " + str(frame_num))

    cap.release()
    print(f"\nCalculating camera parameters using {used_frame_nums} images")

    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, grayColor.shape[::-1], None, None)

    CamParams = {'distortion':distortion,'intrinsicMat':matrix,'imageSize':imageSize, 'rms':ret}
    print(f"movie: {mov_dir}")
    print(f"再投影誤差のRMS[ピクセル]: {ret}")
    print(f"焦点距離(fx, fy): {matrix[0,0],matrix[1,1]}")
    print(f"光学中心(cx, cy): {matrix[0,2],matrix[1,2]}")
    print(f"歪み係数: {distortion}")

    saveFileName = str(int_cali_dir / f"Intrinsic_sg.pickle")
    saveCameraParameters(saveFileName,CamParams)
    # print(f"Camera parameters saved to {saveFileName} !")

    # # 3D座標と2D座標を確認
    # threedpoints_array = np.array(threedpoints)
    # twodpoints_array = np.array(twodpoints)
    # threedpoints_array = np.squeeze(threedpoints_array, axis=1)
    # twodpoints_array = np.squeeze(twodpoints_array, axis=2)
    # print(f"3d_points: {threedpoints_array}\n2d_points: {twodpoints_array}")
    # print(f"3d_points_shape: {threedpoints_array.shape}\n2d_points_shape: {twodpoints_array.shape}")

if __name__ == '__main__':
    main()