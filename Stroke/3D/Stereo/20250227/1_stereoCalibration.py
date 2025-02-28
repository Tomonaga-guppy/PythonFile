import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle


# ステレオキャリブレーションのオプション（設定なしだとステレオキャリブレーション時に内部パラメータ及び歪み係数を求める）
#CALIB_FIX_INTRINSIC:内部パラメータを固定して外部パラメータを求める
#CALIB_USE_INTRINSIC_GUESS:内部パラメータを初期値として外部パラメータを求める
stereo_cali_option = [0, "cv2.CALIB_FIX_INTRINSIC", "cv2.CALIB_USE_INTRINSIC_GUESS"]
flags_num = 0  #0:内部パラメータも求める, 1,2:内部パラメータを読み込む

adjust_frame_num = 4  #右カメラのフレーム番号を調整するための値

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

def evaluate_epipolar_error(img_points1, img_points2, M1, D1, M2, D2, F):
    """
    エピポーラ幾何拘束に基づく誤差を計算する関数 (C++コードに対応)

    Args:
        img_points1: 左カメラの画像点 (リストのリスト)
        img_points2: 右カメラの画像点 (リストのリスト)
        M1: 左カメラの内部パラメータ行列
        D1: 左カメラの歪み係数
        M2: 右カメラの内部パラメータ行列
        D2: 右カメラの歪み係数
        F: 基礎行列

    Returns:
        avg_err: 平均エピポーラ誤差
    """

    nframes = len(img_points1)  # フレーム数 (チェスボードの数)
    N = img_points1[0].shape[0] #1フレームあたりの点の数
    avg_err = 0.0

    for i in range(nframes):
        pt0 = img_points1[i].reshape(-1, 1, 2)  # (N, 1, 2) の形に
        pt1 = img_points2[i].reshape(-1, 1, 2)

        # 歪み補正
        pt0_undistorted = cv2.undistortPoints(pt0, M1, D1, P=M1)
        pt1_undistorted = cv2.undistortPoints(pt1, M2, D2, P=M2)

        # エピポーラ線の計算
        lines0 = cv2.computeCorrespondEpilines(pt0_undistorted, 1, F).reshape(-1, 3) # pt0に対応する右カメラのエピポーラ線
        lines1 = cv2.computeCorrespondEpilines(pt1_undistorted, 2, F).reshape(-1, 3) # pt1に対応する左カメラのエピポーラ線

        # 各点とエピポーラ線の距離を計算し、合計誤差を計算
        for j in range(N):
            # 点と直線の距離公式を用いてエピポーラ誤差を計算
            # （公式においてcv2.computeCorrespondEpilinesは正規化された形で返すため分母は省略）
            # lines1[j] は pt0_undistorted[j] に対応するエピポーラ線
            err1 = abs(pt0_undistorted[j, 0, 0] * lines1[j, 0] +
                       pt0_undistorted[j, 0, 1] * lines1[j, 1] +
                       lines1[j, 2])

            # lines0[j] は pt1_undistorted[j] に対応するエピポーラ線
            err2 = abs(pt1_undistorted[j, 0, 0] * lines0[j, 0] +
                       pt1_undistorted[j, 0, 1] * lines0[j, 1] +
                       lines0[j, 2])
            avg_err += err1 + err2

    avg_err /= (nframes * N)  # 平均誤差に
    return avg_err

def main():
    root_dir = Path(r"G:\gait_pattern")
    target_facility = "ota"  #解析対象施設：tkrzk_9g か ota
    if target_facility == "ota":  mov = "20241114_ota_test"

    int_cali_dir = root_dir / "int_cali" / target_facility  #内部キャリブレーション結果を保存するフォルダ
    target_cameras = ["fl","fr"]
    src_dirs = [Path(fr"G:\gait_pattern\{mov}\gopro\{target_camera}") for target_camera in target_cameras]

    cali_mov_l = src_dirs[0]/"int_cali.MP4"
    cali_mov_r = src_dirs[1]/"int_cali.MP4"
    print(f"cali_mov_l: {cali_mov_l}")
    print(f"cali_mov_r: {cali_mov_r}")
    cap_l = cv2.VideoCapture(str(cali_mov_l))
    cap_r = cv2.VideoCapture(str(cali_mov_r))
    check_frame_range_l = [[1382, 2620], [2782, 4027], [4196, 5558]]  #左カメラにおいておおよそ-1m, 0m, 1mの位置にいる画像のフレーム範囲
    check_frame_list = [np.linspace(check_frame_range[0], check_frame_range[1], 20, dtype=int) for check_frame_range in check_frame_range_l]
    # print(f"check_frame_list: {check_frame_list}")

    checker_pattern = (4, 5)
    squareSize = 35  # mm
    img_points1 = []
    img_points2 = []
    object_points = []

    img_save_dir = root_dir / "stero_cali" / target_facility / "Checkerboard"
    img_save_dir.mkdir(parents=True, exist_ok=True)

    found_num = 0  #検出したチェッカーボードの数
    looped_num = 0

    if flags_num == 0:
        m1 = np.eye(3, 3, dtype=np.float32)
        m2 = np.eye(3, 3, dtype=np.float32)
        d1 = np.zeros((5, 1), np.float32)
        d2 = np.zeros((5, 1), np.float32)
    elif flags_num == 1 or flags_num == 2:
        # カメラの内部パラメータを読み込む場合
        camera_intrinsics = [f'{int_cali_dir}\\Intrinsic_{target_camera}.pickle' for target_camera in target_cameras]
        print(f"camera_intrinsics: {camera_intrinsics}")
        print(f"str(camera_intrinsics[0]):{str(camera_intrinsics[0])}")
        try:
            with open(str(camera_intrinsics[0]), "rb") as f:
                camera_params1 = pickle.load(f)
            with open(str(camera_intrinsics[1]), "rb") as f:
                camera_params2 = pickle.load(f)
        except:
            print("Intrinsic parameters are not found.")
            return
        m1 = camera_params1['intrinsicMat']
        d1 = camera_params1['distortion']
        m2 = camera_params2['intrinsicMat']
        d2 = camera_params2['distortion']

    for iBlock, _ in enumerate(check_frame_range_l):
        for iFrame, frame_num in enumerate(check_frame_list[iBlock]):
            looped_num += 1
            print(f"{iBlock+1}/{len(check_frame_range_l)} {looped_num}/{len(check_frame_list[iBlock])*len(check_frame_range_l)} {frame_num}frame")
            cap_l.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            cap_r.set(cv2.CAP_PROP_POS_FRAMES, frame_num+adjust_frame_num)
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()
            if not ret_l or not ret_r:
                print(f"frame_num: {frame_num} is not found.")
                continue
            frame_gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            frame_gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            found_l , corner_l = cv2.findChessboardCorners(frame_gray_l, (4, 5))
            found_r , corner_r = cv2.findChessboardCorners(frame_gray_r, (4, 5))
            if found_l and found_r:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                # cornerSubPix関数を使ってサブピクセル精度でコーナー位置を求める(corner_l, corner_rが更新されている)
                cv2.cornerSubPix(frame_gray_l, corner_l, (5,5), (-1,-1), term)
                cv2.cornerSubPix(frame_gray_r, corner_r, (5,5), (-1,-1), term)
                cv2.drawChessboardCorners(frame_l, (3,4), corner_l,found_l)
                cv2.drawChessboardCorners(frame_r, (3,4), corner_r,found_r)
                cv2.imwrite(str(img_save_dir / f"{frame_num}_l.jpg"), frame_l)
                cv2.imwrite(str(img_save_dir / f"{frame_num}_r.jpg"), frame_r)
            else:
                print(f"Chessboard not found in {frame_num}frame.")
                continue

            img_points1.append(corner_l.reshape(-1, 2))
            img_points2.append(corner_r.reshape(-1, 2))
            object_points.append(generate3Dgrid(checker_pattern, squareSize))

            found_num += 1

    imgSize = (frame_gray_l.shape[1], frame_l.shape[0])


    print(f"使用した画像の枚数: {found_num}")
    retval, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(object_points, img_points1, img_points2, m1, d1, m2, d2, imgSize, flags=stereo_cali_option[0], criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-5))
    print(f"再投影誤差: {retval}")
    print(f"カメラ1の内部パラメータ: {M1}")
    print(f"カメラ1の歪み係数: {D1}")
    print(f"カメラ2の内部パラメータ: {M2}")
    print(f"カメラ2の歪み係数: {D2}")
    print(f"回転行列: {R}")
    print(f"並進ベクトル: {T}")
    print(f"基本行列: {E}")
    print(f"基礎行列: {F}")

    # ステレオキャリブレーションで取得したパラメータを保存
    stereo_params_dict = {"M1": M1, "D1": D1, "M2": M2, "D2": D2, "R": R, "T": T, "E": E, "F": F}
    params_save_dir = root_dir / "stero_cali" / target_facility
    saveCameraParameters(params_save_dir / "stereo_params.pickle", stereo_params_dict)

    avg_error = evaluate_epipolar_error(img_points1, img_points2, M1, D1, M2, D2, F)
    print(f"平均エピポーラ誤差: {avg_error}")










if __name__ == "__main__":
    main()