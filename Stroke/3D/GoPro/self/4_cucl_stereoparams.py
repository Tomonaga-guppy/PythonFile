import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

def main():
    # --- パラメータ設定 ---
    # 画像の場所をユーザー指定のパスに修正
    stereo_cali_dir = Path(r"G:\gait_pattern\stero_cali\9g_6x5")
    # 各カメラの内部パラメータが保存されている場所
    int_cali_dir = Path(r"G:\gait_pattern\int_cali\9g_6x5")

    left_cam_dir_name = 'fl'
    right_cam_dir_name = 'fr'
    checker_pattern = (5, 4)
    square_size = 35.0  # mm単位

    # チェッカーボードの3D座標を準備
    objp = np.zeros((checker_pattern[0] * checker_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checker_pattern[0], 0:checker_pattern[1]].T.reshape(-1, 2)
    objp *= square_size

    # 終了基準
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"\n{'='*60}")
    print(f"ステレオキャリブレーションを開始します: ({left_cam_dir_name} と {right_cam_dir_name})")
    print(f"{'='*60}")

    # --- 1. 各カメラの内部パラメータを読み込む ---
    # 注意: 内部パラメータは元の 'int_cali' ディレクトリから読み込む想定
    left_params_path = int_cali_dir / left_cam_dir_name / "camera_params.json"
    right_params_path = int_cali_dir / right_cam_dir_name / "camera_params.json"

    # 結果の出力先はステレオキャリブレーション用ディレクトリ
    output_params_path = stereo_cali_dir / "stereo_params.json"

    if not (left_params_path.exists() and right_params_path.exists()):
        print("エラー: 左右どちらかの camera_params.json が見つかりません。")
        print(f"  - 参照先: {int_cali_dir}")
        return

    print("各カメラの内部パラメータを読み込んでいます...")
    with open(left_params_path, 'r') as f:
        left_params = json.load(f)
        mtx_l = np.array(left_params['intrinsics'])
        dist_l = np.array(left_params['distortion'])

    with open(right_params_path, 'r') as f:
        right_params = json.load(f)
        mtx_r = np.array(right_params['intrinsics'])
        dist_r = np.array(right_params['distortion'])
    print("読み込み完了。")

    # --- 2. 対応する画像ペアからコーナーを検出 ---
    # <--- 変更点: ユーザー指定のステレオ画像用ディレクトリを参照
    left_img_folder = stereo_cali_dir / left_cam_dir_name / "cali_imgs"
    right_img_folder = stereo_cali_dir / right_cam_dir_name / "cali_imgs"
    print(f"\n左カメラ画像フォルダ: {left_img_folder}")
    print(f"右カメラ画像フォルダ: {right_img_folder}")

    left_imgs = sorted(list(left_img_folder.glob("*.png")))
    if not left_imgs:
        print(f"エラー: 左カメラの画像フォルダにPNGファイルが見つかりません: {left_img_folder}")
        return

    image_pairs = []
    for left_img_path in left_imgs:
        right_img_path = right_img_folder / left_img_path.name
        if right_img_path.exists():
            image_pairs.append((left_img_path, right_img_path))

    if not image_pairs:
        print("エラー: 対応する画像ペアが見つかりませんでした。")
        return

    print(f"\n{len(image_pairs)} 組の画像ペアを検出しました。コーナー検出を開始します...")

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    for left_path, right_path in tqdm(image_pairs, desc="コーナー検出中"):
        img_l = cv2.imread(str(left_path))
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        img_r = cv2.imread(str(right_path))
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, checker_pattern, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, checker_pattern, None)

        if ret_l and ret_r:
            corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints_l.append(corners2_l)
            imgpoints_r.append(corners2_r)

    if not objpoints:
        print("エラー: 全ての画像ペアでコーナー検出に失敗しました。")
        return

    print(f"\n{len(objpoints)} 組の有効なコーナーを検出しました。")
    img_size = gray_l.shape[::-1]

    # --- 3. ステレオキャリブレーションを実行 ---
    print("\nステレオキャリブレーションを実行中...")
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r,
        img_size, flags=flags, criteria=criteria
    )

    if not ret:
        print("ステレオキャリブレーションに失敗しました。")
        return

    # --- 4. 結果の表示と保存 ---
    print("\n【ステレオキャリブレーション結果】")
    print(f"再投影誤差(RMS): {ret:.4f} pixels")
    print("\n回転行列 (R):")
    print(R)
    print("\n並進ベクトル (T) [mm]:")
    print(T)

    stereo_params = {
        "reprojection_error_rms": ret,
        "camera_matrix_left": mtx_l.tolist(),
        "distortion_left": dist_l.tolist(),
        "camera_matrix_right": mtx_r.tolist(),
        "distortion_right": dist_r.tolist(),
        "R": R.tolist(), "T": T.tolist(), "E": E.tolist(), "F": F.tolist(),
        "image_size": img_size
    }

    with open(output_params_path, 'w') as f:
        json.dump(stereo_params, f, indent=4)

    print(f"\n結果をJSONファイルに保存しました: {output_params_path}")


if __name__ == '__main__':
    main()