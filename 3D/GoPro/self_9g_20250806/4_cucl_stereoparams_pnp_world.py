import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

def main():
    # --- 1. パラメータ設定 ---
    root_dir = Path(r"G:\gait_pattern")
    cali_dir = root_dir / "int_cali" / "9g_20250807_6x5_35"
    left_cam_dir_name = 'fl'
    right_cam_dir_name = 'fr'
    intrinsic_filename = "camera_params.json"
    stereo_img_folder_name = "cali_imgs"
    checker_pattern = (5, 4)
    square_size = 35.0

    objp = np.zeros((checker_pattern[0] * checker_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checker_pattern[0], 0:checker_pattern[1]].T.reshape(-1, 2)
    objp *= square_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # --- 2. 内部パラメータの読み込み ---
    print("\n各カメラの内部パラメータを読み込みます...")
    try:
        with open(cali_dir / left_cam_dir_name / intrinsic_filename, 'r') as f:
            params_l = json.load(f)
            mtx_l, dist_l = np.array(params_l['intrinsics']), np.array(params_l['distortion'])
        with open(cali_dir / right_cam_dir_name / intrinsic_filename, 'r') as f:
            params_r = json.load(f)
            mtx_r, dist_r = np.array(params_r['intrinsics']), np.array(params_r['distortion'])
    except FileNotFoundError as e:
        print(f"エラー: 内部パラメータファイルが見つかりません: {e.filename}")
        return

    # --- 3. チェッカーボードの検出 ---
    objpoints, imgpoints_l, imgpoints_r = [], [], []
    img_path_l = sorted(list((cali_dir / left_cam_dir_name / stereo_img_folder_name).glob("*.png")))
    img_path_r = sorted(list((cali_dir / right_cam_dir_name / stereo_img_folder_name).glob("*.png")))
    files_r_map = {p.name: p for p in img_path_r}
    img_pairs = [(p, files_r_map[p.name]) for p in img_path_l if p.name in files_r_map]

    img_size = None
    for fname_l, fname_r in tqdm(img_pairs, desc="チェッカーボード検出中"):
        img_l = cv2.imread(str(fname_l))
        if img_size is None: img_size = img_l.shape[:2][::-1]
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(cv2.imread(str(fname_r)), cv2.COLOR_BGR2GRAY)
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, checker_pattern, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, checker_pattern, None)
        if ret_l and ret_r:
            objpoints.append(objp)
            imgpoints_l.append(cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria))
            imgpoints_r.append(cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria))

    # --- 4. 外部パラメータの計算 ---
    rvecs_l, tvecs_l, rvecs_r, tvecs_r = [], [], [], []
    for i in range(len(objpoints)):
        ret_l, rvec_l, tvec_l = cv2.solvePnP(objpoints[i], imgpoints_l[i], mtx_l, dist_l)
        ret_r, rvec_r, tvec_r = cv2.solvePnP(objpoints[i], imgpoints_r[i], mtx_r, dist_r)
        if ret_l and ret_r:
            rvecs_l.append(rvec_l); tvecs_l.append(tvec_l)
            rvecs_r.append(rvec_r); tvecs_r.append(tvec_r)

    # ★★★ 修正点: ベクトルを正しく平均化 ★★★
    rvec_l_avg = np.mean(np.array(rvecs_l), axis=0)
    tvec_l_avg = np.mean(np.array(tvecs_l), axis=0)
    rvec_r_avg = np.mean(np.array(rvecs_r), axis=0)
    tvec_r_avg = np.mean(np.array(tvecs_r), axis=0)

    R_l, _ = cv2.Rodrigues(rvec_l_avg)
    R_r, _ = cv2.Rodrigues(rvec_r_avg)

    R_rel = R_r @ R_l.T
    T_rel = tvec_r_avg - (R_rel @ tvec_l_avg)

    # x軸について180度回転
    R_board_to_world = np.array([
                                [1., 0., 0.],
                                [0.,-1., 0.],
                                [0., 0.,-1.]
                                ])
    R_world_to_board = R_board_to_world.T
    R_world_to_left_final = R_l @ R_world_to_board

    # --- 5. 結果の保存 ---
    stereo_params = {
        "camera_matrix_left": mtx_l.tolist(), "distortion_left": dist_l.tolist(),
        "camera_matrix_right": mtx_r.tolist(), "distortion_right": dist_r.tolist(),
        "rotation_matrix": R_rel.tolist(),
        # ★★★ 修正点: ベクトルを1次元リストに変換して保存 ★★★
        "translation_vector": T_rel.flatten().tolist(),
        "R_world_to_left": R_world_to_left_final.tolist(),
        "T_world_to_left": tvec_l_avg.flatten().tolist(),
    }
    output_file = cali_dir / f"stereo_params_{left_cam_dir_name}_{right_cam_dir_name}_world.json"
    with open(output_file, 'w') as f:
        json.dump(stereo_params, f, indent=4)

    print(f"\nワールド座標変換情報を含むステレオパラメータを保存しました:\n-> {output_file}")

if __name__ == '__main__':
    main()
