import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

def main():
    # --- 1. パラメータ設定 ---
    # プロジェクトのルートディレクトリ
    root_dir = Path(r"G:\gait_pattern")

    # 内部パラメータとステレオキャリブレーション用画像があるディレクトリ
    cali_dir = root_dir / "int_cali" / "9g_20250807_6x5_35"

    # 左右カメラのディレクトリ名
    left_cam_dir_name = 'fl'
    right_cam_dir_name = 'fr'

    # 内部パラメータファイル名 (1_culc_intparams_SB.py の出力に合わせる)
    intrinsic_filename = "camera_params.json"

    # ステレオキャリブレーション用画像の入ったフォルダ名 (0_cut_intmp4.py の出力に合わせる)
    stereo_img_folder_name = "cali_imgs"

    # チェッカーボードの物理的な設定
    checker_pattern = (5, 4)
    square_size = 35.0  # mm単位
    print(f"チェッカーボードのパターン: {checker_pattern[0]}x{checker_pattern[1]}, 正方形のサイズ: {square_size} mm")

    # チェッカーボードの3D座標を準備
    objp = np.zeros((checker_pattern[0] * checker_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checker_pattern[0], 0:checker_pattern[1]].T.reshape(-1, 2)
    objp *= square_size

    # 終了基準
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # --- 2. 内部パラメータの読み込み ---
    print("\n各カメラの内部パラメータを読み込みます...")
    try:
        # 左カメラ
        with open(cali_dir / left_cam_dir_name / intrinsic_filename, 'r') as f:
            params_l = json.load(f)
            mtx_l = np.array(params_l['intrinsics'])
            dist_l = np.array(params_l['distortion'])
            print(f"左カメラ ({left_cam_dir_name}) のパラメータを読み込みました。")

        # 右カメラ
        with open(cali_dir / right_cam_dir_name / intrinsic_filename, 'r') as f:
            params_r = json.load(f)
            mtx_r = np.array(params_r['intrinsics'])
            dist_r = np.array(params_r['distortion'])
            print(f"右カメラ ({right_cam_dir_name}) のパラメータを読み込みました。")
    except FileNotFoundError as e:
        print(f"エラー: 内部パラメータファイルが見つかりません。パスを確認してください。")
        print(f"-> {e.filename}")
        print("-> 1_culc_intparams_SB.py を先に実行してください。")
        return
    except KeyError as e:
        print(f"エラー: 内部パラメータファイルのキーが不正です: {e}")
        print("-> 1_culc_intparams_SB.py の出力形式を確認してください。")
        return

    # --- 3. チェッカーボードの検出 ---
    objpoints = []  # 3D点
    imgpoints_l = []  # 左カメラの2D点
    imgpoints_r = []  # 右カメラの2D点

    img_path_l = sorted(list((cali_dir / left_cam_dir_name / stereo_img_folder_name).glob("*.png")))
    img_path_r = sorted(list((cali_dir / right_cam_dir_name / stereo_img_folder_name).glob("*.png")))

    if not img_path_l or not img_path_r:
        print("エラー: キャリブレーション用の画像が見つかりません。")
        print(f"-> 左カメラ探索パス: {cali_dir / left_cam_dir_name / stereo_img_folder_name}")
        print(f"-> 右カメラ探索パス: {cali_dir / right_cam_dir_name / stereo_img_folder_name}")
        print("-> 0_cut_intmp4.py を先に実行してください。")
        return

    # ファイル名のペアを作成 (0_cut_intmp4.py の出力形式を想定)
    img_pairs = []
    files_r_map = {p.name: p for p in img_path_r}
    for path_l in img_path_l:
        if path_l.name in files_r_map:
            img_pairs.append((path_l, files_r_map[path_l.name]))

    print(f"\n{len(img_pairs)} 組の同期された画像ペアからチェッカーボードを検出します...")

    img_size = None
    successful_pairs = []

    for fname_l, fname_r in tqdm(img_pairs, desc="チェッカーボード検出中"):
        img_l = cv2.imread(str(fname_l))
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        img_r = cv2.imread(str(fname_r))
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = gray_l.shape[::-1]

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, checker_pattern, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, checker_pattern, None)

        if ret_l and ret_r:
            objpoints.append(objp)
            corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            imgpoints_l.append(corners2_l)
            corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            imgpoints_r.append(corners2_r)
            successful_pairs.append((fname_l, fname_r))

    print(f"\nチェッカーボードが両方の画像で検出されたのは {len(objpoints)} 組でした。")

    if len(objpoints) < 10:
        print(f"エラー: キャリブレーションに必要な画像ペアが少なすぎます ({len(objpoints)}組)。10組以上を推奨します。")
        return

    # --- 4. 外部パラメータの計算 (solvePnP を使用) ---
    print(f"\n{len(objpoints)} 組の画像ペアで外部パラメータを計算します...")

    rvecs_l, tvecs_l = [], []
    rvecs_r, tvecs_r = [], []

    # 各画像ペアに対して solvePnP を実行
    for i in range(len(objpoints)):
        ret_l, rvec_l, tvec_l = cv2.solvePnP(objpoints[i], imgpoints_l[i], mtx_l, dist_l)
        ret_r, rvec_r, tvec_r = cv2.solvePnP(objpoints[i], imgpoints_r[i], mtx_r, dist_r)

        if ret_l and ret_r:
            rvecs_l.append(rvec_l)
            tvecs_l.append(tvec_l)
            rvecs_r.append(rvec_r)
            tvecs_r.append(tvec_r)

    if len(rvecs_l) < 1:
        print("エラー: 有効な画像ペアで外部パラメータを計算できませんでした。")
        return

    print(f"成功した {len(rvecs_l)} 組の外部パラメータを平均化します。")
    # 計算された外部パラメータを平均化
    rvec_l_avg = np.mean(rvecs_l, axis=0)
    tvec_l_avg = np.mean(tvecs_l, axis=0)
    rvec_r_avg = np.mean(rvecs_r, axis=0)
    tvec_r_avg = np.mean(tvecs_r, axis=0)

    # 回転ベクトルを回転行列に変換
    R_l, _ = cv2.Rodrigues(rvec_l_avg)
    R_r, _ = cv2.Rodrigues(rvec_r_avg)

    # 相対的な回転行列 R と並進ベクトル T を計算
    R = R_r @ R_l.T
    T = tvec_r_avg - (R @ tvec_l_avg)

    # --- 5. 結果の表示と保存 ---
    print("\n【最終的なステレオキャリブレーション結果 (solvePnPベース)】")
    print("\n回転行列 (R):")
    print(R)
    print("\n並進ベクトル (T) [mm]:")
    print(T)

    stereo_params = {
        "camera_matrix_left": mtx_l.tolist(),
        "distortion_left": dist_l.tolist(),
        "camera_matrix_right": mtx_r.tolist(),
        "distortion_right": dist_r.tolist(),
        "rotation_matrix": R.tolist(),
        "translation_vector": T.tolist(),
        "image_size": img_size,
        "method": "solvePnP_based",
        "used_pairs_count": len(objpoints)
    }

    # 出力先は内部パラメータと同じディレクトリ階層にする
    output_dir = cali_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"stereo_params_{left_cam_dir_name}_{right_cam_dir_name}.json"

    with open(output_file, 'w') as f:
        json.dump(stereo_params, f, indent=4)

    print(f"\nステレオパラメータを {output_file} に保存しました。")
    print("後続の3D復元処理 (5_reconstruct_3d.py) でこのファイルを使用します。")
    print("\n処理が完了しました。")


if __name__ == '__main__':
    main()
