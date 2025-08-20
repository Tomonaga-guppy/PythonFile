import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

def main():
    # --- パラメータ設定 ---
    # ステレオキャリブレーション用の画像が保存されているディレクトリ
    stereo_cali_dir = Path(r"G:\gait_pattern\stereo_cali\9g_20250807_6x5_35")
    # 各カメラの内部パラメータが保存されているディレクトリ
    int_cali_dir = Path(r"G:\gait_pattern\int_cali\9g_20250807_6x5_35")

    left_cam_dir_name = 'fl'
    right_cam_dir_name = 'fr'
    checker_pattern = (5, 4)
    square_size = 35.0  # mm単位
    print(f"チェッカーボードのパターン: {checker_pattern[0]}x{checker_pattern[1]}, 正方形のサイズ: {square_size} mm")

    # チェッカーボードの3D座標を準備
    objp = np.zeros((checker_pattern[0] * checker_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checker_pattern[0], 0:checker_pattern[1]].T.reshape(-1, 2)
    objp *= square_size

    # 終了基準
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"\n{'='*80}")
    print(f"ステレオキャリブレーションを開始します: ({left_cam_dir_name} と {right_cam_dir_name})")
    print(f"{'='*80}")

    # --- 1. 各カメラの内部パラメータを読み込む ---
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
    left_img_folder = stereo_cali_dir / left_cam_dir_name / "cali_imgs"
    right_img_folder = stereo_cali_dir / right_cam_dir_name / "cali_imgs"

    left_imgs = sorted([p for p in left_img_folder.glob("*.png")])

    if not left_imgs:
        print(f"エラー: 左カメラの画像フォルダに有効なPNGファイルが見つかりません: {left_img_folder}")
        return

    # 検出成功画像を保存するフォルダを作成
    success_folder_l = stereo_cali_dir / left_cam_dir_name / "success"
    success_folder_r = stereo_cali_dir / right_cam_dir_name / "success"
    success_folder_l.mkdir(exist_ok=True)
    success_folder_r.mkdir(exist_ok=True)

    image_pairs = []
    for left_img_path in left_imgs:
        left_img_name = left_img_path.name
        # left_img_sign_parts = left_img_path.stem.split('_')
        # left_img_sign = '_'.join(left_img_sign_parts[0:2])

        # right_img_path = list(right_img_folder.glob(f"{left_img_sign}_*.png"))[0]
        right_img_path = right_img_folder / left_img_name
        if right_img_path.exists():
            image_pairs.append((left_img_path, right_img_path))

    if not image_pairs:
        print("エラー: 対応する画像ペアが見つかりませんでした。")
        return

    print(f"\n{len(image_pairs)} 組の画像ペアを検出しました。コーナー検出を開始します...")

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []
    used_pairs = [] # 検出に成功したペアのパスを保存

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
            used_pairs.append((left_path, right_path))

            # 成功した画像にコーナーを描画して保存
            corner_img_l = cv2.drawChessboardCorners(img_l.copy(), checker_pattern, corners2_l, ret_l)
            cv2.imwrite(str(success_folder_l / left_path.name), corner_img_l)
            corner_img_r = cv2.drawChessboardCorners(img_r.copy(), checker_pattern, corners2_r, ret_r)
            cv2.imwrite(str(success_folder_r / right_path.name), corner_img_r)


    if not objpoints:
        print("エラー: 全ての画像ペアでコーナー検出に失敗しました。")
        return

    print(f"\n{len(objpoints)} 組のペアでコーナー検出に成功しました。")
    print(f"成功画像は 'success' フォルダに保存されました。")

    # --- 3. ユーザーによる目視での画像ペア除外 ---
    print("\n--- 画像ペア除外リスト ---")
    for i, (left_path, right_path) in enumerate(used_pairs):
        print(f"  [{i:2d}] {left_path.name}")

    excluded_indices = []
    while True:
        try:
            print("\n最終キャリブレーションから『除外する』ペアの番号をカンマ区切りで入力してください。")
            user_input = input("何も除外しない場合は、そのままEnterキーを押してください: ")
            if not user_input: break
            excluded_indices = [int(i.strip()) for i in user_input.split(',')]
            if all(0 <= i < len(used_pairs) for i in excluded_indices): break
            else: print("エラー: 範囲外の番号が入力されました。")
        except ValueError: print("エラー: 不正な入力です。")

    if excluded_indices:
        print("\n以下の画像ペアを除外対象としてファイル名を変更します:")
        excluded_indices_set = set(excluded_indices)
        for i in sorted(list(excluded_indices_set), reverse=True):
            left_to_exclude, right_to_exclude = used_pairs[i]

            # successフォルダ内のファイルもリネーム
            success_path_l = success_folder_l / left_to_exclude.name
            success_path_r = success_folder_r / right_to_exclude.name
            for path_to_exclude in [success_path_l, success_path_r]:
                 new_name = f"excluded_{path_to_exclude.name}"
                 new_path = path_to_exclude.with_name(new_name)
                 if path_to_exclude.exists() and not new_path.exists():
                     path_to_exclude.rename(new_path)

    # 除外されなかったデータを整理
    objpoints_clean, imgpoints_l_clean, imgpoints_r_clean = [], [], []
    kept_paths = []
    for i in range(len(used_pairs)):
        if i not in excluded_indices:
            objpoints_clean.append(objpoints[i])
            imgpoints_l_clean.append(imgpoints_l[i])
            imgpoints_r_clean.append(imgpoints_r[i])
            kept_paths.append(used_pairs[i])

    if len(objpoints_clean) < 10:
        print(f"警告: 残った優良な画像ペアが少なすぎます ({len(objpoints_clean)}組)。")
        if not objpoints_clean:
            print("使用可能な画像ペアがなくなったため、処理を中断します。")
            return

    print("\n以下の画像ペアが最終キャリブレーションに使用されます:")
    for left_path, _ in kept_paths:
        print(f"  - {left_path.name}")

    # --- 4. ステレオキャリブレーションを実行 ---
    print(f"\n{len(objpoints_clean)} 組の画像ペアで最終キャリブレーションを実行します...")
    img_size = gray_l.shape[::-1]
    flags = cv2.CALIB_USE_INTRINSIC_GUESS  # 内部パラメータを初期値として使用
    # flags = cv2.CALIB_FIX_INTRINSIC  #内部パラメータを初期値で固定
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints_clean, imgpoints_l_clean, imgpoints_r_clean, mtx_l, dist_l, mtx_r, dist_r,
        img_size, flags=flags, criteria=criteria
    )

    if not ret:
        print("最終キャリブレーションに失敗しました。")
        return

    # --- 5. 結果の表示と保存 ---
    print("\n【最終的なステレオキャリブレーション結果】")
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
        "image_size": [img_size[0], img_size[1]],
        "excluded_images_indices": excluded_indices,
        "used_images_count_final": len(objpoints_clean),
        "used_checkerboard_pattern": checker_pattern,
        "used_square_size_mm": square_size
    }

    with open(output_params_path, 'w') as f:
        json.dump(stereo_params, f, indent=4)

    print(f"\n結果をJSONファイルに保存しました: {output_params_path}")


if __name__ == '__main__':
    main()
