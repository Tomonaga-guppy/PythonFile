import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

def generate3Dgrid(CheckerBoardParams):
    """
    チェッカーボードの3D座標を生成する
    """
    dimensions = CheckerBoardParams['dimensions']  # (width, height)
    square_size = CheckerBoardParams['squareSize']

    # 3D座標の準備
    objp = np.zeros((dimensions[0] * dimensions[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:dimensions[0], 0:dimensions[1]].T.reshape(-1, 2)
    objp *= square_size

    return objp

def detect_chessboard_corners_stereo_sb(img_left, img_right, checker_pattern, square_size):
    """
    ステレオ画像ペアでのSBWithMeta方式によるコーナー検出

    Args:
        img_left: 左画像
        img_right: 右画像
        checker_pattern: 期待するチェッカーボードパターン (width, height)
        square_size: 正方形のサイズ

    Returns:
        success: 両方で検出成功したかどうか
        corners_left: 左画像のコーナー座標
        corners_right: 右画像のコーナー座標
        objp: 対応する3D座標
        meta_left: 左画像のメタデータ
        meta_right: 右画像のメタデータ
        actual_pattern_left: 左画像で検出されたパターン
        actual_pattern_right: 右画像で検出されたパターン
    """
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 左画像でSBWithMeta検出
    ret_left, corners_left, meta_left = cv2.findChessboardCornersSBWithMeta(
        gray_left,
        checker_pattern,
        cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_LARGER
    )

    # 右画像でSBWithMeta検出
    ret_right, corners_right, meta_right = cv2.findChessboardCornersSBWithMeta(
        gray_right,
        checker_pattern,
        cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_LARGER
    )

    if ret_left and ret_right:
        # 両方で検出された実際のパターンサイズを取得
        actual_pattern_left = meta_left.shape[::-1]  # (width, height)
        actual_pattern_right = meta_right.shape[::-1]  # (width, height)

        # パターンサイズが一致するかチェック
        if actual_pattern_left == actual_pattern_right:
            # 一致する場合、3D座標を生成
            CheckerBoardParams = {
                'dimensions': actual_pattern_left,
                'squareSize': square_size
            }
            objp = generate3Dgrid(CheckerBoardParams)

            return True, corners_left, corners_right, objp, meta_left, meta_right, actual_pattern_left, actual_pattern_right
        else:
            # パターンサイズが不一致の場合は失敗扱い
            return False, None, None, None, None, None, None, None
    else:
        return False, None, None, None, None, None, None, None

def main():
    # --- パラメータ設定 ---
    # ステレオキャリブレーション用の画像が保存されているディレクトリ
    stereo_cali_dir = Path(r"G:\gait_pattern\stereo_cali\9g_20250807_6x5_35")
    # 各カメラの内部パラメータが保存されているディレクトリ
    int_cali_dir = Path(r"G:\gait_pattern\int_cali\9g_20250807_6x5_35")

    left_cam_dir_name = 'fl'
    right_cam_dir_name = 'fr'
    checker_pattern = (5, 4)  # 期待するパターン
    square_size = 35.0  # mm単位

    print(f"期待するチェッカーボードパターン: {checker_pattern[0]}x{checker_pattern[1]}, 正方形のサイズ: {square_size} mm")
    print("注意: SBWithMeta方式では実際に検出されたパターンサイズが自動調整されます")

    print(f"\n{'='*80}")
    print(f"ステレオキャリブレーション (SB方式) を開始します: ({left_cam_dir_name} と {right_cam_dir_name})")
    print(f"{'='*80}")

    # --- 1. 各カメラの内部パラメータを読み込む ---
    left_params_path = int_cali_dir / left_cam_dir_name / "camera_params.json"
    right_params_path = int_cali_dir / right_cam_dir_name / "camera_params.json"

    # 結果の出力先はステレオキャリブレーション用ディレクトリ
    output_params_path = stereo_cali_dir / "stereo_params.json"

    if not (left_params_path.exists() and right_params_path.exists()):
        print("エラー: 左右どちらかの camera_params.json が見つかりません。")
        print(f"  - 参照先: {int_cali_dir}")
        print("  - 優先順位: camera_params_sb.json > camera_params.json")
        return

    print("各カメラの内部パラメータを読み込んでいます...")
    print(f"  - 左カメラ: {left_params_path}")
    print(f"  - 右カメラ: {right_params_path}")

    with open(left_params_path, 'r') as f:
        left_params = json.load(f)
        mtx_l = np.array(left_params['intrinsics'])
        dist_l = np.array(left_params['distortion'])
        left_method = left_params.get('calibration_method', 'unknown')

    with open(right_params_path, 'r') as f:
        right_params = json.load(f)
        mtx_r = np.array(right_params['intrinsics'])
        dist_r = np.array(right_params['distortion'])
        right_method = right_params.get('calibration_method', 'unknown')

    print(f"読み込み完了 - 左: {left_method}, 右: {right_method}")

    # --- 2. 対応する画像ペアからコーナーを検出 ---
    left_img_folder = stereo_cali_dir / left_cam_dir_name / "cali_imgs"
    right_img_folder = stereo_cali_dir / right_cam_dir_name / "cali_imgs"

    left_imgs = sorted([p for p in left_img_folder.glob("*.png") if not p.name.startswith("excluded_")])

    if not left_imgs:
        print(f"エラー: 左カメラの画像フォルダに有効なPNGファイルが見つかりません: {left_img_folder}")
        return

    # 検出成功画像を保存するフォルダを作成
    success_folder_l = stereo_cali_dir / left_cam_dir_name / "success"
    success_folder_r = stereo_cali_dir / right_cam_dir_name / "success"
    success_folder_l.mkdir(exist_ok=True)
    success_folder_r.mkdir(exist_ok=True)

    # 画像ペアを構築
    image_pairs = []
    for left_img_path in left_imgs:
        left_img_name = left_img_path.name
        right_img_path = right_img_folder / left_img_name
        if right_img_path.exists() and not right_img_path.name.startswith("excluded_"):
            image_pairs.append((left_img_path, right_img_path))

    if not image_pairs:
        print("エラー: 対応する画像ペアが見つかりませんでした。")
        return

    print(f"\n{len(image_pairs)} 組の画像ペアを検出しました。SB方式によるコーナー検出を開始します...")

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []
    used_pairs = []  # 検出に成功したペアのパスを保存
    pattern_info = []  # 検出されたパターン情報
    detection_errors = []  # 検出エラー情報

    successful_pairs = 0
    for left_path, right_path in tqdm(image_pairs, desc="SBコーナー検出中"):
        try:
            img_l = cv2.imread(str(left_path))
            img_r = cv2.imread(str(right_path))

            if img_l is None or img_r is None:
                detection_errors.append(f"画像読み込み失敗: {left_path.name}")
                continue

            # SBWithMeta方式でコーナー検出
            success, corners_l, corners_r, objp, meta_l, meta_r, pattern_l, pattern_r = detect_chessboard_corners_stereo_sb(
                img_l, img_r, checker_pattern, square_size
            )

            if success:
                objpoints.append(objp)
                imgpoints_l.append(corners_l)
                imgpoints_r.append(corners_r)
                used_pairs.append((left_path, right_path))
                pattern_info.append(pattern_l)
                successful_pairs += 1

                # 成功した画像にコーナーを描画して保存
                corner_img_l = cv2.drawChessboardCorners(img_l.copy(), pattern_l, corners_l, True)
                cv2.imwrite(str(success_folder_l / left_path.name), corner_img_l)
                corner_img_r = cv2.drawChessboardCorners(img_r.copy(), pattern_r, corners_r, True)
                cv2.imwrite(str(success_folder_r / right_path.name), corner_img_r)

                print(f"  成功: {left_path.name} - パターン: {pattern_l[0]}x{pattern_l[1]}")
            else:
                detection_errors.append(f"SB検出失敗またはパターン不一致: {left_path.name}")

        except Exception as e:
            detection_errors.append(f"処理エラー: {left_path.name} - {str(e)}")

    if not objpoints:
        print("エラー: 全ての画像ペアでコーナー検出に失敗しました。")
        if detection_errors:
            print("\n検出エラーの詳細:")
            for error in detection_errors[:10]:  # 最初の10個のエラーを表示
                print(f"  - {error}")
        return

    print(f"\n{len(objpoints)} 組のペアでコーナー検出に成功しました。")
    print(f"成功画像は 'success' フォルダに保存されました。")

    if detection_errors:
        print(f"\n{len(detection_errors)} 組のペアで検出に失敗しました。")

    # パターンサイズの統計を表示
    unique_patterns = list(set(pattern_info))
    print(f"\n検出されたパターンサイズの種類: {len(unique_patterns)}")
    for pattern in unique_patterns:
        count = pattern_info.count(pattern)
        print(f"  {pattern[0]}x{pattern[1]}: {count}組")

    # --- 3. 初期ステレオキャリブレーションと誤差評価 ---
    if len(objpoints) >= 10:
        print("\nステップ: 初期ステレオキャリブレーションで各ペアの誤差を評価します...")

        # 画像サイズを取得
        first_img = cv2.imread(str(used_pairs[0][0]))
        img_size = (first_img.shape[1], first_img.shape[0])  # (width, height)

        try:
            # 初期キャリブレーション
            # flags = cv2.CALIB_USE_INTRINSIC_GUESS
            flags = cv2.CALIB_FIX_INTRINSIC
            ret_initial, _, _, _, _, R_initial, T_initial, E_initial, F_initial = cv2.stereoCalibrate(
                objpoints, imgpoints_l, imgpoints_r,
                mtx_l, dist_l, mtx_r, dist_r,
                img_size, flags=flags
            )

            print(f"初期ステレオキャリブレーション完了 - RMS誤差: {ret_initial:.4f} pixels")
        except cv2.error as e:
            print(f"初期ステレオキャリブレーションでエラー: {e}")
            print("誤差評価をスキップして除外選択に進みます。")
            ret_initial = None

    # --- 4. ユーザーによる目視での画像ペア除外 ---
    print("\n--- 画像ペア除外リスト (SB方式) ---")
    for i, (left_path, right_path) in enumerate(used_pairs):
        pattern = pattern_info[i]
        print(f"  [{i:2d}] {left_path.name} (パターン: {pattern[0]}x{pattern[1]})")

    excluded_indices = []
    while True:
        try:
            print(f"\n最終キャリブレーションから『除外する』ペアの番号をカンマ区切りで入力してください。")
            print("何も除外しない場合は、そのままEnterキーを押してください。")
            user_input = input("入力: ")
            if not user_input:
                break
            excluded_indices = [int(i.strip()) for i in user_input.split(',')]
            if all(0 <= i < len(used_pairs) for i in excluded_indices):
                break
            else:
                print("エラー: 範囲外の番号が入力されました。")
        except ValueError:
            print("エラー: 不正な入力です。")

    if excluded_indices:
        print(f"\n以下の{len(excluded_indices)}組の画像ペアを除外対象としてファイル名を変更します:")
        excluded_indices_set = set(excluded_indices)
        for i in sorted(list(excluded_indices_set), reverse=True):
            left_to_exclude, right_to_exclude = used_pairs[i]

            # successフォルダ内のファイルをリネーム
            success_path_l = success_folder_l / left_to_exclude.name
            success_path_r = success_folder_r / right_to_exclude.name

            for path_to_exclude in [success_path_l, success_path_r]:
                new_name = f"excluded_{path_to_exclude.name}"
                new_path = path_to_exclude.with_name(new_name)
                if path_to_exclude.exists() and not new_path.exists():
                    path_to_exclude.rename(new_path)
                    print(f"  - {path_to_exclude.name} -> {new_name}")

    # 除外されなかったデータを整理
    objpoints_clean, imgpoints_l_clean, imgpoints_r_clean = [], [], []
    kept_pairs = []
    kept_patterns = []

    for i in range(len(used_pairs)):
        if i not in excluded_indices:
            objpoints_clean.append(objpoints[i])
            imgpoints_l_clean.append(imgpoints_l[i])
            imgpoints_r_clean.append(imgpoints_r[i])
            kept_pairs.append(used_pairs[i])
            kept_patterns.append(pattern_info[i])

    if len(objpoints_clean) < 10:
        print(f"警告: 残った優良な画像ペアが少なすぎます ({len(objpoints_clean)}組)。")
        if not objpoints_clean:
            print("使用可能な画像ペアがなくなったため、処理を中断します。")
            return

    print(f"\n以下の{len(kept_pairs)}組の画像ペアが最終キャリブレーションに使用されます:")
    for i, (left_path, _) in enumerate(kept_pairs):
        pattern = kept_patterns[i]
        print(f"  - {left_path.name} (パターン: {pattern[0]}x{pattern[1]})")

    # --- 5. 最終ステレオキャリブレーションを実行 ---
    print(f"\n{len(objpoints_clean)} 組の画像ペアで最終ステレオキャリブレーション (SB方式) を実行します...")

    # キャリブレーションフラグ
    # flags = cv2.CALIB_USE_INTRINSIC_GUESS  # 内部パラメータを初期値として使用
    flags = cv2.CALIB_FIX_INTRINSIC  # 内部パラメータを初期値で固定する場合

    try:
        ret, mtx_l_final, dist_l_final, mtx_r_final, dist_r_final, R, T, E, F = cv2.stereoCalibrate(
            objpoints_clean, imgpoints_l_clean, imgpoints_r_clean,
            mtx_l, dist_l, mtx_r, dist_r,
            img_size, flags=flags
        )
    except cv2.error as e:
        print(f"最終ステレオキャリブレーションでエラー: {e}")
        return

    if not ret:
        print("最終ステレオキャリブレーションに失敗しました。")
        return

    # --- 6. 結果の表示と保存 ---
    print("\n" + "="*70)
    print("【最終的なステレオキャリブレーション結果 (SB方式)】")
    print("="*70)
    print(f"再投影誤差(RMS): {ret:.4f} pixels")
    print(f"使用画像ペア数: {len(objpoints_clean)}組")
    print(f"画像サイズ: {img_size[0]} x {img_size[1]}")
    print(f"キャリブレーション方法: SBWithMeta")

    print(f"\n回転行列 (R):")
    for i, row in enumerate(R):
        print(f"  [{' '.join(f'{val:8.5f}' for val in row)}]")

    print(f"\n並進ベクトル (T) [mm]:")
    for i, val in enumerate(T.flatten()):
        print(f"  T{i+1} = {val:8.3f}")

    # ベースライン距離を計算
    baseline = np.linalg.norm(T)
    print(f"\nベースライン距離: {baseline:.3f} mm")

    # 最終的な内部パラメータの表示
    print(f"\n最終内部パラメータ (左カメラ):")
    print(f"  fx = {mtx_l_final[0,0]:.2f}, fy = {mtx_l_final[1,1]:.2f}")
    print(f"  cx = {mtx_l_final[0,2]:.2f}, cy = {mtx_l_final[1,2]:.2f}")

    print(f"\n最終内部パラメータ (右カメラ):")
    print(f"  fx = {mtx_r_final[0,0]:.2f}, fy = {mtx_r_final[1,1]:.2f}")
    print(f"  cx = {mtx_r_final[0,2]:.2f}, cy = {mtx_r_final[1,2]:.2f}")

    # 結果をJSONファイルに保存
    stereo_params = {
        "calibration_method": "SBWithMeta_Stereo",
        "reprojection_error_rms": float(ret),
        "camera_matrix_left": mtx_l_final.tolist(),
        "distortion_left": dist_l_final.tolist(),
        "camera_matrix_right": mtx_r_final.tolist(),
        "distortion_right": dist_r_final.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
        "baseline_mm": float(baseline),
        "image_size": [img_size[0], img_size[1]],
        "excluded_images_indices": excluded_indices,
        "used_images_count_final": len(objpoints_clean),
        "total_pairs_processed": len(image_pairs),
        "successful_detections": successful_pairs,
        "detection_errors_count": len(detection_errors),
        "expected_checkerboard_pattern": checker_pattern,
        "detected_pattern_types": unique_patterns,
        "square_size_mm": square_size,
        "calibration_flags": int(flags),
        "left_intrinsic_source": str(left_params_path.name),
        "right_intrinsic_source": str(right_params_path.name),
        "left_calibration_method": left_method,
        "right_calibration_method": right_method
    }

    with open(output_params_path, 'w') as f:
        json.dump(stereo_params, f, indent=4)

    print(f"\nSB方式による高精度ステレオパラメータを保存しました: {output_params_path}")

    if detection_errors:
        error_log_path = stereo_cali_dir / "detection_errors_sb.log"
        with open(error_log_path, 'w') as f:
            for error in detection_errors:
                f.write(error + '\n')
        print(f"検出エラーログを保存しました: {error_log_path}")

    print("="*70)

if __name__ == '__main__':
    main()