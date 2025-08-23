import cv2
import numpy as np
import json
from pathlib import Path
import copy
import matplotlib.pyplot as plt

def generate3Dgrid(CheckerBoardParams):
    """
    チェッカーボードの3D座標を生成する
    """
    dimensions = CheckerBoardParams['dimensions']  # (width, height)
    square_size = CheckerBoardParams['squareSize']

    # 3D座標の準備 (z=0平面に配置)
    objp = np.zeros((dimensions[0] * dimensions[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:dimensions[0], 0:dimensions[1]].T.reshape(-1, 2)
    objp *= square_size

    return objp

def detect_chessboard_corners(image, checker_pattern, square_size, 
                            imageUpsampleFactor=1, visualize=False):
    """
    チェッカーボードのコーナーを検出する
    
    Args:
        image: 入力画像
        checker_pattern: チェッカーボードパターン (width, height)
        square_size: 正方形のサイズ
        imageUpsampleFactor: 画像のアップサンプリング係数
        visualize: 結果を表示するかどうか
    
    Returns:
        ret: 検出成功フラグ
        corners2: 検出されたコーナー座標
        objp: 対応する3D座標
        image_with_corners: コーナーが描画された画像
    """
    # 画像のアップサンプリング
    if imageUpsampleFactor != 1:
        dim = (int(imageUpsampleFactor * image.shape[1]), 
               int(imageUpsampleFactor * image.shape[0]))
        imageUpsampled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    else:
        imageUpsampled = image.copy()

    gray_color = cv2.cvtColor(imageUpsampled, cv2.COLOR_BGR2GRAY)

    # チェッカーボードコーナーの検出
    ret, corners = cv2.findChessboardCorners(
        gray_color, checker_pattern,
        cv2.CALIB_CB_ADAPTIVE_THRESH
    )

    if ret:
        # サブピクセル精度でコーナーを精密化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(
            gray_color, corners, (11, 11), (-1, -1), criteria
        ) / imageUpsampleFactor

        # 3D座標を生成
        CheckerBoardParams = {
            'dimensions': checker_pattern,
            'squareSize': square_size
        }
        objp = generate3Dgrid(CheckerBoardParams)

        # コーナーを描画
        image_with_corners = image.copy()
        
        # 小さな円でコーナーを描画（utilsChecker.pyのスタイル）
        square_size_pixels = np.linalg.norm((corners2[1, 0, :] - corners2[0, 0, :]).squeeze())
        circle_size = 2 if square_size_pixels > 12 else 1

        for i in range(corners2.shape[0]):
            pt = corners2[i, :, :].squeeze()
            cv2.circle(image_with_corners, tuple(pt.astype(int)), circle_size, (255, 255, 0), 2)

        if visualize:
            image_with_corners_mini = cv2.resize(image_with_corners, (3840 // 2, 2160 // 2))
            cv2.imshow('Detected Corners', image_with_corners_mini)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return ret, corners2, objp, image_with_corners
    else:
        print("チェッカーボードが検出できませんでした。")
        return ret, None, None, image

def calculate_extrinsics(image_path, camera_params, checker_pattern, square_size,
                        imageUpsampleFactor=2, useSecondExtrinsicsSolution=False,
                        visualize=False, save_images=True):
    """
    単一画像から外部パラメータを計算する
    
    Args:
        image_path: 画像ファイルパス
        camera_params: 内部パラメータ辞書
        checker_pattern: チェッカーボードパターン (width, height)
        square_size: 正方形のサイズ
        imageUpsampleFactor: 画像のアップサンプリング係数
        useSecondExtrinsicsSolution: 第二解を使用するかどうか
        visualize: 結果を表示するかどうか
        save_images: 結果画像を保存するかどうか
    
    Returns:
        camera_params_with_extrinsics: 外部パラメータを含む辞書
    """
    
    # 画像を読み込み
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"画像を読み込めません: {image_path}")
        return None

    # チェッカーボードコーナーを検出
    ret, corners2, objp, image_with_corners = detect_chessboard_corners(
        image, checker_pattern, square_size, imageUpsampleFactor, visualize
    )

    if not ret:
        print("チェッカーボードの検出に失敗しました。")
        return None

    # 内部パラメータを取得
    intrinsic_mat = np.array(camera_params['intrinsics'])
    distortion = np.array(camera_params['distortion'])

    # solvePnPGenericを使用して外部パラメータを計算（2つの解を取得）
    try:
        rets, rvecs, tvecs, reprojError = cv2.solvePnPGeneric(
            objp, corners2, intrinsic_mat, distortion, 
            flags=cv2.SOLVEPNP_IPPE
        )
    except cv2.error as e:
        print(f"solvePnPGeneric失敗: {e}")
        print("solvePnPRansacで再試行します...")
        
        ret_ransac, rvec, tvec, inliers = cv2.solvePnPRansac(
            objp, corners2, intrinsic_mat, distortion
        )
        
        if ret_ransac:
            rvecs = [rvec]
            tvecs = [tvec]
            rets = 1
        else:
            print("外部パラメータの計算に失敗しました。")
            return None

    if rets < 1:
        print("外部パラメータの計算に失敗しました。")
        return None

    # 使用する解を選択
    solution_to_use = 1 if useSecondExtrinsicsSolution and rets > 1 else 0
    
    # 結果を保存するディレクトリを作成
    output_dir = image_path.parent
    
    # 各解について処理
    camera_params_solutions = []
    
    for i in range(rets):
        rvec = rvecs[i]
        tvec = tvecs[i]
        
        # 回転行列を計算
        R_worldFromCamera = cv2.Rodrigues(rvec)[0]
        
        # カメラパラメータのコピーを作成
        camera_params_copy = copy.deepcopy(camera_params)
        camera_params_copy['rotation'] = R_worldFromCamera.tolist()
        camera_params_copy['translation'] = tvec.tolist()
        camera_params_copy['rotation_EulerAngles'] = rvec.tolist()
        
        camera_params_solutions.append(camera_params_copy)
        
        if save_images:
            # 再投影点を計算
            img_points, _ = cv2.projectPoints(
                objp, rvec, tvec, intrinsic_mat, distortion
            )
            
            # 座標軸を描画
            image_with_axes = cv2.drawFrameAxes(
                image_with_corners.copy(), intrinsic_mat, distortion, 
                rvec, tvec, square_size * 3, 4
            )
            
            # トリミング（utilsChecker.pyのスタイル）
            ht, wd = image.shape[:2]
            buffer_val = 0.05 * np.mean([ht, wd])
            
            top_edge = int(np.max([np.squeeze(np.min(img_points, axis=0))[1] - buffer_val, 0]))
            left_edge = int(np.max([np.squeeze(np.min(img_points, axis=0))[0] - buffer_val, 0]))
            bottom_edge = int(np.min([np.squeeze(np.max(img_points, axis=0))[1] + buffer_val, ht]))
            right_edge = int(np.min([np.squeeze(np.max(img_points, axis=0))[0] + buffer_val, wd]))
            
            image_cropped = image_with_axes[top_edge:bottom_edge, left_edge:right_edge, :]
            
            # 画像を保存
            save_path = output_dir / f"extrinsic_calibration_solution_{i}.jpg"
            cv2.imwrite(str(save_path), image_cropped)
            print(f"外部キャリブレーション画像を保存: {save_path}")
            
            if visualize:
                cv2.imshow(f'Extrinsic Calibration - Solution {i}', image_cropped)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    # 選択された解を返す
    selected_solution = camera_params_solutions[solution_to_use]
    
    print(f"解 {solution_to_use} を選択しました")
    print(f"回転ベクトル: {selected_solution['rotation_EulerAngles']}")
    print(f"並進ベクトル: {selected_solution['translation']}")
    
    return selected_solution, camera_params_solutions

def main():
    """
    メイン処理
    """
    # --- パラメータ設定 ---
    int_dir = Path(r"G:\gait_pattern\int_cali\9g_20250807_6x5")
    ext_dir = Path(r"G:\gait_pattern\stereo_cali\9g_20250811")
    directions = ['fl', 'fr']
    checker_pattern = (5, 4)  # (width, height)
    square_size = 35  # mm単位
    
    # 外部キャリブレーション用画像名（基準となる画像）
    reference_image_name = "origin.png"  
    
    print("外部パラメータキャリブレーション開始")
    print(f"チェッカーボードパターン: {checker_pattern[0]}x{checker_pattern[1]}")
    print(f"正方形のサイズ: {square_size} mm")
    print(f"基準画像名: {reference_image_name}")
    
    all_results = {}
    
    for direction in directions:
        print(f"\n{'='*80}")
        print(f"カメラ '{direction}' の外部パラメータを計算中...")
        print(f"{'='*80}")
        
        # 内部パラメータを読み込み
        params_file = int_dir / direction / "camera_params.json"
        if not params_file.exists():
            print(f"エラー: 内部パラメータファイルが見つかりません: {params_file}")
            continue
            
        with open(params_file, 'r') as f:
            camera_params = json.load(f)
        
        print(f"内部パラメータを読み込み: {params_file}")
        
        # 基準画像のパスを設定
        reference_image_path = ext_dir / direction / reference_image_name
        if not reference_image_path.exists():
            print(f"エラー: 基準画像が見つかりません: {reference_image_path}")
            # 代替として最初の画像を使用
            cali_imgs = sorted(list((ext_dir / direction / "cali_imgs").glob("*.png")))
            cali_imgs = [p for p in cali_imgs if not p.name.startswith("excluded_")]
            if cali_imgs:
                reference_image_path = cali_imgs[0]
                print(f"代替として最初の画像を使用: {reference_image_path}")
            else:
                print("使用可能な画像がありません。")
                continue
        
        # 外部パラメータを計算
        result = calculate_extrinsics(
            reference_image_path, 
            camera_params,
            checker_pattern,
            square_size,
            imageUpsampleFactor=2,
            useSecondExtrinsicsSolution=False,
            visualize=True,
            save_images=True
        )
        
        if result is None:
            print(f"カメラ '{direction}' の外部パラメータ計算に失敗しました。")
            continue
            
        selected_solution, all_solutions = result
        
        # 結果を保存
        output_file = ext_dir / direction / "camera_params_with_extrinsics.json"
        
        # 完全なカメラパラメータを作成
        complete_params = {
            **camera_params,  # 内部パラメータを含む
            'extrinsics': {
                'rotation_matrix': selected_solution['rotation'],
                'translation_vector': selected_solution['translation'],
                'rotation_euler_angles': selected_solution['rotation_EulerAngles']
            },
            'reference_image': str(reference_image_path.name),
            'checkerboard_pattern': checker_pattern,
            'square_size_mm': square_size,
            'num_solutions_found': len(all_solutions),
            'selected_solution_index': 0
        }
        
        # 全ての解も保存
        complete_params['all_extrinsic_solutions'] = []
        for i, solution in enumerate(all_solutions):
            complete_params['all_extrinsic_solutions'].append({
                'solution_index': i,
                'rotation_matrix': solution['rotation'],
                'translation_vector': solution['translation'],
                'rotation_euler_angles': solution['rotation_EulerAngles']
            })
        
        with open(output_file, 'w') as f:
            json.dump(complete_params, f, indent=4)
        
        print(f"完全なカメラパラメータを保存: {output_file}")
        
        # 結果の表示
        print("\n--- 外部パラメータ結果 ---")
        print(f"回転行列:")
        R = np.array(selected_solution['rotation'])
        for row in R:
            print(f"  [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]")
        
        print(f"並進ベクトル: {selected_solution['translation']}")
        print(f"オイラー角: {selected_solution['rotation_EulerAngles']}")
        
        all_results[direction] = complete_params
    
    # 全体の結果をまとめて保存
    summary_file = ext_dir / "all_camera_parameters.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n全カメラのパラメータをまとめて保存: {summary_file}")
    print("\n外部パラメータキャリブレーション完了！")

if __name__ == '__main__':
    main()