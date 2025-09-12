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

        # if visualize:
        #     image_with_corners_mini = cv2.resize(image_with_corners, (3840 // 2, 2160 // 2))
        #     cv2.imshow('Detected Corners', image_with_corners_mini)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        return ret, corners2, objp, image_with_corners
    else:
        print("チェッカーボードが検出できませんでした。")
        return ret, None, None, image

def calculate_reprojection_error(objp, corners2, rvec, tvec, intrinsic_mat, distortion):
    """
    再投影誤差を計算する

    Args:
        objp: 3D座標点
        corners2: 検出された2D角点
        rvec: 回転ベクトル
        tvec: 並進ベクトル
        intrinsic_mat: 内部パラメータ行列
        distortion: 歪み係数

    Returns:
        mean_error: 平均再投影誤差
        max_error: 最大再投影誤差
        errors: 各点の誤差
        projected_points: 再投影された点
    """
    # 3D点を2Dに再投影
    projected_points, _ = cv2.projectPoints(objp, rvec, tvec, intrinsic_mat, distortion)

    # 検出点と再投影点の差を計算
    errors = []
    for i in range(len(corners2)):
        detected = corners2[i, 0, :]
        projected = projected_points[i, 0, :]
        error = np.linalg.norm(detected - projected)
        errors.append(error)

    errors = np.array(errors)
    mean_error = np.mean(errors)
    max_error = np.max(errors)

    return mean_error, max_error, errors, projected_points

def calculate_true_3d_distance(objp, idx1, idx2):
    """
    3D座標を直接使用した正確な距離計算

    Args:
        objp: 真の3D座標点 (mm)
        idx1, idx2: 測定する2点のインデックス

    Returns:
        distance: 2点間の3D距離 (mm)
    """
    point1_3d = objp[idx1]  # 3D座標を直接使用
    point2_3d = objp[idx2]
    return np.linalg.norm(point2_3d - point1_3d)

def calculate_measured_3d_distance(corners2, rvec, tvec, intrinsic_mat, distortion, idx1, idx2):
    """
    検出された2D点から逆算した3D座標を使った距離計算

    Args:
        corners2: 検出された2D角点
        rvec, tvec: 外部パラメータ
        intrinsic_mat, distortion: 内部パラメータ
        idx1, idx2: 測定する2点のインデックス

    Returns:
        distance: 2点間の3D距離 (mm)
    """
    # 2D点を取得
    point1_2d = corners2[idx1, 0, :].reshape(1, 1, 2)
    point2_2d = corners2[idx2, 0, :].reshape(1, 1, 2)

    # チェッカーボード平面上（z=0）で3D座標を逆算
    # undistortPointsを使用して歪み補正
    point1_undist = cv2.undistortPoints(point1_2d, intrinsic_mat, distortion, P=intrinsic_mat)
    point2_undist = cv2.undistortPoints(point2_2d, intrinsic_mat, distortion, P=intrinsic_mat)

    # カメラ座標系での正規化座標
    point1_norm = np.array([(point1_undist[0,0,0] - intrinsic_mat[0,2]) / intrinsic_mat[0,0],
                           (point1_undist[0,0,1] - intrinsic_mat[1,2]) / intrinsic_mat[1,1], 1.0])
    point2_norm = np.array([(point2_undist[0,0,0] - intrinsic_mat[0,2]) / intrinsic_mat[0,0],
                           (point2_undist[0,0,1] - intrinsic_mat[1,2]) / intrinsic_mat[1,1], 1.0])

    # 回転・並進変換の逆変換
    R = cv2.Rodrigues(rvec)[0]
    R_inv = R.T
    t_inv = -R_inv @ tvec.flatten()

    # チェッカーボード平面（z=0）との交点を計算
    # カメラ中心からの光線とz=0平面の交点
    depth1 = -t_inv[2] / (R_inv @ point1_norm)[2]
    depth2 = -t_inv[2] / (R_inv @ point2_norm)[2]

    # 3D座標を計算
    point1_3d = depth1 * (R_inv @ point1_norm) + t_inv
    point2_3d = depth2 * (R_inv @ point2_norm) + t_inv

    return np.linalg.norm(point2_3d - point1_3d)

def calculate_3d_accuracy_metrics(objp, corners2, rvec, tvec, intrinsic_mat, distortion, square_size):
    """
    3次元空間での精度指標を計算する（3D復元のみ）

    Args:
        objp: 真の3D座標点 (mm)
        corners2: 検出された2D角点
        rvec, tvec: 外部パラメータ
        intrinsic_mat, distortion: 内部パラメータ
        square_size: チェッカーボードの正方形サイズ (mm)

    Returns:
        3D精度指標の辞書
    """
    # 再投影誤差（ピクセル）
    projected_points, _ = cv2.projectPoints(objp, rvec, tvec, intrinsic_mat, distortion)
    reprojection_errors = []
    for i in range(len(corners2)):
        detected = corners2[i, 0, :]
        projected = projected_points[i, 0, :]
        error = np.linalg.norm(detected - projected)
        reprojection_errors.append(error)

    # 3D復元による距離精度評価
    true_distances = []
    measured_distances_3d_reconstructed = []

    # チェッカーボードのパターンサイズ
    width, height = 5, 4  # checker_pattern = (5, 4)
    corners_2d = corners2.reshape(-1, 2)

    print(f"チェッカーボード構造: {width}×{height} = {width*height}点")
    print(f"検出された点数: {len(corners_2d)}")

    # 水平方向の隣接点間距離（各行内での隣接点）
    horizontal_pairs = 0

    for i in range(height):  # 各行について
        for j in range(width-1):  # 行内の隣接ペア
            idx1 = i * width + j
            idx2 = i * width + j + 1

            if idx1 < len(corners_2d) and idx2 < len(corners_2d):
                # 検出点から3D復元
                try:
                    measured_distance_3d = calculate_measured_3d_distance(
                        corners2, rvec, tvec, intrinsic_mat, distortion, idx1, idx2
                    )
                except:
                    measured_distance_3d = np.nan

                # 理論値（35mm）との比較
                true_distances.append(square_size)
                measured_distances_3d_reconstructed.append(measured_distance_3d)
                horizontal_pairs += 1

                # デバッグ情報
                if horizontal_pairs <= 3:
                    error_3d_reconstructed = abs(measured_distance_3d - square_size) if not np.isnan(measured_distance_3d) else np.nan

                    print(f"  水平ペア {idx1}-{idx2}:")
                    print(f"    3D復元: {measured_distance_3d:.1f}mm (誤差: {error_3d_reconstructed:.2f}mm)")
                    print(f"    理論値: {square_size}mm")

    # 垂直方向の隣接点間距離（列内での隣接点）
    vertical_pairs = 0

    for i in range(height-1):  # 各行について（最後の行以外）
        for j in range(width):  # 各列について
            idx1 = i * width + j
            idx2 = (i+1) * width + j

            if idx1 < len(corners_2d) and idx2 < len(corners_2d):
                # 検出点から3D復元
                try:
                    measured_distance_3d = calculate_measured_3d_distance(
                        corners2, rvec, tvec, intrinsic_mat, distortion, idx1, idx2
                    )
                except:
                    measured_distance_3d = np.nan

                # 理論値（35mm）との比較
                true_distances.append(square_size)
                measured_distances_3d_reconstructed.append(measured_distance_3d)
                vertical_pairs += 1

                # デバッグ情報
                if vertical_pairs <= 3:
                    error_3d_reconstructed = abs(measured_distance_3d - square_size) if not np.isnan(measured_distance_3d) else np.nan

                    print(f"  垂直ペア {idx1}-{idx2}:")
                    print(f"    3D復元: {measured_distance_3d:.1f}mm (誤差: {error_3d_reconstructed:.2f}mm)")
                    print(f"    理論値: {square_size}mm")

    print(f"測定ペア数: 水平{horizontal_pairs}個 + 垂直{vertical_pairs}個 = 計{horizontal_pairs + vertical_pairs}個")

    # 統計計算
    if len(true_distances) > 0:
        # 3D復元の統計（NaNを除外）
        true_distances_array = np.array(true_distances)
        measured_distances_3d_array = np.array(measured_distances_3d_reconstructed)
        valid_mask = ~np.isnan(measured_distances_3d_array)

        if np.any(valid_mask):
            distance_errors_3d = np.abs(measured_distances_3d_array[valid_mask] - true_distances_array[valid_mask])
            relative_errors_3d = distance_errors_3d / true_distances_array[valid_mask] * 100;

            metrics_3d = {
                'mean_error': float(np.mean(distance_errors_3d)),
                'max_error': float(np.max(distance_errors_3d)),
                'std_error': float(np.std(distance_errors_3d)),
                'relative_error_percent': float(np.mean(relative_errors_3d)),
                'horizontal_pairs': horizontal_pairs,
                'vertical_pairs': vertical_pairs
            }
        else:
            metrics_3d = None

        metrics = {
            'reprojection_error_pixels': {
                'mean': float(np.mean(reprojection_errors)),
                'max': float(np.max(reprojection_errors)),
                'std': float(np.std(reprojection_errors))
            },
            'distance_accuracy_3d_reconstructed': metrics_3d,
            'depth_estimate_mm': float(np.linalg.norm(tvec)),
            'num_distance_measurements': len(true_distances)
        }

        # 統計の表示
        print(f"\n距離精度統計:")
        if metrics_3d:
            print(f"3D復元による計算:")
            print(f"  平均誤差: {metrics_3d['mean_error']:.3f} mm")
            print(f"  最大誤差: {metrics_3d['max_error']:.3f} mm")
            print(f"  相対誤差: {metrics_3d['relative_error_percent']:.2f} %")
        else:
            print(f"3D復元による計算: 失敗")

    else:
        metrics = {
            'reprojection_error_pixels': {
                'mean': float(np.mean(reprojection_errors)),
                'max': float(np.max(reprojection_errors)),
                'std': float(np.std(reprojection_errors))
            },
            'distance_accuracy_3d_reconstructed': None,
            'depth_estimate_mm': float(np.linalg.norm(tvec)),
            'num_distance_measurements': 0
        }

    return metrics

# calculate_extrinsics関数内で使用
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
    accuracy_metrics_all = []

    print(f"\n{rets}個の解が見つかりました。各解の精度を計算中...")

    for i in range(rets):
        rvec = rvecs[i]
        tvec = tvecs[i]

        # 包括的な精度評価（3D精度指標を含む）
        accuracy_metrics = calculate_3d_accuracy_metrics(
            objp, corners2, rvec, tvec, intrinsic_mat, distortion, square_size
        )

        accuracy_metrics_all.append(accuracy_metrics)

        # 結果表示
        print(f"\n解 {i}:")
        print(f"  再投影誤差: {accuracy_metrics['reprojection_error_pixels']['mean']:.3f} pixels (max: {accuracy_metrics['reprojection_error_pixels']['max']:.3f})")

        if accuracy_metrics['distance_accuracy_3d_reconstructed']:
            print(f"  距離精度(3D復元): {accuracy_metrics['distance_accuracy_3d_reconstructed']['mean_error']:.2f} mm (max: {accuracy_metrics['distance_accuracy_3d_reconstructed']['max_error']:.2f} mm)")
            print(f"  相対誤差(3D復元): {accuracy_metrics['distance_accuracy_3d_reconstructed']['relative_error_percent']:.2f} %")
            print(f"  測定点数: {accuracy_metrics['num_distance_measurements']}")
        else:
            print(f"  距離精度: 計算不可")

        print(f"  推定深度: {accuracy_metrics['depth_estimate_mm']:.1f} mm")

        # 回転行列を計算
        R_worldFromCamera = cv2.Rodrigues(rvec)[0]

        # カメラパラメータのコピーを作成
        camera_params_copy = copy.deepcopy(camera_params)
        camera_params_copy['rotation'] = R_worldFromCamera.tolist()
        camera_params_copy['translation'] = tvec.tolist()
        camera_params_copy['rotation_EulerAngles'] = rvec.tolist()
        camera_params_copy['accuracy_metrics'] = accuracy_metrics

        camera_params_solutions.append(camera_params_copy)

        if save_images:
            # 再投影点を計算
            projected_points, _ = cv2.projectPoints(objp, rvec, tvec, intrinsic_mat, distortion)

            # 再投影点を描画した画像を作成
            image_with_reprojection = image_with_corners.copy()

            # 検出されたコーナーを緑で描画
            for j in range(corners2.shape[0]):
                pt = corners2[j, 0, :].astype(int)
                cv2.circle(image_with_reprojection, tuple(pt), 3, (0, 255, 0), -1)  # 緑色

            # 再投影されたコーナーを赤で描画
            for j in range(projected_points.shape[0]):
                pt = projected_points[j, 0, :].astype(int)
                cv2.circle(image_with_reprojection, tuple(pt), 2, (0, 0, 255), -1)  # 赤色

            # エラーが大きい点を強調
            reprojection_errors = []
            for j in range(len(corners2)):
                detected = corners2[j, 0, :]
                projected = projected_points[j, 0, :]
                error = np.linalg.norm(detected - projected)
                reprojection_errors.append(error)

            mean_error = np.mean(reprojection_errors)
            error_threshold = mean_error + np.std(reprojection_errors)
            for j, error in enumerate(reprojection_errors):
                if error > error_threshold:
                    pt = corners2[j, 0, :].astype(int)
                    cv2.circle(image_with_reprojection, tuple(pt), 5, (0, 0, 255), 2)  # 赤い輪

            # 座標軸を描画
            image_with_axes = cv2.drawFrameAxes(
                image_with_reprojection, intrinsic_mat, distortion,
                rvec, tvec, square_size * 3, 4
            )

            # 精度情報をテキストで追加
            text_lines = [
                f"Solution {i}",
                f"Reproj Error: {accuracy_metrics['reprojection_error_pixels']['mean']:.3f} px",
                f"Depth: {accuracy_metrics['depth_estimate_mm']:.1f} mm"
            ]

            if accuracy_metrics['distance_accuracy_3d_reconstructed']:
                text_lines.append(f"Distance Error: {accuracy_metrics['distance_accuracy_3d_reconstructed']['mean_error']:.2f} mm")
                text_lines.append(f"Relative Error: {accuracy_metrics['distance_accuracy_3d_reconstructed']['relative_error_percent']:.1f}%")

            y_offset = 30
            for line in text_lines:
                cv2.putText(image_with_axes, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 25

            # トリミング
            ht, wd = image.shape[:2]
            buffer_val = 0.05 * np.mean([ht, wd])

            top_edge = int(np.max([np.squeeze(np.min(projected_points, axis=0))[1] - buffer_val, 0]))
            left_edge = int(np.max([np.squeeze(np.min(projected_points, axis=0))[0] - buffer_val, 0]))
            bottom_edge = int(np.min([np.squeeze(np.max(projected_points, axis=0))[1] + buffer_val, ht]))
            right_edge = int(np.min([np.squeeze(np.max(projected_points, axis=0))[0] + buffer_val, wd]))

            image_cropped = image_with_axes[top_edge:bottom_edge, left_edge:right_edge, :]

            # 画像を保存（距離誤差も含むファイル名）
            reproj_error = accuracy_metrics['reprojection_error_pixels']['mean']
            if accuracy_metrics['distance_accuracy_3d_reconstructed']:
                dist_error = accuracy_metrics['distance_accuracy_3d_reconstructed']['mean_error']
                save_path = output_dir / f"extrinsic_calibration_solution_{i}_reproj{reproj_error:.3f}px_dist{dist_error:.2f}mm.jpg"
            else:
                save_path = output_dir / f"extrinsic_calibration_solution_{i}_reproj{reproj_error:.3f}px.jpg"

            cv2.imwrite(str(save_path), image_cropped)
            print(f"外部キャリブレーション画像を保存: {save_path}")

            # if visualize:
            #     cv2.imshow(f'Extrinsic Calibration - Solution {i}', image_cropped)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

    # 最も良い解を選択（再投影誤差で判定）
    best_solution_idx = np.argmin([metrics['reprojection_error_pixels']['mean'] for metrics in accuracy_metrics_all])

    if useSecondExtrinsicsSolution and rets > 1:
        solution_to_use = 1
        print(f"\n手動設定により解 {solution_to_use} を選択しました")
    else:
        solution_to_use = best_solution_idx
        print(f"\n最小再投影誤差により解 {solution_to_use} を自動選択しました")

    selected_solution = camera_params_solutions[solution_to_use]
    selected_metrics = accuracy_metrics_all[solution_to_use]

    print(f"\n選択された解の詳細:")
    print(f"  平均再投影誤差: {selected_metrics['reprojection_error_pixels']['mean']:.3f} pixels")
    print(f"  最大再投影誤差: {selected_metrics['reprojection_error_pixels']['max']:.3f} pixels")
    if selected_metrics['distance_accuracy_3d_reconstructed']:
        print(f"  平均距離誤差(3D復元): {selected_metrics['distance_accuracy_3d_reconstructed']['mean_error']:.3f} mm")
        print(f"  最大距離誤差(3D復元): {selected_metrics['distance_accuracy_3d_reconstructed']['max_error']:.3f} mm")
        print(f"  相対距離誤差(3D復元): {selected_metrics['distance_accuracy_3d_reconstructed']['relative_error_percent']:.2f} %")
    print(f"  推定深度: {selected_metrics['depth_estimate_mm']:.1f} mm")
    print(f"  回転ベクトル: {selected_solution['rotation_EulerAngles']}")
    print(f"  並進ベクトル: {selected_solution['translation']}")

    return selected_solution, camera_params_solutions, accuracy_metrics_all

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

        selected_solution, all_solutions, accuracy_metrics = result

        # 結果を保存
        output_file = ext_dir / direction / "camera_params_with_ext_OC.json"

        # 完全なカメラパラメータを作成
        complete_params = {
            **camera_params,  # 内部パラメータを含む
            'extrinsics': {
                'rotation_matrix': selected_solution['rotation'],
                'translation_vector': selected_solution['translation'],
                'rotation_euler_angles': selected_solution['rotation_EulerAngles'],
                'accuracy_metrics': selected_solution['accuracy_metrics']
            },
            'reference_image': str(reference_image_path.name),
            'checkerboard_pattern': checker_pattern,
            'square_size_mm': square_size,
            'num_solutions_found': len(all_solutions),
            'selected_solution_index': 0
        }

        # 全ての解も保存
        complete_params['all_extrinsic_solutions'] = []
        for i, (solution, metrics) in enumerate(zip(all_solutions, accuracy_metrics)):
            complete_params['all_extrinsic_solutions'].append({
                'solution_index': i,
                'rotation_matrix': solution['rotation'],
                'translation_vector': solution['translation'],
                'rotation_euler_angles': solution['rotation_EulerAngles'],
                'accuracy_metrics': metrics
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

        # 精度サマリー
        metrics = selected_solution['accuracy_metrics']
        print(f"\n--- 精度評価サマリー ---")
        print(f"再投影誤差:")
        print(f"  平均: {metrics['reprojection_error_pixels']['mean']:.3f} pixels")
        print(f"  最大: {metrics['reprojection_error_pixels']['max']:.3f} pixels")
        print(f"  標準偏差: {metrics['reprojection_error_pixels']['std']:.3f} pixels")

        if metrics['distance_accuracy_3d_reconstructed']:
            print(f"3D距離精度(3D復元による計算):")
            print(f"  平均誤差: {metrics['distance_accuracy_3d_reconstructed']['mean_error']:.3f} mm")
            print(f"  最大誤差: {metrics['distance_accuracy_3d_reconstructed']['max_error']:.3f} mm")
            print(f"  相対誤差: {metrics['distance_accuracy_3d_reconstructed']['relative_error_percent']:.2f} %")

        print(f"その他:")
        print(f"  推定深度: {metrics['depth_estimate_mm']:.1f} mm")

        all_results[direction] = {
            'solution': selected_solution,
            'accuracy_metrics': metrics
        }

    print("\n" + "="*80)
    print("外部パラメータキャリブレーション完了！")
    print("="*80)

    # 全体のサマリー
    if all_results:
        print("\n--- 全体サマリー ---")
        for direction, result in all_results.items():
            metrics = result['accuracy_metrics']
            reproj_error = metrics['reprojection_error_pixels']['mean']
            if metrics['distance_accuracy_3d_reconstructed']:
                dist_error = metrics['distance_accuracy_3d_reconstructed']['mean_error']
                rel_error = metrics['distance_accuracy_3d_reconstructed']['relative_error_percent']
                print(f"カメラ {direction}: 再投影誤差 {reproj_error:.3f} px, 距離誤差(3D復元) {dist_error:.2f} mm ({rel_error:.1f}%)")
            else:
                print(f"カメラ {direction}: 再投影誤差 {reproj_error:.3f} px, 距離精度 計算不可")

if __name__ == '__main__':
    main()