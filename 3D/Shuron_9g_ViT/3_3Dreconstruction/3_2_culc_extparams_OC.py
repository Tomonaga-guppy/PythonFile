"""
- sub*/cali/{fl,fr,sagi} にあるキャリブレーション用画像（チェッカーボードが映る画像）から、外部パラメータ（回転・並進）を推定
- 内部パラメータ（intrinsics / distortion）は既存の保存場所から読み込み
- 推定結果（json）と可視化画像（解ごとの reprojection 等）は sub*/cali 配下にまとめて出力

入力フォルダ構成
root_dir/
    sub*/cali/
        fl/*.png (9号館実験の場合はfr-fl用に1枚)
        fr/*.png (9号館実験の場合はfr-fl用途fr-sagi用で2枚)
        sagi/*.png (9号館実験の場合はfr-sagi用に1枚)

内部パラメータ
int_dir/
    fl/camera_params.json
    fr/camera_params.json
    sagi/camera_params.json
    

処理内容
- frame0 からチェッカーボード角を検出（サブピクセル補正あり）
- solvePnPGeneric（IPPE）で外部パラメータ候補を推定（失敗時は solvePnPRansac）
- 各解について以下を評価・保存
    - 再投影誤差（平均/最大/分散）
    - 3D復元による隣接点距離誤差（mm, 相対誤差%）
    - 推定深度（|t|）
    - 角・再投影点・軸の描画、必要に応じてトリミングした画像保存
- 再投影誤差が最小の解を自動選択（useSecondExtrinsicsSolution=True の場合は2解目を選択）

出力
sub*/cali/extparams/{fl,fr,sagi}/
    camera_params_with_ext_OC_<動画stem>.json         # 動画ごとに1つ
    <動画stem>/origin.png                             # frame0
    <動画stem>/extrinsic_calibration_solution_...jpg  # 解ごとの可視化
"""


import cv2
import numpy as np
import json
from pathlib import Path
import copy
import matplotlib.pyplot as plt  # 元3_2にあるので保持

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

        return ret, corners2, objp, image_with_corners
    else:
        print("チェッカーボードが検出できませんでした。")
        return ret, None, None, image

def calculate_reprojection_error(objp, corners2, rvec, tvec, intrinsic_mat, distortion):
    """
    再投影誤差を計算する
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
    """
    point1_3d = objp[idx1]  # 3D座標を直接使用
    point2_3d = objp[idx2]
    return np.linalg.norm(point2_3d - point1_3d)

def calculate_measured_3d_distance(corners2, rvec, tvec, intrinsic_mat, distortion, idx1, idx2):
    """
    検出された2D点から逆算した3D座標を使った距離計算
    """
    # 2D点を取得
    point1_2d = corners2[idx1, 0, :].reshape(1, 1, 2)
    point2_2d = corners2[idx2, 0, :].reshape(1, 1, 2)

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
    depth1 = -t_inv[2] / (R_inv @ point1_norm)[2]
    depth2 = -t_inv[2] / (R_inv @ point2_norm)[2]

    # 3D座標を計算
    point1_3d = depth1 * (R_inv @ point1_norm) + t_inv
    point2_3d = depth2 * (R_inv @ point2_norm) + t_inv

    return np.linalg.norm(point2_3d - point1_3d)

def calculate_3d_accuracy_metrics(objp, corners2, rvec, tvec, intrinsic_mat, distortion, square_size):
    """
    3次元空間での精度指標を計算する（3D復元のみ）
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

    # チェッカーボードのパターンサイズ（元3_2のまま）
    width, height = 5, 4  # checker_pattern = (5, 4)
    corners_2d = corners2.reshape(-1, 2)

    print(f"チェッカーボード構造: {width}×{height} = {width*height}点")
    print(f"検出された点数: {len(corners_2d)}")

    # 水平方向の隣接点間距離
    horizontal_pairs = 0
    for i in range(height):
        for j in range(width-1):
            idx1 = i * width + j
            idx2 = i * width + j + 1
            if idx1 < len(corners_2d) and idx2 < len(corners_2d):
                try:
                    measured_distance_3d = calculate_measured_3d_distance(
                        corners2, rvec, tvec, intrinsic_mat, distortion, idx1, idx2
                    )
                except:
                    measured_distance_3d = np.nan

                true_distances.append(square_size)
                measured_distances_3d_reconstructed.append(measured_distance_3d)
                horizontal_pairs += 1

                if horizontal_pairs <= 3:
                    error_3d_reconstructed = abs(measured_distance_3d - square_size) if not np.isnan(measured_distance_3d) else np.nan
                    print(f"  水平ペア {idx1}-{idx2}:")
                    print(f"    3D復元: {measured_distance_3d:.1f}mm (誤差: {error_3d_reconstructed:.2f}mm)")
                    print(f"    理論値: {square_size}mm")

    # 垂直方向の隣接点間距離
    vertical_pairs = 0
    for i in range(height-1):
        for j in range(width):
            idx1 = i * width + j
            idx2 = (i+1) * width + j
            if idx1 < len(corners_2d) and idx2 < len(corners_2d):
                try:
                    measured_distance_3d = calculate_measured_3d_distance(
                        corners2, rvec, tvec, intrinsic_mat, distortion, idx1, idx2
                    )
                except:
                    measured_distance_3d = np.nan

                true_distances.append(square_size)
                measured_distances_3d_reconstructed.append(measured_distance_3d)
                vertical_pairs += 1

                if vertical_pairs <= 3:
                    error_3d_reconstructed = abs(measured_distance_3d - square_size) if not np.isnan(measured_distance_3d) else np.nan
                    print(f"  垂直ペア {idx1}-{idx2}:")
                    print(f"    3D復元: {measured_distance_3d:.1f}mm (誤差: {error_3d_reconstructed:.2f}mm)")
                    print(f"    理論値: {square_size}mm")

    print(f"測定ペア数: 水平{horizontal_pairs}個 + 垂直{vertical_pairs}個 = 計{horizontal_pairs + vertical_pairs}個")

    # 統計計算
    if len(true_distances) > 0:
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

# calculate_extrinsics関数内で使用（元3_2のまま）
def calculate_extrinsics(image_path, camera_params, checker_pattern, square_size,
                        imageUpsampleFactor=2, useSecondExtrinsicsSolution=False,
                        visualize=False, save_images=True):
    """
    単一画像から外部パラメータを計算する
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"画像を読み込めません: {image_path}")
        return None

    ret, corners2, objp, image_with_corners = detect_chessboard_corners(
        image, checker_pattern, square_size, imageUpsampleFactor, visualize
    )
    if not ret:
        print("チェッカーボードの検出に失敗しました。")
        return None

    intrinsic_mat = np.array(camera_params['intrinsics'])
    distortion = np.array(camera_params['distortion'])

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

    solution_to_use = 1 if useSecondExtrinsicsSolution and rets > 1 else 0

    output_dir = image_path.parent

    camera_params_solutions = []
    accuracy_metrics_all = []

    print(f"\n{rets}個の解が見つかりました。各解の精度を計算中...")

    for i in range(rets):
        rvec = rvecs[i]
        tvec = tvecs[i]

        accuracy_metrics = calculate_3d_accuracy_metrics(
            objp, corners2, rvec, tvec, intrinsic_mat, distortion, square_size
        )

        accuracy_metrics_all.append(accuracy_metrics)

        print(f"\n解 {i}:")
        print(f"  再投影誤差: {accuracy_metrics['reprojection_error_pixels']['mean']:.3f} pixels (max: {accuracy_metrics['reprojection_error_pixels']['max']:.3f})")

        if accuracy_metrics['distance_accuracy_3d_reconstructed']:
            print(f"  距離精度(3D復元): {accuracy_metrics['distance_accuracy_3d_reconstructed']['mean_error']:.2f} mm (max: {accuracy_metrics['distance_accuracy_3d_reconstructed']['max_error']:.2f} mm)")
            print(f"  相対誤差(3D復元): {accuracy_metrics['distance_accuracy_3d_reconstructed']['relative_error_percent']:.2f} %")
            print(f"  測定点数: {accuracy_metrics['num_distance_measurements']}")
        else:
            print(f"  距離精度: 計算不可")

        print(f"  推定深度: {accuracy_metrics['depth_estimate_mm']:.1f} mm")

        R_worldFromCamera = cv2.Rodrigues(rvec)[0]

        camera_params_copy = copy.deepcopy(camera_params)
        camera_params_copy['rotation'] = R_worldFromCamera.tolist()
        camera_params_copy['translation'] = tvec.tolist()
        camera_params_copy['rotation_EulerAngles'] = rvec.tolist()
        camera_params_copy['accuracy_metrics'] = accuracy_metrics

        camera_params_solutions.append(camera_params_copy)

        if save_images:
            projected_points, _ = cv2.projectPoints(objp, rvec, tvec, intrinsic_mat, distortion)

            image_with_reprojection = image_with_corners.copy()

            for j in range(corners2.shape[0]):
                pt = corners2[j, 0, :].astype(int)
                cv2.circle(image_with_reprojection, tuple(pt), 3, (0, 255, 0), -1)

            for j in range(projected_points.shape[0]):
                pt = projected_points[j, 0, :].astype(int)
                cv2.circle(image_with_reprojection, tuple(pt), 2, (0, 0, 255), -1)

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
                    cv2.circle(image_with_reprojection, tuple(pt), 5, (0, 0, 255), 2)

            image_with_axes = cv2.drawFrameAxes(
                image_with_reprojection, intrinsic_mat, distortion,
                rvec, tvec, square_size * 3, 4
            )

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

            ht, wd = image.shape[:2]
            buffer_val = 0.05 * np.mean([ht, wd])

            top_edge = int(np.max([np.squeeze(np.min(projected_points, axis=0))[1] - buffer_val, 0]))
            left_edge = int(np.max([np.squeeze(np.min(projected_points, axis=0))[0] - buffer_val, 0]))
            bottom_edge = int(np.min([np.squeeze(np.max(projected_points, axis=0))[1] + buffer_val, ht]))
            right_edge = int(np.min([np.squeeze(np.max(projected_points, axis=0))[0] + buffer_val, wd]))

            image_cropped = image_with_axes[top_edge:bottom_edge, left_edge:right_edge, :]

            reproj_error = accuracy_metrics['reprojection_error_pixels']['mean']
            if accuracy_metrics['distance_accuracy_3d_reconstructed']:
                dist_error = accuracy_metrics['distance_accuracy_3d_reconstructed']['mean_error']
                save_path = output_dir / f"extrinsic_calibration_solution_{i}_reproj{reproj_error:.3f}px_dist{dist_error:.2f}mm.jpg"
            else:
                save_path = output_dir / f"extrinsic_calibration_solution_{i}_reproj{reproj_error:.3f}px.jpg"

            cv2.imwrite(str(save_path), image_cropped)
            print(f"外部キャリブレーション画像を保存: {save_path}")

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


# =========================================================
# 追加：MP4の先頭フレームを読む（変更点）
# =========================================================
def read_first_frame(mp4_path: Path):
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def natural_sort_key(p: Path):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


# =========================================================
# 追加：外部パラメータ（board->camera）を同次変換で扱う
# =========================================================
def rt_to_T(R, t):
    """R(3x3), t(3,) or (3,1) -> 4x4 homogeneous transform (board -> camera)."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_inv(T):
    """Invert homogeneous transform."""
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def extract_selected_extrinsic(selected_solution: dict):
    """selected_solution から R(3x3), t(3,) を取り出す（board->camera）。"""
    R = np.asarray(selected_solution['rotation'], dtype=float).reshape(3, 3)
    t = np.asarray(selected_solution['translation'], dtype=float).reshape(3)
    return R, t


def make_complete_params(camera_params, selected_solution, all_solutions, accuracy_metrics,
                         ext_img_name: str, frame0_name: str, checker_pattern, square_size):
    """保存用のjson dict を作る（元の形式を維持しつつ、動画ごとに分ける）。"""
    complete_params = {
        **camera_params,
        'extrinsics': {
            'rotation_matrix': selected_solution['rotation'],
            'translation_vector': selected_solution['translation'],
            'rotation_euler_angles': selected_solution['rotation_EulerAngles'],
            'accuracy_metrics': selected_solution['accuracy_metrics']
        },
        'source_image': str(ext_img_name),
        'reference_image': str(frame0_name),
        'checkerboard_pattern': checker_pattern,
        'square_size_mm': square_size,
        'num_solutions_found': len(all_solutions),
        'selected_solution_index': int(np.argmin([m['reprojection_error_pixels']['mean'] for m in accuracy_metrics])),
    }

    complete_params['all_extrinsic_solutions'] = []
    for i, (solution, metrics) in enumerate(zip(all_solutions, accuracy_metrics)):
        complete_params['all_extrinsic_solutions'].append({
            'solution_index': i,
            'rotation_matrix': solution['rotation'],
            'translation_vector': solution['translation'],
            'rotation_euler_angles': solution['rotation_EulerAngles'],
            'accuracy_metrics': metrics
        })
    return complete_params


def compute_sagi_aligned_to_flfr_pair(result_by_dir: dict):
    """sagi(B) を fl-fr のペア（A）にそろえた外部（board=A -> sagi）を作る。

    想定：
    - fl と fr は少なくとも 1 本ずつ（index 0 が fl-fr 用）
    - fl または fr のどちらかが 2 本あり、index 1 が (bridge)-(sagi) 用
    - sagi は 1 本（board=B）

    戻り値：
      (R_sagi_A, t_sagi_A, bridge_dir)
    """
    if 'sagi' not in result_by_dir or len(result_by_dir['sagi']) != 1:
        return None
    if 'fl' not in result_by_dir or 'fr' not in result_by_dir:
        return None
    if len(result_by_dir['fl']) < 1 or len(result_by_dir['fr']) < 1:
        return None

    bridge_dir = None
    fl_n = len(result_by_dir.get('fl', []))
    fr_n = len(result_by_dir.get('fr', []))
    # 動画本数で橋渡し方向を決める（2本ある方が bridge）
    if fl_n == 2 and fr_n == 1:
        bridge_dir = 'fl'
    elif fr_n == 2 and fl_n == 1:
        bridge_dir = 'fr'
    else:
        return None

    # A: fl-fr 用（index 0） / B: bridge-sagi 用（bridge index 1 と sagi index 0）
    sel_bridge_A = result_by_dir[bridge_dir][0]['selected_solution']
    sel_bridge_B = result_by_dir[bridge_dir][1]['selected_solution']
    sel_sagi_B = result_by_dir['sagi'][0]['selected_solution']

    R_bridge_A, t_bridge_A = extract_selected_extrinsic(sel_bridge_A)
    R_bridge_B, t_bridge_B = extract_selected_extrinsic(sel_bridge_B)
    R_sagi_B, t_sagi_B = extract_selected_extrinsic(sel_sagi_B)

    T_bridge_A = rt_to_T(R_bridge_A, t_bridge_A)   # board=A -> bridge cam
    T_bridge_B = rt_to_T(R_bridge_B, t_bridge_B)   # board=B -> bridge cam
    T_sagi_B = rt_to_T(R_sagi_B, t_sagi_B)         # board=B -> sagi cam

    # A_from_B = inv(T_bridge_A) * T_bridge_B
    T_A_from_B = T_inv(T_bridge_A) @ T_bridge_B
    T_B_from_A = T_inv(T_A_from_B)

    # sagi_A = sagi_B * (B_from_A)
    T_sagi_A = T_sagi_B @ T_B_from_A

    R_sagi_A = T_sagi_A[:3, :3]
    t_sagi_A = T_sagi_A[:3, 3]
    return R_sagi_A, t_sagi_A, bridge_dir
def main():
    """
    メイン処理（ここだけフォルダ構成に合わせて変更）
    """
    # --- パラメータ設定（必要ならここだけ変更） ---
    root_dir = Path(r"G:\gait_pattern\2025_shuron_BR9G")  # ★あなたの現状
    int_dir = Path(r"G:\gait_pattern\int_cali\9g_20250807_6x5")  # ★内部パラメータ位置は据え置き
    directions = ['fl', 'fr', 'sagi']
    checker_pattern = (5, 4)  # (width, height)
    square_size = 35  # mm

    # 出力は sub*/cali/extparams/{fl,fr,sagi}/{動画stem}/... にまとめる
    out_dirname = "extparams"

    print("外部パラメータキャリブレーション開始（MP4の1フレーム目を使用）")
    print(f"root_dir: {root_dir}")
    print(f"チェッカーボードパターン: {checker_pattern[0]}x{checker_pattern[1]}")
    print(f"正方形のサイズ: {square_size} mm")
    print(f"int_dir: {int_dir}")

    subs = sorted([p for p in root_dir.glob("sub*") if p.is_dir()], key=natural_sort_key)
    if not subs:
        print(f"sub* が見つかりません: {root_dir}")
        return

    for sub in subs:
        cali_dir = sub / "cali"
        if not cali_dir.exists():
            continue

        print("\n" + "="*80)
        print(f"[SUB] {sub.name}")
        print("="*80)

        # この sub 内での結果を保持（後で sagi 変換に使う）
        result_by_dir = {d: [] for d in directions}

        for direction in directions:
            ext_imgs_dir = cali_dir / direction
            ext_imgs = sorted(list(ext_imgs_dir.glob("*.png")), key=natural_sort_key)
            if not ext_imgs:
                print(f"[SKIP] {sub.name} {direction}: PNGがありません")
                continue

            # 内部パラメータ
            params_file = int_dir / direction / "camera_params.json"
            if not params_file.exists():
                print(f"エラー: 内部パラメータファイルが見つかりません: {params_file}")
                continue

            with open(params_file, 'r') as f:
                camera_params = json.load(f)

            print(f"\n[CAM] {direction} | img: {len(ext_imgs)} | intrinsics: {params_file}")

            # 出力ベース
            out_base = cali_dir / out_dirname / direction
            out_base.mkdir(parents=True, exist_ok=True)

            for vid_idx, ext_img_path in enumerate(ext_imgs):
                tag = f"{direction}_{vid_idx+1:02d}"  # stemは使わない
                print(f"\n  - {ext_img_path.name} -> frame0")
                
                frame0 = cv2.imread(str(ext_img_path))

                if frame0 is None:
                    print(f"    [ERROR] frame0 を読み出せません: {ext_img_path}")
                    continue

                # 動画ごとに専用フォルダ
                img_out_dir = out_base / tag
                img_out_dir.mkdir(parents=True, exist_ok=True)

                frame0_path = img_out_dir / "origin.png"
                cv2.imwrite(str(frame0_path), frame0)

                result = calculate_extrinsics(
                    frame0_path,
                    camera_params,
                    checker_pattern,
                    square_size,
                    imageUpsampleFactor=2,
                    useSecondExtrinsicsSolution=False,
                    visualize=True,
                    save_images=True
                )

                if result is None:
                    print(f"    [FAIL] 外部パラメータ計算に失敗: {ext_img_path.name}")
                    continue

                selected_solution, all_solutions, accuracy_metrics = result

                # jsonは動画ごとに保存（上書き防止）
                output_file = out_base / f"camera_params_with_ext_OC_{tag}.json"
                complete_params = make_complete_params(
                    camera_params,
                    selected_solution,
                    all_solutions,
                    accuracy_metrics,
                    ext_img_name=str(ext_img_path.name),
                    frame0_name=str(frame0_path.name),
                    checker_pattern=checker_pattern,
                    square_size=square_size,
                )
                with open(output_file, 'w') as f:
                    json.dump(complete_params, f, indent=4)

                print(f"    [OK] 保存: {output_file}")

                # 後で橋渡し変換に使うため保持（名前順のインデックスを維持）
                result_by_dir[direction].append({
                    'ext_img_path': ext_img_path,
                    'tag': tag,
                    'output_file': output_file,
                    'selected_solution': selected_solution,
                    'camera_params': camera_params,
                    'all_solutions': all_solutions,
                    'accuracy_metrics': accuracy_metrics,
                })

        # -------------------------------------------------
        # 追加：sagi を fl-fr の結果（ペア＝index0）に変換して保存
        # 判定ルール：fl または fr のどちらかが 2 本（=橋渡し）。
        #            名前順で 1本目が fl-fr 用、2本目が (bridge)-sagi 用。
        # -------------------------------------------------
        aligned = compute_sagi_aligned_to_flfr_pair(result_by_dir)
        if aligned is None:
            print("\n[INFO] sagi の橋渡し条件が満たされないため、変換保存はスキップします")
            continue

        R_sagi_A, t_sagi_A, bridge_dir = aligned
        print(f"\n[INFO] sagi を fl-fr ペアに変換します（橋渡し: {bridge_dir}）")

        # 元の sagi json（B基準）も残しつつ、A基準版を別ファイルで保存
        sagi_item = result_by_dir['sagi'][0]
        sagi_out_base = (sub / "cali" / out_dirname / "sagi")
        sagi_out_base.mkdir(parents=True, exist_ok=True)

        tag_sagi = sagi_item['tag']
        output_file_aligned = sagi_out_base / f"camera_params_with_ext_OC_{tag_sagi}_aligned_to_flfr.json"

        # 既存の complete_params をベースに差し替え
        base_params = make_complete_params(
            sagi_item['camera_params'],
            sagi_item['selected_solution'],
            sagi_item['all_solutions'],
            sagi_item['accuracy_metrics'],
            ext_img_name=str(sagi_item['ext_img_path'].name),
            frame0_name="origin.png",
            checker_pattern=checker_pattern,
            square_size=square_size,
        )

        base_params['extrinsics_aligned_to_flfr_pair'] = {
            'rotation_matrix': R_sagi_A.tolist(),
            'translation_vector': t_sagi_A.reshape(3, 1).tolist(),
            'note': 'Aligned using bridge camera that was recorded twice; index0 is fl-fr board(A), index1 is bridge-sagi board(B).'
        }
        base_params['alignment_info'] = {
            'bridge_camera': bridge_dir,
            'pair_board': 'A (fl-fr images index0)',
            'sagi_board': 'B (bridge image index1 + sagi image index0)',
            'pair_images': {
                'fl': result_by_dir['fl'][0]['ext_img_path'].name if len(result_by_dir['fl']) > 0 else None,
                'fr': result_by_dir['fr'][0]['ext_img_path'].name if len(result_by_dir['fr']) > 0 else None,
            },
            'bridge_images': {
                bridge_dir: {
                    'index0': result_by_dir[bridge_dir][0]['ext_img_path'].name,
                    'index1': result_by_dir[bridge_dir][1]['ext_img_path'].name,
                }
            },
            'sagi_image': result_by_dir['sagi'][0]['ext_img_path'].name,
        }

        with open(output_file_aligned, 'w') as f:
            json.dump(base_params, f, indent=4)
        print(f"[OK] 変換版 sagi を保存: {output_file_aligned}")

    print("\n" + "="*80)
    print("外部パラメータキャリブレーション完了！")
    print("="*80)

if __name__ == '__main__':
    main()