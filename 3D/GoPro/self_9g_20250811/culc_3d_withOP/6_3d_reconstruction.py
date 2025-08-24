import cv2
import numpy as np
import json
from pathlib import Path
import copy
import glob

def rotate_coordinates_x_axis(points_3d, angle_degrees=180):
    """
    3D座標をX軸について回転する
    
    Args:
        points_3d: 3D座標 (N, 3) または (3, N)
        angle_degrees: 回転角度（度）
    
    Returns:
        rotated_points: 回転後の3D座標
    """
    # 角度をラジアンに変換
    angle_rad = np.radians(angle_degrees)
    
    # X軸周りの回転行列
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # 入力の形状を確認
    original_shape = points_3d.shape
    
    if len(original_shape) == 2:
        if original_shape[1] == 3:
            # (N, 3) の場合
            rotated_points = np.dot(points_3d, rotation_matrix.T)
        elif original_shape[0] == 3:
            # (3, N) の場合
            rotated_points = np.dot(rotation_matrix, points_3d)
        else:
            raise ValueError(f"Invalid shape: {original_shape}. Expected (N, 3) or (3, N)")
    else:
        raise ValueError(f"Invalid shape: {original_shape}. Expected 2D array")
    
    print(f"X軸について{angle_degrees}度回転を適用しました")
    return rotated_points

def load_camera_parameters(params_file):
    """
    カメラパラメータを読み込む
    """
    with open(params_file, 'r') as f:
        params = json.load(f)
    return params

def p2e(projective):
    """
    projective座標からeuclidean座標に変換
    """
    return (projective / projective[-1, :])[0:-1, :]

def e2p(euclidean):
    """
    euclidean座標からprojective座標に変換
    """
    return np.vstack((euclidean, np.ones((1, euclidean.shape[1]))))

def create_projection_matrix(camera_params):
    """
    カメラパラメータからプロジェクション行列を作成
    """
    K = np.array(camera_params['intrinsics'])
    R = np.array(camera_params['extrinsics']['rotation_matrix'])
    t = np.array(camera_params['extrinsics']['translation_vector']).reshape(3, 1)
    
    P = K @ np.hstack([R, t])
    return P

def construct_D_block(P, uv, w=1):
    """
    三角測量用のD行列のブロックを構築
    utilsCameraPy3.pyの_construct_D_blockと同じ実装
    """
    return w * np.vstack((
        uv[0] * P[2, :] - P[0, :],
        uv[1] * P[2, :] - P[1, :]
    ))

def weighted_linear_triangulation(projection_matrices, correspondences, weights=None):
    """
    重み付き線形三角測量を実行
    utilsCameraPy3.pyのnview_linear_triangulationを参考にした実装
    
    Args:
        projection_matrices: プロジェクション行列のリスト
        correspondences: 対応点座標 (2, n_cameras)
        weights: 重み（信頼度）のリスト
    
    Returns:
        point_3d: 3D座標 (3, 1)
        confidence: 信頼度
    """
    n_cameras = len(projection_matrices)
    
    # 重みの処理
    if weights is None:
        w = np.ones(n_cameras)
        weights = [1 for _ in range(n_cameras)]
    else:
        # NaNを0.5に変換
        w = [np.nan_to_num(wi, nan=0.5) for wi in weights]
    
    # D行列を構築
    D = np.zeros((n_cameras * 2, 4))
    for cam_idx in range(n_cameras):
        P = projection_matrices[cam_idx]
        uv = correspondences[:, cam_idx]
        D[cam_idx * 2:cam_idx * 2 + 2, :] = construct_D_block(P, uv, w=w[cam_idx])
    
    # 最小二乗法で解く
    Q = D.T.dot(D)
    u, s, vh = np.linalg.svd(Q)
    point_3d = p2e(u[:, -1, np.newaxis])
    
    # 信頼度の計算
    weight_array = np.asarray(weights)
    if np.count_nonzero(weights) < 2:
        # 2台未満のカメラの信頼度がある場合は0を返す
        point_3d = np.zeros_like(point_3d)
        conf = 0
    else:
        # 全てがNaNの場合（全カメラがスプライン補間）
        if all(np.isnan(weight_array[weight_array != 0])):
            conf = 0.5  # NaNは0.5の信頼度
        else:
            conf = np.nanmean(weight_array[weight_array != 0])
    
    return point_3d, conf

def triangulate_multiple_points(projection_matrices, correspondences_list, weights_list=None):
    """
    複数の点について重み付き線形三角測量を実行
    
    Args:
        projection_matrices: プロジェクション行列のリスト
        correspondences_list: 対応点座標のリスト [point1(2, n_cameras), point2(2, n_cameras), ...]
        weights_list: 重みのリスト [weights1, weights2, ...]
    
    Returns:
        points_3d: 3D座標 (3, n_points)
        confidences: 信頼度 (n_points,)
    """
    n_points = len(correspondences_list)
    points_3d = np.zeros((3, n_points))
    confidences = np.zeros(n_points)
    
    for i, correspondences in enumerate(correspondences_list):
        if weights_list is not None:
            weights = weights_list[i]
        else:
            weights = None
        
        point_3d, conf = weighted_linear_triangulation(
            projection_matrices, correspondences, weights
        )
        
        points_3d[:, i] = point_3d.flatten()
        confidences[i] = conf
    
    return points_3d, confidences

def triangulate_points_weighted(camera_params_list, points_2d_list, confidences_list=None):
    """
    重み付き三角測量を使用して2D点から3D点を計算
    
    Args:
        camera_params_list: カメラパラメータのリスト
        points_2d_list: 各カメラの2D点座標のリスト
        confidences_list: 各カメラの信頼度のリスト
    
    Returns:
        points_3d: 3D座標 (N, 3)
        confidences: 信頼度 (N,)
    """
    if len(camera_params_list) < 2:
        raise ValueError("少なくとも2台のカメラが必要です")
    
    # プロジェクション行列を作成
    projection_matrices = []
    for camera_params in camera_params_list:
        P = create_projection_matrix(camera_params)
        projection_matrices.append(P)
    
    # 対応点を準備
    n_points = len(points_2d_list[0])
    correspondences_list = []
    weights_list = []
    
    for point_idx in range(n_points):
        # 各カメラからの対応点を収集
        correspondences = np.zeros((2, len(camera_params_list)))
        weights = []
        
        for cam_idx in range(len(camera_params_list)):
            correspondences[:, cam_idx] = points_2d_list[cam_idx][point_idx]
            
            if confidences_list is not None:
                weights.append(confidences_list[cam_idx][point_idx])
            else:
                weights.append(1.0)
        
        correspondences_list.append(correspondences)
        weights_list.append(weights)
    
    # 三角測量を実行
    try:
        points_3d, confidences = triangulate_multiple_points(
            projection_matrices, correspondences_list, weights_list
        )
        
        # 3D座標を取得（形状: (3, N) -> (N, 3)）
        points_3d_array = points_3d.T
        
        # X軸について180度回転を適用
        points_3d_rotated = rotate_coordinates_x_axis(points_3d_array, angle_degrees=180)
        
        return points_3d_rotated, confidences
        
    except Exception as e:
        print(f"重み付き三角測量でエラーが発生: {e}")
        raise

def triangulate_points_undistorted(camera_params_list, points_2d_list):
    """
    既に歪み補正済みの2D点から3D点を三角測量する（OpenCV使用）
    
    Args:
        camera_params_list: カメラパラメータのリスト
        points_2d_list: 各カメラの歪み補正済み2D点座標のリスト
    
    Returns:
        points_3d: 3D座標 (N, 3)
    """
    if len(camera_params_list) < 2:
        raise ValueError("少なくとも2台のカメラが必要です")
    
    # 最初の2台のカメラで三角測量
    cam1_params = camera_params_list[0]
    cam2_params = camera_params_list[1]
    
    # 内部パラメータ（歪み補正済み画像用）
    K1 = np.array(cam1_params['intrinsics'])
    K2 = np.array(cam2_params['intrinsics'])
    
    # 外部パラメータ
    R1 = np.array(cam1_params['extrinsics']['rotation_matrix'])
    t1 = np.array(cam1_params['extrinsics']['translation_vector']).reshape(3, 1)
    R2 = np.array(cam2_params['extrinsics']['rotation_matrix'])
    t2 = np.array(cam2_params['extrinsics']['translation_vector']).reshape(3, 1)
    
    # プロジェクション行列を計算
    P1 = K1 @ np.hstack([R1, t1])
    P2 = K2 @ np.hstack([R2, t2])
    
    # 既に歪み補正済みなので、そのまま使用
    points1 = points_2d_list[0]
    points2 = points_2d_list[1]
    
    # 三角測量
    try:
        points_4d = cv2.triangulatePoints(
            P1, P2, 
            points1.T, 
            points2.T
        )
        
        # 同次座標から3D座標に変換
        points_3d = points_4d[:3] / points_4d[3]
        points_3d_array = points_3d.T  # (N, 3) の形状に変換
        
        # X軸について180度回転を適用
        points_3d_rotated = rotate_coordinates_x_axis(points_3d_array, angle_degrees=180)
        
        return points_3d_rotated
        
    except Exception as e:
        print(f"三角測量でエラーが発生: {e}")
        raise

def load_openpose_json(json_file_path):
    """
    OpenPoseのJSON結果から2Dキーポイントを読み込む
    
    Args:
        json_file_path: OpenPoseのJSONファイルパス
    
    Returns:
        keypoints_2d: 2Dキーポイント座標 (N, 2)
        confidence: 信頼度 (N,)
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    if not data['people']:
        return None, None
    
    # 最初の人物のキーポイントを取得
    person_data = data['people'][0]
    pose_keypoints = person_data['pose_keypoints_2d']
    
    # [x, y, confidence] の形式から分離
    keypoints_2d = []
    confidence = []
    
    for i in range(0, len(pose_keypoints), 3):
        x = pose_keypoints[i]
        y = pose_keypoints[i + 1]
        conf = pose_keypoints[i + 2]
        
        keypoints_2d.append([x, y])
        confidence.append(conf)
    
    return np.array(keypoints_2d), np.array(confidence)

def get_all_frame_files(openpose_dirs):
    """
    OpenPose結果ディレクトリから存在するすべてのフレームファイルを取得する
    
    Args:
        openpose_dirs: OpenPose結果ディレクトリのリスト
    
    Returns:
        common_frames: すべてのカメラに共通して存在するフレームファイル名のリスト
    """
    all_frame_sets = []
    
    for i, openpose_dir in enumerate(openpose_dirs):
        if not openpose_dir.exists():
            print(f"OpenPoseディレクトリが存在しません: {openpose_dir}")
            return []
        
        # JSONファイルをすべて取得
        json_files = list(openpose_dir.glob("*_keypoints.json"))
        frame_files = []
        
        for json_file in json_files:
            frame_files.append(json_file.name)
        
        frame_set = set(frame_files)
        all_frame_sets.append(frame_set)
        print(f"カメラ{i+1} ({openpose_dir.parent.name}): {len(frame_set)} フレーム")
    
    if not all_frame_sets:
        return []
    
    # すべてのカメラに共通するフレームファイル名を取得
    common_frames = set.intersection(*all_frame_sets)
    common_frames_list = sorted(list(common_frames))
    
    print(f"共通フレーム数: {len(common_frames_list)}")
    if len(common_frames_list) > 0:
        print(f"フレーム例: {common_frames_list[0]} ~ {common_frames_list[-1]}")
    
    return common_frames_list

def match_keypoints_by_frame_file(openpose_dirs, frame_file, confidence_threshold=0.3):
    """
    指定フレームファイルで複数カメラ間のキーポイントをマッチング
    
    Args:
        openpose_dirs: OpenPose結果ディレクトリのリスト
        frame_file: フレームファイル名（例：frame_00100_keypoints.json）
        confidence_threshold: 信頼度の閾値
    
    Returns:
        matched_keypoints: マッチしたキーポイント
        matched_confidences: マッチした信頼度
        valid_indices: 有効なキーポイントのインデックス
        camera_data: 各カメラのデータ
    """
    camera_data = []
    
    # 各カメラのデータを読み込み
    for i, openpose_dir in enumerate(openpose_dirs):
        json_file = openpose_dir / frame_file
        
        if json_file.exists():
            keypoints, confidence = load_openpose_json(json_file)
            if keypoints is not None:
                camera_data.append({
                    'keypoints': keypoints,
                    'confidence': confidence,
                    'valid': True
                })
            else:
                camera_data.append({'valid': False})
        else:
            camera_data.append({'valid': False})
    
    # 有効なカメラが2台未満の場合は処理しない
    valid_cameras = [data for data in camera_data if data.get('valid', False)]
    if len(valid_cameras) < 2:
        return None, None, None, camera_data
    
    # キーポイントのマッチング
    num_keypoints = 25  # OpenPose COCO形式
    matched_keypoints = []
    matched_confidences = []
    valid_indices = []
    
    for kp_idx in range(num_keypoints):
        camera_points = []
        camera_confs = []
        all_valid = True
        
        for cam_data in camera_data:
            if cam_data.get('valid', False):
                conf = cam_data['confidence'][kp_idx]
                if conf > confidence_threshold:
                    camera_points.append(cam_data['keypoints'][kp_idx])
                    camera_confs.append(conf)
                else:
                    all_valid = False
                    break
            else:
                all_valid = False
                break
        
        if all_valid and len(camera_points) >= 2:
            matched_keypoints.append(camera_points)
            matched_confidences.append(camera_confs)
            valid_indices.append(kp_idx)
    
    return matched_keypoints, matched_confidences, valid_indices, camera_data

def get_openpose_keypoint_name(index):
    """
    OpenPoseキーポイントインデックスから名前を取得
    """
    openpose_keypoint_names = [
        "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
        "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
        "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
    ]
    if 0 <= index < len(openpose_keypoint_names):
        return openpose_keypoint_names[index]
    else:
        return f"Unknown_{index}"

def convert_numpy_to_python(obj):
    """
    NumPy配列をPython標準データ型に変換する
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj

def save_3d_pose_results(points_3d, keypoint_indices, output_path, additional_info=None):
    """
    3Dポーズ結果を保存する（NumPy配列対応）
    """
    # NumPy配列をPythonリストに変換
    points_3d_list = convert_numpy_to_python(points_3d)
    keypoint_indices_list = convert_numpy_to_python(keypoint_indices)
    
    results = {
        'points_3d': points_3d_list,
        'keypoint_indices': keypoint_indices_list,
        'keypoint_names': [get_openpose_keypoint_name(i) for i in keypoint_indices],
        'num_points': len(points_3d),
        'statistics': {
            'mean': convert_numpy_to_python(points_3d.mean(axis=0)),
            'std': convert_numpy_to_python(points_3d.std(axis=0)),
            'min': convert_numpy_to_python(points_3d.min(axis=0)),
            'max': convert_numpy_to_python(points_3d.max(axis=0))
        },
        'coordinate_system': 'rotated_180_degrees_around_x_axis'  # 回転適用を記録
    }
    
    if additional_info:
        # additional_infoもNumPy配列が含まれている可能性があるので変換
        additional_info_converted = convert_numpy_to_python(additional_info)
        results.update(additional_info_converted)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"3Dポーズ結果を保存（X軸180度回転済み）: {output_path}")
    except Exception as e:
        print(f"JSON保存エラー: {e}")
        print(f"出力パス: {output_path}")
        # デバッグ用：問題のあるデータ型を特定
        for key, value in results.items():
            print(f"  {key}: {type(value)}")

def process_single_frame_weighted(openpose_dirs, frame_file, camera_params_dict, confidence_threshold=0.3):
    """
    単一フレームについて重み付き3D再構成を行う
    """
    directions = ["fl", "fr"]  # sagiは除外（2カメラのみ使用）
    
    # キーポイントをマッチング
    matched_keypoints, matched_confidences, valid_indices, camera_data = match_keypoints_by_frame_file(
        openpose_dirs, frame_file, confidence_threshold
    )
    
    if matched_keypoints is None or len(matched_keypoints) < 5:
        return None
    
    # カメラパラメータを取得
    camera_params_list = []
    for direction in directions:
        if direction in camera_params_dict:
            camera_params_list.append(camera_params_dict[direction])
        else:
            return None
    
    # 重み付き三角測量を実行
    try:
        # データを適切な形式に変換
        points_2d_list = []
        confidences_list = []
        
        for cam_idx in range(len(camera_params_list)):
            cam_points = np.array([match[cam_idx] for match in matched_keypoints])
            cam_confs = np.array([conf[cam_idx] for conf in matched_confidences])
            
            points_2d_list.append(cam_points)
            confidences_list.append(cam_confs)
        
        # 重み付き三角測量（X軸180度回転込み）
        points_3d, point_confidences = triangulate_points_weighted(
            camera_params_list, points_2d_list, confidences_list
        )
        
        # 結果を返す
        result = {
            'points_3d': points_3d,
            'keypoint_indices': valid_indices,
            'point_confidences': point_confidences,
            'frame_file': frame_file,
            'cameras_used': directions,
            'num_matched_keypoints': len(matched_keypoints),
            'triangulation_method': 'weighted_linear',
            'coordinate_transformation': 'x_axis_180_rotation_applied'
        }
        
        return result
        
    except Exception as e:
        print(f"重み付き三角測量でエラー: {e}")
        return None

def process_single_frame(openpose_dirs, frame_file, camera_params_dict, confidence_threshold=0.3):
    """
    単一フレームについて3D再構成を行う（OpenCV三角測量使用）
    """
    directions = ["fl", "fr"]  # sagiは除外（2カメラのみ使用）
    
    # キーポイントをマッチング
    matched_keypoints, matched_confidences, valid_indices, camera_data = match_keypoints_by_frame_file(
        openpose_dirs, frame_file, confidence_threshold
    )
    
    if matched_keypoints is None or len(matched_keypoints) < 5:
        return None
    
    # カメラパラメータを取得
    camera_params_list = []
    for direction in directions:
        if direction in camera_params_dict:
            camera_params_list.append(camera_params_dict[direction])
        else:
            return None
    
    # 3次元三角測量
    try:
        points_2d_cam1 = np.array([match[0] for match in matched_keypoints])
        points_2d_cam2 = np.array([match[1] for match in matched_keypoints])
        
        # OpenCV三角測量（X軸180度回転込み）
        points_3d = triangulate_points_undistorted(
            camera_params_list, [points_2d_cam1, points_2d_cam2]
        )
        
        # 結果を返す
        result = {
            'points_3d': points_3d,
            'keypoint_indices': valid_indices,
            'frame_file': frame_file,
            'cameras_used': directions,
            'num_matched_keypoints': len(matched_keypoints),
            'triangulation_method': 'opencv',
            'coordinate_transformation': 'x_axis_180_rotation_applied'
        }
        
        return result
        
    except Exception as e:
        return None

def main():
    """
    メイン処理：OpenPoseデータからの3次元ポーズ推定（全フレーム）
    重み付き線形三角測量とOpenCV三角測量の両方をサポート
    X軸について180度回転を適用
    """
    # --- パラメータ設定 ---
    root_dir = Path(r"G:\gait_pattern\20250811_br")
    stereo_cali_dir = Path(r"G:\gait_pattern\stereo_cali\9g_20250811")
    
    confidence_threshold = 0.3
    use_weighted_triangulation = True  # 重み付き三角測量を使用するかどうか
    
    method_name = "weighted_linear" if use_weighted_triangulation else "opencv"
    print(f"OpenPoseデータからの3次元ポーズ推定開始（{method_name}三角測量 + X軸180度回転）")
    print(f"ルートディレクトリ: {root_dir}")
    print(f"信頼度閾値: {confidence_threshold}")
    print("注意: すべての3D座標にX軸について180度回転が適用されます")
    
    # カメラパラメータを読み込み
    directions = ["fl", "fr"]
    camera_params_dict = {}
    
    for direction in directions:
        params_file = stereo_cali_dir / direction / "camera_params_with_ext_OC.json"
        if params_file.exists():
            camera_params_dict[direction] = load_camera_parameters(params_file)
            print(f"✓ カメラパラメータを読み込み: {direction}")
        else:
            print(f"✗ エラー: カメラパラメータが見つかりません: {params_file}")
            return
    
    # 各被験者・セラピスト・フレームについて処理
    subject_dir_list = [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("sub")]
    print(f"対象被験者ディレクトリ: {[d.name for d in subject_dir_list]}")
    
    total_processed = 0
    total_successful = 0
    
    for subject_dir in subject_dir_list:
        therapist_dir_list = [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("thera")]
        print(f"\n{subject_dir.name} の対象セラピストディレクトリ: {[d.name for d in therapist_dir_list]}")
        
        for thera_dir in therapist_dir_list:
            if thera_dir.name in ["thera0-15"]:
                print(f"Mocap課題用の動画でスキップ: {thera_dir.name}")
                continue
            
            print(f"\n{'='*80}")
            print(f"処理開始: {subject_dir.name}/{thera_dir.name}")
            print(f"{'='*80}")
            
            # OpenPose結果ディレクトリを確認
            openpose_dirs = []
            for direction in directions:
                openpose_dir = thera_dir / direction / "openpose.json"
                if openpose_dir.exists():
                    openpose_dirs.append(openpose_dir)
                else:
                    print(f"OpenPose結果が見つかりません: {openpose_dir}")
                    break
            
            if len(openpose_dirs) < 2:
                print("2つのカメラのOpenPose結果が必要です。スキップします。")
                continue
            
            # 存在するすべてのフレームファイルを取得
            all_frame_files = get_all_frame_files(openpose_dirs)
            
            if not all_frame_files:
                print("共通するフレームファイルが見つかりません。スキップします。")
                continue
            
            print(f"処理対象フレーム数: {len(all_frame_files)}")
            
            # 結果保存ディレクトリを作成
            
            ext_method = params_file.name.replace("camera_params_with_ext_", "").replace(".json", "")
            
            
            output_dir_name = f"3d_pose_results_{method_name}_{ext_method}"
            output_dir = thera_dir / output_dir_name
            output_dir.mkdir(exist_ok=True)
            
            # 進捗カウンター
            trial_processed = 0
            trial_successful = 0
            
            for frame_idx, frame_file in enumerate(all_frame_files):
                trial_processed += 1
                total_processed += 1
                
                # 進捗表示（10フレームごと）
                if frame_idx % 10 == 0:
                    progress = (frame_idx + 1) / len(all_frame_files) * 100
                    print(f"進捗: {frame_idx + 1}/{len(all_frame_files)} ({progress:.1f}%) - {frame_file}")
                
                # 出力ファイル名を生成（frame_fileから拡張子とキーワードを除去）
                base_name = frame_file.replace("_keypoints.json", "")
                file_prefix = f"3d_pose_{method_name}"
                output_file = output_dir / f"{file_prefix}{base_name}.json"
                
                # 既に処理済みかチェック
                if output_file.exists():
                    trial_successful += 1
                    total_successful += 1
                    continue
                
                # 3D再構成を実行
                if use_weighted_triangulation:
                    result = process_single_frame_weighted(
                        openpose_dirs, frame_file, camera_params_dict, confidence_threshold
                    )
                else:
                    result = process_single_frame(
                        openpose_dirs, frame_file, camera_params_dict, confidence_threshold
                    )
                
                if result is not None:
                    # フレーム情報を追加
                    result['subject'] = subject_dir.name
                    result['therapist'] = thera_dir.name
                    
                    # JSON結果を保存
                    save_3d_pose_results(
                        result['points_3d'], 
                        result['keypoint_indices'], 
                        output_file, 
                        result
                    )
                    
                    trial_successful += 1
                    total_successful += 1
                    
                    if use_weighted_triangulation and 'point_confidences' in result:
                        avg_conf = np.mean(result['point_confidences'])
                        print(f"✓ 成功: フレーム {base_name} - {len(result['keypoint_indices'])} キーポイント, 平均信頼度: {avg_conf:.3f}")
                    else:
                        print(f"✓ 成功: フレーム {base_name} - {len(result['keypoint_indices'])} キーポイント")
                else:
                    print(f"✗ 失敗: フレーム {frame_file}")
                            
            # 試行の統計情報
            success_rate = trial_successful / trial_processed * 100 if trial_processed > 0 else 0
            print(f"\n{subject_dir.name}/{thera_dir.name} 完了:")
            print(f"  処理数: {trial_processed}/{len(all_frame_files)}")
            print(f"  成功数: {trial_successful}")
            print(f"  成功率: {success_rate:.1f}%")
            print(f"  結果保存先: {output_dir}")
    
    # 全体の統計情報
    print(f"\n{'='*80}")
    print(f"全処理完了（{method_name}三角測量 + X軸180度回転）")
    print(f"総処理数: {total_processed}")
    print(f"総成功数: {total_successful}")
    print(f"全体成功率: {total_successful/total_processed*100:.1f}%" if total_processed > 0 else "全体成功率: 0%")
    print(f"結果は各セラピストディレクトリ内の{output_dir_name}フォルダに保存されました")
    print("注意: すべての3D座標にX軸について180度回転が適用されています")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()