import cv2
import numpy as np
import json
from pathlib import Path
import copy
import glob

def load_camera_parameters(params_file):
    """
    カメラパラメータを読み込む
    """
    with open(params_file, 'r') as f:
        params = json.load(f)
    return params

def triangulate_points_undistorted(camera_params_list, points_2d_list):
    """
    既に歪み補正済みの2D点から3D点を三角測量する
    
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
        
        return points_3d.T
        
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
        return None, None, camera_data
    
    # キーポイントのマッチング
    num_keypoints = 25  # OpenPose COCO形式
    matched_keypoints = []
    valid_indices = []
    
    for kp_idx in range(num_keypoints):
        camera_points = []
        all_valid = True
        
        for cam_data in camera_data:
            if cam_data.get('valid', False):
                conf = cam_data['confidence'][kp_idx]
                if conf > confidence_threshold:
                    camera_points.append(cam_data['keypoints'][kp_idx])
                else:
                    all_valid = False
                    break
            else:
                all_valid = False
                break
        
        if all_valid and len(camera_points) >= 2:
            matched_keypoints.append(camera_points)
            valid_indices.append(kp_idx)
    
    return matched_keypoints, valid_indices, camera_data

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
        }
    }
    
    if additional_info:
        # additional_infoもNumPy配列が含まれている可能性があるので変換
        additional_info_converted = convert_numpy_to_python(additional_info)
        results.update(additional_info_converted)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"3Dポーズ結果を保存: {output_path}")
    except Exception as e:
        print(f"JSON保存エラー: {e}")
        print(f"出力パス: {output_path}")
        # デバッグ用：問題のあるデータ型を特定
        for key, value in results.items():
            print(f"  {key}: {type(value)}")

def process_single_frame(openpose_dirs, frame_file, camera_params_dict, confidence_threshold=0.3):
    """
    単一フレームについて3D再構成を行う
    """
    directions = ["fl", "fr"]  # sagiは除外（2カメラのみ使用）
    
    # キーポイントをマッチング
    matched_keypoints, valid_indices, camera_data = match_keypoints_by_frame_file(
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
        
        points_3d = triangulate_points_undistorted(
            camera_params_list, [points_2d_cam1, points_2d_cam2]
        )
        
        # 結果を返す
        result = {
            'points_3d': points_3d,
            'keypoint_indices': valid_indices,
            'frame_file': frame_file,
            'cameras_used': directions,
            'num_matched_keypoints': len(matched_keypoints)
        }
        
        return result
        
    except Exception as e:
        return None

def main():
    """
    メイン処理：OpenPoseデータからの3次元ポーズ推定（全フレーム）
    """
    # --- パラメータ設定 ---
    root_dir = Path(r"G:\gait_pattern\20250811_br")
    stereo_cali_dir = Path(r"G:\gait_pattern\stereo_cali\9g_20250811")
    
    confidence_threshold = 0.3
    
    print("OpenPoseデータからの3次元ポーズ推定開始（全フレーム処理）")
    print(f"ルートディレクトリ: {root_dir}")
    print(f"信頼度閾値: {confidence_threshold}")
    
    # カメラパラメータを読み込み
    directions = ["fl", "fr"]
    camera_params_dict = {}
    
    for direction in directions:
        params_file = stereo_cali_dir / direction / "camera_params_with_extrinsics.json"
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
            output_dir = thera_dir / "3d_pose_results"
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
                output_file = output_dir / f"3d_pose_{base_name}.json"
                
                # 既に処理済みかチェック
                if output_file.exists():
                    trial_successful += 1
                    total_successful += 1
                    continue
                
                # 3D再構成を実行
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
    print("全処理完了")
    print(f"総処理数: {total_processed}")
    print(f"総成功数: {total_successful}")
    print(f"全体成功率: {total_successful/total_processed*100:.1f}%" if total_processed > 0 else "全体成功率: 0%")
    print(f"結果は各セラピストディレクトリ内の3d_pose_resultsフォルダに保存されました")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()