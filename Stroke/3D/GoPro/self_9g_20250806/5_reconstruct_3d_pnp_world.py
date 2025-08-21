import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import csv

KEYPOINT_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
    "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye",
    "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
]

def load_openpose_json(json_path):
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        if not data['people']: return None, None
        keypoints = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
        return keypoints[:, :2], keypoints[:, 2]
    except (FileNotFoundError, json.JSONDecodeError): return None, None

def main():
    # --- 1. パラメータ設定 ---
    project_root = Path(r"G:\gait_pattern")
    analysis_dir = project_root / "20250807_br" / "ngait"
    cali_dir = project_root / "int_cali" / "9g_20250807_6x5_35"
    left_cam_dir_name, right_cam_dir_name = 'fl', 'fr'

    stereo_params_filename = f"stereo_params_{left_cam_dir_name}_{right_cam_dir_name}_world.json"
    stereo_params_path = cali_dir / stereo_params_filename
    openpose_json_dir_name = "openpose_35_sb.json"
    output_3d_csv_path = analysis_dir / "keypoints_3d_world_origin.csv"

    # --- 2. ステレオパラメータの読み込み ---
    print(f"ステレオパラメータを読み込み中: {stereo_params_path}")
    try:
        with open(stereo_params_path, 'r') as f: params = json.load(f)
        mtx_l, dist_l = np.array(params['camera_matrix_left']), np.array(params['distortion_left'])
        mtx_r, dist_r = np.array(params['camera_matrix_right']), np.array(params['distortion_right'])
        R_rel = np.array(params['rotation_matrix'])

        # ★★★ 修正点: T_relを(3, 1)の列ベクトルに強制的に変形 ★★★
        T_rel = np.array(params['translation_vector']).reshape(3, 1)

        R_world_to_left = np.array(params['R_world_to_left'])
        T_world_to_left = np.array(params['T_world_to_left']).reshape(3, 1)
    except (FileNotFoundError, KeyError) as e:
        print(f"エラー: パラメータ読み込みに失敗しました。: {e}")
        print("-> 修正版の 4_culc_stereoparams_world.py を先に実行してください。")
        return

    # --- 3. 投影行列の作成 ---
    P1 = mtx_l @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = mtx_r @ np.hstack((R_rel, T_rel))

    # --- 4. OpenPose JSONファイルのリストを取得 ---
    json_dir_l = analysis_dir / left_cam_dir_name / openpose_json_dir_name
    json_dir_r = analysis_dir / right_cam_dir_name / openpose_json_dir_name
    json_files_l = {p.stem.split('_')[-2]: p for p in sorted(json_dir_l.glob("*.json"))}
    json_files_r = {p.stem.split('_')[-2]: p for p in sorted(json_dir_r.glob("*.json"))}
    common_frames = sorted(list(set(json_files_l.keys()) & set(json_files_r.keys())))

    # --- 5. 3D復元とCSV書き込み ---
    with open(output_3d_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "keypoint_id", "keypoint_name", "x", "y", "z", "confidence_L", "confidence_R"])

        for frame_key in tqdm(common_frames, desc="3D復元・CSV書き込み中"):
            points2d_l, conf_l = load_openpose_json(json_files_l[frame_key])
            points2d_r, conf_r = load_openpose_json(json_files_r[frame_key])
            if points2d_l is None or points2d_r is None: continue

            # 三角測量 (結果は左カメラ座標系)
            points4d_hom = cv2.triangulatePoints(P1, P2, points2d_l.T, points2d_r.T)
            points3d_cam_left = (points4d_hom[:3] / points4d_hom[3]).T

            # 左カメラ座標系からワールド(ボード)座標系へ変換
            R_left_to_world = R_world_to_left.T
            points3d_world = (R_left_to_world @ (points3d_cam_left.T - T_world_to_left)).T

            # CSVに書き込み
            for i, p_world in enumerate(points3d_world):
                row = [int(frame_key), i, KEYPOINT_NAMES[i], *p_world, conf_l[i], conf_r[i]]
                writer.writerow(row)

    print(f"\nボード原点の3Dキーポイントを保存しました:\n-> {output_3d_csv_path}")

if __name__ == '__main__':
    main()
