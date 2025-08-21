import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import csv

# OpenPose BODY_25 modelのキーポイント名
KEYPOINT_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe",
    "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
]

def load_openpose_json(json_path):
    """
    指定されたパスからOpenPoseのJSONファイルを読み込み、
    最初の人物のキーポイントと信頼度を返す。
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if not data['people']:
            return None, None
        keypoints = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
        return keypoints[:, :2], keypoints[:, 2]
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None

def main():
    # --- 1. パラメータ設定 ---
    project_root = Path(r"G:\gait_pattern")
    analysis_dir = project_root / "20250807_br" / "ngait"
    cali_dir = project_root / "int_cali" / "9g_20250807_6x5_35"

    left_cam_dir_name = 'fl'
    right_cam_dir_name = 'fr'

    stereo_params_filename = f"stereo_params_{left_cam_dir_name}_{right_cam_dir_name}.json"
    stereo_params_path = cali_dir / stereo_params_filename

    openpose_json_dir_name = "openpose_35_sb.json"

    # --- 出力ファイル名をCSVに変更 ---
    output_3d_csv_path = analysis_dir / "keypoints_3d.csv"

    print("--- 3D座標復元処理を開始します ---")

    # --- 2. ステレオパラメータの読み込み ---
    print(f"ステレオパラメータを読み込み中: {stereo_params_path}")
    try:
        with open(stereo_params_path, 'r') as f:
            params = json.load(f)

        mtx_l = np.array(params['camera_matrix_left'])
        dist_l = np.array(params['distortion_left'])
        mtx_r = np.array(params['camera_matrix_right'])
        dist_r = np.array(params['distortion_right'])
        R = np.array(params['rotation_matrix'])
        T = np.array(params['translation_vector'])
    except FileNotFoundError:
        print(f"エラー: ステレオパラメータファイルが見つかりません: {stereo_params_path}")
        print("-> 4_culc_stereoparams.py を先に実行してください。")
        return
    except KeyError as e:
        print(f"エラー: ステレオパラメータファイルに必要なキーがありません: {e}")
        return

    # --- 3. 投影行列の作成 ---
    P1 = mtx_l @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = mtx_r @ np.hstack((R, T))
    print("\n左右カメラの投影行列を作成しました。")

    # --- 4. OpenPose JSONファイルのリストを取得 ---
    json_dir_l = analysis_dir / left_cam_dir_name / openpose_json_dir_name
    json_dir_r = analysis_dir / right_cam_dir_name / openpose_json_dir_name

    if not json_dir_l.exists() or not json_dir_r.exists():
        print(f"エラー: OpenPoseのJSONディレクトリが見つかりません。")
        print(f"-> 3_openpose.py を先に実行してください。")
        return

    json_files_l = {p.stem.split('_')[-2]: p for p in sorted(json_dir_l.glob("*.json"))}
    json_files_r = {p.stem.split('_')[-2]: p for p in sorted(json_dir_r.glob("*.json"))}

    common_frames = sorted(list(set(json_files_l.keys()) & set(json_files_r.keys())))

    if not common_frames:
        print("エラー: 左右で対応するOpenPoseのJSONファイルが見つかりません。")
        return

    print(f"\n{len(common_frames)} フレーム分の対応するJSONファイルを検出しました。")

    # --- 5 & 6. 3D復元ループとCSVへの書き込み ---
    try:
        with open(output_3d_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # ヘッダーを書き込む
            header = ["frame", "keypoint_id", "keypoint_name", "x", "y", "z", "confidence_L", "confidence_R"]
            writer.writerow(header)

            reconstructed_frames_count = 0

            for frame_key in tqdm(common_frames, desc="3D復元・CSV書き込み中"):
                json_path_l = json_files_l[frame_key]
                json_path_r = json_files_r[frame_key]

                points2d_l, conf_l = load_openpose_json(json_path_l)
                points2d_r, conf_r = load_openpose_json(json_path_r)

                if points2d_l is None or points2d_r is None:
                    continue

                # 三角測量
                points4d_hom = cv2.triangulatePoints(P1, P2, points2d_l.T, points2d_r.T)
                points3d = (points4d_hom[:3] / points4d_hom[3]).T

                frame_idx = int(frame_key)

                # 各キーポイントのデータをCSVに書き込む
                for i in range(len(points3d)):
                    row = [
                        frame_idx,
                        i,
                        KEYPOINT_NAMES[i],
                        points3d[i, 0], # x
                        points3d[i, 1], # y
                        points3d[i, 2], # z
                        conf_l[i],      # 左カメラの信頼度
                        conf_r[i]       # 右カメラの信頼度
                    ]
                    writer.writerow(row)

                reconstructed_frames_count += 1

        print(f"\n{reconstructed_frames_count} フレームの3D座標を復元しました。")
        print(f"\n3Dキーポイントを {output_3d_csv_path} に保存しました。")
        print("--- 全ての処理が完了しました ---")

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")


if __name__ == '__main__':
    main()
