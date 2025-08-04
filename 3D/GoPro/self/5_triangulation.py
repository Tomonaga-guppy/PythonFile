import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm

# OpenPose BODY_25モデルのキーポイント名とインデックスのマッピング
# 出力されるCSVの可読性を高めるために使用
BODY_25_MAPPING = {
    0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
    10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
    15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
    20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel",
}

def load_keypoints_from_json(json_path):
    """
    単一のOpenPose JSONファイルを読み込み、キーポイントデータを返す。
    人が検出されなかった場合はNoneを返す。
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    if not data['people']:
        return None
    # 最初の人物のキーポイントデータを取得 (x, y, confidence)
    keypoints = data['people'][0]['pose_keypoints_2d']
    return np.array(keypoints).reshape(-1, 3)

def main():
    # --- 1. パス設定 ---
    # Tposeビデオデータが格納されているルートディレクトリ
    video_dir = Path(r"G:\gait_pattern\20250717_br\ngait")
    # ステレオキャリブレーションのパラメータファイル
    stereo_params_path = Path(r"G:\gait_pattern\stero_cali\9g_6x5\stereo_params.json")

    left_cam_name = 'fl'
    right_cam_name = 'fr'

    # OpenPoseが出力したJSONファイルが格納されているディレクトリ
    left_json_dir = video_dir / left_cam_name / "openpose.json"
    right_json_dir = video_dir / right_cam_name / "openpose.json"

    # 3D座標の出力先CSVファイル
    output_csv_path = video_dir / "keypoints_3d.csv"

    # 信頼度の閾値。これより低い信頼度のキーポイントは計算に使用しない
    CONFIDENCE_THRESHOLD = 0.5

    print(f"\n{'='*60}")
    print("3D三角測量を開始します。")
    print(f"{'='*60}")
    print(f"入力 (左): {left_json_dir}")
    print(f"入力 (右): {right_json_dir}")
    print(f"出力: {output_csv_path}")

    # --- 2. ステレオパラメータの読み込み ---
    if not stereo_params_path.exists():
        print(f"エラー: ステレオパラメータファイルが見つかりません: {stereo_params_path}")
        return

    with open(stereo_params_path, 'r') as f:
        params = json.load(f)

    mtx_l = np.array(params['camera_matrix_left'])
    dist_l = np.array(params['distortion_left'])
    mtx_r = np.array(params['camera_matrix_right'])
    dist_r = np.array(params['distortion_right'])
    R = np.array(params['R'])
    T = np.array(params['T'])

    # --- 3. 射影行列の計算 ---
    # 左カメラの射影行列 P1 = K1 * [I|0]
    P1 = mtx_l @ np.hstack((np.eye(3), np.zeros((3, 1))))
    # 右カメラの射影行列 P2 = K2 * [R|T]
    P2 = mtx_r @ np.hstack((R, T))

    # --- 4. JSONファイルのリストを取得 ---
    # 左カメラのJSONリストを基準に処理を進める
    json_files_l = sorted(list(left_json_dir.glob("*_keypoints.json")))
    if not json_files_l:
        print(f"エラー: 左カメラのJSONファイルが見つかりません: {left_json_dir}")
        return

    # --- 5. フレームごとに3D座標を計算 ---
    all_3d_points = []

    for json_path_l in tqdm(json_files_l, desc="3D座標を計算中"):
        frame_name = json_path_l.stem.replace('_keypoints', '')
        frame_number = int(frame_name.split('_')[-1]) # "frame_00000" -> 0

        json_path_r = right_json_dir / json_path_l.name
        if not json_path_r.exists():
            continue # 対応する右カメラのJSONがなければスキップ

        # キーポイントを読み込む
        keypoints_l = load_keypoints_from_json(json_path_l)
        keypoints_r = load_keypoints_from_json(json_path_r)

        # どちらかのカメラで人が検出されなかった場合はスキップ
        if keypoints_l is None or keypoints_r is None:
            continue

        # 2D座標は歪み補正済みのはずだが、念のためundistortPointsを適用することも可能
        # 今回はOpenPoseを歪み補正後の画像にかけているため、この処理は不要
        points_2d_l = keypoints_l[:, :2].astype(np.float32)
        points_2d_r = keypoints_r[:, :2].astype(np.float32)

        # --- 三角測量 ---
        # cv2.triangulatePointsは (4, N) の同次座標を返す
        points_4d_hom = cv2.triangulatePoints(P1, P2, points_2d_l.T, points_2d_r.T)

        # 同次座標を3D座標に変換 (X/W, Y/W, Z/W)
        points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
        points_3d = points_3d.T # (N, 3) の形状に転置

        # --- 結果を保存 ---
        for i in range(len(BODY_25_MAPPING)):
            conf_l = keypoints_l[i, 2]
            conf_r = keypoints_r[i, 2]

            # 両方の信頼度が閾値を超えている場合のみ座標を記録
            if conf_l >= CONFIDENCE_THRESHOLD and conf_r >= CONFIDENCE_THRESHOLD:
                x, y, z = points_3d[i]
            else:
                x, y, z = np.nan, np.nan, np.nan

            all_3d_points.append({
                'frame': frame_number,
                'keypoint_id': i,
                'keypoint_name': BODY_25_MAPPING.get(i, 'Unknown'),
                'x': x,
                'y': y,
                'z': z,
                'confidence_L': conf_l,
                'confidence_R': conf_r
            })

    # --- 6. CSVファイルに書き出し ---
    if not all_3d_points:
        print("警告: 有効な3Dポイントが1つも計算されませんでした。")
        return

    df = pd.DataFrame(all_3d_points)
    df.to_csv(output_csv_path, index=False, float_format='%.4f')

    print(f"\n処理が完了しました。3DキーポイントをCSVに保存しました。")
    print(f"-> {output_csv_path}")

if __name__ == '__main__':
    main()
