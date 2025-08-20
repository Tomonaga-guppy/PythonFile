import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm

# OpenPose BODY_25モデルのキーポイント名とインデックスのマッピング
BODY_25_MAPPING = {
    0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
    10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
    15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
    20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel",
}

def load_keypoints_from_json(json_path):
    """単一のOpenPose JSONファイルを読み込み、キーポイントデータを返す。"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    if not data['people']:
        return None
    keypoints = data['people'][0]['pose_keypoints_2d']
    return np.array(keypoints).reshape(-1, 3)

# ★★★ ここからが追加/変更箇所 ★★★
def define_world_origin(stereo_params_path, left_cam_name, right_cam_name, origin_image_name):
    """
    指定された画像を世界の原点として定義し、その変換行列を返す。
    """
    print(f"\n世界の原点を '{origin_image_name}' のチェッカーボードに設定します...")

    # ステレオキャリブレーション用の画像があった場所
    stereo_cali_dir = Path(r"G:\gait_pattern\stereo_cali\9g_20250807_6x5_49d5")
    left_img_path = stereo_cali_dir / left_cam_name / "cali_imgs" / origin_image_name
    right_img_path = stereo_cali_dir / right_cam_name / "cali_imgs" / origin_image_name

    if not left_img_path.exists() or not right_img_path.exists():
        print(f"エラー: 原点設定用の画像が見つかりません。")
        print(f"  - {left_img_path}")
        print(f"  - {right_img_path}")
        return None, None

    # カメラパラメータを読み込む
    with open(stereo_params_path, 'r') as f:
        params = json.load(f)
    mtx_l = np.array(params['camera_matrix_left'])
    dist_l = np.array(params['distortion_left'])

    # チェッカーボード情報を設定
    checker_pattern = (5, 4)
    square_size = 49.5  # mm
    objp = np.zeros((checker_pattern[0] * checker_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checker_pattern[0], 0:checker_pattern[1]].T.reshape(-1, 2)
    objp *= square_size

    print(f"チェッカーボードのパターン: {checker_pattern}, 正方形のサイズ: {square_size} mm")

    # 左カメラの画像からコーナーを検出
    img_l = cv2.imread(str(left_img_path))
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_l, checker_pattern, None)

    if not ret:
        print(f"エラー: '{origin_image_name}' でチェッカーボードを検出できませんでした。")
        return None, None

    # コーナーの精度を高める
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray_l, corners, (11, 11), (-1, -1), criteria)

    # solvePnPを使って、カメラから見たチェッカーボードの姿勢を計算
    # これが、元の座標系（左カメラ基準）における、新しい原点の姿勢（位置と向き）になる
    _, rvec, tvec = cv2.solvePnP(objp, corners2, mtx_l, dist_l)

    # 回転ベクトルを回転行列に変換
    R_origin, _ = cv2.Rodrigues(rvec)

    print("原点の設定完了。")
    # 新しい原点への変換に必要な回転行列と並進ベクトルを返す
    return R_origin, tvec

def main():
    # --- 1. パス設定 ---
    video_dir = Path(r"G:\gait_pattern\20250807_br\Tpose")
    stereo_params_path = Path(r"G:\gait_pattern\stereo_cali\9g_20250807_6x5_35\stereo_params.json")

    left_cam_name = 'fl'
    right_cam_name = 'fr'

    # ★★★ 原点としたい画像ファイル名を指定 ★★★
    origin_image_name = "0031.png"  #35mm
    # origin_image_name = "0027.png"  #49.5mm

    left_json_dir = video_dir / left_cam_name / "openpose_35_sb.json"
    right_json_dir = video_dir / right_cam_name / "openpose_35_sb.json"
    output_csv_path = video_dir / "keypoints_3d_ori_35_sb.csv" # 出力ファイル名を変更
    CONFIDENCE_THRESHOLD = 0.5

    # --- 2. ワールド座標系の原点を定義 ---
    R_origin, t_origin = define_world_origin(stereo_params_path, left_cam_name, right_cam_name, origin_image_name)
    if R_origin is None:
        return # 原点設定に失敗した場合は終了

    print(f"\n{'='*60}")
    print("3D三角測量を開始します。（原点補正あり）")
    print(f"{'='*60}")

    # --- 3. ステレオパラメータの読み込み ---
    with open(stereo_params_path, 'r') as f:
        params = json.load(f)
    mtx_l, dist_l = np.array(params['camera_matrix_left']), np.array(params['distortion_left'])
    mtx_r, dist_r = np.array(params['camera_matrix_right']), np.array(params['distortion_right'])
    R, T = np.array(params['R']), np.array(params['T'])

    # --- 4. 射影行列の計算 ---
    P1 = mtx_l @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = mtx_r @ np.hstack((R, T))

    # --- 5. JSONファイルのリストを取得 ---
    json_files_l = sorted(list(left_json_dir.glob("*_keypoints.json")))
    if not json_files_l:
        print(f"エラー: 左カメラのJSONファイルが見つかりません: {left_json_dir}")
        return

    # --- 6. フレームごとに3D座標を計算し、原点を補正 ---
    all_3d_points = []
    for json_path_l in tqdm(json_files_l, desc="3D座標を計算中"):
        frame_name = json_path_l.stem.replace('_keypoints', '')
        frame_number = int(frame_name.split('_')[-1])

        json_path_r = right_json_dir / json_path_l.name
        if not json_path_r.exists(): continue

        keypoints_l = load_keypoints_from_json(json_path_l)
        keypoints_r = load_keypoints_from_json(json_path_r)
        if keypoints_l is None or keypoints_r is None: continue

        points_2d_l = keypoints_l[:, :2].astype(np.float32)
        points_2d_r = keypoints_r[:, :2].astype(np.float32)

        points_4d_hom = cv2.triangulatePoints(P1, P2, points_2d_l.T, points_2d_r.T)
        points_3d_cam_coord = (points_4d_hom[:3, :] / points_4d_hom[3, :]).T

        # ★★★ ここで座標変換を実行 ★★★
        # 各3Dポイントを、新しい原点（チェッカーボード）を基準とした座標に変換する
        # p_new = R_origin^T * (p_old - t_origin)
        points_3d_world_coord = (R_origin.T @ (points_3d_cam_coord.T - t_origin)).T

        for i in range(len(BODY_25_MAPPING)):
            conf_l, conf_r = keypoints_l[i, 2], keypoints_r[i, 2]
            if conf_l >= CONFIDENCE_THRESHOLD and conf_r >= CONFIDENCE_THRESHOLD:
                x, y, z = points_3d_world_coord[i]
            else:
                x, y, z = np.nan, np.nan, np.nan

            all_3d_points.append({
                'frame': frame_number, 'keypoint_id': i,
                'keypoint_name': BODY_25_MAPPING.get(i, 'Unknown'),
                'x': x, 'y': y, 'z': z,
                'confidence_L': conf_l, 'confidence_R': conf_r
            })

    # --- 7. CSVファイルに書き出し ---
    if not all_3d_points:
        print("警告: 有効な3Dポイントが1つも計算されませんでした。")
        return

    df = pd.DataFrame(all_3d_points)
    df.to_csv(output_csv_path, index=False, float_format='%.4f')

    print(f"\n処理が完了しました。原点補正済みの3DキーポイントをCSVに保存しました。")
    print(f"-> {output_csv_path}")

if __name__ == '__main__':
    main()