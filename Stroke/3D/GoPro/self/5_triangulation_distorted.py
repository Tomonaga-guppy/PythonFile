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

def reconstruct_3d_from_distorted_keypoints(kp_left_distorted, kp_right_distorted,
                                           K_left, D_left, K_right, D_right, R, T):
    """
    歪んだキーポイントから3D座標を再構成する（参考コードのワークフローBに基づく）。

    Args:
        kp_left_distorted (np.ndarray): 左カメラの歪んだ2Dキーポイント (N, 2)
        kp_right_distorted (np.ndarray): 右カメラの歪んだ2Dキーポイント (N, 2)
        K_left (np.ndarray): 左カメラの内部パラメータ行列 (3, 3)
        D_left (np.ndarray): 左カメラの歪み係数
        K_right (np.ndarray): 右カメラの内部パラメータ行列 (3, 3)
        D_right (np.ndarray): 右カメラの歪み係数
        R (np.ndarray): 左カメラから右カメラへの回転行列 (3, 3)
        T (np.ndarray): 左カメラから右カメラへの並進ベクトル (3, 1)

    Returns:
        np.ndarray: 再構成された3D座標 (N, 3)
    """
    # 1. キーポイントの歪み補正
    # cv2.undistortPoints に渡すため、(N, 1, 2) の形状にリシェイプ
    points_distorted_left = kp_left_distorted.reshape(-1, 1, 2).astype(np.float32)
    points_distorted_right = kp_right_distorted.reshape(-1, 1, 2).astype(np.float32)

    # P引数を指定しないことで、補正済みの「正規化座標」を取得
    # これにより、後の計算でK行列が不要になる
    points_normalized_left = cv2.undistortPoints(points_distorted_left, K_left, D_left)
    points_normalized_right = cv2.undistortPoints(points_distorted_right, K_right, D_right)

    # 2. 投影行列の構築
    # キーポイントが正規化座標系にあるため、投影行列にKは含めない
    # 左カメラはワールド座標系の原点にあると仮定
    projMat_left = np.hstack((np.eye(3), np.zeros((3, 1))))
    # 右カメラの投影行列は、外部パラメータそのもの
    projMat_right = np.hstack((R, T.reshape(-1, 1)))

    # 3. キーポイントの形状を triangulatePoints の入力形式 (2, N) に変換
    points_left = points_normalized_left.reshape(-1, 2).T
    points_right = points_normalized_right.reshape(-1, 2).T

    # 4. 三角測量を実行
    points_4d_hom = cv2.triangulatePoints(projMat_left, projMat_right, points_left, points_right)

    # 5. 同次座標を3D座標に変換
    points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]

    return points_3d.T  # (N, 3)の形状で返す

def main():
    # --- 1. パス設定 ---
    # Tposeビデオデータが格納されているルートディレクトリ
    video_dir = Path(r"G:\gait_pattern\20250807_br\Tpose")
    # ステレオキャリブレーションのパラメータファイル
    stereo_params_path = Path(r"G:\gait_pattern\stereo_cali\9g_20250807_6x5_49d5\stereo_params.json")

    left_cam_name = 'fl'
    right_cam_name = 'fr'

    # OpenPoseが出力したJSONファイルが格納されているディレクトリ
    # 注意: これらは歪んだ生画像から検出されたキーポイント
    left_json_dir = video_dir / left_cam_name / "openpose_49d5.json"
    right_json_dir = video_dir / right_cam_name / "openpose_49d5.json"

    # 3D座標の出力先CSVファイル
    output_csv_path = video_dir / "keypoints_3d_49d5_udOP.csv"

    # 信頼度の閾値。これより低い信頼度のキーポイントは計算に使用しない
    CONFIDENCE_THRESHOLD = 0.6

    print(f"\n{'='*60}")
    print("3D三角測量（歪み補正付き）を開始します。")
    print(f"{'='*60}")
    print(f"入力 (左): {left_json_dir}")
    print(f"入力 (右): {right_json_dir}")
    print(f"出力: {output_csv_path}")
    print("処理フロー:")
    print("1. 歪んだ画像からOpenPoseでキーポイント検出")
    print("2. 検出されたキーポイントの歪み補正")
    print("3. 補正済みキーポイントで三角測量")

    # --- 2. ステレオパラメータの読み込み ---
    if not stereo_params_path.exists():
        print(f"エラー: ステレオパラメータファイルが見つかりません: {stereo_params_path}")
        return

    with open(stereo_params_path, 'r') as f:
        params = json.load(f)

    # カメラパラメータ
    mtx_l = np.array(params['camera_matrix_left'])
    dist_l = np.array(params['distortion_left'])
    mtx_r = np.array(params['camera_matrix_right'])
    dist_r = np.array(params['distortion_right'])

    # ステレオパラメータ
    R = np.array(params['R'])
    T = np.array(params['T'])

    print(f"\nカメラパラメータを読み込みました:")
    print(f"左カメラ内部パラメータ: {mtx_l[0,0]:.2f}(fx), {mtx_l[1,1]:.2f}(fy)")
    print(f"右カメラ内部パラメータ: {mtx_r[0,0]:.2f}(fx), {mtx_r[1,1]:.2f}(fy)")
    print(f"歪み係数 左: {len(dist_l)}個, 右: {len(dist_r)}個")

    # --- 3. ステレオパラメータの準備 ---
    # T が1次元配列の場合は形状を調整
    if T.ndim == 1:
        T = T.reshape(-1, 1)

    # --- 4. JSONファイルのリストを取得 ---
    # 左カメラのJSONリストを基準に処理を進める
    json_files_l = sorted(list(left_json_dir.glob("*_keypoints.json")))
    if not json_files_l:
        print(f"エラー: 左カメラのJSONファイルが見つかりません: {left_json_dir}")
        return

    print(f"\n処理するフレーム数: {len(json_files_l)}")

    # --- 5. フレームごとに3D座標を計算 ---
    all_3d_points = []

    for json_path_l in tqdm(json_files_l, desc="3D座標を計算中"):
        frame_name = json_path_l.stem.replace('_keypoints', '')
        frame_number = int(frame_name.split('_')[-1]) # "frame_00000" -> 0

        json_path_r = right_json_dir / json_path_l.name
        if not json_path_r.exists():
            continue # 対応する右カメラのJSONがなければスキップ

        # --- キーポイント読み込み（歪んだ座標系） ---
        keypoints_l_distorted = load_keypoints_from_json(json_path_l)
        keypoints_r_distorted = load_keypoints_from_json(json_path_r)

        # どちらかのカメラで人が検出されなかった場合はスキップ
        if keypoints_l_distorted is None or keypoints_r_distorted is None:
            continue

        # 歪んだ2D座標と信頼度を取得
        points_2d_l_distorted = keypoints_l_distorted[:, :2]
        points_2d_r_distorted = keypoints_r_distorted[:, :2]
        conf_l = keypoints_l_distorted[:, 2]
        conf_r = keypoints_r_distorted[:, 2]

        # --- 有効なキーポイントのマスクを作成 ---
        valid_mask = (conf_l >= CONFIDENCE_THRESHOLD) & (conf_r >= CONFIDENCE_THRESHOLD)
        valid_mask = valid_mask & ~np.isnan(points_2d_l_distorted).any(axis=1)
        valid_mask = valid_mask & ~np.isnan(points_2d_r_distorted).any(axis=1)

        # 全てのキーポイントの3D座標を初期化（無効な場合はNaN）
        points_3d = np.full((len(BODY_25_MAPPING), 3), np.nan)

        if np.any(valid_mask):
            # 有効なキーポイントのみを抽出
            valid_points_l = points_2d_l_distorted[valid_mask]
            valid_points_r = points_2d_r_distorted[valid_mask]

            # --- 3D再構成（歪み補正 + 三角測量） ---
            try:
                valid_points_3d = reconstruct_3d_from_distorted_keypoints(
                    valid_points_l, valid_points_r,
                    mtx_l, dist_l, mtx_r, dist_r, R, T
                )

                # 有効な3D座標を元の配列に代入
                points_3d[valid_mask] = valid_points_3d

            except Exception as e:
                print(f"警告: フレーム {frame_number} で3D再構成に失敗: {e}")
                # エラーの場合はNaNを維持

        # --- 結果を保存 ---
        for i in range(len(BODY_25_MAPPING)):
            conf_l_val = conf_l[i]
            conf_r_val = conf_r[i]

            x, y, z = points_3d[i]

            all_3d_points.append({
                'frame': frame_number,
                'keypoint_id': i,
                'keypoint_name': BODY_25_MAPPING.get(i, 'Unknown'),
                'x': x,
                'y': y,
                'z': z,
                'confidence_L': conf_l_val,
                'confidence_R': conf_r_val,
                'valid': not np.isnan(x)  # 有効な3D座標かどうかのフラグ
            })

    # --- 6. CSVファイルに書き出し ---
    if not all_3d_points:
        print("警告: 有効な3Dポイントが1つも計算されませんでした。")
        return

    df = pd.DataFrame(all_3d_points)
    df.to_csv(output_csv_path, index=False, float_format='%.4f')

    # 統計情報を表示
    total_points = len(df)
    valid_points = df['valid'].sum()
    print(f"\n処理が完了しました。3DキーポイントをCSVに保存しました。")
    print(f"-> {output_csv_path}")
    print(f"\n統計情報:")
    print(f"総キーポイント数: {total_points}")
    print(f"有効な3Dポイント数: {valid_points} ({valid_points/total_points*100:.1f}%)")

if __name__ == '__main__':
    main()