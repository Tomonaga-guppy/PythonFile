import cv2
import numpy as np
import json
from pathlib import Path
from scipy.signal import butter, lfilter
import traceback

# --- ユーティリティ関数 (変更なし) ---

def load_camera_parameters(params_file):
    """カメラパラメータ (internal/external) をJSONファイルから読み込む"""
    with open(params_file, 'r') as f:
        params = json.load(f)
    return params

def create_projection_matrix(camera_params):
    """カメラパラメータから3x4のプロジェクション行列を作成する"""
    K = np.array(camera_params['intrinsics'])
    R = np.array(camera_params['extrinsics']['rotation_matrix'])
    t = np.array(camera_params['extrinsics']['translation_vector']).reshape(3, 1)
    P = K @ np.hstack([R, t])
    return P

def load_openpose_json(json_file_path):
    """単一のOpenPose JSONファイルからキーポイントと信頼度を読み込む"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    if not data.get('people'):
        return np.full((25, 2), np.nan), np.full((25,), np.nan)
    
    person_data = data['people'][0]
    keypoints_raw = np.array(person_data['pose_keypoints_2d']).reshape(-1, 3)
    keypoints_2d = keypoints_raw[:, :2]
    confidence = keypoints_raw[:, 2]
    keypoints_2d[confidence == 0] = np.nan
    return keypoints_2d, confidence

def triangulate_points(P1, P2, points1, points2):
    """2組の2D点群から3D点群を三角測量する (OpenCV)"""
    if points1.shape[0] == 0 or points2.shape[0] == 0:
        return np.array([])
    
    points1_t = points1.T
    points2_t = points2.T
    points_4d_hom = cv2.triangulatePoints(P1, P2, points1_t, points2_t)
    points_3d = points_4d_hom[:3] / points_4d_hom[3]
    return points_3d.T

def rotate_coordinates_x_axis(points_3d, angle_degrees=180):
    """3D座標をX軸周りに回転させる"""
    angle_rad = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return np.dot(points_3d, rotation_matrix.T)

def calculate_acceleration(p_prev2, p_prev1, p_curr):
    """3点間の二階差分で加速度の大きさを計算"""
    if np.isnan(p_prev2).any() or np.isnan(p_prev1).any() or np.isnan(p_curr).any():
        return np.inf
    v1 = p_prev1 - p_prev2
    v2 = p_curr - p_prev1
    acceleration_vec = v2 - v1
    return np.linalg.norm(acceleration_vec)

def swap_left_right_keypoints(keypoints):
    """キーポイント配列の左右の部位を入れ替える"""
    swapped = keypoints.copy()
    l_indices = [5, 6, 7, 12, 13, 14, 16, 18, 19, 20, 21]
    r_indices = [2, 3, 4,  9, 10, 11, 15, 17, 22, 23, 24]
    swapped[l_indices + r_indices] = swapped[r_indices + l_indices]
    return swapped

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """時系列データにバターワースローパスフィルタを適用する"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    not_nan = ~np.isnan(data)
    filtered_data = np.full_like(data, np.nan)
    if np.any(not_nan) and len(data[not_nan]) > order * 3:
        filtered_data[not_nan] = lfilter(b, a, data[not_nan])
    elif np.any(not_nan):
        filtered_data[not_nan] = data[not_nan]
    return filtered_data

class StableKalmanFilter:
    """
    「加速度がランダムウォークする」という思想を安定的に実装したカルマンフィルタ
    状態: [位置, 速度, 加速度]
    """
    def __init__(self, dt=1/60, q=100.0, r=10.0):
        self.dt = dt
        self.F = np.array([[1, dt, 0.5*dt**2],
                           [0, 1, dt],
                           [0, 0, 1]])
        self.H = np.array([[1, 0, 0]])
        G = np.array([[0.5*dt**2], [dt], [1]])
        self.Q = G @ G.T * q
        self.R = np.array([[r]])
        self.x = np.zeros((3, 1))
        self.P = np.eye(3) * 100.0
        self.initialized = False

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # 常に予測ステップを先に実行する
        self.predict()

        # 観測値がない(NaN)場合は、更新せず予測値のみを返し、
        # 次の有効な値が来た時にリセットするようにフラグを降ろす
        if np.isnan(z):
            self.initialized = False
            return self.x[0, 0]

        # 最初の有効な観測値、またはNaNの後の最初の有効な観測値で状態をリセット
        if not self.initialized:
            self.x.fill(0) # 速度と加速度をリセット
            self.x[0, 0] = z # 位置を観測値に設定
            self.P = np.eye(3) * 100.0 # 不確かさをリセット
            self.initialized = True
            return self.x[0, 0]
        
        # 通常の更新ステップ
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        if S[0,0] < 1e-6:
             S[0,0] = 1e-6
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(3)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x[0, 0]

# --- メイン処理 ---
def main():
    root_dir = Path(r"G:\gait_pattern\20250811_br")
    stereo_cali_dir = Path(r"G:\gait_pattern\stereo_cali\9g_20250811")
    
    ACCELERATION_THRESHOLD = 150.0
    BUTTERWORTH_CUTOFF = 6.0
    FRAME_RATE = 60
    
    ### 変更点: パラメータを調整 ###
    # Q: プロセスノイズ。モデルの信頼度。小さいほど「動きは滑らか」だと仮定する。
    # R: 観測ノイズ。観測値の信頼度。大きいほど「観測値はノイジー」だと仮定する。
    # R/Q の比率が重要。この比率が大きいほど、フィルタは観測値よりも自身の予測を信じる。
    STABLE_KF_Q = 1.0    # 動きは非常に滑らかであると仮定
    STABLE_KF_R = 1000.0  # 観測値はかなりノイジーであると仮定
    
    directions = ["fl", "fr"]

    print("統合版マーカーレス歩行解析を開始します。")
    try:
        params_cam1 = load_camera_parameters(stereo_cali_dir / directions[0] / "camera_params_with_ext_OC.json")
        params_cam2 = load_camera_parameters(stereo_cali_dir / directions[1] / "camera_params_with_ext_OC.json")
        P1 = create_projection_matrix(params_cam1)
        P2 = create_projection_matrix(params_cam2)
        print("✓ カメラパラメータを正常に読み込みました。")
    except FileNotFoundError as e:
        print(f"✗ エラー: カメラパラメータファイルが見つかりません。{e}")
        return

    subject_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("sub")])
    if not subject_dirs:
        print(f"✗ エラー: {root_dir} 内に 'sub' で始まる被験者ディレクトリが見つかりません。")
        return

    for subject_dir in subject_dirs:
        thera_dirs = sorted([d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("thera")])
        for thera_dir in thera_dirs:
            print(f"\n{'='*80}")
            print(f"処理開始: {thera_dir.relative_to(root_dir)}")
            
            if subject_dir.name != "sub0" or thera_dir.name != "thera0-16":
                print(subject_dir.name, thera_dir.name)
                print("今は0-0-16以外はスキップ")
                continue
            
            openpose_dir1 = thera_dir / directions[0] / "openpose.json"
            openpose_dir2 = thera_dir / directions[1] / "openpose.json"
            if not (openpose_dir1.exists() and openpose_dir2.exists()): continue
            files1 = {f.name for f in openpose_dir1.glob("*_keypoints.json")}
            files2 = {f.name for f in openpose_dir2.glob("*_keypoints.json")}
            common_frames = sorted(list(files1 & files2))
            if not common_frames: continue
            
            print(f"  - {len(common_frames)} フレームを処理します。")
            output_dir = thera_dir / "3d_gait_analysis_stable_kf"
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"{thera_dir.name}_3d_results.json"

            print("  - ステップ1: 全フレームの観測値を収集中...")
            all_raw_points = []
            all_kf_input_points = []
            history_for_accel = []

            for frame_idx, frame_name in enumerate(common_frames):
                kp2d_cam1, _ = load_openpose_json(openpose_dir1 / frame_name)
                kp2d_cam2, _ = load_openpose_json(openpose_dir2 / frame_name)

                best_points_3d = None
                min_avg_acceleration = np.inf
                pattern0_points_3d = np.full((25, 3), np.nan)

                for pattern_id in range(4):
                    kp1_trial = swap_left_right_keypoints(kp2d_cam1) if pattern_id in [1, 3] else kp2d_cam1
                    kp2_trial = swap_left_right_keypoints(kp2d_cam2) if pattern_id in [2, 3] else kp2d_cam2

                    valid_indices = np.where(~np.isnan(kp1_trial).any(axis=1) & ~np.isnan(kp2_trial).any(axis=1))[0]
                    if len(valid_indices) == 0: continue

                    points_3d_trial_raw = triangulate_points(P1, P2, kp1_trial[valid_indices], kp2_trial[valid_indices])
                    points_3d_trial = rotate_coordinates_x_axis(points_3d_trial_raw)
                    full_points_3d = np.full((25, 3), np.nan)
                    full_points_3d[valid_indices] = points_3d_trial
                    
                    if pattern_id == 0:
                        pattern0_points_3d = full_points_3d

                    current_avg_accel = 0
                    count = 0
                    if len(history_for_accel) >= 2:
                        eval_indices = [10, 11, 13, 14]
                        for idx in eval_indices:
                            accel = calculate_acceleration(history_for_accel[-2][idx], history_for_accel[-1][idx], full_points_3d[idx])
                            if accel != np.inf:
                                current_avg_accel += accel; count += 1
                        current_avg_accel = current_avg_accel / count if count > 0 else np.inf
                    
                    if current_avg_accel < min_avg_acceleration:
                        min_avg_acceleration = current_avg_accel
                        best_points_3d = full_points_3d

                all_raw_points.append(pattern0_points_3d)
                is_error_frame = best_points_3d is None or min_avg_acceleration >= ACCELERATION_THRESHOLD
                kf_input = np.full((25, 3), np.nan) if is_error_frame else best_points_3d
                all_kf_input_points.append(kf_input)
                history_for_accel.append(best_points_3d if best_points_3d is not None else np.full((25, 3), np.nan))

            # --- ステップ2: カルマンフィルタ適用 ---
            print("  - ステップ2: カルマンフィルタを適用中...")
            kalman_filters = [[StableKalmanFilter(dt=1/FRAME_RATE, q=STABLE_KF_Q, r=STABLE_KF_R) for _ in range(3)] for _ in range(25)]
            kf_points_list = []
            for t in range(len(common_frames)):
                kf_output = np.full((25, 3), np.nan)
                for i in range(25):
                    for j in range(3):
                        z = all_kf_input_points[t][i, j]
                        kf_output[i, j] = kalman_filters[i][j].update(z)
                kf_points_list.append(kf_output)

            # --- ステップ3: バターワースフィルタ適用 ---
            print("  - ステップ3: バターワースフィルタで最終平滑化中...")
            kf_points_array = np.array(kf_points_list)
            final_points_array = np.full_like(kf_points_array, np.nan)
            for kp_idx in range(kf_points_array.shape[1]):
                for axis_idx in range(kf_points_array.shape[2]):
                    sequence = kf_points_array[:, kp_idx, axis_idx]
                    final_points_array[:, kp_idx, axis_idx] = butter_lowpass_filter(sequence, BUTTERWORTH_CUTOFF, FRAME_RATE)

            # --- ステップ4: 全ての結果を結合して保存 ---
            analysis_results = []
            for t, frame_name in enumerate(common_frames):
                analysis_results.append({
                    "frame_name": frame_name,
                    "points_3d_raw": all_raw_points[t].tolist(),
                    "points_3d_kf": kf_points_list[t].tolist(),
                    "points_3d_final": final_points_array[t].tolist()
                })

            try:
                with open(output_file, 'w') as f:
                    json.dump(analysis_results, f, indent=4)
                print(f"  ✓ 処理完了。結果を {output_file.relative_to(root_dir)} に保存しました。")
            except Exception as e:
                print(f"  ✗ JSON保存エラー: {e}")
                traceback.print_exc()

    print(f"\n{'='*80}")
    print("全ての処理が完了しました。")

if __name__ == '__main__':
    main()
