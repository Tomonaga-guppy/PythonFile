"""
パラメータ推定: 常に**「未加工の生データ (raw_3d)」**の履歴を使って、その瞬間の「正直なノイズ特性」を学習します。

フィルター処理: 上記で学習したパラメータを使い、**「補正済みのデータ (corrected_3d)」**の履歴を滑らかにします。
"""

import json
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize
from tqdm import tqdm
import warnings

"""
パラメータは毎フレーム調整
エラーが起きるごとにrawからKFパラメータを算出してフィルタリングの対象はcorrectedにする
"""


# m_triangulationモジュールから、指定された関数をインポート
try:
    from m_triangulation import triangulate_and_rotate
except ImportError:
    print("エラー: m_triangulation.pyが見つかりません。同じディレクトリに配置してください。")
    # フォールバックとしてダミー関数を定義
    def triangulate_and_rotate(P1, P2, points1, points2, confidences1, confidences2):
        print("警告: ダミーのtriangulate_and_rotate関数を使用しています。")
        num_points = points1.shape[0] if points1 is not None else 0
        # ★★★ 変更点: ダミー関数も信頼度配列を返すように修正 ★★★
        return np.full((25, 3), np.nan), np.full((25,), np.nan)

# 数値計算の警告を抑制
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# 0. 設定と定数
# =============================================================================
ROOT_DIR = Path(r"G:\gait_pattern\20250811_br")
STEREO_CALI_DIR = Path(r"G:\gait_pattern\stereo_cali\9g_20250811")
BUTTERWORTH_CUTOFF = 6.0
FRAME_RATE = 60
ACCELERATION_THRESHOLD = 100.0
MAX_INTERPOLATION_GAP = 15


KEYPOINT_PAIRS = [
    (11, 14, "Ankle"), (10, 13, "Knee"), (9, 12, "Hip"),
    (22, 19, "BigToe"), (23, 20, "SmallToe"), (24, 21, "Heel")
]

# =============================================================================
# 1. データの読み込みと準備
# =============================================================================

def load_camera_parameters(params_file):
    """カメラパラメータをJSONファイルから読み込む"""
    with open(params_file, 'r') as f:
        return json.load(f)

def create_projection_matrix(params):
    """カメラパラメータから3x4のプロジェクション行列を作成する"""
    K = np.array(params['intrinsics'])
    R = np.array(params['extrinsics']['rotation_matrix'])
    t = np.array(params['extrinsics']['translation_vector']).reshape(3, 1)
    return K @ np.hstack([R, t])

def load_2d_data(openpose_dir1, openpose_dir2):
    """2台のカメラのOpenPose JSONを読み込み、共通フレームのデータを返す"""
    files1 = {p.name: p for p in openpose_dir1.glob("*_keypoints.json")}
    files2 = {p.name: p for p in openpose_dir2.glob("*_keypoints.json")}
    common_frames = sorted(list(files1.keys() & files2.keys()))

    all_kps1, all_kps2 = [], []
    print(f"共通フレームを読み込み中 (全{len(common_frames)}フレーム)...")
    for frame_name in tqdm(common_frames):
        with open(files1[frame_name], 'r') as f: data1 = json.load(f)
        with open(files2[frame_name], 'r') as f: data2 = json.load(f)
        kps1 = np.full((25, 3), np.nan)
        if data1.get('people'):
            kps1 = np.array(data1['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
        kps2 = np.full((25, 3), np.nan)
        if data2.get('people'):
            kps2 = np.array(data2['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
        all_kps1.append(kps1)
        all_kps2.append(kps2)
    return np.array(all_kps1), np.array(all_kps2), common_frames

# =============================================================================
# 2. 3D座標の計算
# =============================================================================

def calculate_raw_3d_coordinates(kps1_seq, kps2_seq, P1, P2):
    """全フレームの生の3D座標と信頼度を一括で計算する"""
    num_frames = len(kps1_seq)
    raw_3d_points = np.full((num_frames, 25, 3), np.nan)
    # ★★★ 変更点: 信頼度を格納する配列を追加 ★★★
    raw_confidences = np.full((num_frames, 25), np.nan)

    print(f"生の3D座標と信頼度を計算中 (全{num_frames}フレーム)...")
    for i in tqdm(range(num_frames)):
        kp1, cf1 = kps1_seq[i][:, :2], kps1_seq[i][:, 2]
        kp2, cf2 = kps2_seq[i][:, :2], kps2_seq[i][:, 2]
        # ★★★ 変更点: 3D座標と信頼度を両方受け取る ★★★
        raw_3d_points[i], raw_confidences[i] = triangulate_and_rotate(P1, P2, kp1, kp2, cf1, cf2)

    # ★★★ 変更点: 3D座標と信頼度の両方を返す ★★★
    return raw_3d_points, raw_confidences

# =============================================================================
# 3. カルマンフィルタ関連関数
# =============================================================================

def moving_average_filter(data, window_size=3):
    if len(data) < window_size: return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def local_level_kf(y, a1, P1, var_eta, var_eps):
    L = len(y)
    a_tt1, P_tt1 = np.zeros(L + 1), np.zeros(L + 1)
    a_tt1[0], P_tt1[0] = a1, P1
    v_t, F_t = np.zeros(L), np.zeros(L)
    a_tt, P_tt = np.zeros(L), np.zeros(L)
    for t in range(L):
        v_t[t] = y[t] - a_tt1[t]
        F_t[t] = P_tt1[t] + var_eps
        if F_t[t] <= 1e-10: F_t[t] = 1e-10
        K_t = P_tt1[t] / F_t[t]
        a_tt[t] = a_tt1[t] + K_t * v_t[t]
        P_tt[t] = P_tt1[t] * (1 - K_t)
        a_tt1[t+1] = a_tt[t]
        P_tt1[t+1] = P_tt[t] + var_eta
    return a_tt, P_tt, F_t, v_t

def calc_log_diffuse_llhd(y, vars):
    try:
        y_valid = y[~np.isnan(y)]
        if len(y_valid) < 2: return -1e10
        psi_eta, psi_eps = np.clip(vars, -10, 10)
        var_eta, var_eps = np.exp(2 * psi_eta), np.exp(2 * psi_eps)
        var_eta, var_eps = np.clip(var_eta, 1e-8, 1e8), np.clip(var_eps, 1e-8, 1e8)
        a1, P1 = y_valid[0], var_eps
        _, _, F_t, v_t = local_level_kf(y_valid, a1, P1, var_eta, var_eps)
        valid_F = np.maximum(F_t[1:], 1e-10)
        tmp = np.sum(np.log(valid_F) + v_t[1:]**2 / valid_F)
        log_ld = -0.5 * len(y_valid) * np.log(2 * np.pi) - 0.5 * tmp
        return log_ld if np.isfinite(log_ld) else -1e10
    except:
        return -1e10

def estimate_kf_parameters(y, initial_value=0.005):
    par = np.clip(initial_value, 1e-6, 0.1)
    x0 = [np.log(np.sqrt(par)), np.log(np.sqrt(par))]
    res = minimize(lambda x: -calc_log_diffuse_llhd(y, x), x0, method='L-BFGS-B', bounds=([-5, -5], [5, 5]))
    x_opt = res.x if res.success else x0
    var_eta_opt = np.clip(np.exp(2 * x_opt[0]), 1e-6, 100.0)
    var_eps_opt = np.clip(np.exp(2 * x_opt[1]), 1e-6, 100.0)
    return var_eta_opt, var_eps_opt

def run_kalman_filter_dual_data(
    estimation_data_series,
    filtering_data_series,
    initial_value=0.005
):
    valid_mask_est = ~np.isnan(estimation_data_series)
    if np.sum(valid_mask_est) < 5:
        return None
    y_est = np.diff(estimation_data_series[valid_mask_est])
    y_maf_est = moving_average_filter(y_est)
    var_eta, var_eps = estimate_kf_parameters(y_maf_est, initial_value)
    var_eps, var_eta = var_eta, var_eps
    a1, P1 = var_eps, var_eta
    valid_mask_filt = ~np.isnan(filtering_data_series)
    if np.sum(valid_mask_filt) < 5:
        return None
    y_filt = np.diff(filtering_data_series[valid_mask_filt])
    y_maf_filt = moving_average_filter(y_filt)
    a_tt, _, _, _ = local_level_kf(y_maf_filt, a1, P1, var_eta, var_eps)
    return a_tt

def apply_frame_by_frame_correction(raw_3d, kps1_seq, kps2_seq, P1, P2):
    print("フレームごとの入れ替わり検出と補正を実行中 (生データでパラメータ推定)...")
    corrected_3d = raw_3d.copy()
    num_frames = len(raw_3d)

    for i in tqdm(range(2, num_frames), desc="フレーム処理"):
        kp1, cf1 = kps1_seq[i][:, :2], kps1_seq[i][:, 2]
        kp2, cf2 = kps2_seq[i][:, :2], kps2_seq[i][:, 2]

        for r_idx, l_idx, name in KEYPOINT_PAIRS:
            prev_2_raw, prev_1_raw, current_raw = raw_3d[i-2], raw_3d[i-1], raw_3d[i]
            if np.any(np.isnan(prev_1_raw[r_idx])) or np.any(np.isnan(prev_2_raw[r_idx])) or \
               np.any(np.isnan(prev_1_raw[l_idx])) or np.any(np.isnan(prev_2_raw[l_idx])) or \
               np.any(np.isnan(current_raw[r_idx])) or np.any(np.isnan(current_raw[l_idx])):
                continue
            accel_r = np.linalg.norm((current_raw[r_idx] - prev_1_raw[r_idx]) - (prev_1_raw[r_idx] - prev_2_raw[r_idx]))
            accel_l = np.linalg.norm((current_raw[l_idx] - prev_1_raw[l_idx]) - (prev_1_raw[l_idx] - prev_2_raw[l_idx]))

            if accel_r < ACCELERATION_THRESHOLD and accel_l < ACCELERATION_THRESHOLD:
                corrected_3d[i, r_idx], corrected_3d[i, l_idx] = current_raw[r_idx], current_raw[l_idx]
            elif accel_r > ACCELERATION_THRESHOLD and accel_l > ACCELERATION_THRESHOLD:
                for axis in range(3):
                    corrected_3d[i, r_idx, axis] = corrected_3d[i-1, r_idx, axis] + (corrected_3d[i-1, r_idx, axis] - corrected_3d[i-2, r_idx, axis])
                    corrected_3d[i, l_idx, axis] = corrected_3d[i-1, l_idx, axis] + (corrected_3d[i-1, l_idx, axis] - corrected_3d[i-2, l_idx, axis])
            elif accel_r > ACCELERATION_THRESHOLD and accel_l < ACCELERATION_THRESHOLD:
                corrected_3d[i, l_idx] = current_raw[l_idx]
                for axis in range(3):
                    vel_r_axis = run_kalman_filter_dual_data(
                        estimation_data_series=raw_3d[:i, r_idx, axis],
                        filtering_data_series=corrected_3d[:i, r_idx, axis]
                    )
                    if vel_r_axis is not None and len(vel_r_axis) > 0:
                        kalman_velocity = vel_r_axis[-1]
                        corrected_3d[i, r_idx, axis] = corrected_3d[i-1, r_idx, axis] + kalman_velocity
                    else:
                        corrected_3d[i, r_idx, axis] = corrected_3d[i-1, r_idx, axis] + (corrected_3d[i-1, r_idx, axis] - corrected_3d[i-2, r_idx, axis])
            elif accel_r < ACCELERATION_THRESHOLD and accel_l > ACCELERATION_THRESHOLD:
                corrected_3d[i, r_idx] = current_raw[r_idx]
                for axis in range(3):
                    vel_l_axis = run_kalman_filter_dual_data(
                        estimation_data_series=raw_3d[:i, l_idx, axis],
                        filtering_data_series=corrected_3d[:i, l_idx, axis]
                    )
                    if vel_l_axis is not None and len(vel_l_axis) > 0:
                        kalman_velocity = vel_l_axis[-1]
                        corrected_3d[i, l_idx, axis] = corrected_3d[i-1, l_idx, axis] + kalman_velocity
                    else:
                        corrected_3d[i, l_idx, axis] = corrected_3d[i-1, l_idx, axis] + (corrected_3d[i-1, l_idx, axis] - corrected_3d[i-2, l_idx, axis])
    return corrected_3d

# =============================================================================
# 4. データの後処理と可視化
# =============================================================================

def calculate_accelerations(points_3d):
    """3D座標の時系列データから加速度の大きさを計算する"""
    num_frames, num_kps, _ = points_3d.shape
    accelerations = np.full((num_frames, num_kps), np.nan)
    accel_vectors = np.diff(points_3d, n=2, axis=0)
    accel_norms = np.linalg.norm(accel_vectors, axis=2)
    accelerations[2:] = accel_norms
    return accelerations

def apply_interpolation_and_filter(points_3d):
    """単一の3D点群データに対して、欠損値補間と平滑化フィルタを適用する"""
    print("データの後処理中 (補間とフィルタリング)...")
    processed_data = points_3d.copy()

    # 1. スプライン補間
    for kp_idx in tqdm(range(processed_data.shape[1]), desc="  スプライン補間"):
        for axis_idx in range(processed_data.shape[2]):
            seq = processed_data[:, kp_idx, axis_idx]
            is_nan, nan_indices = np.isnan(seq), np.where(np.isnan(seq))[0]
            if not len(nan_indices): continue

            gaps = np.split(nan_indices, np.where(np.diff(nan_indices) != 1)[0] + 1)
            for gap in gaps:
                if len(gap) > MAX_INTERPOLATION_GAP: continue

                start, end = gap[0], gap[-1]
                prev_indices = np.where(~is_nan & (np.arange(len(seq)) < start))[0]
                next_indices = np.where(~is_nan & (np.arange(len(seq)) > end))[0]

                if len(prev_indices) < 2 or len(next_indices) < 2: continue

                interp_indices = np.concatenate([prev_indices[-5:], next_indices[:5]])
                if len(np.unique(interp_indices)) < 4: continue

                cs = CubicSpline(interp_indices, seq[interp_indices])
                processed_data[gap, kp_idx, axis_idx] = cs(gap)

    # 2. バターワースフィルタ
    for kp_idx in tqdm(range(processed_data.shape[1]), desc="  バターワースフィルタ"):
        for axis_idx in range(processed_data.shape[2]):
            seq = processed_data[:, kp_idx, axis_idx]
            valid_mask = ~np.isnan(seq)
            if np.sum(valid_mask) > 8: # フィルタ適用に必要な最低限のデータ数
                b, a = butter(4, BUTTERWORTH_CUTOFF / (FRAME_RATE / 2), btype='low')
                processed_data[valid_mask, kp_idx, axis_idx] = filtfilt(b, a, seq[valid_mask])

    return processed_data

# ★★★ 変更点: 引数にraw_confidencesを追加 ★★★
def save_and_visualize_results(output_dir, file_prefix, frames, raw_unprocessed, raw_processed, corrected_unprocessed, final_corrected, raw_accelerations, corrected_accelerations, raw_confidences):
    """
    結果をJSONファイルに保存し、比較グラフを生成
    """
    print("結果の保存と可視化...")
    output_dir.mkdir(exist_ok=True, parents=True)

    # JSONにも4種類のデータを保存
    output_data = []
    for i, frame_name in enumerate(frames):
        person_data = {
            "raw_unprocessed_3d": raw_unprocessed[i].tolist(),
            "raw_processed_3d": raw_processed[i].tolist(),
            "corrected_unprocessed_3d": corrected_unprocessed[i].tolist(),
            "final_corrected_3d": final_corrected[i].tolist()
        }
        # ★★★ 変更点: 信頼度情報を確実に追加 ★★★
        person_data["confidence"] = raw_confidences[i].tolist()

        output_data.append({"frame_name": frame_name, "person_1": person_data})

    with open(output_dir / f"{file_prefix}_3d_results.json", 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"  ✓ JSONファイルを保存しました: {file_prefix}_3d_results.json")

    graphs_dir = output_dir / "graphs"; graphs_dir.mkdir(exist_ok=True)
    time_axis = np.arange(len(frames))
    for r_idx, l_idx, name in KEYPOINT_PAIRS:
        fig, axes = plt.subplots(3, 2, figsize=(20, 15), sharex=True)
        fig.suptitle(f"{name} (KP {r_idx} & {l_idx}) Trajectory Analysis", fontsize=16)
        for axis in range(3):
            ax0 = axes[axis, 0]; ax1 = axes[axis, 1]
            # --- 左列: 未処理データの比較 (Raw vs Corrected) ---
            ax0.plot(time_axis, raw_unprocessed[:, r_idx, axis], 'o', color='lightcoral', markersize=2, alpha=0.5, label=f'Right {name} (Raw Unprocessed)')
            ax0.plot(time_axis, raw_unprocessed[:, l_idx, axis], 'o', color='lightblue', markersize=2, alpha=0.5, label=f'Left {name} (Raw Unprocessed)')
            ax0.plot(time_axis, corrected_unprocessed[:, r_idx, axis], 'r-', label=f'Right {name} (Corrected Unprocessed)')
            ax0.plot(time_axis, corrected_unprocessed[:, l_idx, axis], 'b-', label=f'Left {name} (Corrected Unprocessed)')
            ax0.set_title(f'Unprocessed Data Comparison - {"XYZ"[axis]} axis'); ax0.set_ylabel('Position (mm)'); ax0.grid(True); ax0.legend()

            # --- 右列: 処理済みデータの比較 (Raw vs Corrected) ---
            ax1.plot(time_axis, raw_processed[:, r_idx, axis], 'r--', alpha=0.7, label=f'Right {name} (Raw Processed)')
            ax1.plot(time_axis, final_corrected[:, r_idx, axis], 'r-', label=f'Right {name} (Final Corrected)')
            ax1.plot(time_axis, raw_processed[:, l_idx, axis], 'b--', alpha=0.7, label=f'Left {name} (Raw Processed)')
            ax1.plot(time_axis, final_corrected[:, l_idx, axis], 'b-', label=f'Left {name} (Final Corrected)')
            ax1.set_title(f'Processed Data Comparison - {"XYZ"[axis]} axis'); ax1.grid(True); ax1.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(graphs_dir / f"{file_prefix}_{name}_trajectory.png"); plt.close()
    print(f"  ✓ 軌道グラフを {graphs_dir} に保存しました。")

    # 加速度プロット
    for r_idx, l_idx, name in KEYPOINT_PAIRS:
        fig, axes = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
        fig.suptitle(f"{name} (KP {r_idx} & {l_idx}) Acceleration Analysis", fontsize=16)
        axes[0].plot(time_axis, raw_accelerations[:, r_idx], 'r-', alpha=0.5, label=f'Right {name} (Raw Accel)')
        axes[0].plot(time_axis, corrected_accelerations[:, r_idx], 'r-', linewidth=2, label=f'Right {name} (Final Accel)')
        axes[0].axhline(y=ACCELERATION_THRESHOLD, color='k', linestyle='--', label='Threshold')
        axes[0].set_title(f'Right {name} Acceleration'); axes[0].set_ylabel('Acceleration (mm/frame^2)'); axes[0].legend(); axes[0].grid(True)
        axes[1].plot(time_axis, raw_accelerations[:, l_idx], 'b-', alpha=0.5, label=f'Left {name} (Raw Accel)')
        axes[1].plot(time_axis, corrected_accelerations[:, l_idx], 'b-', linewidth=2, label=f'Left {name} (Final Accel)')
        axes[1].axhline(y=ACCELERATION_THRESHOLD, color='k', linestyle='--', label='Threshold')
        axes[1].set_title(f'Left {name} Acceleration'); axes[1].set_xlabel('Frame'); axes[1].set_ylabel('Acceleration (mm/frame^2)'); axes[1].legend(); axes[1].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(graphs_dir / f"{file_prefix}_{name}_acceleration.png"); plt.close()
    print(f"  ✓ 加速度グラフを {graphs_dir} に保存しました。")


# =============================================================================
# メイン実行部
# =============================================================================
def main():
    """メインの処理パイプライン"""
    directions = ["fl", "fr"]
    try:
        params_cam1 = load_camera_parameters(STEREO_CALI_DIR / directions[0] / "camera_params_with_ext_OC.json")
        params_cam2 = load_camera_parameters(STEREO_CALI_DIR / directions[1] / "camera_params_with_ext_OC.json")
        P1, P2 = create_projection_matrix(params_cam1), create_projection_matrix(params_cam2)
    except FileNotFoundError as e:
        print(f"✗ エラー: カメラパラメータファイルが見つかりません。{e}"); return
    subject_dirs = sorted([d for d in ROOT_DIR.iterdir() if d.is_dir() and d.name.startswith("sub")])
    for subject_dir in subject_dirs:
        thera_dirs = sorted([d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("thera")])
        for thera_dir in thera_dirs:
            # NOTE: デバッグ用に処理対象を限定
            if subject_dir.name != "sub1" or thera_dir.name != "thera0-2": continue
            print(f"\n{'='*80}\n処理開始: {thera_dir.relative_to(ROOT_DIR)}")
            openpose_dir1, openpose_dir2 = thera_dir / directions[0] / "openpose.json", thera_dir / directions[1] / "openpose.json"
            if not (openpose_dir1.exists() and openpose_dir2.exists()):
                print(f"  - スキップ: OpenPoseディレクトリが見つかりません。"); continue
            kps1_seq, kps2_seq, frames = load_2d_data(openpose_dir1, openpose_dir2)
            if not frames:
                print(f"  - スキップ: 共通フレームが見つかりません。"); continue

            # --- データ処理のフロー ---

            # 1. 未処理の生データ (Unprocessed Raw) と信頼度
            # ★★★ 変更点: 信頼度も受け取る ★★★
            raw_unprocessed, raw_confidences = calculate_raw_3d_coordinates(kps1_seq, kps2_seq, P1, P2)

            # 2. 未処理の補正済みデータ (Unprocessed Corrected)
            corrected_unprocessed = apply_frame_by_frame_correction(raw_unprocessed, kps1_seq, kps2_seq, P1, P2)

            # 3. 処理済みの生データ (Processed Raw) - 未処理データを基に生成
            raw_processed = apply_interpolation_and_filter(raw_unprocessed)

            # 4. 最終的な補正済みデータ (Final Corrected) - 未処理の補正済みデータを基に生成
            final_corrected = apply_interpolation_and_filter(corrected_unprocessed)

            # --------------------------------------------------------------------

            output_dir = thera_dir / "3d_gait_analysis_kalman_v3"
            # 加速度は、処理前の純粋な生データと、最終的な処理済みデータで計算
            raw_accelerations = calculate_accelerations(raw_unprocessed)
            corrected_accelerations = calculate_accelerations(final_corrected)

            save_and_visualize_results(
                output_dir, thera_dir.name, frames,
                raw_unprocessed=raw_unprocessed,
                raw_processed=raw_processed,
                corrected_unprocessed=corrected_unprocessed,
                final_corrected=final_corrected,
                raw_accelerations=raw_accelerations,
                corrected_accelerations=corrected_accelerations,
                # ★★★ 変更点: 信頼度を渡す ★★★
                raw_confidences=raw_confidences
            )
            print(f"処理が正常に完了しました: {thera_dir.name}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nエラーが発生しました: {e}"); traceback.print_exc()
