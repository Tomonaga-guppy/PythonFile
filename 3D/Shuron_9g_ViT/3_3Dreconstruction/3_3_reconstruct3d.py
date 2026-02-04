"""
3_3_reconstruct3d.py (parallel)
==============================
fl, fr, sagi の3視点から3D三角測量を行い、補間・フィルタリングして npz 保存

※「出力に影響する処理」は一切変えず、処理単位（sub/thera/method）を並列化して高速化します。
  - 計算ロジック（triangulation / conf filter / spline / butter / plot / save npz）は同一
  - 変更点は「実行の仕方（並列実行）」と「進捗表示（全体）」のみ

並列化の粒度
------------
1つのジョブ = 1 subject(sub) × 1 trial(thera) × 1 method(例: ViTPose) の組
各ジョブは出力ファイルが別なので競合しません。

Windows注意
-----------
ProcessPoolExecutor を使うため、必ず
if __name__ == "__main__": main()
の形で実行してください（このファイルはそうなっています）。
"""

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# optional（プロットを使う場合のみ）
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

from concurrent.futures import ProcessPoolExecutor, as_completed

from m_triangulation import triangulate_and_rotate_multi


# =============================================================================
# 設定（ここだけ調整）
# =============================================================================
ROOT_DIR = Path(r"G:\gait_pattern\2025_shuron_BR9G")  # ★あなたの現状

EXT_DIRNAME = "extparams"
EXT_JSON_PART_NAME = "camera_params_with_ext"

# 3視点（fl, fr, sagi）で三角測量します
DIRECTIONS = ["fl", "fr", "sagi"]

CSV_SUFFIXES = [
    "_PA_spline.csv",
    "_PT_spline.csv",
    "_PA.csv",
    "_PT.csv",
]

TARGET_METHOD = "ViTPose"  # 対象とする手法名（OpenPoseの結果は使用しない）

CONF_TH_3D = 0.4  # 3D信頼度閾値（この値以下の3D点はNaNにする）
VALID_RANGE_Z = (-2000, 2000)  # MidHipのZ座標による有効範囲[mm] およそ+-2mになるように

# 1フレームでの3Dジャンプがこの閾値[mm]を超える点は外れ値としてNaN化
OUTLIER_JUMP_MM = 100.0

USE_BUTTERWORTH = True
BUTTERWORTH_CUTOFF = 6.0
FRAME_RATE = 60

SAVE_TIMESERIES_PLOTS = True

# 並列数（出力は変わりません。PC負荷と相談して調整）
# 目安: CPUコア数-1。0や負数は自動調整。
MAX_WORKERS = 4


# =============================================================================
# ユーティリティ
# =============================================================================
def is_target_thera(name: str) -> bool:
    if not name.startswith("thera"):
        return False
    return True


def natural_sort_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def load_camera_parameters(params_file: Path) -> dict:
    with open(params_file, "r", encoding="utf-8") as f:
        return json.load(f)


def create_projection_matrix(params: dict, extr_key: str = "extrinsics") -> np.ndarray:
    K = np.array(params["intrinsics"], dtype=float)
    ex = params[extr_key]
    R = np.array(ex["rotation_matrix"], dtype=float)
    t = np.array(ex["translation_vector"], dtype=float).reshape(3, 1)
    return K @ np.hstack([R, t])



def load_csv_2d_data_multi(csv_paths: List[Path]):
    """複数視点の2DキーポイントCSVを読み込み、長さを揃えて返す。

    戻り値
    ------
    kps_list : list[np.ndarray]
        各視点の (num_frames, 25, 3)
    frames : list[int]
    """

    # frame列がある場合は落とす（76列→75列）
    def drop_frame_if_needed(df: pd.DataFrame) -> pd.DataFrame:
        if "frame" in df.columns:
            return df.drop(columns=["frame"])
        if df.shape[1] != 75 and df.shape[1] > 75:
            first = df.iloc[:10, 0].to_numpy()
            if np.all(np.diff(first) >= 0):
                return df.iloc[:, 1:]
        return df

    arrays = []
    lengths = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df = drop_frame_if_needed(df)
        a = df.to_numpy(dtype=float)
        if a.shape[1] != 75:
            raise ValueError(f"CSV columns must be 75 after dropping frame: {p} (shape={a.shape})")
        arrays.append(a)
        lengths.append(len(a))

    n = int(min(lengths)) if lengths else 0
    if n == 0:
        return None, None

    kps_list = [a[:n].reshape(-1, 25, 3) for a in arrays]
    frames = list(range(n))
    return kps_list, frames


# =============================================================================
# 3D（三角測量）
# =============================================================================
def calculate_raw_3d_coordinates_multi(kps_seq_list, P_list):
    """複数視点の2D系列から3D系列を計算。"""
    num_frames = len(kps_seq_list[0])
    raw_3d = np.full((num_frames, 25, 3), np.nan, dtype=float)
    conf_3d = np.full((num_frames, 25), np.nan, dtype=float)

    for i in range(num_frames):
        pts_list = [kps_seq_list[v][i][:, :2] for v in range(len(kps_seq_list))]
        cfs_list = [kps_seq_list[v][i][:, 2] for v in range(len(kps_seq_list))]
        raw_3d[i], conf_3d[i] = triangulate_and_rotate_multi(P_list, pts_list, cfs_list)

    return raw_3d, conf_3d


def confidence_filter_keypoints(data_3d, confidences, conf_threshold=0.4):
    filtered = data_3d.copy()
    low = confidences < conf_threshold
    filtered[low] = np.nan
    return filtered

def remove_jump_outliers_3d(data_3d: np.ndarray, jump_th_mm: float = 100.0) -> np.ndarray:
    """フレーム間の急激なジャンプを外れ値としてNaN化する。

    連続フレーム間で ||p_t - p_{t-1}|| > jump_th_mm のとき、そのフレームtの点をNaN化する。
    NaNをまたぐ比較は行わず、次に有効な点から再開する。
    """
    out = data_3d.copy()
    num_frames, num_kp, _ = out.shape

    for kp in range(num_kp):
        prev = out[0, kp].copy()
        prev_ok = np.isfinite(prev).all()

        for i in range(1, num_frames):
            cur = out[i, kp]
            cur_ok = np.isfinite(cur).all()

            if prev_ok and cur_ok:
                if np.linalg.norm(cur - prev) > jump_th_mm:
                    out[i, kp] = np.nan
                    # prev は更新しない（外れ値に引っ張られないようにする）
                    prev_ok = False
                    continue
                prev = cur.copy()
                prev_ok = True
            elif cur_ok:
                # ここから新しい連続区間として開始
                prev = cur.copy()
                prev_ok = True
            # curがNaNなら何もしない（prevは維持）

    return out

def remove_short_valid_runs_3d(data_3d: np.ndarray, min_len: int = 12) -> np.ndarray:
    """各キーポイントの「連続して有効な区間」がmin_len未満なら、その区間をNaN化する。

    - 有効判定: xyzすべてがfiniteのフレーム
    - 連続区間長 < min_len を削除（NaN化）
    """
    out = data_3d.copy()
    n, k, _ = out.shape

    for kp in range(k):
        valid = np.isfinite(out[:, kp, :]).all(axis=1)
        i = 0
        while i < n:
            if not valid[i]:
                i += 1
                continue
            j = i
            while j < n and valid[j]:
                j += 1
            run_len = j - i
            if run_len < min_len:
                out[i:j, kp, :] = np.nan
            i = j

    return out


def detect_valid_frame_range(data_3d, z_min=-2000, z_max=2000, midhip_idx=8):
    num_frames = len(data_3d)
    midhip_z = data_3d[:, midhip_idx, 2]
    valid = (~np.isnan(midhip_z)) & (midhip_z >= z_min) & (midhip_z <= z_max)

    if not np.any(valid):
        return 0, num_frames - 1

    idx = np.where(valid)[0]
    s, e = int(idx[0]), int(idx[-1])
    return s, e


# =============================================================================
# 補間 & フィルタ
# =============================================================================
def _fill_short_gaps_cubic_1d(y: np.ndarray, max_gap: int, neighbor_pts: int = 2) -> np.ndarray:
    """
    1D系列のNaN欠損を、連続欠損run長 < max_gap の区間だけ局所CubicSplineで埋める。
    neighbor_pts: run前後から使う有効点数（片側）。2なら最大4点でcubic、足りなければ線形。
    """
    out = y.copy()
    n = len(out)
    i = 0
    while i < n:
        if not np.isnan(out[i]):
            i += 1
            continue

        # 欠損run [i, j)
        j = i
        while j < n and np.isnan(out[j]):
            j += 1

        run_len = j - i
        if run_len >= max_gap:
            i = j
            continue  # 長欠損は埋めない

        # 前後の有効点を集める
        left_idx = []
        k = i - 1
        while k >= 0 and len(left_idx) < neighbor_pts:
            if np.isfinite(out[k]):
                left_idx.append(k)
            k -= 1
        left_idx = left_idx[::-1]

        right_idx = []
        k = j
        while k < n and len(right_idx) < neighbor_pts:
            if np.isfinite(out[k]):
                right_idx.append(k)
            k += 1

        xs = np.array(left_idx + right_idx, dtype=int)
        ys = out[xs]
        x_fill = np.arange(i, j, dtype=int)

        if len(xs) >= 4 and np.all(np.diff(xs) > 0):
            cs = CubicSpline(xs, ys)
            out[x_fill] = cs(x_fill)
        elif len(xs) >= 2:
            out[x_fill] = np.interp(x_fill, xs, ys)
        # それすら無理なら埋めない（NaNのまま）

        i = j

    return out

def expand_nan_gaps_1d(y: np.ndarray, max_gap: int, expand: int = 2) -> np.ndarray:
    """
    連続NaN run長 < max_gap の欠損について，
    その前後 expand フレームも含めて NaN 化する（非カスケード版）
    """
    n = len(y)
    nan0 = np.isnan(y)  # ★元のマスクを固定
    if not nan0.any():
        return y.copy()

    to_nan = nan0.copy()
    i = 0
    while i < n:
        if not nan0[i]:
            i += 1
            continue

        j = i
        while j < n and nan0[j]:
            j += 1

        run_len = j - i
        if run_len < max_gap:
            s = max(0, i - expand)
            e = min(n, j + expand)
            to_nan[s:e] = True

        i = j

    out = y.copy()
    out[to_nan] = np.nan
    return out

def spline_interpolate(data, max_gap: int = 30):
    """
    3DデータのNaN欠損を、連続欠損run長 < max_gap(0.2秒：12フレーム以内 の区間だけ補間する。
    →したかったけど欠損がそれよりも長くなることが多いので結局30フレームまでは補間することにした。（ただ根拠が弱い）
    長欠損(>=max_gap)はNaNのまま残す。
    """
    interp = np.copy(data)
    num_frames, num_kp, _ = interp.shape

    for kp in range(num_kp):
        for c in range(3):
            s = interp[:, kp, c].astype(float)

            # 有効点が2点未満なら何もできない
            if np.isfinite(s).sum() < 2:
                continue

            # 短欠損だけ埋める（局所補間）
            interp[:, kp, c] = _fill_short_gaps_cubic_1d(
                s, max_gap=max_gap, neighbor_pts=2
            )

            # # 短欠損の前後2フレームも削除
            # s2 = expand_nan_gaps_1d(s, max_gap=max_gap, expand=2)

            # interp[:, kp, c] = _fill_short_gaps_cubic_1d(
            #     s2, max_gap=max_gap, neighbor_pts=2
            # )

    return interp


def butterworth_filter(data, cutoff, fs, order=4):
    out = data.copy()
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")

    num_frames, num_kp, _ = out.shape
    for kp in range(num_kp):
        for c in range(3):
            s = out[:, kp, c]
            m = ~np.isnan(s)
            if np.sum(m) < (order * 3 + 1):
                continue
            try:
                out[m, kp, c] = filtfilt(b, a, s[m])
            except Exception:
                pass
    return out


# =============================================================================
# 3D時系列プロット（必要なら）
# =============================================================================
KEYPOINT_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe",
    "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
]


def plot_keypoint_timeseries(data_3d_dict, conf_3d, save_dir: Path, frame_range=None):
    save_dir.mkdir(parents=True, exist_ok=True)

    any_key = next(iter(data_3d_dict.keys()))
    num_frames = data_3d_dict[any_key].shape[0]

    if frame_range is None:
        s, e = 0, num_frames - 1
    else:
        s, e = frame_range

    frames = np.arange(s, e + 1)
    coord_labels = ["X (mm)", "Y (mm)", "Z (mm)"]

    for kp_idx in range(25):
        kp_name = KEYPOINT_NAMES[kp_idx]
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f"{kp_name} (kp{kp_idx:02d}) - 3D time series", fontsize=14)

        for c in range(3):
            ax = axes[c]
            for name, arr in data_3d_dict.items():
                series = arr[s:e+1, kp_idx, c]
                if name == "raw":
                    label_width = 10
                    _alpha = 0.4
                elif name == "outlier_filt":
                    label_width = 5
                    _alpha = 0.6
                else:
                    label_width = 1.2
                    _alpha = 0.9
                ax.plot(frames, series, linewidth=label_width, label=name, alpha=_alpha)
            ax.set_ylabel(coord_labels[c])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=4)

        axc = axes[3]
        cc = conf_3d[s:e+1, kp_idx]
        axc.plot(frames, cc, linewidth=1.2, label="conf")
        axc.set_ylim(0, 1.05)
        axc.set_ylabel("Confidence")
        axc.set_xlabel("Frame")
        axc.grid(True, alpha=0.3)
        axc.legend()

        out = save_dir / f"kp{kp_idx:02d}_{kp_name}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        # if kp_name == "RHeel" and save_dir.name.endswith("_PA"):
        #     plt.show()
        plt.close(fig)


# =============================================================================
# 入力探索
# =============================================================================
def collect_pairs_for_thera(thera_dir: Path, csv_suffix: str):
    gopro = thera_dir / "gopro"
    fl_root = gopro / "fl"
    fr_root = gopro / "fr"
    sagi_root = gopro / "sagi"
    if not (fl_root.exists() and fr_root.exists() and sagi_root.exists()):
        return []

    fl_methods, fr_methods, sagi_methods = {}, {}, {}

    for d in fl_root.iterdir():
        if not d.is_dir():
            continue
        c = d / f"{d.name}{csv_suffix}"
        if c.exists():
            fl_methods[d.name] = c

    for d in fr_root.iterdir():
        if not d.is_dir():
            continue
        c = d / f"{d.name}{csv_suffix}"
        if c.exists():
            fr_methods[d.name] = c

    for d in sagi_root.iterdir():
        if not d.is_dir():
            continue
        c = d / f"{d.name}{csv_suffix}"
        if c.exists():
            sagi_methods[d.name] = c

    # 3つすべてに同じmethodフォルダがあるものだけ採用
    common = set(fl_methods) & set(fr_methods) & set(sagi_methods)
    
    # 対象をViTPoseのみに限定
    common = [m for m in common if m == TARGET_METHOD]

    common = sorted(common, key=lambda x: natural_sort_key(Path(x)))
    return [(m, fl_methods[m], fr_methods[m], sagi_methods[m]) for m in sorted(common)]


def load_projection_matrices_for_subject(sub_dir: Path):
    # fl/fr は通常そのまま、sagi は _aligned が付く場合があるので両対応
    fl_dir = sub_dir / "cali" / EXT_DIRNAME / "fl"
    fr_dir = sub_dir / "cali" / EXT_DIRNAME / "fr"
    sagi_dir = sub_dir / "cali" / EXT_DIRNAME / "sagi"
    # print(f"Loading extparams from: {fl_dir}, {fr_dir}, {sagi_dir}")

    fl_json = list(fl_dir.glob(f"{EXT_JSON_PART_NAME}*01.json"))[0]
    fr_json = list(fr_dir.glob(f"{EXT_JSON_PART_NAME}*01.json"))[0]
    sagi_json = list(sagi_dir.glob(f"{EXT_JSON_PART_NAME}*01_aligned*.json"))[0]

    params_fl = load_camera_parameters(fl_json)
    params_fr = load_camera_parameters(fr_json)
    params_sagi = load_camera_parameters(sagi_json)

    P1 = create_projection_matrix(params_fl, extr_key="extrinsics")
    P2 = create_projection_matrix(params_fr, extr_key="extrinsics")
    # sagiは aligned キーがあればそれを使う（なければ通常）
    sagi_extr_key = "extrinsics_aligned_to_flfr_pair" if "extrinsics_aligned_to_flfr_pair" in params_sagi else "extrinsics"
    P3 = create_projection_matrix(params_sagi, sagi_extr_key)
    
    # print(f"fl_json: {fl_json}")
    # print(f"fr_json: {fr_json}")
    # print(f"sagi_json: {sagi_json}")
    return [P1, P2, P3], [fl_json, fr_json, sagi_json]


# =============================================================================
# 並列ジョブ定義
# =============================================================================
@dataclass(frozen=True)
class Job:
    sub_dir: str
    thera_dir: str
    method_name: str
    csv_fl: str
    csv_fr: str
    csv_sagi: str
    csv_suffix: str


def _run_one_job(job: Job) -> Tuple[bool, str, float]:
    """
    1ジョブを実行して結果を返す。
    返り値: (ok, message, seconds)
    """
    t0 = time.time()
    sub = Path(job.sub_dir)
    thera = Path(job.thera_dir)
    method_name = job.method_name
    csv_fl = Path(job.csv_fl)
    csv_fr = Path(job.csv_fr)
    csv_sagi = Path(job.csv_sagi)

    try:
        P_list, json_paths = load_projection_matrices_for_subject(sub)

        kps_list, frames = load_csv_2d_data_multi([csv_fl, csv_fr, csv_sagi])
        if frames is None or len(frames) == 0:
            return False, f"[SKIP] empty frames: {thera.name} / {method_name}", time.time() - t0

        raw_3d, conf_3d = calculate_raw_3d_coordinates_multi(kps_list, P_list)

        # 信頼度によるフィルタリング
        conf_filt_3d = confidence_filter_keypoints(raw_3d, conf_3d, conf_threshold=CONF_TH_3D)

        # 1フレームで100mm以上のジャンプがある点を外れ値として除外（NaN化）
        outlier_filt_3d_0 = remove_jump_outliers_3d(conf_filt_3d, jump_th_mm=OUTLIER_JUMP_MM)
        # outlier_filt後に残った「連続して有効な区間」が短すぎるものは削除（NaN化）
        outlier_filt_3d = remove_short_valid_runs_3d(outlier_filt_3d_0, min_len=3)


        spline_3d = spline_interpolate(outlier_filt_3d)

        butter_3d = None
        if USE_BUTTERWORTH:
            butter_3d = butterworth_filter(spline_3d, BUTTERWORTH_CUTOFF, FRAME_RATE)

        if VALID_RANGE_Z is not None:
            s, e = detect_valid_frame_range(
                butter_3d, z_min=VALID_RANGE_Z[0], z_max=VALID_RANGE_Z[1]
            )
        else:
            s, e = 0, len(frames) - 1
        # print(f"Valid frame range for {csv_sagi.name} {sub.name}/{thera.name}/{method_name}: {s} - {e} / {len(frames)}")

        csv_suffix = job.csv_suffix
        csv_tag = csv_suffix.replace(".csv", "").lstrip("_")
        
        out_npz = thera / f"3d_kp_{method_name}_{csv_tag}.npz"
        np.savez(
            out_npz,
            frame=np.array(frames, dtype=int),
            raw=raw_3d,
            conf_filt=conf_filt_3d,
            outlier_filt=outlier_filt_3d,
            spline=spline_3d,
            butter=(butter_3d if butter_3d is not None else np.array([])),
            conf=conf_3d,
            valid_frame_range=np.array([s, e], dtype=int),
            meta=np.array(
                [
                    f"sub={sub.name}",
                    f"thera={thera.name}",
                    f"method={method_name}",
                    f"csv_suffix={csv_suffix}",
                    f"ext_fl={json_paths[0].name}",
                    f"ext_fr={json_paths[1].name}",
                    f"ext_sagi={json_paths[2].name}",
                ],
                dtype=object,
            ),
        )

        if SAVE_TIMESERIES_PLOTS:
            ts_dir = thera / f"keypoint_timeseries_3d_{method_name}_{csv_tag}"
            data_dict = {"raw": raw_3d, "conf_filt": conf_filt_3d, "outlier_filt": outlier_filt_3d, "spline": spline_3d}
            if butter_3d is not None and getattr(butter_3d, "size", 0) != 0:
                data_dict["butter"] = butter_3d
            plot_keypoint_timeseries(data_dict, conf_3d, ts_dir, frame_range=(s, e))

        sec = time.time() - t0
        return True, f"[OK] {sub.name}/{thera.name}/{method_name}/{csv_tag}  ({out_npz.name})", sec

    except Exception as e:
        sec = time.time() - t0
        return False, f"[ERR] {sub.name}/{thera.name}/{method_name}: {e}", sec

def _build_jobs(root_dir: Path) -> List[Job]:
    jobs: List[Job] = []

    subject_dirs = sorted(
        [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("sub")],
        key=natural_sort_key
    )

    for sub in subject_dirs:
        try:
            _ = load_projection_matrices_for_subject(sub)
        except Exception:
            continue

        thera_dirs = sorted(
            [d for d in sub.iterdir() if d.is_dir() and is_target_thera(d.name)],
            key=natural_sort_key
        )

        for thera in thera_dirs:
            for csv_suffix in CSV_SUFFIXES:
                pairs = collect_pairs_for_thera(thera, csv_suffix)
                for method_name, csv_fl, csv_fr, csv_sagi in pairs:
                    jobs.append(Job(
                        sub_dir=str(sub),
                        thera_dir=str(thera),
                        method_name=method_name,
                        csv_fl=str(csv_fl),
                        csv_fr=str(csv_fr),
                        csv_sagi=str(csv_sagi),
                        csv_suffix=csv_suffix,
                    ))
    return jobs


# =============================================================================
# メイン
# =============================================================================
def main():
    jobs = _build_jobs(ROOT_DIR)
    if not jobs:
        print("No jobs found. Check folder structure / extparams / thera filter.")
        return

    # worker数決定
    cpu = os.cpu_count() or 1
    if MAX_WORKERS and MAX_WORKERS > 0:
        workers = MAX_WORKERS
    else:
        workers = max(1, cpu - 1)

    print("=" * 80)
    print(f"ROOT: {ROOT_DIR}")
    print(f"Jobs: {len(jobs)}")
    print(f"Workers: {workers} (cpu={cpu})")
    print(f"CSV_SUFFIXES: {CSV_SUFFIXES}")
    print(f"thera filter: name startswith 'thera'")
    print("=" * 80)

    ok_count = 0
    ng_count = 0
    total_sec = 0.0

    # 全体進捗（ジョブ単位）
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_run_one_job, job) for job in jobs]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="All jobs", unit="job"):
            ok, msg, sec = fut.result()
            total_sec += sec
            if ok:
                ok_count += 1
            else:
                ng_count += 1

            # 進捗が見やすいように、最後に短いログだけ出す（大量ならコメントアウト可）
            tqdm.write(f"{msg}  [{sec:.2f}s]")

    print("=" * 80)
    print(f"DONE. ok={ok_count}  ng/skip={ng_count}  sum_worker_time={total_sec:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
