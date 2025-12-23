"""
3_3_reconstruct3d.py (parallel)
==============================
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

from concurrent.futures import ProcessPoolExecutor, as_completed

from m_triangulation import triangulate_and_rotate


# =============================================================================
# 設定（ここだけ調整）
# =============================================================================
ROOT_DIR = Path(r"G:\gait_pattern\BR9G_shuron")

EXT_DIRNAME = "extparams"
EXT_JSON_NAME = "camera_params_with_ext.json"

DIRECTIONS = ["fl", "fr"]

CSV_SUFFIX = "_pa_spline.csv"  # 例: openpose_facemasked_pa_spline.csv

CONF_TH_3D = 0.4
VALID_RANGE_X = (-2000, 2000)

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
    if not name.endswith("-0"):
        return False
    if "_" in name:
        return False
    return True


def natural_sort_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def load_camera_parameters(params_file: Path) -> dict:
    with open(params_file, "r", encoding="utf-8") as f:
        return json.load(f)


def create_projection_matrix(params: dict) -> np.ndarray:
    K = np.array(params["intrinsics"], dtype=float)
    R = np.array(params["extrinsics"]["rotation_matrix"], dtype=float)
    t = np.array(params["extrinsics"]["translation_vector"], dtype=float).reshape(3, 1)
    return K @ np.hstack([R, t])


def load_csv_2d_data(csv_path1: Path, csv_path2: Path):
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)

    # frame列がある場合は落とす（76列→75列）
    def drop_frame_if_needed(df: pd.DataFrame) -> pd.DataFrame:
        if "frame" in df.columns:
            return df.drop(columns=["frame"])
        if df.shape[1] != 75 and df.shape[1] > 75:
            first = df.iloc[:10, 0].to_numpy()
            if np.all(np.diff(first) >= 0):
                return df.iloc[:, 1:]
        return df

    df1 = drop_frame_if_needed(df1)
    df2 = drop_frame_if_needed(df2)

    a1 = df1.to_numpy(dtype=float)
    a2 = df2.to_numpy(dtype=float)

    n = min(len(a1), len(a2))
    if n == 0:
        return None, None, None

    a1 = a1[:n]
    a2 = a2[:n]

    if a1.shape[1] != 75 or a2.shape[1] != 75:
        raise ValueError(
            f"CSV columns must be 75 after dropping frame. "
            f"csv1={a1.shape}, csv2={a2.shape}"
        )

    kps1 = a1.reshape(-1, 25, 3)
    kps2 = a2.reshape(-1, 25, 3)

    frames = list(range(n))
    return kps1, kps2, frames


# =============================================================================
# 3D（三角測量）
# =============================================================================
def calculate_raw_3d_coordinates(kps1_seq, kps2_seq, P1, P2):
    num_frames = len(kps1_seq)
    raw_3d = np.full((num_frames, 25, 3), np.nan, dtype=float)
    conf_3d = np.full((num_frames, 25), np.nan, dtype=float)

    # NOTE: 並列時に内部 tqdm を出すとログが崩れるため、表示は外側の全体進捗に統一。
    for i in range(num_frames):
        kp1, cf1 = kps1_seq[i][:, :2], kps1_seq[i][:, 2]
        kp2, cf2 = kps2_seq[i][:, :2], kps2_seq[i][:, 2]
        raw_3d[i], conf_3d[i] = triangulate_and_rotate(P1, P2, kp1, kp2, cf1, cf2)

    return raw_3d, conf_3d


def confidence_filter_keypoints(data_3d, confidences, conf_threshold=0.5):
    filtered = data_3d.copy()
    low = confidences < conf_threshold
    filtered[low] = np.nan
    return filtered


def detect_valid_frame_range(data_3d, x_min=-2500, x_max=2500, midhip_idx=8):
    num_frames = len(data_3d)
    midhip_x = data_3d[:, midhip_idx, 0]
    valid = (~np.isnan(midhip_x)) & (midhip_x >= x_min) & (midhip_x <= x_max)

    if not np.any(valid):
        return 0, num_frames - 1

    idx = np.where(valid)[0]
    s, e = int(idx[0]), int(idx[-1])
    return s, e


# =============================================================================
# 補間 & フィルタ
# =============================================================================
def spline_interpolate(data):
    interp = np.copy(data)
    num_frames, num_kp, _ = interp.shape

    for kp in range(num_kp):
        for c in range(3):
            s = interp[:, kp, c]
            m = ~np.isnan(s)
            if np.sum(m) < 2:
                continue
            xs = np.where(m)[0]
            try:
                cs = CubicSpline(xs, s[m])
                interp[:, kp, c] = cs(np.arange(num_frames))
            except Exception:
                ser = pd.Series(s)
                interp[:, kp, c] = ser.interpolate(limit_direction="both").to_numpy()

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
                ax.plot(frames, series, linewidth=1.2, label=name)
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
        plt.close(fig)


# =============================================================================
# 入力探索
# =============================================================================
def collect_pairs_for_thera(thera_dir: Path):
    gopro = thera_dir / "gopro"
    fl_root = gopro / "fl"
    fr_root = gopro / "fr"
    if not (fl_root.exists() and fr_root.exists()):
        return []

    fl_methods = {}
    for d in sorted([p for p in fl_root.iterdir() if p.is_dir()], key=natural_sort_key):
        c = d / f"{d.name}{CSV_SUFFIX}"
        if c.exists():
            fl_methods[d.name] = c

    pairs = []
    for d in sorted([p for p in fr_root.iterdir() if p.is_dir()], key=natural_sort_key):
        c = d / f"{d.name}{CSV_SUFFIX}"
        if c.exists() and d.name in fl_methods:
            pairs.append((d.name, fl_methods[d.name], c))

    return pairs


def load_projection_matrices_for_subject(sub_dir: Path):
    fl_json = sub_dir / "cali" / EXT_DIRNAME / DIRECTIONS[0] / EXT_JSON_NAME
    fr_json = sub_dir / "cali" / EXT_DIRNAME / DIRECTIONS[1] / EXT_JSON_NAME

    if not fl_json.exists():
        raise FileNotFoundError(f"ext json not found: {fl_json}")
    if not fr_json.exists():
        raise FileNotFoundError(f"ext json not found: {fr_json}")

    params_fl = load_camera_parameters(fl_json)
    params_fr = load_camera_parameters(fr_json)

    P1 = create_projection_matrix(params_fl)
    P2 = create_projection_matrix(params_fr)
    return P1, P2, fl_json, fr_json


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

    try:
        P1, P2, fl_json, fr_json = load_projection_matrices_for_subject(sub)

        kps1, kps2, frames = load_csv_2d_data(csv_fl, csv_fr)
        if frames is None or len(frames) == 0:
            return False, f"[SKIP] empty frames: {thera.name} / {method_name}", time.time() - t0

        raw_3d, conf_3d = calculate_raw_3d_coordinates(kps1, kps2, P1, P2)

        conf_filt_3d = confidence_filter_keypoints(raw_3d, conf_3d, conf_threshold=CONF_TH_3D)

        if VALID_RANGE_X is not None:
            s, e = detect_valid_frame_range(
                conf_filt_3d, x_min=VALID_RANGE_X[0], x_max=VALID_RANGE_X[1]
            )
        else:
            s, e = 0, len(frames) - 1

        spline_3d = spline_interpolate(conf_filt_3d)

        butter_3d = None
        if USE_BUTTERWORTH:
            butter_3d = butterworth_filter(spline_3d, BUTTERWORTH_CUTOFF, FRAME_RATE)

        csv_tag = CSV_SUFFIX.replace(".csv", "").lstrip("_")
        out_npz = thera / f"3d_kp_{method_name}_{csv_tag}.npz"
        np.savez(
            out_npz,
            frame=np.array(frames, dtype=int),
            raw=raw_3d,
            conf_filt=conf_filt_3d,
            spline=spline_3d,
            butter=(butter_3d if butter_3d is not None else np.array([])),
            conf=conf_3d,
            valid_frame_range=np.array([s, e], dtype=int),
            meta=np.array(
                [
                    f"sub={sub.name}",
                    f"thera={thera.name}",
                    f"method={method_name}",
                    f"csv_suffix={CSV_SUFFIX}",
                    f"ext_fl={fl_json.name}",
                    f"ext_fr={fr_json.name}",
                ],
                dtype=object,
            ),
        )

        if SAVE_TIMESERIES_PLOTS:
            ts_dir = thera / f"keypoint_timeseries_3d_{method_name}_{csv_tag}"
            data_dict = {"raw": raw_3d, "conf_filt": conf_filt_3d, "spline": spline_3d}
            if butter_3d is not None and getattr(butter_3d, "size", 0) != 0:
                data_dict["butter"] = butter_3d
            plot_keypoint_timeseries(data_dict, conf_3d, ts_dir, frame_range=(s, e))

        sec = time.time() - t0
        return True, f"[OK] {sub.name}/{thera.name}/{method_name}  ({out_npz.name})", sec

    except Exception as e:
        sec = time.time() - t0
        return False, f"[ERR] {sub.name}/{thera.name}/{method_name}: {e}", sec


def _build_jobs(root_dir: Path) -> List[Job]:
    jobs: List[Job] = []
    subject_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("sub")],
                          key=natural_sort_key)

    for sub in subject_dirs:
        # extparams が無ければ全体スキップ（ログは main 側で出す）
        fl_json = sub / "cali" / EXT_DIRNAME / DIRECTIONS[0] / EXT_JSON_NAME
        fr_json = sub / "cali" / EXT_DIRNAME / DIRECTIONS[1] / EXT_JSON_NAME
        if not (fl_json.exists() and fr_json.exists()):
            continue

        thera_dirs = sorted([d for d in sub.iterdir() if d.is_dir() and is_target_thera(d.name)],
                            key=natural_sort_key)

        for thera in thera_dirs:
            pairs = collect_pairs_for_thera(thera)
            for method_name, csv_fl, csv_fr in pairs:
                jobs.append(Job(
                    sub_dir=str(sub),
                    thera_dir=str(thera),
                    method_name=str(method_name),
                    csv_fl=str(csv_fl),
                    csv_fr=str(csv_fr),
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
    print(f"CSV_SUFFIX: {CSV_SUFFIX}")
    print(f"thera filter: name startswith 'thera' and endswith '-0' and no '_'")
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
