"""
OpenPose / ViTPose の JSON 出力（people[...].pose_keypoints_2d）を
フレーム時系列の CSV に変換するスクリプト。

主な処理内容:
- 各フレームで最大2人までの人物を抽出
- MidHip（なければ bbox 中心）を代表点とした簡易トラッキングにより
  フレーム間で人物ID（PA / PT）を安定化
- bbox 面積・有効キーポイント数・移動量に基づく人物フィルタリング
- 各人物ごとに BODY_25 キーポイントを CSV 出力

欠損処理・補間:
- x または y が 0 の座標を欠損（NaN）として扱う
- 欠損座標は 3 次スプライン補間で補完
- スプライン補間によって補完された座標点については，
  対応する confidence（p）値を 0.6 として補完する
- 補間不可能な場合は座標を 0 のまま保持し，p も 0 とする

出力:
- {base}_PA.csv / {base}_PA_spline.csv
- {base}_PT.csv / {base}_PT_spline.csv
- 各キーポイントの時系列プロット（Raw / Spline / Confidence）

用途:
- 多視点RGBカメラによる2D姿勢推定結果の前処理
- 3次元再構成・歩行解析・臨床動作解析の入力データ生成
"""

import json
import glob
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import CubicSpline

# =========================
# 設定（ここだけ調整）
# =========================
root_dir = Path(r"G:\gait_pattern\2025_shuron_tkrzk")
directions = ["fl", "fr", "sagi"]

# 採用条件（元のまま）
CONF_TH = 0.4  # confidence閾値
AREA_TH = 200 * 300  # bbox面積閾値
AREA_TH_SAGI = 450 * 200  # sagi方向のみbbox面積閾値ざっくり
MIN_VALID_KPTS = 5  # 有効キーポイント数閾値
MIN_DET_RATIO = 0.10  # 検出フレーム割合閾値

# 移動量フィルタの閾値（MIN_RANGE）はprocess_one_method内で個別設定

# 2トラックのタグ
OUT_PERSON_TAGS = ["PA", "PT"]   # 一人目, 二人目

# トラッキング閾値（画素距離）
MAX_TRACK_DIST = 100.0

# 画像座標レンジ
XLIM = (0, 3840)
YLIM = (0, 2160)

# 既存CSVスキップフラグ
SKIP_IF_CSV_EXISTS = False

# OpenPose BODY_25
keypoint_names = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
    "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel"
]

SKIP_KEYPOINTS = {
    "Nose", "Neck",
    "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist",
    "REye", "REar", "LEye", "LEar",
}

MAX_WORKERS = 4

# =========================
# 探索
# =========================
def iter_poseestimate_json_dirs(root: Path, directions_list):
    for sub in sorted(root.glob("sub*")):
        if not sub.is_dir():
            continue
        for thera in sub.iterdir():
            if not (thera.is_dir() and thera.name.startswith("thera")):
                continue
            for d in directions_list:
                gopro_dir = thera / "gopro" / d
                if not gopro_dir.exists():
                    continue
                for poseestimate_dir in gopro_dir.iterdir():
                    if not poseestimate_dir.is_dir():
                        continue
                    json_dir = poseestimate_dir / "json"
                    if json_dir.exists() and json_dir.is_dir():
                        yield json_dir, d

# =========================
# bbox/代表点
# =========================
def prefilter_kpts_by_conf(kpts, conf_th: float):
    """
    conf_th 未満の点を (0,0,0) にして無効化する（bbox/valid/MidHipの判定に使わせない）
    """
    if not kpts or len(kpts) < len(keypoint_names) * 3:
        return kpts

    k = list(kpts)  # copy
    for i in range(len(keypoint_names)):
        p = float(k[i * 3 + 2])
        if p < conf_th:
            k[i * 3 + 0] = 0.0
            k[i * 3 + 1] = 0.0
            k[i * 3 + 2] = 0.0
    return k


def person_bbox_area_from_kpts(kpts):
    if not kpts or len(kpts) < len(keypoint_names) * 3:
        return 0.0, 0, None  # area, valid, (xmin,ymin,xmax,ymax)

    xs, ys = [], []
    for i in range(len(keypoint_names)):
        x = kpts[i * 3 + 0]
        y = kpts[i * 3 + 1]
        p = kpts[i * 3 + 2]
        if x is None or y is None:
            continue
        x = float(x); y = float(y); p = float(p)
        if x == 0 or y == 0:
            continue
        xs.append(x); ys.append(y)

    valid = len(xs)
    if valid < MIN_VALID_KPTS:
        return 0.0, valid, None

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w = xmax - xmin
    h = ymax - ymin
    area = float(max(0.0, w) * max(0.0, h))
    return area, valid, (xmin, ymin, xmax, ymax)

def get_midhip_xy(kpts):
    # MidHip index
    try:
        idx = keypoint_names.index("MidHip")
    except ValueError:
        return None
    if len(kpts) < len(keypoint_names) * 3:
        return None
    x = float(kpts[idx * 3 + 0])
    y = float(kpts[idx * 3 + 1])
    p = float(kpts[idx * 3 + 2])
    if x == 0 or y == 0:
        return None
    if p <= 0.0:
        return None
    return np.array([x, y], dtype=float)

def representative_point(kpts, bbox):
    # MidHip 優先、なければ bbox中心
    mh = get_midhip_xy(kpts)
    if mh is not None:
        return mh
    if bbox is None:
        return None
    xmin, ymin, xmax, ymax = bbox
    return np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0], dtype=float)

# =========================
# 3人トラッキング本体
# =========================
def load_sequence_theree_tracks(json_dir: Path):
    json_files = sorted(glob.glob(str(json_dir / "*.json")))
    frame_count = len(json_files)
    if frame_count == 0:
        return None, None, None, 0, 0, 0

    # track state
    # last_pos: np.array([x,y]) or None
    tracks = [{"last_pos": None, "traj": []},
              {"last_pos": None, "traj": []},
              {"last_pos": None, "traj": []}]


    rows_by_track = [[], [], []]
    any_det_frames = 0  # フレームに1人以上入った回数（フォルダskip判定用）

    for frame_idx, fp in enumerate(json_files):
        with open(fp, "r") as f:
            val = json.load(f)

        people = val.get("people", []) or []

        # 候補生成
        candidates = []
        for person in people:
            kpts = person.get("pose_keypoints_2d", []) or []
            kpts = prefilter_kpts_by_conf(kpts, conf_th=CONF_TH)
            area, valid, bbox = person_bbox_area_from_kpts(kpts)
            if area < (AREA_TH_SAGI if json_dir.parent.parent.name == "sagi" else AREA_TH):
                continue
            rp = representative_point(kpts, bbox)
            if rp is None:
                continue
            candidates.append({
                "person": person,
                "kpts": kpts,
                "area": area,
                "rp": rp,
            })

        if len(candidates) > 0:
            any_det_frames += 1

        # まず面積順で軽く整列（初期化の安定化）
        candidates.sort(key=lambda c: c["area"], reverse=True)

        # 割当 予備で3トラック分
        assigned = [None, None, None]  # candidate index or None
        used = set()

        # 1) last_pos があるトラックを先に割当
        for ti in [0, 1, 2]:
            if tracks[ti]["last_pos"] is None:
                continue
            best_j = None
            best_d = 1e18
            for j, c in enumerate(candidates):
                if j in used:
                    continue
                d = float(np.linalg.norm(c["rp"] - tracks[ti]["last_pos"]))
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j is not None and best_d <= MAX_TRACK_DIST:
                assigned[ti] = best_j
                used.add(best_j)

        # 2) last_pos がないトラック or まだ未割当を、残り候補から面積順(面積閾値を超えている場合のみ)で埋める
        for ti in [0, 1, 2]:
            if assigned[ti] is not None:
                continue
            # 空いてる候補の先頭
            for j in range(len(candidates)):
                if j in used:
                    continue
                assigned[ti] = j
                used.add(j)
                break

        # 行作成
        for ti in [0, 1, 2]:
            row = {"frame": frame_idx}
            j = assigned[ti]
            if j is None:
                # 欠損
                for name in keypoint_names:
                    row[f"{name}_x"] = 0
                    row[f"{name}_y"] = 0
                    row[f"{name}_p"] = 0
                # last_pos は更新しない（瞬断に強くしたいなら保持）
            else:
                c = candidates[j]
                kpts = c["kpts"]
                if len(kpts) >= len(keypoint_names) * 3:
                    for i, name in enumerate(keypoint_names):
                        row[f"{name}_x"] = kpts[i * 3 + 0]
                        row[f"{name}_y"] = kpts[i * 3 + 1]
                        row[f"{name}_p"] = kpts[i * 3 + 2]
                else:
                    for name in keypoint_names:
                        row[f"{name}_x"] = 0
                        row[f"{name}_y"] = 0
                        row[f"{name}_p"] = 0

                tracks[ti]["last_pos"] = c["rp"]
                tracks[ti]["traj"].append(c["rp"])

            rows_by_track[ti].append(row)

    df0 = pd.DataFrame(rows_by_track[0])
    df1 = pd.DataFrame(rows_by_track[1])
    df2 = pd.DataFrame(rows_by_track[2])
    
    def spatial_range(traj):
        if len(traj) < 2:
            return 0.0, 0.0
        P = np.vstack(traj)
        dx = np.nanmax(P[:,0]) - np.nanmin(P[:,0])
        dy = np.nanmax(P[:,1]) - np.nanmin(P[:,1])
        return dx, dy

    dx0, dy0 = spatial_range(tracks[0]["traj"])
    dx1, dy1 = spatial_range(tracks[1]["traj"])
    dx2, dy2 = spatial_range(tracks[2]["traj"])

    return df0, df1, df2, frame_count, any_det_frames, (dx0, dy0, dx1, dy1, dx2, dy2)

# =========================
# 0→NaN → 3次スプライン補間（×は0由来のみ）
# =========================
def _short_nan_runs_mask(is_nan: np.ndarray, max_len: int) -> np.ndarray:
    """
    短いNaNのrunをTrueにするマスクを返す
    """
    n = len(is_nan)
    out = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if not is_nan[i]:
            i += 1
            continue
        j = i
        while j < n and is_nan[j]:
            j += 1
        run_len = j - i
        if run_len < max_len:
            out[i:j] = True
        i = j
    return out

def _fill_short_gaps_cubic(y: np.ndarray, short_gap_mask: np.ndarray, neighbor_pts: int) -> np.ndarray:
    """
    短い欠損を3次スプライン補間
    """
    out = y.copy()
    n = len(out)
    i = 0
    while i < n:
        if not (np.isnan(out[i]) and short_gap_mask[i]):
            i += 1
            continue

        j = i
        while j < n and np.isnan(out[j]) and short_gap_mask[j]:
            j += 1

        # runの前後から有効点を集める
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
            # 点が足りないなら線形
            out[x_fill] = np.interp(x_fill, xs, ys)
        # それすら無理なら埋めない（NaNのまま）

        i = j

    return out


def zero_nan_and_spline(df_raw: pd.DataFrame):
    """
    欠損が0.2秒(60Hzだと12フレーム)未満は3次スプライン補間を実行（論文doi：10.1080/10255841003664701）
    """
    MAX_SPLINE_GAP = 12  # 補完する最大欠損フレーム
    NEIGHBOR_PTS   = 2      # 欠損runの前後から何点ずつ使うか（2→合計4点狙い）
    
    df0 = df_raw.copy()
    mask0_by_kp = {}

    # 1) 0 → NaN（x,y）
    for k in keypoint_names:
        xcol = f"{k}_x"
        ycol = f"{k}_y"
        pcol = f"{k}_p"

        x = df0[xcol].astype(float).to_numpy()
        y = df0[ycol].astype(float).to_numpy()

        m0 = (x == 0) | (y == 0)
        if m0.any():
            df0.loc[m0, xcol] = np.nan
            df0.loc[m0, ycol] = np.nan
        mask0_by_kp[k] = m0

    df_s = df0.copy()

    # 2) 座標補間：短い欠損run（< MAX_SPLINE_GAP）だけ局所CubicSplineで埋める
    for k in keypoint_names:
        xcol = f"{k}_x"
        ycol = f"{k}_y"
        pcol = f"{k}_p"

        x = df_s[xcol].astype(float).to_numpy()
        y = df_s[ycol].astype(float).to_numpy()

        # 0由来欠損（NaN化した場所）だけを対象に、短欠損runマスクを作る
        m0 = mask0_by_kp[k]  # True = 0由来欠損
        short_gap_mask = _short_nan_runs_mask(m0, MAX_SPLINE_GAP)

        # 局所補間（短欠損runのみ）
        x_filled = _fill_short_gaps_cubic(x, short_gap_mask, NEIGHBOR_PTS)
        y_filled = _fill_short_gaps_cubic(y, short_gap_mask, NEIGHBOR_PTS)

        # 「実際に埋まった点」→ p=0.6（XもYもNaN→値になったところ）
        filled_mask = short_gap_mask & np.isnan(x) & np.isfinite(x_filled) & np.isnan(y) & np.isfinite(y_filled)

        # 仕上げ：残NaNは0
        df_s[xcol] = np.where(np.isfinite(x_filled), x_filled, 0.0)
        df_s[ycol] = np.where(np.isfinite(y_filled), y_filled, 0.0)

        df_s.loc[filled_mask, pcol] = 0.6  
        df_s[pcol] = df_s[pcol].fillna(0)

    return df_s, mask0_by_kp

# =========================
# プロット（元の関数を流用）
# =========================
def _save_plot(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def kp_pair_name(rkp: str, lkp: str) -> str:
    if rkp.startswith("R") and lkp.startswith("L") and rkp[1:] == lkp[1:]:
        return rkp[1:]
    return f"{rkp}_{lkp}"

def save_plots_single_person(df_raw, df_spline, mask0_by_kp, out_root: Path, plot_name: str):
    plot_folder = out_root / f"{plot_name}_plots"
    plot_folder.mkdir(parents=True, exist_ok=True)
    frames = df_raw["frame"].to_numpy()

    for kp in keypoint_names:
        if kp in SKIP_KEYPOINTS:
            continue

        m0 = mask0_by_kp.get(kp, np.zeros_like(frames, dtype=bool))
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        x_col = f"{kp}_x"
        axes[0].plot(frames, df_raw[x_col], label="Raw", alpha=0.6, linewidth=1)
        axes[0].plot(frames, df_spline[x_col], label="Spline", alpha=0.9, linewidth=1.5)
        if m0.any():
            axes[0].scatter(frames[m0], df_raw.loc[m0, x_col], marker="x", s=35, label="Zero", zorder=4)
        axes[0].set_ylabel("X (px)")
        axes[0].set_ylim(*XLIM)
        axes[0].set_title(f"{kp} - X")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        y_col = f"{kp}_y"
        axes[1].plot(frames, df_raw[y_col], label="Raw", alpha=0.6, linewidth=1)
        axes[1].plot(frames, df_spline[y_col], label="Spline", alpha=0.9, linewidth=1.5)
        if m0.any():
            axes[1].scatter(frames[m0], df_raw.loc[m0, y_col], marker="x", s=35, label="Zero", zorder=4)
        axes[1].set_ylabel("Y (px)")
        axes[1].set_ylim(*YLIM)
        axes[1].set_title(f"{kp} - Y")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        p_col = f"{kp}_p"
        axes[2].plot(frames, df_raw[p_col].astype(float), label="Raw", alpha=0.6, linewidth=1)
        axes[2].plot(frames, df_spline[p_col].astype(float), label="Spline", alpha=0.9, linewidth=1.5)
        axes[2].set_xlabel("Frame")
        axes[2].set_ylabel("p")
        axes[2].set_ylim(0, 1.05)
        axes[2].set_title(f"{kp} - Confidence")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        _save_plot(fig, plot_folder / f"{kp}.png")

    lr_pairs = []
    for kp in keypoint_names:
        if kp.startswith("R"):
            lk = "L" + kp[1:]
            if lk in keypoint_names:
                lr_pairs.append((kp, lk))

    for rkp, lkp in lr_pairs:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        rx, lx = f"{rkp}_x", f"{lkp}_x"
        ry, ly = f"{rkp}_y", f"{lkp}_y"

        axes[0].plot(frames, df_raw[rx],    label="R raw",    color="red",  alpha=0.35, linewidth=1)
        axes[0].plot(frames, df_spline[rx], label="R spline", color="red",  alpha=0.95, linewidth=1.6)
        axes[0].plot(frames, df_raw[lx],    label="L raw",    color="blue", alpha=0.35, linewidth=1)
        axes[0].plot(frames, df_spline[lx], label="L spline", color="blue", alpha=0.95, linewidth=1.6)
        axes[0].set_ylabel("X (px)")
        axes[0].set_ylim(*XLIM)
        axes[0].set_title(f"{rkp} / {lkp} - X")
        axes[0].legend(ncol=2)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(frames, df_raw[ry],    label="R raw",    color="red",  alpha=0.35, linewidth=1)
        axes[1].plot(frames, df_spline[ry], label="R spline", color="red",  alpha=0.95, linewidth=1.6)
        axes[1].plot(frames, df_raw[ly],    label="L raw",    color="blue", alpha=0.35, linewidth=1)
        axes[1].plot(frames, df_spline[ly], label="L spline", color="blue", alpha=0.95, linewidth=1.6)
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Y (px)")
        axes[1].set_ylim(*YLIM)
        axes[1].set_title(f"{rkp} / {lkp} - Y")
        axes[1].legend(ncol=2)
        axes[1].grid(True, alpha=0.3)

        name = kp_pair_name(rkp, lkp)
        _save_plot(fig, plot_folder / f"{name}_LR.png")

# =========================
# 1フォルダ処理
# =========================
def process_one_json_dir(json_dir: Path, direction: str):
    out_root = json_dir.parent
    base = out_root.name

    # 2人分のCSVが揃っていればスキップ（任意）
    if SKIP_IF_CSV_EXISTS:
        ok = True
        for tag in OUT_PERSON_TAGS:
            if not (out_root / f"{base}_{tag}.csv").exists():
                ok = False
            if not (out_root / f"{base}_{tag}_spline.csv").exists():
                ok = False
        if ok:
            return "skip_exists"

    df_a, df_b, df_c, frame_count, any_det_frames, (dx0, dy0, dx1, dy1, dx2, dy2) = load_sequence_theree_tracks(json_dir)
    if df_a is None:
        return "skip"

    # フォルダskip判定：少なくとも「どちらかが検出できたフレーム」が一定割合未満ならskip
    if any_det_frames < max(1, int(np.ceil(frame_count * MIN_DET_RATIO))):
        return "skip_low_detection"
    
    # ---- 静止人物フィルタ & PA/PT決定 ----

    tracks = [
        {"df": df_a, "dx": dx0, "dy": dy0},
        {"df": df_b, "dx": dx1, "dy": dy1},
        {"df": df_c, "dx": dx2, "dy": dy2},
    ]

    # 静止判定（共通）
    def is_static(dx, dy):
        return (dx < 1000) and (dy < 250)

    active = [tr for tr in tracks if not is_static(tr["dx"], tr["dy"])]

    if len(active) < 2:
        return "skip_static_person"

    # --- PA/PT 判定用の代表フレーム ---
    mid_frame = frame_count // 2

    def midhip_x_at(df, frame_idx):
        row = df[df["frame"] == frame_idx]
        if row.empty:
            return None
        return float(row["MidHip_x"].values[0])

    # MidHip が取得できる2人を探す
    pairs = []
    for i in range(len(active)):
        for j in range(i+1, len(active)):
            x0 = midhip_x_at(active[i]["df"], mid_frame)
            x1 = midhip_x_at(active[j]["df"], mid_frame)
            if x0 is not None and x1 is not None:
                pairs.append((active[i], active[j], x0, x1))

    if not pairs:
        return "skip_no_midhip"

    # PAPTの位置関係判定に使うフレーム
    judge_frames = [mid_frame, mid_frame + 30, mid_frame + 60, mid_frame + 90, mid_frame + 120]
    judge_frames = [f for f in judge_frames if f < frame_count]

    def midhip_x_at(df, frame_idx):
        row = df[df["frame"] == frame_idx]
        if row.empty:
            return None
        x = row["MidHip_x"].values[0]
        if x == 0 or np.isnan(x):
            return None
        return float(x)

    print("DEBUG sub_name =", out_root.parent.parent.parent.parent.name,
        " direction=", direction,
        " out_root=", out_root)

    def is_pa_first(x0, x1, direction):  #PAPTの位置関係判定
        if direction == "sagi" and out_root.parent.parent.parent.parent.name != "sub15":  #矢状面右側にカメラ
            return x0 >= x1   # x 大 → PA
        elif direction == "sagi" and out_root.parent.parent.parent.parent.name == "sub15":  #矢状面左側にカメラ
            return x0 <= x1   # x 小 → PA
        elif direction == "fr":
            return x0 >= x1   # x 大 → PA
        elif direction == "fl":
            return x0 <= x1   # x 小 → PA
        else:
            return None
    
    # 最初に見つかったペアを使用（1人は自然に捨てられる）
    tr0, tr1 = pairs[0][0], pairs[0][1]

    votes = []

    for f in judge_frames:
        x0 = midhip_x_at(tr0["df"], f)
        x1 = midhip_x_at(tr1["df"], f)
        if x0 is None or x1 is None:
            continue
        vote = is_pa_first(x0, x1, direction)
        if vote is not None:
            votes.append(vote)

    if len(votes) == 0:
        return "skip_no_valid_vote"

    # True が多ければ tr0 が PA
    if sum(votes) >= (len(votes) / 2):
        PA_df, PT_df = tr0["df"], tr1["df"]
    else:
        PA_df, PT_df = tr1["df"], tr0["df"]

    valid_tracks = [("PA", PA_df), ("PT", PT_df)]

    if len(valid_tracks) == 0:
        return "skip_static_person"
    
    for tag, df_raw in valid_tracks:
        df_orig = df_raw.copy()
        df_spline, mask0_by_kp = zero_nan_and_spline(df_raw)

        csv_raw = out_root / f"{base}_{tag}.csv"
        csv_spline = out_root / f"{base}_{tag}_spline.csv"
        df_raw.to_csv(csv_raw, index=False)
        df_spline.to_csv(csv_spline, index=False)

        save_plots_single_person(df_orig, df_spline, mask0_by_kp, out_root, plot_name=tag)

    return "ok"

# =========================
# 並列処理用
# =========================
def worker(args):
    json_dir, direction = args
    try:
        rel = json_dir.relative_to(root_dir)
        parts = rel.parts
        display = " / ".join(parts[:5])
        print(f"[START] {display}")
        return process_one_json_dir(json_dir, direction)
    except Exception as e:
        return f"error: {json_dir} {e}"

# =========================
# メイン
# =========================
def main():
    targets = list(iter_poseestimate_json_dirs(root_dir, directions))
    print(f"root: {root_dir}")
    print(f"found json dirs: {len(targets)}")
    if not targets:
        print("No target json directories found.")
        return

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(worker, t): t for t in targets}
        for fut in tqdm(as_completed(futures), total=len(targets), desc="All poseestimate folders"):
            res = fut.result()
            if isinstance(res, str) and res.startswith("error"):
                print(res)

    dt = time.time() - t0
    print("\n" + "=" * 60)
    print("done.")
    print(f"folders: {len(targets)}")
    print(f"elapsed: {dt:.1f} sec")
    print("=" * 60)

if __name__ == "__main__":
    main()
