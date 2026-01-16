"""
plot_camera_poses_from_board_hardcoded.py
========================================
fl–fr で利用したキャリブレーションボード座標系（board座標）を基準に、
fl / fr / sagi のカメラ中心位置と姿勢（カメラ座標軸）を3D描画してPNG保存する。
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# ★ 設定（ここだけ編集）
# =============================================================================
SUB_DIR = Path(r"G:\gait_pattern\2025_shuron_tkrzk\sub2")

# チェッカーボード設定（fl–fr キャリブレーションで使ったもの）
CHECKER_W = 5          # 内部コーナー数（width）
CHECKER_H = 4          # 内部コーナー数（height）
SQUARE_MM = 35.0       # 正方形サイズ [mm]

# 描画設定
AXIS_LEN_MM = 200.0    # 各カメラ座標軸の描画長 [mm]

# 出力
OUT_PNG = SUB_DIR / "cali" / "extparams" / "camera_pose_board_frame.png"

# カメラ方向
DIRECTIONS = ["fl", "fr", "sagi"]

# =============================================================================
# utilities
# =============================================================================
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_R_t(j: dict, prefer_aligned=False):
    """
    json から (R, t, mode) を抽出
    X_cam = R X_board + t を仮定
    """
    if prefer_aligned and "extrinsics_aligned_to_flfr_pair" in j:
        ex = j["extrinsics_aligned_to_flfr_pair"]
        mode = "aligned_to_flfr"
    else:
        ex = j["extrinsics"]
        mode = "raw"

    R = np.asarray(ex["rotation_matrix"], dtype=float).reshape(3, 3)
    t = np.asarray(ex["translation_vector"], dtype=float).reshape(3,)
    return R, t, mode


def camera_center_in_board(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """C = -R^T t"""
    return -R.T @ t


def camera_axes_in_board(R: np.ndarray) -> np.ndarray:
    """
    board 座標系でのカメラ座標軸（3x3）
    columns = camera x, y, z axis in board frame
    """
    return R.T


def make_checkerboard_points(w, h, square_mm):
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.stack(
        [xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1
    ).astype(float)
    pts *= square_mm
    return pts


def set_axes_equal(ax):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])

    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def find_ext_json(sub_dir: Path, direction: str) -> Path:
    base = sub_dir / "cali" / "extparams" / direction
    if direction == "sagi":
        aligned = sorted(base.glob("*aligned_to_flfr*.json"))
        if aligned:
            return aligned[0]
    cand = sorted(base.glob("camera_params_with_ext*.json"))
    if not cand:
        cand = sorted(base.rglob("camera_params_with_ext*.json"))
    if not cand:
        raise FileNotFoundError(f"ext json not found: {base}")
    return cand[0]


# =============================================================================
# main
# =============================================================================
def main():
    if not SUB_DIR.exists():
        raise FileNotFoundError(SUB_DIR)

    # --- load cameras
    cams = []
    for d in DIRECTIONS:
        jp = find_ext_json(SUB_DIR, d)
        j = load_json(jp)
        R, t, mode = extract_R_t(j, prefer_aligned=(d == "sagi"))
        C = camera_center_in_board(R, t)
        A = camera_axes_in_board(R)
        cams.append((d, mode, C, A, jp))

        print(f"[{d}] {jp.name} ({mode})")
        print(f"  camera center (board) [mm]: {C}")
        print(f"  optical axis (board): {A[:,2]}")

    # --- checkerboard
    board_pts = make_checkerboard_points(CHECKER_W, CHECKER_H, SQUARE_MM)

    # --- plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Camera poses in board frame ({SUB_DIR.name})")

    # board points
    ax.scatter(board_pts[:, 0], board_pts[:, 1], board_pts[:, 2], s=10)

    # board axes
    Lb = max(CHECKER_W, CHECKER_H) * SQUARE_MM * 0.8
    ax.quiver(0, 0, 0, Lb, 0, 0)
    ax.quiver(0, 0, 0, 0, Lb, 0)
    ax.quiver(0, 0, 0, 0, 0, Lb)

    # cameras
    for name, mode, C, A, jp in cams:
        ax.scatter(C[0], C[1], C[2], s=60)
        ax.text(C[0], C[1], C[2], f" {name}", fontsize=10)

        ax.quiver(C[0], C[1], C[2], *(A[:, 0] * AXIS_LEN_MM))
        ax.quiver(C[0], C[1], C[2], *(A[:, 1] * AXIS_LEN_MM))
        ax.quiver(C[0], C[1], C[2], *(A[:, 2] * AXIS_LEN_MM))

    ax.set_xlabel("X_board [mm]")
    ax.set_ylabel("Y_board [mm]")
    ax.set_zlabel("Z_board [mm]")
    ax.grid(True)

    # bounds
    all_xyz = np.vstack([board_pts] + [c[2].reshape(1, 3) for c in cams])
    pad = 200.0
    ax.set_xlim(all_xyz[:, 0].min() - pad, all_xyz[:, 0].max() + pad)
    ax.set_ylim(all_xyz[:, 1].min() - pad, all_xyz[:, 1].max() + pad)
    ax.set_zlim(all_xyz[:, 2].min() - pad, all_xyz[:, 2].max() + pad)
    set_axes_equal(ax)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    
    
    fig.tight_layout()
    plt.show()

    print(f"\n[OK] saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
