#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4_4_check_vit_mocap_anim_v2.py
==============================
ViTPose(Body25) の 3Dキーポイント(npz) と、OptiTrack mocap マーカー(csv) を
同一3D空間上でアニメーション比較して mp4 を出力する。

ポイント
- mocap 側は marker 名が Body25 と一致しないため、mocapマーカーから Body25の“近似点”を作る
  (例: MidHip=ASIS/PSISの平均, Knee=(KNE+KNE2)/2, Ankle=(ANK+ANK2)/2 など)
- mocap 座標は [m] の想定なので、描画前に 1000 を掛けて [mm] に変換する（ユーザ指定）
- ViTPose の単位はパイプライン依存。一般に [mm] を想定（必要なら VIT_SCALE を調整）

入力（想定）
- mocap_keypoints_60hz_*.csv : 4_2_mocap_result.py が出力（ヘッダ2行のMultiIndex）
- 3d_kp_ViTPose_*PA.npz      : 3D再構成結果（key "butter" 等）

出力
- <vit_npz_parent>/ViTPose_results/compare_mocap_results/compare_vitpose_mocap_<tag>.mp4
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# =========================
# 設定（必要ならここだけ変更）
# =========================
ROOT_DIR = Path(r"G:\gait_pattern\2025_shuron_BR9G")  # 環境に合わせて
ANIM_TARGET_KEY = "butter"  # npz 内で描画するキー
FPS = 60

# スケール
MOCAP_SCALE = 1000.0   # [m] -> [mm]
VIT_SCALE = 1.0        # ViTPoseが[m]なら1000にする、[mm]なら1.0

# 描画
POINT_SIZE = 18
LINE_WIDTH = 2.0

# ViTPose の conf がある場合、これ未満は描画しない（npzに "conf" が無ければ無視）
CONF_DRAW_TH = 0.4


# =========================
# Skeleton connections
# =========================
def get_body25_connections():
    # BODY25の関節接続リストを返す
    return [
        (1, 8), (1, 2), (1, 5), (2, 3), (3, 4),
        (5, 6), (6, 7), (8, 9), (8, 12), (9, 10),
        (10, 11), (12, 13), (13, 14), (1, 0),
        (0, 15), (15, 17), (0, 16), (16, 18),
        (11, 24), (11, 22), (22, 23),
        (14, 21), (14, 19), (19, 20),
    ]

def get_mocap_connections():
    # mocap マーカー接続リスト（OptiTrack MarkerSet 01ベース）
    return [
        ()
    ]


# =========================
# Utilities
# =========================
def _ordered_unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def read_mocap_multiheader_csv(csv_path: Path) -> tuple[np.ndarray, list[str], int]:
    """
    4_2_mocap_result.py が出力する mocap_keypoints_60hz_*.csv を読む。
    - header=[0,1] の MultiIndex になっている前提
    戻り:
      mocap_xyz: (N, M, 3)  [m]のまま
      marker_names: 長さ M（例 'MarkerSet 01:C7' 等）
      mocap_start_frame_60hz: 60Hzの絶対フレーム（最初のフレーム番号）
    """
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
    mocap_start_frame_60hz = df.index[0]

    # 念のため、X/Y/Z 以外の列は落とす
    df = df.loc[:, df.columns.get_level_values(1).isin(["X", "Y", "Z"])].copy()

    marker_names = _ordered_unique(df.columns.get_level_values(0))
    m = len(marker_names)
    n = len(df)

    xyz = np.full((n, m, 3), np.nan, dtype=np.float64)
    for j, mn in enumerate(marker_names):
        cols = [(mn, "X"), (mn, "Y"), (mn, "Z")]
        if all(c in df.columns for c in cols):
            xyz[:, j, :] = df[cols].to_numpy(dtype=np.float64)
    return xyz, marker_names, mocap_start_frame_60hz


def apply_conf_mask(kp: np.ndarray, conf: np.ndarray | None, th: float) -> np.ndarray:
    """
    conf があれば th 未満を NaN にする。
    kp: (N,25,3), conf: (N,25)
    """
    if conf is None:
        return kp
    out = kp.copy()
    mask = conf < th
    out[mask] = np.nan
    return out


def compute_axis_limits(*arrays: np.ndarray, pad_ratio: float = 0.08):
    """
    (N,25,3) の配列群から xyz の min/max を計算（NaN無視）。
    """
    all_xyz = np.concatenate([a.reshape(-1, 3) for a in arrays], axis=0)
    ok = np.isfinite(all_xyz).all(axis=1)
    if not np.any(ok):
        # フォールバック
        return (-1000, 1000), (-1000, 1000), (-1000, 1000)

    v = all_xyz[ok]
    mn = v.min(axis=0)
    mx = v.max(axis=0)
    span = mx - mn
    pad = span * pad_ratio
    mn = mn - pad
    mx = mx + pad
    return (mn[0], mx[0]), (mn[1], mx[1]), (mn[2], mx[2])


# =========================
# Animation
# =========================
def update_frame_compare(i,
                         data_v, conf_v,
                         data_m,  # mocap (already Body25 approx)
                         scat_v, lines_v, scat_m, lines_m,
                         ax, frame_text, frame_offset):
    kp_v = data_v[i].copy()
    kp_m = data_m[i].copy()

    # conf マスク（ViTPoseのみ）
    if conf_v is not None:
        kp_v[conf_v[i] < CONF_DRAW_TH] = np.nan

    # 3_4 と同じ軸変換: [x,y,z] -> (z, x, y)
    xyz_v = np.column_stack([kp_v[:, 2], kp_v[:, 0], kp_v[:, 1]])
    xyz_m = np.column_stack([kp_m[:, 2], kp_m[:, 0], kp_m[:, 1]])

    valid_v = ~np.isnan(xyz_v).any(axis=1)
    valid_m = ~np.isnan(xyz_m).any(axis=1)

    # scatter 更新
    if valid_v.any():
        scat_v._offsets3d = (xyz_v[valid_v, 0], xyz_v[valid_v, 1], xyz_v[valid_v, 2])
    else:
        scat_v._offsets3d = ([], [], [])

    if valid_m.any():
        scat_m._offsets3d = (xyz_m[valid_m, 0], xyz_m[valid_m, 1], xyz_m[valid_m, 2])
    else:
        scat_m._offsets3d = ([], [], [])

    # line 更新
    conns_vit = get_body25_connections()
    for l, (a, b) in zip(lines_v, conns_vit):
        if not np.isnan(kp_v[a]).any() and not np.isnan(kp_v[b]).any():
            l.set_data_3d(
                [xyz_v[a, 0], xyz_v[b, 0]],
                [xyz_v[a, 1], xyz_v[b, 1]],
                [xyz_v[a, 2], xyz_v[b, 2]],
            )
        else:
            l.set_data_3d([], [], [])

    # conns_mocap = get_mocap_connections()
    # for l, (a, b) in zip(lines_m, conns_mocap):
    #     if not np.isnan(kp_m[a]).any() and not np.isnan(kp_m[b]).any():
    #         l.set_data_3d(
    #             [xyz_m[a, 0], xyz_m[b, 0]],
    #             [xyz_m[a, 1], xyz_m[b, 1]],
    #             [xyz_m[a, 2], xyz_m[b, 2]],
    #         )
    #     else:
    #         l.set_data_3d([], [], [])

    # 表示範囲（3_4 準拠）
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(-2000, 2000)
    ax.set_zlim(0, 2000)

    # フレーム番号表示
    original_frame = i + frame_offset
    frame_text.set_text(f"Frame: {original_frame:04d}")

    return [scat_v, *lines_v, scat_m, *lines_m]


def make_compare_animation(vit_kp: np.ndarray,
                           vit_conf: np.ndarray | None,
                           mocap_body25_mm: np.ndarray,
                           out_mp4: Path,
                           title: str,
                           frame_offset: int = 0):
    # 同じ長さに揃える
    n = min(vit_kp.shape[0], mocap_body25_mm.shape[0])
    vit_kp = vit_kp[:n]
    mocap_body25_mm = mocap_body25_mm[:n]
    vit_conf = vit_conf[:n] if vit_conf is not None and len(vit_conf) >= n else vit_conf

    conns_vit = get_body25_connections()
    # conns_mocap = get_mocap_connections()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # ViTPose: red
    scat_v = ax.scatter([], [], [], s=40, c="r", depthshade=True)
    lines_v = [ax.plot([], [], [], lw=2)[0] for _ in conns_vit]

    # Mocap: blue
    scat_m = ax.scatter([], [], [], s=40, c="b", depthshade=True)
    # lines_m = [ax.plot([], [], [], lw=2)[0] for _ in conns_mocap]
    lines_m = []  # mocapの線は描かない

    # フレーム番号表示（画面固定）
    frame_text = ax.text(
        0.02, 0.95, 0.98, "",
        transform=ax.transAxes,
        fontsize=14,
        color="black"
    )

    ax.set_title(title)
    ax.set_xlabel("Z (Forward) [mm]")
    ax.set_ylabel("X (Side) [mm]")
    ax.set_zlabel("Y (Up) [mm]")
    ax.view_init(elev=10, azim=45)
    ax.set_autoscale_on(False)

    ani = animation.FuncAnimation(
        fig,
        update_frame_compare,
        frames=n,
        fargs=(vit_kp, vit_conf, mocap_body25_mm,
               scat_v, lines_v, scat_m, lines_m,
               ax, frame_text, int(frame_offset)),
        interval=1000 / FPS,
        blit=False
    )

    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # mp4 保存（ffmpegが無い環境向けに gif フォールバック）
    try:
        writer = animation.FFMpegWriter(fps=FPS, bitrate=3600)
        ani.save(str(out_mp4), writer=writer)
    except Exception as e:
        out_gif = out_mp4.with_suffix(".gif")
        print(f"[WARN] mp4保存に失敗したため gif で保存します: {e}")
        ani.save(str(out_gif), writer=animation.PillowWriter(fps=FPS))

    plt.close(fig)


# =========================
# Main compare function
# =========================
def compare_vit_mocap_anim(mocap_kp_path: Path, vit_kp_array_path: Path, frame_diff: int):
    """
    mocap と vitpose を読み込み、比較アニメーション(mp4)を保存。
    """
    mocap_kp_path = Path(mocap_kp_path)
    vit_kp_array_path = Path(vit_kp_array_path)

    if not mocap_kp_path.exists():
        print(f"[SKIP] missing mocap csv: {mocap_kp_path}")
        return
    if not vit_kp_array_path.exists():
        print(f"[SKIP] missing vit npz: {vit_kp_array_path}")
        return

    # ---- mocap ----
    mocap_xyz_m, marker_names, mocap_start_frame_60hz = read_mocap_multiheader_csv(mocap_kp_path)
    mocap_xyz_m = mocap_xyz_m * MOCAP_SCALE  # [m] -> [mm]
    
    # # フレーム差を考慮してシフト不要だった
    # # frame_diff_fin = mocap_start_frame_60hz
    # frame_diff_fin = mocap_start_frame_60hz - frame_diff
    # # frame_diff_fin  = 100
    # print(f"mocap_data shape: {mocap_xyz_m.shape}")
    # print(f"mocap_start_frame_60hz: {mocap_start_frame_60hz}, frame_diff: {frame_diff}, frame_diff_fin: {frame_diff_fin}")
    # mocap_xyz_mm = mocap_xyz_m[frame_diff_fin:]

    # ---- vitpose ----
    gopro_gait_cycle_dir = vit_kp_array_path.parent / "ViTPose_results"
    gopro_gait_cycle_path = next(gopro_gait_cycle_dir.glob("gait_cycles_*.npz"), None)
    if gopro_gait_cycle_path is None:
        print(f"[WARN] no gopro gait cycle {gopro_gait_cycle_path}")
        return
    vit_gait_cycle = np.load(gopro_gait_cycle_path, allow_pickle=True)
    vit_gait_cycles_r = vit_gait_cycle["gait_cycle_r"]
    vit_gait_cycles_l = vit_gait_cycle["gait_cycle_l"]
    
    vit_all_frames_r = [frame for cycle in vit_gait_cycles_r for frame in cycle]
    vit_all_frames_l = [frame for cycle in vit_gait_cycles_l for frame in cycle]
    vit_start_frame = int(min(vit_all_frames_r+vit_all_frames_l))
    vit_end_frame = int(max(vit_all_frames_r+vit_all_frames_l))
        
    vit_npz = np.load(vit_kp_array_path, allow_pickle=True)
    if ANIM_TARGET_KEY not in vit_npz.files:
        print(f"[SKIP] {vit_kp_array_path.name} lacks key '{ANIM_TARGET_KEY}' (has {vit_npz.files})")
        return

    print(f"vit start_frame: {vit_start_frame}, vit_end_frame: {vit_end_frame}), frame_range={vit_end_frame - vit_start_frame + 1}")
    
    vit_kp = vit_npz[ANIM_TARGET_KEY].astype(np.float64) * VIT_SCALE  # (N,25,3)
    print(f"vit_kp original shape: {vit_kp.shape}")
    vit_kp = vit_kp[vit_start_frame:vit_end_frame + 1]  # 有効フレーム範囲にトリム

    vit_conf = vit_npz["conf"] if "conf" in vit_npz.files else None

    print(f"vit_data shape: {vit_kp.shape}")
    out_dir = vit_kp_array_path.parent / "ViTPose_results" / "compare_mocap_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = mocap_kp_path.stem.replace("mocap_keypoints_60hz_", "").replace("mocap_keypoints_60Hz_", "")
    out_mp4 = out_dir / f"compare_vitpose_mocap.mp4"
    title = f"ViTPose vs Mocap ({tag})"

    print(f"[INFO] mocap: {mocap_kp_path}")
    print(f"[INFO] vit  : {vit_kp_array_path}")
    print(f"[INFO] out  : {out_mp4}")

    make_compare_animation(vit_kp, vit_conf, mocap_xyz_m, out_mp4, title)


def main():
    """
    ROOT_DIR 以下を走査して、各 sub*/thera*/ で
    - mocap/mocap_keypoints_60hz_*.csv
    - 3d_kp_ViTPose_*PA.npz
    が揃っていれば比較アニメーションを作成する。
    """
    for sub_dir in sorted(ROOT_DIR.glob("sub*")):
        thera_dirs = sorted(sub_dir.glob("thera*-0"))
        for thera_dir in thera_dirs:
            mocap_kp_path = next((thera_dir / "mocap").glob("mocap_keypoints_60hz_*.csv"), None)
            if mocap_kp_path is None:
                mocap_kp_path = next((thera_dir / "mocap").glob("mocap_keypoints_60Hz_*.csv"), None)
            if mocap_kp_path is None:
                continue

            # PA only
            vit_kp_npz_path = next(thera_dir.glob("3d_kp_ViTPose_*PA.npz"), None)
            if vit_kp_npz_path is None:
                continue
            
            frame_diff_csv = thera_dir / "mocap" / "frame_diff.csv"
            if not frame_diff_csv.exists():
                print(f"[SKIP] missing: {frame_diff_csv}")
                continue
            
            frame_diff_df = pd.read_csv(frame_diff_csv)
            frame_diff = frame_diff_df['frame_diff'].values[0]

            print(f"\n[RUN] {thera_dir}")
            compare_vit_mocap_anim(mocap_kp_path, vit_kp_npz_path, frame_diff)


if __name__ == "__main__":
    main()
