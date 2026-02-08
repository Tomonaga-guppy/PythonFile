"""
目的:
- 3_3_reconstruct3d.py が出力した 3D姿勢(OpenPose/ViTPose等) の解析を行う

入出力（想定フォルダ）:
ROOT/
  subX/
    theraX-0/
      mocap/
        *_opti_intermediate.npz  (4_1が作成)
      3d_kp_<method>_<csv_tag>.npz  (3_3が作成)

出力:
theraX-0/
  Pose3D_results/
    <method>/
      normalized_cycle_R_mean_<tag>_<method>.csv
      normalized_cycle_L_mean_<tag>_<method>.csv
      gait_parameters_R_<tag>_<method>.csv
      gait_parameters_L_<tag>_<method>.csv
      angle_error_summary_<tag>_<method>.csv
      gait_cycle_FlEx_R_<tag>_<method>.png など（主要角度プロット）

重要:
- 3_3の出力（3D再計算・補間・フィルタ）には一切手を入れない（解析だけ）
- タイミング合わせは同期用IMUで行う
依存:
- numpy, pandas, matplotlib
- m_openpose.py（culc_angle_all_frames を使用）
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import m_openpose as op  # culc_angle_all_frames
import matplotlib.animation as animation

# =========================
# 設定（ここだけ調整）
# =========================
ROOT_DIR = Path(r"G:\gait_pattern\2025_shuron_tkrzk")

# 3_3出力npzを使う優先順位（最初に見つかったものを使う）
PREFERRED_3D_KEY_ORDER = ["butter", "spline", "conf_filt", "raw"]

# 3Dのサンプリング周波数（3_3のFRAME_RATE=60前提）
FS_3D = 60.0

# 麻痺側情報
PARALYZED_SIDES = {"sub2": "R",    "sub3": "R",    "sub15": "L",    "sub16": "R"}

# =========================
# 角度プロットのY範囲（関節ごとに設定）
# =========================
YLIM_BY_ANGLE = {
    # Hip
    "Hip_FlEx":  (-40,  60),
    "Hip_AdAb":  (-20,  20),
    "Hip_InEx":  (-20,  20),

    # Knee
    "Knee_FlEx": (-20, 80),

    # Ankle  ※あなたの系列名は Ankle_PlDo (背屈/底屈)
    "Ankle_PlDo": (-30,  50),
}

def ylims_for_angle_key(k: str):
    """
    例: 'R_Hip_FlEx' -> (-70, 70)
        'L_Knee_FlEx' -> (0, 120)
    見つからなければ None
    """
    base = k.split("_", 1)[1]  # 'Hip_FlEx' など
    return YLIM_BY_ANGLE.get(base, None)


# =========================
# 小物
# =========================
def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def pick_3d_array(npz, preferred_keys):
    # 3_3: butterが無効のとき np.array([]) を入れていることがある
    for k in preferred_keys:
        if k in npz.files:
            arr = npz[k]
            if isinstance(arr, np.ndarray) and arr.size == 0:
                continue
            return k, arr
    raise KeyError(f"no usable 3d key found. available={npz.files}")

def clamp_cycle_indices(cycle, n_frames):
    """
    cycle = [ic, ic_opp, to, ic_end] (float index)
    0..n_frames-1 に丸め、単調増加が破綻するものは None
    """
    ic, ic_opp, to, ic_end = [int(np.round(x)) for x in cycle]
    ic = max(0, min(n_frames - 1, ic))
    ic_opp = max(0, min(n_frames - 1, ic_opp))
    to = max(0, min(n_frames - 1, to))
    ic_end = max(0, min(n_frames - 1, ic_end))
    if not (ic < ic_opp < to < ic_end):
        return None
    return [ic, ic_opp, to, ic_end]


def normalize_cycle(angle_series: np.ndarray, ic: int, to: int, ic_end: int, n_points=101):
    """
    角度時系列 angle_series から [ic..ic_end] を 0..100% に線形補間して返す。
    """
    if ic_end <= ic:
        return None
    orig_frames = np.arange(ic, ic_end + 1)
    x_new = np.linspace(0, 100, n_points)
    x_old = np.linspace(0, 100, len(orig_frames))
    y_old = angle_series[orig_frames]
    # NaNを含む場合は補間のために埋める（解析用、3D自体は変更しない）
    if np.any(~np.isfinite(y_old)):
        s = pd.Series(y_old).interpolate(limit_direction="both").to_numpy()
        y_old = s
    y_new = np.interp(x_new, x_old, y_old)
    stance_pct = ((to - ic) / (ic_end - ic)) * 100.0
    return x_new, y_new, stance_pct


def mean_std_cycles(cycles_ys: list[np.ndarray]):
    arr = np.array(cycles_ys, dtype=float)  # (n_cycles, n_points)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    return float(np.mean(np.abs(a[m] - b[m])))


# =========================
# 角度計算（BODY_25 index前提）
# =========================
def compute_angles_from_body25(kp3d: np.ndarray):
    """
    kp3d: (n_frames, 25, 3)
    returns dict of angle series (deg)
    """
    n = kp3d.shape[0]
    # 必要点抽出（4_ref_opti_opの定義に合わせる）
    neck = kp3d[:, 1, :]
    midhip = kp3d[:, 8, :]

    rhip = kp3d[:, 9, :]
    rknee = kp3d[:, 10, :]
    rankle = kp3d[:, 11, :]
    rhee = kp3d[:, 24, :]
    rtoe = (kp3d[:, 22, :] + kp3d[:, 23, :]) / 2.0

    lhip = kp3d[:, 12, :]
    lknee = kp3d[:, 13, :]
    lankle = kp3d[:, 14, :]
    lhee = kp3d[:, 21, :]
    ltoe = (kp3d[:, 19, :] + kp3d[:, 20, :]) / 2.0

    # ベクトル
    pel_up = neck - midhip
    thigh_r = rknee - rhip
    shank_r = rankle - rknee
    foot_r = rtoe - rhee

    thigh_l = lknee - lhip
    shank_l = lankle - lknee
    foot_l = ltoe - lhee

    # 回転軸など
    n_axis = rhip - lhip  # 左→右

    # 股関節屈伸/膝屈伸/足関節背屈底屈
    rhip_flex = op.culc_angle_all_frames(pel_up, thigh_r, n_axis, degrees=True, angle_type="hip")
    lhip_flex = op.culc_angle_all_frames(pel_up, thigh_l, n_axis, degrees=True, angle_type="hip")

    rknee_flex = op.culc_angle_all_frames(thigh_r, shank_r, n_axis, degrees=True, angle_type="knee")
    lknee_flex = op.culc_angle_all_frames(thigh_l, shank_l, n_axis, degrees=True, angle_type="knee")

    rankle_pldo = op.culc_angle_all_frames(shank_r, foot_r, n_axis, degrees=True, angle_type="ankle")
    lankle_pldo = op.culc_angle_all_frames(shank_l, foot_l, n_axis, degrees=True, angle_type="ankle")

    # 内転外転: 進行方向相当の軸を cross(pel_up, n_axis) とする（旧スクリプト互換）
    n_axis_adab = np.cross(pel_up, n_axis)
    # 0割対策（長さ0のときは少しだけ逃がす）
    eps = 1e-9
    n_norm = np.linalg.norm(n_axis_adab, axis=1)
    bad = n_norm < eps
    if np.any(bad):
        n_axis_adab[bad, :] = np.array([0.0, 0.0, 1.0])

    rhip_adab = op.culc_angle_all_frames(pel_up, thigh_r, n_axis_adab, degrees=True, angle_type="hip_adab")
    lhip_adab = op.culc_angle_all_frames(pel_up, thigh_l, n_axis_adab, degrees=True, angle_type="hip_adab")

    # 内旋外旋: （旧スクリプト互換）vector1=thigh, vector2=n_axis_adab, n=pel_up
    rhip_inex = op.culc_angle_all_frames(thigh_r, n_axis_adab, pel_up, degrees=True, angle_type="hip_inex")
    lhip_inex = op.culc_angle_all_frames(thigh_l, n_axis_adab, pel_up, degrees=True, angle_type="hip_inex")

    # 返す（必要なら増やせる）
    return dict(
        midhip=midhip,
        rhee=rhee,
        lhee=lhee,
        # angles
        R_Hip_FlEx=rhip_flex,
        R_Knee_FlEx=rknee_flex,
        R_Ankle_PlDo=rankle_pldo,
        R_Hip_AdAb=rhip_adab,
        R_Hip_InEx=rhip_inex,
        L_Hip_FlEx=lhip_flex,
        L_Knee_FlEx=lknee_flex,
        L_Ankle_PlDo=lankle_pldo,
        L_Hip_AdAb=lhip_adab,
        L_Hip_InEx=lhip_inex,
    )


# =========================
# 歩行パラメータ（3D）
# =========================
def calculate_gait_parameters_3d(gait_cycles, midhip, rhee, lhee, side, sampling_freq=60):
    """
    回帰に使用するパラメータを算出する
    gait_cycles: [[ic, ic_opp, to, ic_end], ...] (int index)
    
    算出する指標:
    - gait_speed: 歩行速度 [m/s]
    - swing_stance_ratio: 時間的対称性（遊脚期比）ここではまだ算出不可
    - stride_time: ストライド時間 [s]
    - stride_width: 歩隔 [m]
    """
    gait_params = []
    
    for cycle_idx, (ic_start, ic_opp, to, ic_end) in enumerate(gait_cycles):
        # 整数インデックスに変換
        ic_start_idx = int(np.round(ic_start))
        ic_opp_idx = int(np.round(ic_opp))
        to_idx = int(np.round(to))
        ic_end_idx = int(np.round(ic_end))
        
        # ストライド時間 [s]
        stride_time = (ic_end_idx - ic_start_idx) / sampling_freq
        
        # 遊脚期時間 [s]
        swing_duration = (ic_end_idx - to_idx) / sampling_freq
        
        # 歩行速度 [m/s]
        hip_displacement = np.linalg.norm(midhip[ic_end_idx] - midhip[ic_start_idx]) / 1000 # mmからmに変換
        gait_speed = hip_displacement / stride_time

        # 歩隔[m](とステップ長[m])
        if side == 'R':
            a = np.linalg.norm(lhee[ic_opp_idx] - rhee[ic_start_idx]) / 1000 # mmからmに変換
            b = np.linalg.norm(rhee[ic_end_idx] - lhee[ic_opp_idx]) / 1000 # mmからmに変換
            c = np.linalg.norm(rhee[ic_end_idx] - rhee[ic_start_idx]) / 1000 # mmからmに変換
            step_length = (b**2 + c**2 -a**2) / (2*c)
            stride_width = np.sqrt(b**2 - step_length**2)
        else:
            a = np.linalg.norm(rhee[ic_opp_idx] - lhee[ic_start_idx]) / 1000 # mmからmに変換
            b = np.linalg.norm(lhee[ic_end_idx] - rhee[ic_opp_idx]) / 1000 # mmからmに変換
            c = np.linalg.norm(lhee[ic_end_idx] - lhee[ic_start_idx]) / 1000 # mmからmに変換
            step_length = (b**2 + c**2 -a**2) / (2*c)
            stride_width = np.sqrt(b**2 - step_length**2)

        params = {
            'cycle_index': cycle_idx,
            'gait_speed': gait_speed,
            'stride_time': stride_time,
            'swing_duration': swing_duration,
            'stride_width': stride_width,
        }
        gait_params.append(params)
    
    return gait_params


# =========================
# 本体（1 trial x 1 method）
# =========================
def process_one_method(thera_dir: Path, npz_3d_path: Path):
    """
    1条件分をここで処理する
    """
    # ---------- 被験者ID・麻痺側 ----------
    sub_id = thera_dir.parent.name  # subX
    paralyzed_side = PARALYZED_SIDES.get(sub_id)
    
    if paralyzed_side is None:
        print(f"[SKIP] {sub_id} not in PARALYZED_SIDES dict. Skipping {thera_dir.name}")
        return
    
    # ---------- 3Dnpzファイルの読み込み ----------
    npz = np.load(npz_3d_path, allow_pickle=True)
    used_key, kp3d = pick_3d_array(npz, PREFERRED_3D_KEY_ORDER)
    
    npz_pt_path = Path(str(npz_3d_path).replace("_PA.npz", "_PT.npz"))
    npz_pt = np.load(npz_pt_path, allow_pickle=True)
    _, kp3d_pt = pick_3d_array(npz_pt, PREFERRED_3D_KEY_ORDER)

    # kp3d: (n_frames, 25, 3)
    n_frames = len(npz["frame"])
    method_tag = npz_3d_path.stem  # 3d_kp_<method(ViTPose)>_<csv_tag>
    # method名を雑に切り出し
    # 例: 3d_kp_ViTPose_pa_spline など
    m = re.match(r"3d_kp_(.+)$", method_tag)
    # 例: ViTPose_pa_spline など
    method_tag = m.group(1) if m else method_tag
    
    # ---------- 結果の保存先ディレクトリ ----------
    tag_short = npz_3d_path.stem.replace("3d_kp_", "")

    out_root = thera_dir / "ViTPose_results"
    out_root.mkdir(parents=True, exist_ok=True)
    
    #  --------- 歩行イベント検出 ----------
    event_frame_dict = op.cucl_gait_event(kp3d)
    gait_cycles_r, gait_cycles_l = op.culc_gait_cycles(event_frame_dict)
    
    #######################################################################################
    #######################################################################################
    # # 重要！
    # # 今回は使用する歩行周期を最後から1つだけに絞る（一番安定して検出可能）
    # gait_cycles_r = gait_cycles_r[-1:] if len(gait_cycles_r) > 0 else []
    # gait_cycles_l = gait_cycles_l[-1:] if len(gait_cycles_l) > 0 else []
    
    #######################################################################################
    #######################################################################################

    if len(gait_cycles_r) == 0 and len(gait_cycles_l) == 0:
        print(f"[SKIP] no valid cycles after conversion: {thera_dir.name} / {npz_3d_path.name}")
        return
    
    # デバック用に初期接地フレームを取得しておく
    ic_r = event_frame_dict.get("ic_r", [])
    ic_l = event_frame_dict.get("ic_l", [])

    # ---------- 関節角度算出 ----------
    angles_dict = compute_angles_from_body25(kp3d)

    def plot_three_timeseries(title, keys_r, keys_l, angles_dict, fname, ic_r=None, ic_l=None):
        """
        keys_r: ["R_Hip_FlEx","R_Knee_FlEx","R_Ankle_PlDo"] など
        keys_l: ["L_Hip_FlEx","L_Knee_FlEx","L_Ankle_PlDo"] など
        ic_rやic_l: 左右の初期接地フレームリスト（プロットに縦線を入れる）
        3段：股→膝→足
        横軸：フレーム番号
        各段に R と L を重ね描き
        """
        assert len(keys_r) == 3 and len(keys_l) == 3

        # フレーム数（R/Lで長さが違う可能性に備えて最小に合わせる）
        n_r = len(angles_dict[keys_r[0]]) if keys_r[0] in angles_dict else 0
        n_l = len(angles_dict[keys_l[0]]) if keys_l[0] in angles_dict else 0
        n = min(n_r, n_l) if (n_r > 0 and n_l > 0) else max(n_r, n_l)
        frames = np.arange(n)

        # 初期接地フレーム（範囲外は除外、重複も潰す）
        ic_r = [] if ic_r is None else sorted({int(x) for x in ic_r if 0 <= int(x) < n})
        ic_l = [] if ic_l is None else sorted({int(x) for x in ic_l if 0 <= int(x) < n})

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

        for ax, kr, kl in zip(axes, keys_r, keys_l):
            has_r = kr in angles_dict
            has_l = kl in angles_dict

            if not has_r and not has_l:
                ax.set_title(f"{kr} / {kl} (missing)")
                ax.grid(True)
                continue

            if has_r:
                y_r = np.asarray(angles_dict[kr], dtype=float)[:n]
                ax.plot(frames, y_r, label=kr, color="tab:orange")

            if has_l:
                y_l = np.asarray(angles_dict[kl], dtype=float)[:n]
                ax.plot(frames, y_l, label=kl, color="tab:blue")

            # ---- IC 縦線（破線） ----
            # 凡例が増えすぎないよう「最初の1本だけ label を付ける」
            for i, f in enumerate(ic_r):
                ax.axvline(f, linestyle="--", linewidth=1.0,
                        color="tab:orange", alpha=0.6,
                        label="IC_R" if i == 0 else None)
            for i, f in enumerate(ic_l):
                ax.axvline(f, linestyle="--", linewidth=1.0,
                        color="tab:blue", alpha=0.6,
                        label="IC_L" if i == 0 else None)
            
            ax.set_ylabel("Angle [deg]")

            # y軸範囲は「R_」「L_」を外したベース名で共通設定（既存関数を流用）
            # 例: R_Hip_FlEx / L_Hip_FlEx のどちらでも同じYlimになる
            ref_key = kr if has_r else kl
            ylims = ylims_for_angle_key(ref_key)
            if ylims is not None:
                ax.set_ylim(*ylims)

            ax.grid(True)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Frame")
        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(out_root / fname)
        plt.close()
    
    plot_three_timeseries(
        f"Timeseries Flex/Ext (R & L) - {method_tag} ({used_key})",
        ["R_Hip_FlEx", "R_Knee_FlEx", "R_Ankle_PlDo"],
        ["L_Hip_FlEx", "L_Knee_FlEx", "L_Ankle_PlDo"],
        angles_dict,
        f"timeseries_FlEx_{tag_short}.png",
        ic_r=ic_r, ic_l=ic_l
    )
    plot_three_timeseries(
        f"Timeseries Ad/Ab (R & L) - {method_tag} ({used_key})",
        ["R_Hip_AdAb", "R_Knee_AdAb", "R_Ankle_AdAb"],
        ["L_Hip_AdAb", "L_Knee_AdAb", "L_Ankle_AdAb"],
        angles_dict,
        f"timeseries_AdAb_{tag_short}.png",
        ic_r=ic_r, ic_l=ic_l
    )
    plot_three_timeseries(
        f"Timeseries In/Ex (R & L) - {method_tag} ({used_key})",
        ["R_Hip_InEx", "R_Knee_InEx", "R_Ankle_InEx"],
        ["L_Hip_InEx", "L_Knee_InEx", "L_Ankle_InEx"],
        angles_dict,
        f"timeseries_InEx_{tag_short}.png",
        ic_r=ic_r, ic_l=ic_l
    )

    # ---------- 歩行パラメータの計算 ----------
    midhip = kp3d[:, 8, :]
    rhee = kp3d[:, 24, :]
    lhee = kp3d[:, 21, :]

    gait_params_r = calculate_gait_parameters_3d(gait_cycles_r, midhip, rhee, lhee, side="R", sampling_freq=FS_3D)
    gait_params_l = calculate_gait_parameters_3d(gait_cycles_l, midhip, rhee, lhee, side="L", sampling_freq=FS_3D)

    # 時間的対称性指標（遊脚期比のSI）および最大関節角度を算出
    if paralyzed_side == "R":
        swing_duration_para = np.array([p['swing_duration'] for p in gait_params_r]).mean()
        swing_duration_nonpara = np.array([p['swing_duration'] for p in gait_params_l]).mean()
        symmetry_index_sw = abs(swing_duration_para - swing_duration_nonpara) / (0.5 * (swing_duration_para + swing_duration_nonpara)) * 100
        # symmetry_index_swも保存　一度の施行で一回算出する値なので各サイクルに同じ値を入れておく
        for i_cycle, cycle_frames in enumerate(gait_cycles_r):
            gait_params_r[i_cycle]['SI_sw'] = symmetry_index_sw
            ic_start = int(cycle_frames[0])
            ic_end = int(cycle_frames[3])
            hip_flex = angles_dict["R_Hip_FlEx"][ic_start:ic_end+1]
            knee_flex = angles_dict["R_Knee_FlEx"][ic_start:ic_end+1]
            ankle_pldo = angles_dict["R_Ankle_PlDo"][ic_start:ic_end+1]
            hip_abad = angles_dict["R_Hip_AdAb"][ic_start:ic_end+1]
            hip_max_ext = - np.min(hip_flex)  # 股関節最大伸展　伸展は負の値になるので正にするために符号反転
            knee_max_flex = np.max(knee_flex)  # 膝関節最大屈曲
            ankle_max_do = np.max(ankle_pldo)  # 足関節最大背屈
            hip_max_ab = np.max(hip_abad)  # 股関節最大外転
            gait_params_r[i_cycle]['hip_max_ext'] = hip_max_ext
            gait_params_r[i_cycle]['knee_max_flex'] = knee_max_flex
            gait_params_r[i_cycle]['ankle_max_do'] = ankle_max_do
            gait_params_r[i_cycle]['hip_max_ab'] = hip_max_ab
        
    elif paralyzed_side == "L":
        swing_duration_para = np.array([p['swing_duration'] for p in gait_params_l]).mean()
        swing_duration_nonpara = np.array([p['swing_duration'] for p in gait_params_r]).mean()
        symmetry_index_sw = abs(swing_duration_para - swing_duration_nonpara) / (0.5 * (swing_duration_para + swing_duration_nonpara)) * 100
        # symmetry_index_swも保存　一度の施行で一回算出する値なので各サイクルに同じ値を入れておく
        for i_cycle, cycle_frames in enumerate(gait_cycles_l):
            gait_params_l[i_cycle]['SI_sw'] = symmetry_index_sw
            ic_start = int(cycle_frames[0])
            ic_end = int(cycle_frames[3])
            hip_flex = angles_dict["L_Hip_FlEx"][ic_start:ic_end+1]
            knee_flex = angles_dict["L_Knee_FlEx"][ic_start:ic_end+1]
            ankle_pldo = angles_dict["L_Ankle_PlDo"][ic_start:ic_end+1]
            hip_abad = angles_dict["L_Hip_AdAb"][ic_start:ic_end+1]
            hip_max_ext = - np.min(hip_flex)  # 股関節最大伸展　伸展は負の値になるので正にするために符号反転
            knee_max_flex = np.max(knee_flex)  # 膝関節最大屈曲
            ankle_max_do = np.max(ankle_pldo)  # 足関節最大背屈
            hip_max_ab = np.max(hip_abad)  # 股関節最大外転
            gait_params_l[i_cycle]['hip_max_ext'] = hip_max_ext
            gait_params_l[i_cycle]['knee_max_flex'] = knee_max_flex
            gait_params_l[i_cycle]['ankle_max_do'] = ankle_max_do
            gait_params_l[i_cycle]['hip_max_ab'] = hip_max_ab
            
    max_angle_list = [hip_max_ext, knee_max_flex, ankle_max_do, hip_max_ab]
    
    # 歩行パラメータのまとめ
    def summarize_gait_parameters(gait_params_r, gait_params_l, symmetry_index_sw, max_angle_list, paralyzed_side):
        """
        歩行パラメータのまとめを作成
        """
        if paralyzed_side == "R":
            gait_speed = np.mean([p['gait_speed'] for p in gait_params_r])
            stride_time = np.mean([p['stride_time'] for p in gait_params_r])
            stride_width = np.mean([p['stride_width'] for p in gait_params_r])
        elif paralyzed_side == "L":
            gait_speed = np.mean([p['gait_speed'] for p in gait_params_l])
            stride_time = np.mean([p['stride_time'] for p in gait_params_l])
            stride_width = np.mean([p['stride_width'] for p in gait_params_l])
        summary = {
            'gait_speed': gait_speed,
            'symmetry_index_sw': symmetry_index_sw,
            'stride_time': stride_time,
            'stride_width': stride_width,
            'hip_max_ext': max_angle_list[0],
            'knee_max_flex': max_angle_list[1],
            'ankle_max_do': max_angle_list[2],
            'hip_max_ab': max_angle_list[3],
        }
        return summary
    
    pa_gait_params = summarize_gait_parameters(gait_params_r, gait_params_l, symmetry_index_sw, max_angle_list, paralyzed_side)
    
    # ---------- 結果保存 ----------
    # 歩行パラメータ保存
    pa_gait_params_df = pd.DataFrame([pa_gait_params])
    pa_gait_params_df.to_csv(out_root / f"pa_gait_parameters.csv", index=False)

    # PTの介助指標を算出  ##############################################################################
    def _nan_safe_zscore(x):
        x = np.asarray(x, float)
        m = np.nanmean(x)
        s = np.nanstd(x)
        if not np.isfinite(s) or s < 1e-12:
            return np.full_like(x, np.nan)
        return (x - m) / s

    def _norm_xcorr_at_lag(a, b, lag: int, min_valid: int = 5):
        """
        zscore 済み 1D 配列 a, b について，指定 lag の正規化相互相関（平均(a*b)）を返す。
        lag > 0: b が遅れる（b を +lag シフト）
        lag < 0: b が先行
        """
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        n = len(a)
        if n != len(b) or n < min_valid:
            return np.nan

        if lag < 0:
            aa, bb = a[-lag:], b[:n + lag]
        elif lag > 0:
            aa, bb = a[:n - lag], b[lag:]
        else:
            aa, bb = a, b

        if len(aa) < min_valid:
            return np.nan

        mask = np.isfinite(aa) & np.isfinite(bb)
        if np.sum(mask) < min_valid:
            return np.nan

        return float(np.nanmean(aa[mask] * bb[mask]))
    
    def _norm_xcorr_xyz_max(pa_xyz, pt_xyz, max_lag: int, min_valid: int = 5):
        """
        3軸(x,y,z)の相互相関を計算し，
        C3D(lag)=sqrt(Cx(lag)^2 + Cy(lag)^2 + Cz(lag)^2) を最大化する lag を返す（方法B）

        Returns:
            ccx, ccy, ccz, c3d, best_lag
        """
        pa_xyz = np.asarray(pa_xyz, float)
        pt_xyz = np.asarray(pt_xyz, float)
        if pa_xyz.ndim != 2 or pt_xyz.ndim != 2 or pa_xyz.shape[1] != 3 or pt_xyz.shape[1] != 3:
            raise ValueError("pa_xyz and pt_xyz must be (N,3)")

        # 軸ごとに z-score（NaN安全）
        pa_x = _nan_safe_zscore(pa_xyz[:, 0])
        pa_y = _nan_safe_zscore(pa_xyz[:, 1])
        pa_z = _nan_safe_zscore(pa_xyz[:, 2])

        pt_x = _nan_safe_zscore(pt_xyz[:, 0])
        pt_y = _nan_safe_zscore(pt_xyz[:, 1])
        pt_z = _nan_safe_zscore(pt_xyz[:, 2])

        best_c3d = -np.inf
        best_lag = 0
        best = (np.nan, np.nan, np.nan, np.nan)

        for lag in range(-max_lag, max_lag + 1):
            ccx = _norm_xcorr_at_lag(pa_x, pt_x, lag, min_valid=min_valid)
            ccy = _norm_xcorr_at_lag(pa_y, pt_y, lag, min_valid=min_valid)
            ccz = _norm_xcorr_at_lag(pa_z, pt_z, lag, min_valid=min_valid)

            if not (np.isfinite(ccx) and np.isfinite(ccy) and np.isfinite(ccz)):
                continue

            c3d = float(np.sqrt(ccx * ccx + ccy * ccy + ccz * ccz))
            if np.isfinite(c3d) and c3d > best_c3d:
                best_c3d = c3d
                best_lag = lag
                best = (ccx, ccy, ccz, c3d)

        if best_c3d == -np.inf:
            return (np.nan, np.nan, np.nan, np.nan, 0)

        ccx, ccy, ccz, c3d = best
        return float(ccx), float(ccy), float(ccz), float(c3d), int(best_lag)

    def calculate_pt_assist_metrics_by_cycle(gait_cycles,pa_midhip,pt_midhip,pa_neck,pt_neck,pt_wrist_para,pt_wrist_nonpara,pt_forearm_length,sampling_freq=60.0,max_lag=30):
        """
        gait_cycles: [[ic, ic_opp, to, ic_end], ...]
        pa_midhip, pt_midhip: (N,3) [mm]
        """
        max_lag_frames = int(max_lag)

        hip_dist_cycle = []
        hip_dist_normalized_cycle = []
        wri_para_s_cycle = []
        wri_nonpara_s_cycle = []
        cos_sim_cycle = []
        hip_cc_x_cycle = []
        hip_cc_y_cycle = []
        hip_cc_z_cycle = []
        hip_cc_3d_cycle = []
        hip_cc_lag_cycle = []
        
        def _save_trunk_anim_2d_and_cossim(
            pa_midhip_seg, pa_neck_seg,
            pt_midhip_seg, pt_neck_seg,
            cos_sim,
            out_path_mp4: Path,
            fps: int = 30,
            dpi: int = 150,
        ):
            """
            4パネル（XY, YZ, ZX, cos_sim）をフレームごとに更新して動画保存。
            各フレームで矢印は「その瞬間の1本」だけ描画（PA=赤, PT=青）。
            """
            pa_midhip_seg = np.asarray(pa_midhip_seg, float)
            pa_neck_seg   = np.asarray(pa_neck_seg, float)
            pt_midhip_seg = np.asarray(pt_midhip_seg, float)
            pt_neck_seg   = np.asarray(pt_neck_seg, float)
            cos_sim       = np.asarray(cos_sim, float)

            n = min(len(pa_midhip_seg), len(pa_neck_seg), len(pt_midhip_seg), len(pt_neck_seg), len(cos_sim))
            if n < 3:
                return

            trunk_pa = pa_neck_seg[:n] - pa_midhip_seg[:n]
            trunk_pt = pt_neck_seg[:n] - pt_midhip_seg[:n]

            # limit を全フレームから自動決定
            def _auto_lim(v, a, b):
                v = np.asarray(v, float)
                m = np.all(np.isfinite(v[:, [a, b]]), axis=1)
                if not np.any(m):
                    return 100.0
                mx = float(np.nanmax(np.abs(v[m][:, [a, b]])))
                return max(50.0, mx * 1.2)

            lim_xy = max(_auto_lim(trunk_pa, 0, 1), _auto_lim(trunk_pt, 0, 1))
            lim_yz = max(_auto_lim(trunk_pa, 1, 2), _auto_lim(trunk_pt, 1, 2))
            lim_zx = max(_auto_lim(trunk_pa, 2, 0), _auto_lim(trunk_pt, 2, 0))

            fig = plt.figure(figsize=(18, 4.5))
            ax_xy = fig.add_subplot(1, 4, 1)
            ax_yz = fig.add_subplot(1, 4, 2)
            ax_zx = fig.add_subplot(1, 4, 3)
            ax_cs = fig.add_subplot(1, 4, 4)

            # axes setup
            def _setup_ax(ax, title, xlabel, ylabel, lim):
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(True)
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.axis("equal")
                ax.scatter([0], [0], c="k", s=20, zorder=3)

            _setup_ax(ax_xy, "Trunk (XY)", "X [mm]", "Y [mm]", lim_xy)
            _setup_ax(ax_yz, "Trunk (YZ)", "Y [mm]", "Z [mm]", lim_yz)
            _setup_ax(ax_zx, "Trunk (ZX)", "Z [mm]", "X [mm]", lim_zx)

            ax_cs.set_title("cos_sim (PA·PT) within cycle")
            ax_cs.set_xlabel("Frame (within cycle)")
            ax_cs.set_ylabel("cos_sim [-]")
            ax_cs.set_ylim(0.965, 1.001)
            ax_cs.grid(True)
            line_cs, = ax_cs.plot([], [], linewidth=2)

            # “現在フレーム”縦線
            vline = ax_cs.axvline(0, linestyle="--", linewidth=1.5)

            # quiver（初期はゼロ矢印）
            q_pa_xy = ax_xy.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0,
                                width=0.01, color="red", alpha=0.8, label="PA")
            q_pt_xy = ax_xy.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0,
                                width=0.01, color="blue", alpha=0.8, label="PT")
            ax_xy.legend(loc="upper right", frameon=False)

            q_pa_yz = ax_yz.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0,
                                width=0.01, color="red", alpha=0.8, label="PA")
            q_pt_yz = ax_yz.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0,
                                width=0.01, color="blue", alpha=0.8, label="PT")
            ax_yz.legend(loc="upper right", frameon=False)

            q_pa_zx = ax_zx.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0,
                                width=0.01, color="red", alpha=0.8, label="PA")
            q_pt_zx = ax_zx.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1.0,
                                width=0.01, color="blue", alpha=0.8, label="PT")
            ax_zx.legend(loc="upper right", frameon=False)

            x_all = np.arange(n)
            line_cs.set_data(x_all, cos_sim[:n])

            def _safe_vec(v):
                return v if np.all(np.isfinite(v)) else np.array([np.nan, np.nan, np.nan], float)

            def init():
                # 何もしなくてもOK（lineは固定で張ってる）
                vline.set_xdata([0, 0])
                return (q_pa_xy, q_pt_xy, q_pa_yz, q_pt_yz, q_pa_zx, q_pt_zx, line_cs, vline)

            def update(t):
                vpa = _safe_vec(trunk_pa[t])
                vpt = _safe_vec(trunk_pt[t])

                # NaNなら矢印を消す（長さ0扱い）
                if not np.all(np.isfinite(vpa)):
                    vpa = np.array([0.0, 0.0, 0.0])
                if not np.all(np.isfinite(vpt)):
                    vpt = np.array([0.0, 0.0, 0.0])

                # quiverの更新：set_UVC(U, V)
                q_pa_xy.set_UVC(vpa[0], vpa[1])
                q_pt_xy.set_UVC(vpt[0], vpt[1])

                q_pa_yz.set_UVC(vpa[1], vpa[2])
                q_pt_yz.set_UVC(vpt[1], vpt[2])

                q_pa_zx.set_UVC(vpa[2], vpa[0])
                q_pt_zx.set_UVC(vpt[2], vpt[0])

                vline.set_xdata([t, t])
                return (q_pa_xy, q_pt_xy, q_pa_yz, q_pt_yz, q_pa_zx, q_pt_zx, line_cs, vline)

            ani = animation.FuncAnimation(fig, update, frames=n, init_func=init, interval=1000/fps, blit=False)

            out_path_mp4 = Path(out_path_mp4)
            out_path_mp4.parent.mkdir(parents=True, exist_ok=True)

            # MP4（ffmpegがあれば）
            try:
                writer = animation.FFMpegWriter(fps=fps)
                ani.save(str(out_path_mp4), writer=writer, dpi=dpi)
            except Exception as e:
                # GIF（pillowがあれば）
                out_gif = out_path_mp4.with_suffix(".gif")
                try:
                    ani.save(str(out_gif), writer=animation.PillowWriter(fps=fps), dpi=dpi)
                except Exception as e2:
                    print(f"[WARN] animation save failed. mp4_err={e} gif_err={e2}")

            plt.close(fig)
        
        def _plot_trunk_vectors_and_cossim(
            pa_midhip_seg, pa_neck_seg, pt_midhip_seg, pt_neck_seg,
            cos_sim, out_path_png, stride=5
        ):
            """
            体幹ベクトル（midhip->neck）を3D矢印で可視化し、cos_sim時系列も並べて保存する。
            stride: 矢印を間引く間隔（例:5なら5フレームごとに描画）
            """
            pa_midhip_seg = np.asarray(pa_midhip_seg, float)
            pa_neck_seg   = np.asarray(pa_neck_seg, float)
            pt_midhip_seg = np.asarray(pt_midhip_seg, float)
            pt_neck_seg   = np.asarray(pt_neck_seg, float)
            cos_sim       = np.asarray(cos_sim, float)

            n = min(len(pa_midhip_seg), len(pa_neck_seg), len(pt_midhip_seg), len(pt_neck_seg), len(cos_sim))
            if n < 3:
                return

            idx = np.arange(0, n, max(1, int(stride)))

            # trunk vectors (midhip -> neck)
            trunk_pa = pa_neck_seg[:n] - pa_midhip_seg[:n]
            trunk_pt = pt_neck_seg[:n] - pt_midhip_seg[:n]

            # pick sampled points
            pa_o = pa_midhip_seg[:n][idx]
            pt_o = pt_midhip_seg[:n][idx]
            pa_v = trunk_pa[idx]
            pt_v = trunk_pt[idx]

            # finite masks
            m_pa = np.all(np.isfinite(pa_o), axis=1) & np.all(np.isfinite(pa_v), axis=1)
            m_pt = np.all(np.isfinite(pt_o), axis=1) & np.all(np.isfinite(pt_v), axis=1)

            # 2D quiver helper
            def _quiver2d(ax, pa_v, m_pa, pt_v, m_pt, a, b, title):
                # 原点
                ax.scatter([0], [0], c="k", s=20, zorder=3)

                # PA（赤）
                if np.any(m_pa):
                    oa = np.zeros(np.sum(m_pa))
                    ob = np.zeros(np.sum(m_pa))
                    va = pa_v[m_pa, a]
                    vb = pa_v[m_pa, b]
                    ax.quiver(
                        oa, ob, va, vb,
                        angles="xy", scale_units="xy", scale=1.0,
                        width=0.004, color="red", alpha=0.7, label="PA"
                    )

                # PT（青）
                if np.any(m_pt):
                    oa = np.zeros(np.sum(m_pt))
                    ob = np.zeros(np.sum(m_pt))
                    va = pt_v[m_pt, a]  
                    vb = pt_v[m_pt, b]
                    ax.quiver(
                        oa, ob, va, vb,
                        angles="xy", scale_units="xy", scale=1.0,
                        width=0.004, color="blue", alpha=0.7, label="PT"
                    )

                ax.set_title(title)
                ax.set_xlabel(["X", "Y", "Z"][a] + " [mm]")
                ax.set_ylabel(["X", "Y", "Z"][b] + " [mm]")
                ax.grid(True)
                ax.axis("equal")
                
                # ---- axis limit: auto from data (mm or whatever unit) ----
                vals = []
                if np.any(m_pa):
                    vals.append(np.abs(pa_v[m_pa][:, [a, b]]))
                if np.any(m_pt):
                    vals.append(np.abs(pt_v[m_pt][:, [a, b]]))

                if len(vals) == 0:
                    lim = 100.0
                else:
                    mx = float(np.nanmax(np.vstack(vals)))
                    lim = max(50.0, mx * 1.2)   # 最低50、データに合わせて拡張

                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)

                # 凡例は1回だけ出るように
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(loc="upper right", frameon=False)
            # -------------------------
            # Figure layout:
            #   row1: PA (XY, YZ, ZX) + cos_sim
            #   row2: PT (XY, YZ, ZX) + cos_sim
            # -------------------------
            fig = plt.figure(figsize=(18, 4.5))

            ax_xy = fig.add_subplot(1, 4, 1)
            ax_yz = fig.add_subplot(1, 4, 2)
            ax_zx = fig.add_subplot(1, 4, 3)
            ax_cs = fig.add_subplot(1, 4, 4)

            _quiver2d(ax_xy, pa_v, m_pa, pt_v, m_pt, 0, 1, "Trunk (XY)")
            _quiver2d(ax_yz, pa_v, m_pa, pt_v, m_pt, 1, 2, "Trunk (YZ)")
            _quiver2d(ax_zx, pa_v, m_pa, pt_v, m_pt, 2, 0, "Trunk (ZX)")

            ax_cs.plot(np.arange(n), cos_sim, linewidth=2)
            ax_cs.set_title("cos_sim (PA·PT) within cycle")
            ax_cs.set_xlabel("Frame (within cycle)")
            ax_cs.set_ylabel("cos_sim [-]")
            ax_cs.set_ylim(0.965, 1.001)
            ax_cs.grid(True)

            plt.tight_layout()
            plt.savefig(out_path_png, dpi=150)
            plt.close()
            
        def _axis_angles_deg_from_vec(v_xyz: np.ndarray, unwrap: bool = True) -> dict:
            """
            v_xyz: (N,3)
            returns: dict(ax_deg, ay_deg, az_deg)
            ax: rotation about X (angle in YZ plane) = atan2(z, y)
            ay: rotation about Y (angle in ZX plane) = atan2(x, z)
            az: rotation about Z (angle in XY plane) = atan2(y, x)
            """
            v = np.asarray(v_xyz, float)
            x, y, z = v[:, 0], v[:, 1], v[:, 2]

            ax = np.arctan2(z, y)
            ay = np.arctan2(x, z)
            az = np.arctan2(y, x)
            
            ax_deg = np.degrees(ax)
            ay_deg = np.degrees(ay)
            az_deg = np.degrees(az)
            
            if unwrap:
                ax_deg = np.unwrap(ax_deg)
                ay_deg = np.unwrap(ay_deg)
                az_deg = np.unwrap(az_deg)
    
            return {"ax_deg": ax_deg, "ay_deg": ay_deg, "az_deg": az_deg}
        
        def _plot_trunk_axis_angles(pa_trunk, pt_trunk, out_path_png, title="Trunk axis angles"):
            """
            pa_trunk, pt_trunk: (N,3) trunk vectors (midhip->neck)
            3軸まわり角度の時系列を、PA/PT重ね描きで保存
            """
            pa_trunk = np.asarray(pa_trunk, float)
            pt_trunk = np.asarray(pt_trunk, float)

            n = min(len(pa_trunk), len(pt_trunk))
            if n < 3:
                return

            pa = pa_trunk[:n]
            pt = pt_trunk[:n]

            # 有効フレーム（両者ともfinite）
            m = np.all(np.isfinite(pa), axis=1) & np.all(np.isfinite(pt), axis=1)
            if np.sum(m) < 3:
                return

            # 欠損はNaNにしておき、プロットで途切れさせる
            pa2 = pa.copy()
            pt2 = pt.copy()
            pa2[~m] = np.nan
            pt2[~m] = np.nan

            pa_ang = _axis_angles_deg_from_vec(pa2, unwrap=True)
            pt_ang = _axis_angles_deg_from_vec(pt2, unwrap=True)

            frames = np.arange(n)

            fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

            axes[0].plot(frames, pa_ang["ax_deg"], label="PA", linewidth=2, color="red")
            axes[0].plot(frames, pt_ang["ax_deg"], label="PT", linewidth=2, color="blue")
            axes[0].set_ylabel("deg")
            axes[0].set_title("about X (atan2(z, y))")
            axes[0].grid(True)
            axes[0].legend()

            axes[1].plot(frames, pa_ang["ay_deg"], label="PA", linewidth=2, color="red")
            axes[1].plot(frames, pt_ang["ay_deg"], label="PT", linewidth=2, color="blue")
            axes[1].set_ylabel("deg")
            axes[1].set_title("about Y (atan2(x, z))")
            axes[1].grid(True)
            axes[1].legend()

            axes[2].plot(frames, pa_ang["az_deg"], label="PA", linewidth=2, color="red")
            axes[2].plot(frames, pt_ang["az_deg"], label="PT", linewidth=2, color="blue")
            axes[2].set_ylabel("deg")
            axes[2].set_title("about Z (atan2(y, x))")
            axes[2].set_xlabel("Frame (within cycle)")
            axes[2].grid(True)
            axes[2].legend()

            fig.suptitle(title)
            plt.tight_layout()
            plt.savefig(out_path_png, dpi=150)
            plt.close()

        for ic, _, _, ic_end in gait_cycles:
            if ic_end <= ic:
                continue

            pa_midhip_seg = pa_midhip[ic:ic_end]
            pt_midhip_seg = pt_midhip[ic:ic_end]
            pa_neck_seg = pa_neck[ic:ic_end]
            pt_neck_seg = pt_neck[ic:ic_end]
            pt_wrist_para_seg = pt_wrist_para[ic:ic_end]
            pt_wrist_nonpara_seg = pt_wrist_nonpara[ic:ic_end]
            if len(pa_midhip_seg) < 5:
                continue
            

            # --- hip_dist [m] ---
            dist = np.linalg.norm(pa_midhip_seg - pt_midhip_seg, axis=1) / 1000.0
            hip_dist_cycle.append(np.nanmedian(dist))
            
            # --- hip_dist normalized [-] ---
            dist_normalized = dist / (pt_forearm_length / 1000.0)
            hip_dist_normalized_cycle.append(np.nanmedian(dist_normalized))
            
            # --- wri_pos_scale [-] ---
            u = pa_neck_seg - pa_midhip_seg
            num_para = np.sum((pt_wrist_para_seg - pa_midhip_seg) * u, axis=1)  # (N,)
            num_nonpara = np.sum((pt_wrist_nonpara_seg - pa_midhip_seg) * u, axis=1)  # (N,)
            den = np.sum(u * u, axis=1)                          # (N,)

            wri_para_s = num_para / den
            wri_nonpara_s = num_nonpara / den
            valid_para = np.isfinite(wri_para_s)
            if np.any(valid_para):
                wri_para_s_cycle.append(np.nanmedian(wri_para_s))
            else:
                wri_para_s_cycle.append(np.nan)
            valid_nonpara = np.isfinite(wri_nonpara_s)
            if np.any(valid_nonpara):
                wri_nonpara_s_cycle.append(np.nanmedian(wri_nonpara_s))
            else:
                wri_nonpara_s_cycle.append(np.nan)
            
            # --- cos_sim ----
            trunk_pa = pa_neck_seg - pa_midhip_seg
            trunk_pt = pt_neck_seg - pt_midhip_seg
            dot = np.sum(trunk_pa * trunk_pt, axis=1)
            den = np.linalg.norm(trunk_pa, axis=1) * np.linalg.norm(trunk_pt, axis=1)
            cos_sim = dot / den
            cos_sim[~np.isfinite(cos_sim)] = np.nan
            cos_sim_cycle.append(np.nanmedian(cos_sim))
            
            save_trunk_angle_fig = False
            if save_trunk_angle_fig:
                cycle_idx = len(hip_dist_cycle)  # 既存の命名と合わせる
                debug_dir = thera_dir / "ViTPose_results"
                debug_dir.mkdir(parents=True, exist_ok=True)
                _plot_trunk_axis_angles(
                    pa_trunk=trunk_pa,
                    pt_trunk=trunk_pt,
                    out_path_png=debug_dir / f"trunk_axis_angles_cycle_{cycle_idx:02d}.png",
                    title=f"Trunk axis angles (Cycle {cycle_idx:02d})",
                )
                
            save_trunk_anim = False
            if save_trunk_anim:
                cycle_idx = len(hip_dist_cycle)
                debug_dir = thera_dir / "ViTPose_results"
                debug_dir.mkdir(parents=True, exist_ok=True)
                _save_trunk_anim_2d_and_cossim(
                    pa_midhip_seg, pa_neck_seg,
                    pt_midhip_seg, pt_neck_seg,
                    cos_sim,
                    out_path_mp4=debug_dir / f"trunk_vec_and_cossim_cycle_{cycle_idx:02d}.mp4",
                    fps=30
                )
                
            
            # --- trunk vectors 可視化（デバッグ）---
            save_trunk_fig = False
            if save_trunk_fig:
                cycle_idx = len(hip_dist_cycle)  # いまの周期番号（既存の付け方に合わせる）
                debug_dir = thera_dir / "ViTPose_results"
                debug_dir.mkdir(parents=True, exist_ok=True)
                _plot_trunk_vectors_and_cossim(
                    pa_midhip_seg, pa_neck_seg,
                    pt_midhip_seg, pt_neck_seg,
                    cos_sim,
                    out_path_png=debug_dir / f"trunk_vec_and_cossim_cycle_{cycle_idx:02d}.png",
                    stride=5
                )


            # --- hip_cc (x,y,z), lag ---
            hip_cc_x_cycle_, hip_cc_y_cycle_, hip_cc_z_cycle_, hip_cc_3d_cycle_, lag = _norm_xcorr_xyz_max(
                pa_midhip_seg, pt_midhip_seg, max_lag_frames
            )
            hip_cc_x_cycle.append(hip_cc_x_cycle_)
            hip_cc_y_cycle.append(hip_cc_y_cycle_)
            hip_cc_z_cycle.append(hip_cc_z_cycle_)
            hip_cc_3d_cycle.append(hip_cc_3d_cycle_)
            hip_cc_lag_cycle.append(lag / sampling_freq)
            
            save_hip_x_fig = True  # デバッグ用にTrueにすると各周期のヒップ座標をプロット
            if save_hip_x_fig:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                frames = np.arange(len(pa_midhip_seg))
                
                # X座標
                axes[0].plot(frames, pa_midhip_seg[:, 0], label='PA', color='tab:red', linewidth=2)
                axes[0].plot(frames, pt_midhip_seg[:, 0], label='PT', color='tab:blue', linewidth=2)
                axes[0].set_xlabel('Frame')
                axes[0].set_ylabel('X [mm]')
                axes[0].set_title(f'Hip X')
                axes[0].legend()
                axes[0].grid(True)
                
                # Y座標
                axes[1].plot(frames, pa_midhip_seg[:, 1], label='PA', color='tab:red', linewidth=2)
                axes[1].plot(frames, pt_midhip_seg[:, 1], label='PT', color='tab:blue', linewidth=2)
                axes[1].set_xlabel('Frame')
                axes[1].set_ylabel('Y [mm]')
                axes[1].set_title('Hip Y')
                axes[1].legend()
                axes[1].grid(True)
                
                # Z座標
                axes[2].plot(frames, pa_midhip_seg[:, 2], label='PA', color='tab:red', linewidth=2)
                axes[2].plot(frames, pt_midhip_seg[:, 2], label='PT', color='tab:blue', linewidth=2)
                axes[2].set_xlabel('Frame')
                axes[2].set_ylabel('Z [mm]')
                axes[2].set_title('Hip Z')
                axes[2].legend()
                axes[2].grid(True)
                
                cycle_idx = len(hip_dist_cycle)
                plt.suptitle(f'MidHip Trajectory (Cycle {cycle_idx}): PA vs PT')
                plt.tight_layout()
                
                # 保存先（out_root はこの関数のスコープ外なので thera_dir から構築）
                debug_dir = thera_dir / "ViTPose_results"
                plt.savefig(debug_dir / f"hip_xyz_cycle_{cycle_idx:02d}.png", dpi=150)
                plt.close()
                
        return dict(
            hip_dist=np.nanmean(hip_dist_cycle),
            hip_dist_n=np.nanmean(hip_dist_normalized_cycle),
            wri_para_s=np.nanmedian(wri_para_s_cycle),  #手の検出はぶれが多いので中央値を使用
            wri_nonpara_s=np.nanmedian(wri_nonpara_s_cycle),  #手の検出はぶれが多いので中央値を使用
            cos_sim=np.nanmean(cos_sim_cycle),
            hip_cc_x=np.nanmean(hip_cc_x_cycle),
            hip_cc_y=np.nanmean(hip_cc_y_cycle),
            hip_cc_z=np.nanmean(hip_cc_z_cycle),
            hip_cc_3d=np.nanmean(hip_cc_3d_cycle),
            hip_cc_lag=np.nanmean(hip_cc_lag_cycle),
        )
        
    if paralyzed_side == "R":
        gait_cycles = gait_cycles_r
        pt_wrist_para = kp3d_pt[:, 4, :]  # 右手首
        pt_wrist_nonpara = kp3d_pt[:, 7, :]  # 左手首
    elif paralyzed_side == "L":
        gait_cycles = gait_cycles_l
        pt_wrist_para = kp3d_pt[:, 7, :]  # 左手首
        pt_wrist_nonpara = kp3d_pt[:, 4, :]  # 右手首
    right_forearm = np.nanmean(np.linalg.norm(kp3d_pt[:, 4, :] - kp3d_pt[:, 3, :], axis=1))  # 右前腕長
    left_forearm = np.nanmean(np.linalg.norm(kp3d_pt[:, 7, :] - kp3d_pt[:, 6, :], axis=1))  # 左前腕長
    pt_forearm_length = (right_forearm + left_forearm) / 2  # 両側の平均を使用[mm]
    
    assist_metrics = calculate_pt_assist_metrics_by_cycle(
    gait_cycles=gait_cycles,
    pa_midhip=kp3d[:, 8, :],
    pt_midhip=kp3d_pt[:, 8, :],
    pa_neck=kp3d[:, 1, :],
    pt_neck=kp3d_pt[:, 1, :],
    pt_wrist_para=pt_wrist_para,
    pt_wrist_nonpara=pt_wrist_nonpara,
    pt_forearm_length=pt_forearm_length,
    sampling_freq=FS_3D,)

    pd.DataFrame([assist_metrics]).to_csv(
        out_root / f"pt_assist_metrics.csv",
        index=False
    )
    
    print(f"[OK] {thera_dir.name} / {npz_3d_path.name}")
    print(f"     -> {out_root}")

def main():
    for sub_dir in sorted(ROOT_DIR.glob("sub*")):
        if not sub_dir.exists():
            print(f"[SKIP] missing: {sub_dir}")
            continue

        for thera_dir in sorted(sub_dir.glob("thera*")):
            if not thera_dir.exists():
                print(f"[SKIP] missing: {thera_dir}")
                continue
            
            # 3_3の3D npz（複数methodがある前提で全部回す）
            npz_pa = list(thera_dir.glob("3d_kp_*_PA.npz"))
            npz_list = sorted(npz_pa, key=lambda p: natural_sort_key(p.name))
            if len(npz_list) == 0:
                print(f"[SKIP] 3D npz not found (run 3_3 first): {thera_dir}")
                continue

            print("=" * 90)
            print(f"[RUN] {thera_dir}  (methods={len(npz_list)})")
            print("=" * 90)

            for npz_3d in npz_list:
                try:
                    process_one_method(thera_dir, npz_3d)
                except Exception as e:
                    print(f"[ERROR] {thera_dir.name} / {npz_3d.name}: {e}")


if __name__ == "__main__":
    main()