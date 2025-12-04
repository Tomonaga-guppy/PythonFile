"""
ISB推奨のJoint Coordinate System (JCS) を使用した関節角度計算プログラム

元のプログラム（4_ref_opti_op.py）をISB推奨方法に修正したバージョンです。

主な変更点:
1. Grood & Suntay (1983) のJCSに基づく角度計算
2. ISB推奨の座標系定義（Wu et al., 2002）
3. 浮動軸を使用した3自由度の関節角度計算

参考文献:
- Wu, G., et al. (2002). ISB recommendation on definitions of joint coordinate system 
  of various joints for the reporting of human joint motion—part I: ankle, hip, and spine.
  Journal of Biomechanics, 35(4), 543-548.
"""

import pandas as pd
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import json

# モーションキャプチャデータ読み込みモジュール（m_opti.pyと同等の機能）
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(data, order, cutoff_freq, frame_list, sampling_freq=100):
    """バターワースローパスフィルタ"""
    nyquist_freq = sampling_freq / 2
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data[frame_list])
    data_filter = np.copy(data)
    data_filter[frame_list] = y
    return data_filter


# =====================================================
# ISB JCS関連の関数（isb_joint_angles.pyからインポート）
# =====================================================

def normalize(v):
    """ベクトルを正規化する"""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros_like(v)
    return v / norm


def calculate_pelvis_coordinate_system_isb(rasi, lasi, rpsi, lpsi):
    """
    ISB推奨の骨盤座標系を計算する
    
    ISB定義 (Wu et al., 2002):
    - 原点: 股関節中心（ここではASIS中点を使用）
    - Z軸: 左右ASISを結ぶ線に平行、右を向く
    - X軸: 2つのASISと2つのPSISの中点で定義される平面内、Z軸に直交、前方を向く
    - Y軸: XとZに垂直、頭側を向く
    """
    origin = (rasi + lasi) / 2
    mid_psis = (rpsi + lpsi) / 2
    
    # Z軸: 右向き
    z_axis = normalize(rasi - lasi)
    
    # 平面の定義
    v1 = rasi - mid_psis
    
    # Y軸: 頭側を向く
    y_axis = normalize(np.cross(z_axis, v1))
    
    # X軸: 前方を向く
    x_axis = normalize(np.cross(y_axis, z_axis))
    
    # Y軸を再計算
    y_axis = normalize(np.cross(z_axis, x_axis))
    
    axes = np.column_stack([x_axis, y_axis, z_axis])
    
    return origin, axes


def calculate_femur_coordinate_system_isb(hip_center, knee_medial, knee_lateral, side='right'):
    """
    ISB推奨の大腿骨座標系を計算する
    
    ISB定義 (Wu et al., 2002):
    - 原点: 股関節中心
    - y軸: 内外側大腿顆の中点と原点を結ぶ線、頭側を向く
    - z軸: y軸に垂直、原点と2つの大腿顆で定義される平面内、右/左を向く
    - x軸: yとzに垂直、前方を向く
    """
    origin = hip_center
    knee_center = (knee_medial + knee_lateral) / 2
    
    # y軸: 頭側を向く
    y_axis = normalize(hip_center - knee_center)
    
    # z軸の計算
    if side == 'right':
        knee_axis = knee_lateral - knee_medial
    else:
        knee_axis = knee_medial - knee_lateral
    
    z_axis = normalize(knee_axis - np.dot(knee_axis, y_axis) * y_axis)
    
    # x軸: 前方を向く
    x_axis = normalize(np.cross(y_axis, z_axis))
    
    # z軸を再計算
    z_axis = normalize(np.cross(x_axis, y_axis))
    
    axes = np.column_stack([x_axis, y_axis, z_axis])
    
    return origin, axes


def calculate_tibia_coordinate_system_isb(knee_center, ankle_medial, ankle_lateral, 
                                          knee_medial, knee_lateral, side='right'):
    """
    ISB推奨の脛骨座標系を計算する
    
    ISB定義 (Wu et al., 2002):
    - 原点: 内外踝の中点 (IM)
    - Z軸: 内踝と外踝を結ぶ線、右を向く
    - Y軸: 脛骨の長軸（頭側を向く）
    - X軸: YとZに垂直、前方を向く
    """
    origin = (ankle_medial + ankle_lateral) / 2
    
    if side == 'right':
        z_axis = normalize(ankle_lateral - ankle_medial)
    else:
        z_axis = normalize(ankle_medial - ankle_lateral)
    
    knee_mid = (knee_medial + knee_lateral) / 2
    y_axis_temp = normalize(knee_mid - origin)
    
    x_axis = normalize(np.cross(y_axis_temp, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    
    axes = np.column_stack([x_axis, y_axis, z_axis])
    
    return origin, axes


def calculate_foot_coordinate_system_isb(ankle_medial, ankle_lateral, toe, heel, side='right'):
    """
    ISB推奨の踵骨（足部）座標系を計算する
    """
    origin = (ankle_medial + ankle_lateral) / 2
    
    x_axis = normalize(toe - heel)
    
    if side == 'right':
        z_temp = ankle_lateral - ankle_medial
    else:
        z_temp = ankle_medial - ankle_lateral
    
    y_axis = normalize(np.cross(z_temp, x_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))
    
    axes = np.column_stack([x_axis, y_axis, z_axis])
    
    return origin, axes


def calculate_jcs_angles_hip(pelvis_axes, femur_axes, side='right'):
    """
    ISB推奨のJCSに基づく股関節角度を計算する
    
    JCS軸定義:
    - e1: 骨盤のZ軸（固定軸）→ 屈曲/伸展 (α)
    - e3: 大腿のy軸（固定軸）→ 内旋/外旋 (γ)
    - e2: 浮動軸（e1 × e3に垂直）→ 内転/外転 (β)
    
    Returns:
    --------
    flexion : float
        屈曲(+)/伸展(-) [degrees]
    adduction : float
        内転(+)/外転(-) [degrees]
    internal_rotation : float
        内旋(+)/外旋(-) [degrees]
    """
    # JCS軸の定義
    e1 = pelvis_axes[:, 2]   # 骨盤のZ軸
    e3 = femur_axes[:, 1]    # 大腿のy軸
    e2 = normalize(np.cross(e3, e1))
    
    # α (屈曲/伸展): e1軸周りの回転
    pelvis_x = pelvis_axes[:, 0]
    cos_alpha = np.clip(np.dot(pelvis_x, e2), -1.0, 1.0)
    sin_alpha = np.dot(np.cross(pelvis_x, e2), e1)
    flexion = np.degrees(np.arctan2(sin_alpha, cos_alpha))
    
    # β (内転/外転): e2軸周りの回転
    beta = np.degrees(np.arcsin(np.clip(-np.dot(e1, e3), -1.0, 1.0)))
    
    if side == 'right':
        adduction = beta
    else:
        adduction = -beta
    
    # γ (内旋/外旋): e3軸周りの回転
    femur_x = femur_axes[:, 0]
    cos_gamma = np.clip(np.dot(femur_x, e2), -1.0, 1.0)
    sin_gamma = np.dot(np.cross(e2, femur_x), e3)
    internal_rotation = np.degrees(np.arctan2(sin_gamma, cos_gamma))
    
    return flexion, adduction, internal_rotation


def calculate_jcs_angles_knee(femur_axes, tibia_axes, side='right'):
    """
    ISB推奨のJCSに基づく膝関節角度を計算する
    """
    e1 = femur_axes[:, 2]
    e3 = tibia_axes[:, 1]
    e2 = normalize(np.cross(e3, e1))
    
    femur_x = femur_axes[:, 0]
    cos_alpha = np.clip(np.dot(femur_x, e2), -1.0, 1.0)
    sin_alpha = np.dot(np.cross(femur_x, e2), e1)
    flexion = np.degrees(np.arctan2(sin_alpha, cos_alpha))
    
    beta = np.degrees(np.arcsin(np.clip(-np.dot(e1, e3), -1.0, 1.0)))
    
    if side == 'right':
        adduction = beta
    else:
        adduction = -beta
    
    tibia_x = tibia_axes[:, 0]
    cos_gamma = np.clip(np.dot(tibia_x, e2), -1.0, 1.0)
    sin_gamma = np.dot(np.cross(e2, tibia_x), e3)
    internal_rotation = np.degrees(np.arctan2(sin_gamma, cos_gamma))
    
    return flexion, adduction, internal_rotation


def calculate_jcs_angles_ankle(tibia_axes, foot_axes, side='right'):
    """
    ISB推奨のJCSに基づく足関節角度を計算する
    """
    e1 = tibia_axes[:, 2]
    e3 = foot_axes[:, 1]
    e2 = normalize(np.cross(e3, e1))
    
    tibia_x = tibia_axes[:, 0]
    cos_alpha = np.clip(np.dot(tibia_x, e2), -1.0, 1.0)
    sin_alpha = np.dot(np.cross(tibia_x, e2), e1)
    dorsiflexion = np.degrees(np.arctan2(sin_alpha, cos_alpha))
    
    beta = np.degrees(np.arcsin(np.clip(-np.dot(e1, e3), -1.0, 1.0)))
    
    if side == 'right':
        inversion = beta
    else:
        inversion = -beta
    
    foot_x = foot_axes[:, 0]
    cos_gamma = np.clip(np.dot(foot_x, e2), -1.0, 1.0)
    sin_gamma = np.dot(np.cross(e2, foot_x), e3)
    internal_rotation = np.degrees(np.arctan2(sin_gamma, cos_gamma))
    
    return dorsiflexion, inversion, internal_rotation


def calculate_hip_joint_center_davis(rasi, lasi, rpsi, lpsi, leg_length, 
                                      marker_radius=0.0127, height=1.76):
    """
    Davisモデルに基づく股関節中心位置の推定
    """
    d_asi = np.linalg.norm(rasi - lasi)
    
    beta = 0.1 * np.pi
    theta = 0.496
    c = 0.115 * leg_length - 0.0153
    x_dis = 0.1288 * leg_length - 0.04856
    r = marker_radius
    
    x_offset = -(x_dis + r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
    z_offset = -(x_dis + r) * np.sin(beta) - c * np.cos(theta) * np.cos(beta)
    
    y_offset_r = +(c * np.sin(theta) - d_asi / 2)
    y_offset_l = -(c * np.sin(theta) - d_asi / 2)
    
    hip_0 = (rasi + lasi) / 2
    sacrum = (rpsi + lpsi) / 2
    
    e_x0 = normalize(hip_0 - sacrum)
    e_y = normalize(lasi - rasi)
    e_z = normalize(np.cross(e_x0, e_y))
    e_x = np.cross(e_y, e_z)
    
    rot_matrix = np.column_stack([e_x, e_y, e_z])
    
    rhip_offset_local = np.array([x_offset, y_offset_r, z_offset])
    lhip_offset_local = np.array([x_offset, y_offset_l, z_offset])
    
    rhip_center = hip_0 + rot_matrix @ rhip_offset_local
    lhip_center = hip_0 + rot_matrix @ lhip_offset_local
    
    return rhip_center, lhip_center


def calculate_isb_joint_angles_for_frame(rasi, lasi, rpsi, lpsi,
                                          rknee, rknee2, lknee, lknee2,
                                          rank, rank2, lank, lank2,
                                          rtoe, rhee, ltoe, lhee,
                                          marker_radius=0.0127, height=1.76):
    """
    1フレームのISB JCS関節角度を計算する
    
    Parameters:
    -----------
    各マーカーの3D座標 (ndarray shape: (3,))
    
    Returns:
    --------
    dict : 各関節角度を含む辞書
    """
    # 下肢長の計算
    leg_length_r = np.linalg.norm(rank - rasi)
    leg_length_l = np.linalg.norm(lank - lasi)
    leg_length = (leg_length_r + leg_length_l) / 2
    
    # 股関節中心の推定
    rhip_center, lhip_center = calculate_hip_joint_center_davis(
        rasi, lasi, rpsi, lpsi, leg_length, marker_radius, height
    )
    
    # 骨盤座標系
    _, pelvis_axes = calculate_pelvis_coordinate_system_isb(rasi, lasi, rpsi, lpsi)
    
    # 大腿骨座標系
    # マーカー配置: RKNE/LKNEが外側、RKNE2/LKNE2が内側
    _, rfemur_axes = calculate_femur_coordinate_system_isb(
        rhip_center, rknee2, rknee, side='right'  # rknee2=内側, rknee=外側
    )
    _, lfemur_axes = calculate_femur_coordinate_system_isb(
        lhip_center, lknee2, lknee, side='left'   # lknee2=内側, lknee=外側
    )
    
    # 膝関節中心
    rknee_center = (rknee + rknee2) / 2
    lknee_center = (lknee + lknee2) / 2
    
    # 脛骨座標系
    # マーカー配置: RANK/LANKが外側、RANK2/LANK2が内側
    _, rtibia_axes = calculate_tibia_coordinate_system_isb(
        rknee_center, rank2, rank, rknee2, rknee, side='right'  # rank2=内側, rank=外側
    )
    _, ltibia_axes = calculate_tibia_coordinate_system_isb(
        lknee_center, lank2, lank, lknee2, lknee, side='left'   # lank2=内側, lank=外側
    )
    
    # 足部座標系
    # マーカー配置: RANK/LANKが外側、RANK2/LANK2が内側
    _, rfoot_axes = calculate_foot_coordinate_system_isb(
        rank2, rank, rtoe, rhee, side='right'  # rank2=内側, rank=外側
    )
    _, lfoot_axes = calculate_foot_coordinate_system_isb(
        lank2, lank, ltoe, lhee, side='left'   # lank2=内側, lank=外側
    )
    
    # 股関節角度
    r_hip_flex, r_hip_add, r_hip_rot = calculate_jcs_angles_hip(
        pelvis_axes, rfemur_axes, side='right'
    )
    l_hip_flex, l_hip_add, l_hip_rot = calculate_jcs_angles_hip(
        pelvis_axes, lfemur_axes, side='left'
    )
    
    # 膝関節角度
    r_knee_flex, r_knee_add, r_knee_rot = calculate_jcs_angles_knee(
        rfemur_axes, rtibia_axes, side='right'
    )
    l_knee_flex, l_knee_add, l_knee_rot = calculate_jcs_angles_knee(
        lfemur_axes, ltibia_axes, side='left'
    )
    
    # 足関節角度
    r_ankle_df, r_ankle_inv, r_ankle_rot = calculate_jcs_angles_ankle(
        rtibia_axes, rfoot_axes, side='right'
    )
    l_ankle_df, l_ankle_inv, l_ankle_rot = calculate_jcs_angles_ankle(
        ltibia_axes, lfoot_axes, side='left'
    )
    
    return {
        # 右股関節
        'R_Hip_FlEx': r_hip_flex,
        'R_Hip_AdAb': r_hip_add,
        'R_Hip_InEx': r_hip_rot,
        # 左股関節
        'L_Hip_FlEx': l_hip_flex,
        'L_Hip_AdAb': l_hip_add,
        'L_Hip_InEx': l_hip_rot,
        # 右膝関節
        'R_Knee_FlEx': r_knee_flex,
        'R_Knee_VrVl': r_knee_add,
        'R_Knee_InEx': r_knee_rot,
        # 左膝関節
        'L_Knee_FlEx': l_knee_flex,
        'L_Knee_VrVl': l_knee_add,
        'L_Knee_InEx': l_knee_rot,
        # 右足関節
        'R_Ankle_DfPf': r_ankle_df,
        'R_Ankle_InEv': r_ankle_inv,
        'R_Ankle_InEx': r_ankle_rot,
        # 左足関節
        'L_Ankle_DfPf': l_ankle_df,
        'L_Ankle_InEv': l_ankle_inv,
        'L_Ankle_InEx': l_ankle_rot,
        # 追加情報（デバッグ用）
        '_rhip_center': rhip_center,
        '_lhip_center': lhip_center,
        '_pelvis_axes': pelvis_axes,
        '_rfemur_axes': rfemur_axes,
        '_lfemur_axes': lfemur_axes,
    }


# =====================================================
# メイン処理（使用例）
# =====================================================

def process_mocap_data_isb(keypoints_mocap, full_range, sampling_freq=100, 
                           filter_order=4, cutoff_freq=6, 
                           marker_radius=0.0127, height=1.76):
    """
    モーションキャプチャデータからISB JCS関節角度を計算する
    
    Parameters:
    -----------
    keypoints_mocap : ndarray
        マーカー座標データ (frames, markers, 3)
    full_range : range
        処理するフレーム範囲
    sampling_freq : int
        サンプリング周波数 [Hz]
    filter_order : int
        フィルタ次数
    cutoff_freq : float
        カットオフ周波数 [Hz]
    marker_radius : float
        マーカー半径 [m]
    height : float
        被験者身長 [m]
    
    Returns:
    --------
    angles_df : DataFrame
        各フレームの関節角度
    """
    # マーカーインデックスの定義（元のコードに合わせる）
    # final_marker_set = ["RASI", "LASI", "RPSI", "LPSI", "RKNE", "LKNE", "RANK", "LANK", 
    #                     "RTOE", "LTOE", "RHEE", "LHEE", "RKNE2", "LKNE2", "RANK2", "LANK2"]
    # インデックス: 
    # 0:LANK, 1:LANK2, 2:LASI, 3:LHEE, 4:LKNE, 5:LKNE2, 6:LPSI, 7:LTOE
    # 8:RANK, 9:RANK2, 10:RASI, 11:RHEE, 12:RKNE, 13:RKNE2, 14:RPSI, 15:RTOE
    
    # マーカーデータをフィルタリング
    rasi = np.array([butter_lowpass_filter(keypoints_mocap[:, 10, x], filter_order, cutoff_freq, full_range, sampling_freq) for x in range(3)]).T
    lasi = np.array([butter_lowpass_filter(keypoints_mocap[:, 2, x], filter_order, cutoff_freq, full_range, sampling_freq) for x in range(3)]).T
    rpsi = np.array([butter_lowpass_filter(keypoints_mocap[:, 14, x], filter_order, cutoff_freq, full_range, sampling_freq) for x in range(3)]).T
    lpsi = np.array([butter_lowpass_filter(keypoints_mocap[:, 6, x], filter_order, cutoff_freq, full_range, sampling_freq) for x in range(3)]).T
    rank = np.array([butter_lowpass_filter(keypoints_mocap[:, 8, x], filter_order, cutoff_freq, full_range, sampling_freq) for x in range(3)]).T
    lank = np.array([butter_lowpass_filter(keypoints_mocap[:, 0, x], filter_order, cutoff_freq, full_range, sampling_freq) for x in range(3)]).T
    rank2 = np.array([butter_lowpass_filter(keypoints_mocap[:, 9, x], filter_order, cutoff_freq, full_range, sampling_freq) for x in range(3)]).T
    lank2 = np.array([butter_lowpass_filter(keypoints_mocap[:, 1, x], filter_order, cutoff_freq, full_range, sampling_freq) for x in range(3)]).T
    rknee = np.array([butter_lowpass_filter(keypoints_mocap[:, 12, x], filter_order, cutoff_freq, full_range, sampling_freq) for x in range(3)]).T
    lknee = np.array([butter_lowpass_filter(keypoints_mocap[:, 4, x], filter_order, cutoff_freq, full_range, sampling_freq) for x in range(3)]).T
    rknee2 = np.array([butter_lowpass_filter(keypoints_mocap[:, 13, x], filter_order, cutoff_freq, full_range, sampling_freq) for x in range(3)]).T
    lknee2 = np.array([butter_lowpass_filter(keypoints_mocap[:, 5, x], filter_order, cutoff_freq, full_range, sampling_freq) for x in range(3)]).T
    rtoe = np.array([butter_lowpass_filter(keypoints_mocap[:, 15, x], filter_order, cutoff_freq, full_range, sampling_freq) for x in range(3)]).T
    ltoe = np.array([butter_lowpass_filter(keypoints_mocap[:, 7, x], filter_order, cutoff_freq, full_range, sampling_freq) for x in range(3)]).T
    rhee = np.array([butter_lowpass_filter(keypoints_mocap[:, 11, x], filter_order, cutoff_freq, full_range, sampling_freq) for x in range(3)]).T
    lhee = np.array([butter_lowpass_filter(keypoints_mocap[:, 3, x], filter_order, cutoff_freq, full_range, sampling_freq) for x in range(3)]).T
    
    # 各フレームで関節角度を計算
    angle_records = []
    
    for frame_num in full_range:
        angles = calculate_isb_joint_angles_for_frame(
            rasi[frame_num], lasi[frame_num], rpsi[frame_num], lpsi[frame_num],
            rknee[frame_num], rknee2[frame_num], lknee[frame_num], lknee2[frame_num],
            rank[frame_num], rank2[frame_num], lank[frame_num], lank2[frame_num],
            rtoe[frame_num], rhee[frame_num], ltoe[frame_num], lhee[frame_num],
            marker_radius, height
        )
        
        # 角度データのみ抽出（プライベート属性を除く）
        angle_data = {k: v for k, v in angles.items() if not k.startswith('_')}
        angle_data['frame'] = frame_num
        angle_records.append(angle_data)
    
    # DataFrameに変換
    angles_df = pd.DataFrame(angle_records)
    
    return angles_df


def compare_methods_visualization(angles_isb, angles_original, save_path=None):
    """
    ISB方式と元の方式の角度を比較するプロットを作成
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # 股関節屈曲/伸展
    ax = axes[0, 0]
    ax.plot(angles_isb['R_Hip_FlEx'], 'b-', label='ISB (Right)', linewidth=2)
    ax.plot(angles_original['R_Hip_FlEx'], 'b--', label='Original (Right)', linewidth=1)
    ax.plot(angles_isb['L_Hip_FlEx'], 'r-', label='ISB (Left)', linewidth=2)
    ax.plot(angles_original['L_Hip_FlEx'], 'r--', label='Original (Left)', linewidth=1)
    ax.set_title('Hip Flexion/Extension')
    ax.set_ylabel('Angle [deg]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 股関節内転/外転
    ax = axes[0, 1]
    ax.plot(angles_isb['R_Hip_AdAb'], 'b-', label='ISB (Right)', linewidth=2)
    ax.plot(angles_original['R_Hip_AdAb'], 'b--', label='Original (Right)', linewidth=1)
    ax.plot(angles_isb['L_Hip_AdAb'], 'r-', label='ISB (Left)', linewidth=2)
    ax.plot(angles_original['L_Hip_AdAb'], 'r--', label='Original (Left)', linewidth=1)
    ax.set_title('Hip Adduction/Abduction')
    ax.set_ylabel('Angle [deg]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 股関節内旋/外旋
    ax = axes[1, 0]
    ax.plot(angles_isb['R_Hip_InEx'], 'b-', label='ISB (Right)', linewidth=2)
    ax.plot(angles_original['R_Hip_InEx'], 'b--', label='Original (Right)', linewidth=1)
    ax.plot(angles_isb['L_Hip_InEx'], 'r-', label='ISB (Left)', linewidth=2)
    ax.plot(angles_original['L_Hip_InEx'], 'r--', label='Original (Left)', linewidth=1)
    ax.set_title('Hip Internal/External Rotation')
    ax.set_ylabel('Angle [deg]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 膝関節屈曲/伸展
    ax = axes[1, 1]
    ax.plot(angles_isb['R_Knee_FlEx'], 'b-', label='ISB (Right)', linewidth=2)
    ax.plot(angles_original['R_Knee_FlEx'], 'b--', label='Original (Right)', linewidth=1)
    ax.plot(angles_isb['L_Knee_FlEx'], 'r-', label='ISB (Left)', linewidth=2)
    ax.plot(angles_original['L_Knee_FlEx'], 'r--', label='Original (Left)', linewidth=1)
    ax.set_title('Knee Flexion/Extension')
    ax.set_ylabel('Angle [deg]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 足関節背屈/底屈
    ax = axes[2, 0]
    ax.plot(angles_isb['R_Ankle_DfPf'], 'b-', label='ISB (Right)', linewidth=2)
    ax.plot(angles_original['R_Ankle_DfPf'], 'b--', label='Original (Right)', linewidth=1)
    ax.plot(angles_isb['L_Ankle_DfPf'], 'r-', label='ISB (Left)', linewidth=2)
    ax.plot(angles_original['L_Ankle_DfPf'], 'r--', label='Original (Left)', linewidth=1)
    ax.set_title('Ankle Dorsiflexion/Plantarflexion')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle [deg]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 足関節内反/外反
    ax = axes[2, 1]
    ax.plot(angles_isb['R_Ankle_InEv'], 'b-', label='ISB (Right)', linewidth=2)
    ax.plot(angles_original['R_Ankle_InEv'], 'b--', label='Original (Right)', linewidth=1)
    ax.plot(angles_isb['L_Ankle_InEv'], 'r-', label='ISB (Left)', linewidth=2)
    ax.plot(angles_original['L_Ankle_InEv'], 'r--', label='Original (Left)', linewidth=1)
    ax.set_title('Ankle Inversion/Eversion')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle [deg]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"プロットを保存しました: {save_path}")
    
    plt.show()


def main_example():
    """
    使用例（実際のデータパスに合わせて修正してください）
    """
    print("=" * 60)
    print("ISB推奨のJoint Coordinate System (JCS) に基づく")
    print("関節角度計算プログラム")
    print("=" * 60)
    print()
    print("参考文献:")
    print("- Wu, G., et al. (2002). ISB recommendation on definitions")
    print("  of joint coordinate system. Journal of Biomechanics.")
    print("- Grood, E.S., & Suntay, W.J. (1983). A joint coordinate")
    print("  system for clinical description of 3D motions.")
    print()
    print("主な特徴:")
    print("1. Grood & Suntayの浮動軸方式を使用")
    print("2. ISB推奨の座標系定義に準拠")
    print("3. 臨床的に解釈しやすい角度表現")
    print()
    print("使用方法:")
    print("  from joint_angles_isb import process_mocap_data_isb")
    print("  angles_df = process_mocap_data_isb(keypoints_mocap, full_range)")
    print()
    
    # テスト用のダミーデータで動作確認
    print("テスト用ダミーデータで動作確認...")
    
    # 中立位のダミーデータを作成（フィルタ要件を満たすため十分なフレーム数）
    n_frames = 50  # filtfiltのpadlen(15)より十分大きくする
    keypoints_mocap = np.zeros((n_frames, 16, 3))
    
    # 静止立位姿勢の座標を設定
    for i in range(n_frames):
        # LANK, LANK2, LASI, LHEE, LKNE, LKNE2, LPSI, LTOE
        # RANK, RANK2, RASI, RHEE, RKNE, RKNE2, RPSI, RTOE
        keypoints_mocap[i, 0] = [-0.15, 0.0, 0.1]   # LANK
        keypoints_mocap[i, 1] = [-0.05, 0.0, 0.1]   # LANK2
        keypoints_mocap[i, 2] = [-0.1, 0.0, 1.0]    # LASI
        keypoints_mocap[i, 3] = [-0.15, -0.1, 0.0]  # LHEE
        keypoints_mocap[i, 4] = [-0.15, 0.0, 0.5]   # LKNE
        keypoints_mocap[i, 5] = [-0.05, 0.0, 0.5]   # LKNE2
        keypoints_mocap[i, 6] = [-0.05, -0.15, 1.0] # LPSI
        keypoints_mocap[i, 7] = [-0.15, 0.2, 0.0]   # LTOE
        keypoints_mocap[i, 8] = [0.15, 0.0, 0.1]    # RANK
        keypoints_mocap[i, 9] = [0.05, 0.0, 0.1]    # RANK2
        keypoints_mocap[i, 10] = [0.1, 0.0, 1.0]    # RASI
        keypoints_mocap[i, 11] = [0.15, -0.1, 0.0]  # RHEE
        keypoints_mocap[i, 12] = [0.15, 0.0, 0.5]   # RKNE
        keypoints_mocap[i, 13] = [0.05, 0.0, 0.5]   # RKNE2
        keypoints_mocap[i, 14] = [0.05, -0.15, 1.0] # RPSI
        keypoints_mocap[i, 15] = [0.15, 0.2, 0.0]   # RTOE
    
    full_range = range(n_frames)
    
    # 関節角度を計算
    angles_df = process_mocap_data_isb(keypoints_mocap, full_range)
    
    print("\n計算結果（最初のフレーム）:")
    print("-" * 40)
    first_frame = angles_df.iloc[0]
    for col in angles_df.columns:
        if col != 'frame':
            print(f"{col}: {first_frame[col]:.2f}°")
    
    print("\n処理完了！")


if __name__ == "__main__":
    main_example()