"""
ISB推奨のJoint Coordinate System (JCS) に基づく関節角度計算モジュール

参考文献:
- Wu, G., et al. (2002). ISB recommendation on definitions of joint coordinate system 
  of various joints for the reporting of human joint motion—part I: ankle, hip, and spine.
  Journal of Biomechanics, 35(4), 543-548.
- Grood, E.S., & Suntay, W.J. (1983). A joint coordinate system for the clinical 
  description of three-dimensional motions: application to the knee. 
  Journal of Biomechanical Engineering, 105(2), 136-144.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize(v):
    """ベクトルを正規化する"""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros_like(v)
    return v / norm


def calculate_pelvis_coordinate_system_isb(rasi, lasi, rpsi, lpsi):
    """
    ISB推奨の骨盤座標系を計算する
    
    Parameters:
    -----------
    rasi, lasi : ndarray
        右・左上前腸骨棘 (ASIS) の3D座標
    rpsi, lpsi : ndarray
        右・左上後腸骨棘 (PSIS) の3D座標
    
    Returns:
    --------
    origin : ndarray
        座標系原点（後で股関節中心に移動）
    axes : ndarray (3x3)
        座標系軸 [X, Y, Z] (各列が軸ベクトル)
        X: 前方, Y: 頭側, Z: 右側
    
    ISB定義 (Wu et al., 2002):
    - 原点: 股関節中心（ここではASIS中点を仮の原点として使用）
    - Z軸: 左右ASISを結ぶ線に平行、右を向く
    - X軸: 2つのASISと2つのPSISの中点で定義される平面内、Z軸に直交、前方を向く
    - Y軸: XとZに垂直、頭側を向く
    """
    # ASIS中点（仮の原点）
    origin = (rasi + lasi) / 2
    
    # PSIS中点
    mid_psis = (rpsi + lpsi) / 2
    
    # Z軸: 右ASIS - 左ASIS方向（右向き）
    z_axis = normalize(rasi - lasi)
    
    # 骨盤平面の定義: 2つのASISと2つのPSISの中点を含む平面
    # 平面の法線ベクトルを計算
    v1 = rasi - mid_psis
    v2 = lasi - mid_psis
    
    # Y軸: 平面の法線（頭側を向く）
    y_axis = normalize(np.cross(z_axis, v1))
    
    # X軸: Y × Z（前方を向く）
    x_axis = normalize(np.cross(y_axis, z_axis))
    
    # Y軸を再計算して直交性を確保
    y_axis = normalize(np.cross(z_axis, x_axis))
    
    axes = np.column_stack([x_axis, y_axis, z_axis])
    
    return origin, axes


def calculate_femur_coordinate_system_isb(hip_center, knee_medial, knee_lateral, side='right'):
    """
    ISB推奨の大腿骨座標系を計算する
    
    Parameters:
    -----------
    hip_center : ndarray
        股関節中心の3D座標
    knee_medial : ndarray
        内側大腿顆の3D座標（またはKNEEマーカー）
    knee_lateral : ndarray
        外側大腿顆の3D座標（またはKNEE2マーカー）
    side : str
        'right' または 'left'
    
    Returns:
    --------
    origin : ndarray
        座標系原点（股関節中心）
    axes : ndarray (3x3)
        座標系軸 [x, y, z] (各列が軸ベクトル)
        x: 前方, y: 近位（頭側）, z: 右側（左脚の場合は左側）
    
    ISB定義 (Wu et al., 2002):
    - 原点: 股関節中心
    - y軸: 内外側大腿顆の中点と原点を結ぶ線、頭側を向く
    - z軸: y軸に垂直、原点と2つの大腿顆で定義される平面内、右を向く（右脚の場合）
    - x軸: yとzに垂直、前方を向く
    """
    origin = hip_center
    
    # 大腿顆の中点
    knee_center = (knee_medial + knee_lateral) / 2
    
    # y軸: 膝中心から股関節中心への方向（頭側を向く）
    y_axis = normalize(hip_center - knee_center)
    
    # z軸の計算: 大腿顆間の方向
    if side == 'right':
        # 右脚: 内側から外側へ（右向き）
        knee_axis = knee_lateral - knee_medial
    else:
        # 左脚: 外側から内側へ（左向き、つまり体の中心から外側）
        knee_axis = knee_medial - knee_lateral
    
    # z軸: y軸に垂直で、膝軸と同じ平面内
    z_axis = normalize(knee_axis - np.dot(knee_axis, y_axis) * y_axis)
    
    # x軸: y × z（前方を向く）
    x_axis = normalize(np.cross(y_axis, z_axis))
    
    # z軸を再計算して直交性を確保
    z_axis = normalize(np.cross(x_axis, y_axis))
    
    axes = np.column_stack([x_axis, y_axis, z_axis])
    
    return origin, axes


def calculate_tibia_coordinate_system_isb(knee_center, ankle_medial, ankle_lateral, 
                                          knee_medial, knee_lateral, side='right'):
    """
    ISB推奨の脛骨座標系を計算する
    
    Parameters:
    -----------
    knee_center : ndarray
        膝関節中心の3D座標
    ankle_medial : ndarray
        内踝の3D座標（ANKマーカー）
    ankle_lateral : ndarray
        外踝の3D座標（ANK2マーカー）
    knee_medial, knee_lateral : ndarray
        内側・外側大腿顆の3D座標
    side : str
        'right' または 'left'
    
    Returns:
    --------
    origin : ndarray
        座標系原点（足関節中心 = 内外踝の中点）
    axes : ndarray (3x3)
        座標系軸 [X, Y, Z]
    
    ISB定義 (Wu et al., 2002) - 足関節複合体:
    - 原点: 内外踝の中点 (IM)
    - Z軸: 内踝と外踝を結ぶ線、右を向く
    - Y軸: 脛骨の長軸（頭側を向く）
    - X軸: YとZに垂直、前方を向く
    """
    # 足関節中心（内外踝の中点）
    origin = (ankle_medial + ankle_lateral) / 2
    
    # Z軸: 踝間の方向
    if side == 'right':
        # 右脚: 内踝から外踝へ（右向き）
        z_axis = normalize(ankle_lateral - ankle_medial)
    else:
        # 左脚: 外踝から内踝へ（左向き）
        z_axis = normalize(ankle_medial - ankle_lateral)
    
    # 膝中心
    knee_mid = (knee_medial + knee_lateral) / 2
    
    # Y軸: 脛骨の長軸（足関節から膝関節方向、頭側を向く）
    y_axis_temp = normalize(knee_mid - origin)
    
    # X軸: Y × Z（前方を向く）
    x_axis = normalize(np.cross(y_axis_temp, z_axis))
    
    # Y軸を再計算
    y_axis = normalize(np.cross(z_axis, x_axis))
    
    axes = np.column_stack([x_axis, y_axis, z_axis])
    
    return origin, axes


def calculate_foot_coordinate_system_isb(ankle_medial, ankle_lateral, toe, heel, side='right'):
    """
    ISB推奨の踵骨（足部）座標系を計算する
    
    Parameters:
    -----------
    ankle_medial : ndarray
        内踝の3D座標
    ankle_lateral : ndarray
        外踝の3D座標
    toe : ndarray
        つま先の3D座標
    heel : ndarray
        踵の3D座標
    side : str
        'right' または 'left'
    
    Returns:
    --------
    origin : ndarray
        座標系原点
    axes : ndarray (3x3)
        座標系軸 [x, y, z]
    
    ISB定義では中立位で脛骨座標系と一致するように定義されますが、
    ここでは足部のランドマークから直接計算します:
    - x軸: 踵からつま先への方向（前方）
    - z軸: 踝間の方向（右向き）
    - y軸: xとzに垂直（上方）
    """
    # 足関節中心
    origin = (ankle_medial + ankle_lateral) / 2
    
    # x軸: 踵からつま先への方向（前方）
    x_axis = normalize(toe - heel)
    
    # z軸の初期ベクトル: 踝間の方向
    if side == 'right':
        z_temp = ankle_lateral - ankle_medial
    else:
        z_temp = ankle_medial - ankle_lateral
    
    # y軸: z × x（上方を向く）
    y_axis = normalize(np.cross(z_temp, x_axis))
    
    # z軸を再計算
    z_axis = normalize(np.cross(x_axis, y_axis))
    
    axes = np.column_stack([x_axis, y_axis, z_axis])
    
    return origin, axes


def calculate_jcs_angles_hip(pelvis_axes, femur_axes, side='right'):
    """
    ISB推奨のJCSに基づく股関節角度を計算する
    
    Parameters:
    -----------
    pelvis_axes : ndarray (3x3)
        骨盤座標系の軸 [X, Y, Z]
    femur_axes : ndarray (3x3)
        大腿骨座標系の軸 [x, y, z]
    side : str
        'right' または 'left'
    
    Returns:
    --------
    flexion : float
        屈曲(+)/伸展(-) [degrees]
    adduction : float
        内転(+)/外転(-) [degrees]
    internal_rotation : float
        内旋(+)/外旋(-) [degrees]
    
    ISB JCS定義:
    - e1: 骨盤のZ軸（固定軸）→ 屈曲/伸展
    - e3: 大腿のy軸（固定軸）→ 内旋/外旋
    - e2: 浮動軸（e1 × e3）→ 内転/外転
    """
    # JCS軸の定義
    # e1: 骨盤のZ軸に一致
    e1 = pelvis_axes[:, 2]  # Z軸
    
    # e3: 大腿のy軸に一致
    e3 = femur_axes[:, 1]   # y軸
    
    # e2: 浮動軸 (e3 × e1を正規化)
    e2 = normalize(np.cross(e3, e1))
    
    # 各角度の計算
    # α (屈曲/伸展): e1軸周りの回転
    # 骨盤のX軸とe2の間の角度
    pelvis_x = pelvis_axes[:, 0]
    
    cos_alpha = np.clip(np.dot(pelvis_x, e2), -1.0, 1.0)
    sin_alpha = np.dot(np.cross(pelvis_x, e2), e1)
    flexion = np.degrees(np.arctan2(sin_alpha, cos_alpha))
    
    # β (内転/外転): e2軸周りの回転
    # e1とe3の間の角度から90度を引く
    cos_beta = np.clip(np.dot(e1, e3), -1.0, 1.0)
    # β = π/2 - arccos(e1・e3) = arcsin(e1・e3) ではない
    # Grood & Suntayの定義に従う
    beta = np.degrees(np.arcsin(np.clip(-np.dot(e1, e3), -1.0, 1.0)))
    
    if side == 'right':
        adduction = beta
    else:
        adduction = -beta  # 左脚は符号反転
    
    # γ (内旋/外旋): e3軸周りの回転
    # 大腿のx軸とe2の間の角度
    femur_x = femur_axes[:, 0]
    
    cos_gamma = np.clip(np.dot(femur_x, e2), -1.0, 1.0)
    sin_gamma = np.dot(np.cross(e2, femur_x), e3)
    internal_rotation = np.degrees(np.arctan2(sin_gamma, cos_gamma))
    
    return flexion, adduction, internal_rotation


def calculate_jcs_angles_knee(femur_axes, tibia_axes, side='right'):
    """
    ISB推奨のJCSに基づく膝関節角度を計算する
    
    Parameters:
    -----------
    femur_axes : ndarray (3x3)
        大腿骨座標系の軸
    tibia_axes : ndarray (3x3)
        脛骨座標系の軸
    side : str
        'right' または 'left'
    
    Returns:
    --------
    flexion : float
        屈曲(+)/伸展(-) [degrees]
    adduction : float
        内転(+)/外転(-) [degrees] (通常は内反/外反)
    internal_rotation : float
        内旋(+)/外旋(-) [degrees]
    
    膝関節のJCSは股関節と同様の構造:
    - e1: 大腿のz軸（固定軸）→ 屈曲/伸展
    - e3: 脛骨のY軸（固定軸）→ 内旋/外旋
    - e2: 浮動軸 → 内転/外転（内反/外反）
    """
    # JCS軸の定義
    e1 = femur_axes[:, 2]   # 大腿のz軸
    e3 = tibia_axes[:, 1]   # 脛骨のY軸
    e2 = normalize(np.cross(e3, e1))
    
    # 屈曲/伸展
    femur_x = femur_axes[:, 0]
    cos_alpha = np.clip(np.dot(femur_x, e2), -1.0, 1.0)
    sin_alpha = np.dot(np.cross(femur_x, e2), e1)
    flexion = np.degrees(np.arctan2(sin_alpha, cos_alpha))
    
    # 内転/外転（内反/外反）
    cos_beta = np.clip(np.dot(e1, e3), -1.0, 1.0)
    beta = np.degrees(np.arcsin(np.clip(-np.dot(e1, e3), -1.0, 1.0)))
    
    if side == 'right':
        adduction = beta
    else:
        adduction = -beta
    
    # 内旋/外旋
    tibia_x = tibia_axes[:, 0]
    cos_gamma = np.clip(np.dot(tibia_x, e2), -1.0, 1.0)
    sin_gamma = np.dot(np.cross(e2, tibia_x), e3)
    internal_rotation = np.degrees(np.arctan2(sin_gamma, cos_gamma))
    
    return flexion, adduction, internal_rotation


def calculate_jcs_angles_ankle(tibia_axes, foot_axes, side='right'):
    """
    ISB推奨のJCSに基づく足関節角度を計算する
    
    Parameters:
    -----------
    tibia_axes : ndarray (3x3)
        脛骨座標系の軸 [X, Y, Z]
    foot_axes : ndarray (3x3)
        足部座標系の軸 [x, y, z]
    side : str
        'right' または 'left'
    
    Returns:
    --------
    dorsiflexion : float
        背屈(+)/底屈(-) [degrees]
    inversion : float
        内反(+)/外反(-) [degrees]
    internal_rotation : float
        内旋(+)/外旋(-) [degrees]
    
    ISB足関節複合体JCS定義:
    - e1: 脛骨のZ軸（固定軸）→ 背屈/底屈
    - e3: 踵骨のy軸（固定軸）→ 内旋/外旋
    - e2: 浮動軸 → 内反/外反
    """
    # JCS軸の定義
    e1 = tibia_axes[:, 2]   # 脛骨のZ軸
    e3 = foot_axes[:, 1]    # 足部のy軸
    e2 = normalize(np.cross(e3, e1))
    
    # 背屈/底屈
    tibia_x = tibia_axes[:, 0]
    cos_alpha = np.clip(np.dot(tibia_x, e2), -1.0, 1.0)
    sin_alpha = np.dot(np.cross(tibia_x, e2), e1)
    dorsiflexion = np.degrees(np.arctan2(sin_alpha, cos_alpha))
    
    # 内反/外反
    cos_beta = np.clip(np.dot(e1, e3), -1.0, 1.0)
    beta = np.degrees(np.arcsin(np.clip(-np.dot(e1, e3), -1.0, 1.0)))
    
    if side == 'right':
        inversion = beta
    else:
        inversion = -beta
    
    # 内旋/外旋
    foot_x = foot_axes[:, 0]
    cos_gamma = np.clip(np.dot(foot_x, e2), -1.0, 1.0)
    sin_gamma = np.dot(np.cross(e2, foot_x), e3)
    internal_rotation = np.degrees(np.arctan2(sin_gamma, cos_gamma))
    
    return dorsiflexion, inversion, internal_rotation


def calculate_hip_joint_center_davis(rasi, lasi, rpsi, lpsi, leg_length, marker_radius=0.0127, height=1.76):
    """
    Davisモデルに基づく股関節中心位置の推定
    
    Parameters:
    -----------
    rasi, lasi : ndarray
        右・左ASISの3D座標
    rpsi, lpsi : ndarray
        右・左PSISの3D座標
    leg_length : float
        下肢長（ASIから足関節まで）
    marker_radius : float
        マーカー半径 [m]（デフォルト: 12.7mm）
    height : float
        被験者身長 [m]
    
    Returns:
    --------
    rhip_center, lhip_center : ndarray
        右・左股関節中心の3D座標
    """
    d_asi = np.linalg.norm(rasi - lasi)
    k = height / 1.7
    
    beta = 0.1 * np.pi  # [rad]
    theta = 0.496  # [rad]
    c = 0.115 * leg_length - 0.0153
    x_dis = 0.1288 * leg_length - 0.04856
    r = marker_radius
    
    # 骨盤座標系での股関節中心オフセット
    x_offset = -(x_dis + r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
    z_offset = -(x_dis + r) * np.sin(beta) - c * np.cos(theta) * np.cos(beta)
    
    y_offset_r = +(c * np.sin(theta) - d_asi / 2)
    y_offset_l = -(c * np.sin(theta) - d_asi / 2)
    
    # 骨盤座標系を計算
    hip_0 = (rasi + lasi) / 2
    sacrum = (rpsi + lpsi) / 2
    
    e_x0 = normalize(hip_0 - sacrum)
    e_y = normalize(lasi - rasi)
    e_z = normalize(np.cross(e_x0, e_y))
    e_x = np.cross(e_y, e_z)
    
    # 変換行列
    rot_matrix = np.column_stack([e_x, e_y, e_z])
    
    # グローバル座標系での股関節中心
    rhip_offset_local = np.array([x_offset, y_offset_r, z_offset])
    lhip_offset_local = np.array([x_offset, y_offset_l, z_offset])
    
    rhip_center = hip_0 + rot_matrix @ rhip_offset_local
    lhip_center = hip_0 + rot_matrix @ lhip_offset_local
    
    return rhip_center, lhip_center


class ISBJointAngles:
    """
    ISB推奨のJCSに基づく関節角度計算クラス
    
    使用例:
    -------
    >>> calculator = ISBJointAngles()
    >>> angles = calculator.calculate_all_angles(
    ...     rasi, lasi, rpsi, lpsi,
    ...     rknee, rknee2, lknee, lknee2,
    ...     rank, rank2, lank, lank2,
    ...     rtoe, rhee, ltoe, lhee,
    ...     leg_length=0.9
    ... )
    """
    
    def __init__(self, marker_radius=0.0127, height=1.76):
        """
        Parameters:
        -----------
        marker_radius : float
            マーカー半径 [m]
        height : float
            被験者身長 [m]
        """
        self.marker_radius = marker_radius
        self.height = height
    
    def calculate_all_angles(self, rasi, lasi, rpsi, lpsi,
                             rknee, rknee2, lknee, lknee2,
                             rank, rank2, lank, lank2,
                             rtoe, rhee, ltoe, lhee,
                             leg_length=None):
        """
        すべての関節角度を計算する
        
        Parameters:
        -----------
        rasi, lasi, rpsi, lpsi : ndarray
            骨盤マーカー位置
        rknee, rknee2 : ndarray
            右膝マーカー位置（内側、外側）
        lknee, lknee2 : ndarray
            左膝マーカー位置（内側、外側）
        rank, rank2 : ndarray
            右足首マーカー位置（内踝、外踝）
        lank, lank2 : ndarray
            左足首マーカー位置（内踝、外踝）
        rtoe, rhee : ndarray
            右足つま先、踵マーカー位置
        ltoe, lhee : ndarray
            左足つま先、踵マーカー位置
        leg_length : float, optional
            下肢長（自動計算される場合はNone）
        
        Returns:
        --------
        dict : 各関節角度を含む辞書
        """
        # 下肢長の計算（必要な場合）
        if leg_length is None:
            leg_length_r = np.linalg.norm(rank - rasi)
            leg_length_l = np.linalg.norm(lank - lasi)
            leg_length = (leg_length_r + leg_length_l) / 2
        
        # 股関節中心の推定
        rhip_center, lhip_center = calculate_hip_joint_center_davis(
            rasi, lasi, rpsi, lpsi, leg_length, 
            self.marker_radius, self.height
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
            'R_Knee_VrVl': r_knee_add,  # 内反/外反
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
            # 座標系（デバッグ用）
            '_pelvis_axes': pelvis_axes,
            '_rfemur_axes': rfemur_axes,
            '_lfemur_axes': lfemur_axes,
            '_rtibia_axes': rtibia_axes,
            '_ltibia_axes': ltibia_axes,
            '_rfoot_axes': rfoot_axes,
            '_lfoot_axes': lfoot_axes,
            '_rhip_center': rhip_center,
            '_lhip_center': lhip_center,
        }


# テスト用関数
def test_isb_angles():
    """簡単なテストケース"""
    # 中立位のダミーデータ
    rasi = np.array([0.1, 0.0, 1.0])
    lasi = np.array([-0.1, 0.0, 1.0])
    rpsi = np.array([0.05, -0.15, 1.0])
    lpsi = np.array([-0.05, -0.15, 1.0])
    
    rknee = np.array([0.1, 0.0, 0.5])
    rknee2 = np.array([0.2, 0.0, 0.5])
    lknee = np.array([-0.2, 0.0, 0.5])
    lknee2 = np.array([-0.1, 0.0, 0.5])
    
    rank = np.array([0.1, 0.0, 0.1])
    rank2 = np.array([0.2, 0.0, 0.1])
    lank = np.array([-0.2, 0.0, 0.1])
    lank2 = np.array([-0.1, 0.0, 0.1])
    
    rtoe = np.array([0.15, 0.2, 0.0])
    rhee = np.array([0.15, -0.1, 0.0])
    ltoe = np.array([-0.15, 0.2, 0.0])
    lhee = np.array([-0.15, -0.1, 0.0])
    
    calculator = ISBJointAngles()
    angles = calculator.calculate_all_angles(
        rasi, lasi, rpsi, lpsi,
        rknee, rknee2, lknee, lknee2,
        rank, rank2, lank, lank2,
        rtoe, rhee, ltoe, lhee
    )
    
    print("ISB JCS 関節角度テスト結果:")
    print("-" * 40)
    for key, value in angles.items():
        if not key.startswith('_'):
            print(f"{key}: {value:.2f}°")
    
    return angles


if __name__ == "__main__":
    test_isb_angles()
