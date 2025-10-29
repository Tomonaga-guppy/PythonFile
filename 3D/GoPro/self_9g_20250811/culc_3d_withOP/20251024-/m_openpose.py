import numpy as np

def culc_angle(vector1, vector2, n_vector, degrees=False, angle_type=None):
    """
    股関節の屈曲伸展角度を計算（矢状面への投影）
    
    Parameters:
    -----------
    vector1 : np.ndarray 
    vector2 : np.ndarray
    n_vector : np.ndarray
        法線方向ベクトル
    degrees : bool, optional
        角度を度数法で返すかどうか。デフォルトはFalse（ラジアン）。
    angle_type : str, optional
        角度の種類（将来の拡張用）。現在は未使用
    Returns:
    --------
    float : 関節角度
    """
    # 矢状面の法線を正規化
    n_norm = n_vector / np.linalg.norm(n_vector)
    
    # ベクトル1とベクトル2を矢状面に投影
    vector1_proj = vector1 - np.dot(vector1, n_norm) * n_norm
    vector2_proj = vector2 - np.dot(vector2, n_norm) * n_norm
    
    # 投影ベクトルの長さチェック
    vector1_norm = np.linalg.norm(vector1_proj)
    vector2_norm = np.linalg.norm(vector2_proj)
    
    if vector1_norm < 1e-8 or vector2_norm < 1e-8:
        return 0.0
    
    # 角度計算
    cos_angle = np.dot(vector1_proj, vector2_proj) / (vector1_norm * vector2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    
    # 符号判定：後方向ベクトルとの外積で判断
    # 矢状面内でベクトル1に直交する後方向ベクトル
    posterior_vec = np.cross(n_norm, vector1_proj)
    posterior_vec = posterior_vec / np.linalg.norm(posterior_vec)

    # ベクトル1が前方にあれば負、後方にあれば正
    sign = np.sign(np.dot(vector2_proj, posterior_vec))

    if angle_type == 'hip':
        angle = - sign * (np.pi - angle_rad)
    elif angle_type == 'knee':
        angle = - sign * angle_rad
    elif angle_type == 'ankle':
        angle = angle_rad - np.pi/2
    elif angle_type == 'hip_adab':
        angle = 180 - angle_rad
    elif angle_type == 'hip_inex':
        angle = angle_rad

    if degrees:
        angle = np.degrees(angle)
    
    return angle

def culc_angle_all_frames(vector1_all, vector2_all, n_vector_all, degrees=False, angle_type=None):
    """
    全フレームで股関節屈曲伸展角度を計算
    """
    angles = np.zeros(vector1_all.shape[0])
    for i in range(vector1_all.shape[0]):
        angles[i] = culc_angle(
            vector1_all[i], 
            vector2_all[i], 
            n_vector_all[i],
            degrees=degrees,
            angle_type=angle_type
        )
    return angles