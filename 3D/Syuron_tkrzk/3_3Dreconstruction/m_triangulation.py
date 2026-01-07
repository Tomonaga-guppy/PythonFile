import numpy as np

"""
★★★ 重み付き三角測量モジュール ★★★
2Dキーポイントから3D座標と信頼度を計算するための関数群
(座標を厳密に評価する場合は平行移動量は計測ごとに毎回変更が必要)
"""

def p2e(projective):
    """projective座標からeuclidean座標に変換"""
    return (projective / projective[-1, :])[0:-1, :]

def construct_D_block(P, uv, w=1):
    """三角測量用のD行列のブロックを構築"""
    return w * np.vstack((
        uv[0] * P[2, :] - P[0, :],
        uv[1] * P[2, :] - P[1, :]
    ))

def rotate_coordinates_x_axis(points_3d, angle_degrees=180):
    """3D座標をX軸周りに回転させた後、平行移動を適用する"""
    angle_rad = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])
    rotated_points = np.dot(points_3d, rotation_matrix.T)
    translation = np.array([-35, 189, 0])
    return rotated_points + translation

# =============================================================================
# 3-view / N-view 対応（後方互換のため既存APIは残す）
# =============================================================================
def weighted_linear_triangulation_multi(P_list, uv_list, weights=None):
    """N視点の重み付き線形三角測量。

    Parameters
    ----------
    P_list : list[np.ndarray]
        各カメラの射影行列 (3x4) のリスト
    uv_list : list[np.ndarray]
        各カメラの 2D 座標 (2,) のリスト
    weights : list[float] | None
        各カメラの重み（=2D信頼度など）。Noneなら全て1.0

    Returns
    -------
    X : np.ndarray shape (3,)
        3D点（失敗時はnan）
    conf : float
        有効視点の重み平均（有効視点<2なら0）
    """
    if weights is None:
        weights = [1.0] * len(P_list)

    D = np.zeros((len(P_list) * 2, 4))
    for cam_idx, P, uv, w in zip(range(len(P_list)), P_list, uv_list, weights):
        if uv is None:
            continue
        uv = np.asarray(uv, dtype=float).reshape(2,)
        w = float(np.nan_to_num(w, nan=0.01))

        # 無効データは除外
        if np.any(np.isnan(uv)) or w <= 0:
            continue

        D[cam_idx * 2:cam_idx * 2 + 2, :] = construct_D_block(P, uv, w=w)

    # 2視点未満は三角測量不可
    if np.count_nonzero(weights) < 2:
        return np.full(3, np.nan), 0.0

    Q = D.T @ D

    # 最小固有値に対応するベクトル（=最小二乗解）
    _, _, vh = np.linalg.svd(Q)
    X_h = vh[-1, :]
    X = p2e(X_h.reshape(-1, 1)).flatten()

    conf = float(np.nanmean(np.asarray(weights))) if len(weights) else 0.0
    return X, conf


def triangulate_and_rotate_multi(P_list, points_list, confidences_list):
    """N視点の三角測量 + 座標回転。

    Parameters
    ----------
    P_list : list[np.ndarray]
        射影行列リスト
    points_list : list[np.ndarray]
        各視点の2D点配列 (25,2)
    confidences_list : list[np.ndarray]
        各視点の信頼度配列 (25,)
    """
    n_kp = 25
    out_xyz = np.full((n_kp, 3), np.nan, dtype=float)
    out_conf = np.full((n_kp,), np.nan, dtype=float)

    for k in range(n_kp):
        uv_list = [pts[k] for pts in points_list]
        w_list = [cfs[k] for cfs in confidences_list]
        X, conf = weighted_linear_triangulation_multi(P_list, uv_list, w_list)
        out_xyz[k] = X
        out_conf[k] = conf

    out_xyz = rotate_coordinates_x_axis(out_xyz)
    return out_xyz, out_conf