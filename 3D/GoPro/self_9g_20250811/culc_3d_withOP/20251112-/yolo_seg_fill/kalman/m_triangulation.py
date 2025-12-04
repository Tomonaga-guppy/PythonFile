import numpy as np

"""
★★★ 重み付き三角測量モジュール ★★★
2Dキーポイントから3D座標と信頼度を計算するための関数群
平行移動量は計測ごとに毎回変更してください
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

def weighted_linear_triangulation(P1, P2, correspondences, weights=None):
    """
    重み付き線形三角測量を実行し、3D座標と信頼度を返す
    """
    projection_matrices = [P1, P2]
    n_cameras = len(projection_matrices)

    if weights is None:
        weights = [1.0] * n_cameras
        w = np.ones(n_cameras)
    else:
        # nanを小さな値に置き換える
        w = [np.nan_to_num(wi, nan=0.1) for wi in weights]

    D = np.zeros((n_cameras * 2, 4))
    for cam_idx in range(n_cameras):
        P = projection_matrices[cam_idx]
        uv = correspondences[:, cam_idx]
        D[cam_idx * 2:cam_idx * 2 + 2, :] = construct_D_block(P, uv, w=w[cam_idx])

    Q = D.T.dot(D)
    u, s, vh = np.linalg.svd(Q)
    point_3d = p2e(u[:, -1, np.newaxis])

    weightArray = np.asarray(weights)
    # 0でない重みが2つ未満の場合、信頼度は0
    if np.count_nonzero(weightArray) < 2:
        conf = 0.0
    else:
        # 0でない重みのみを抽出
        valid_weights = weightArray[weightArray != 0]
        # 有効な重みがすべてnanの場合、信頼度は0.5
        if np.all(np.isnan(valid_weights)):
             conf = 0.5
        else:
             # nanを無視して平均を計算
             conf = np.nanmean(valid_weights)
    # ----------------------------------------------------------------

    return point_3d.flatten(), conf

def triangulate_points_weighted(P1, P2, points1, points2, confidences1, confidences2):
    """
    重み付き三角測量を使用して2D点から3D点と信頼度を計算
    """
    points_3d_list = []
    confidences_list = []
    for i in range(points1.shape[0]):
        correspondences = np.column_stack([points1[i], points2[i]])
        weights = [confidences1[i], confidences2[i]]
        try:
            point_3d, conf = weighted_linear_triangulation(P1, P2, correspondences, weights)
            points_3d_list.append(point_3d)
            confidences_list.append(conf)
        except Exception:
            points_3d_list.append(np.full(3, np.nan))
            confidences_list.append(np.nan)
    return np.array(points_3d_list), np.array(confidences_list)

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

def triangulate_and_rotate(P1, P2, points1, points2, confidences1, confidences2):
    """三角測量と座標回転をまとめて行い、3D座標と信頼度を返すヘルパー関数"""
    valid_indices = np.where(~np.isnan(points1).any(axis=1) & ~np.isnan(points2).any(axis=1))[0]
    if len(valid_indices) == 0:
        return np.full((25, 3), np.nan), np.full((25,), np.nan)

    points_3d_raw, confidences_raw = triangulate_points_weighted(
        P1, P2,
        points1[valid_indices],
        points2[valid_indices],
        confidences1[valid_indices],
        confidences2[valid_indices]
    )
    points_3d_rotated = rotate_coordinates_x_axis(points_3d_raw)

    full_points_3d = np.full((25, 3), np.nan)
    full_points_3d[valid_indices] = points_3d_rotated

    full_confidences = np.full((25,), np.nan)
    full_confidences[valid_indices] = confidences_raw

    return full_points_3d, full_confidences
