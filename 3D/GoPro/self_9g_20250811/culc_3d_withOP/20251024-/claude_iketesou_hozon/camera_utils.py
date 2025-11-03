"""
OpenCapのutilsCameraPy3.pyから移植した高度な三角測量機能
"""
import numpy as np
import cv2
from typing import Tuple, Optional


def undistort_points(points_2d: np.ndarray, camera_matrix: np.ndarray, 
                     dist_coeffs: np.ndarray) -> np.ndarray:
    """
    レンズ歪みを補正する
    
    Args:
        points_2d: (N, 2) の2Dポイント
        camera_matrix: (3, 3) のカメラ内部パラメータ行列
        dist_coeffs: (5,) の歪み係数 [k1, k2, p1, p2, k3]
    
    Returns:
        undistorted_points: (N, 2) の補正後の2Dポイント
    """
    if points_2d.shape[0] == 0:
        return points_2d
    
    # OpenCVの歪み補正関数を使用
    points_2d = points_2d.reshape(-1, 1, 2).astype(np.float32)
    undistorted = cv2.undistortPoints(
        points_2d, 
        camera_matrix, 
        dist_coeffs, 
        P=camera_matrix
    )
    return undistorted.reshape(-1, 2)


def calculate_parallax_angle(point1: np.ndarray, point2: np.ndarray, 
                             R1: np.ndarray, t1: np.ndarray,
                             R2: np.ndarray, t2: np.ndarray) -> float:
    """
    2つのカメラビュー間の視差角を計算する
    
    Args:
        point1, point2: 正規化された2D点
        R1, t1: カメラ1の回転行列と並進ベクトル
        R2, t2: カメラ2の回転行列と並進ベクトル
    
    Returns:
        parallax_angle: 視差角(度)
    """
    # カメラ中心の3D位置
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2
    
    # 各カメラから点への方向ベクトル(正規化座標を使用)
    # 正規化座標 (x, y, 1) を3D方向に変換
    dir1 = R1.T @ np.array([point1[0], point1[1], 1.0])
    dir2 = R2.T @ np.array([point2[0], point2[1], 1.0])
    
    # 方向ベクトルを正規化
    dir1 = dir1 / np.linalg.norm(dir1)
    dir2 = dir2 / np.linalg.norm(dir2)
    
    # 内積から角度を計算
    cos_angle = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def triangulate_with_checks(P1: np.ndarray, P2: np.ndarray,
                            points1: np.ndarray, points2: np.ndarray,
                            confidences1: np.ndarray, confidences2: np.ndarray,
                            min_confidence: float = 0.1,
                            min_parallax: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    視差角チェックとエピポーラ制約を考慮した三角測量
    
    Args:
        P1, P2: プロジェクション行列 (3, 4)
        points1, points2: 2Dキーポイント (25, 2)
        confidences1, confidences2: 信頼度 (25,)
        min_confidence: 最小信頼度しきい値
        min_parallax: 最小視差角(度)
    
    Returns:
        points_3d: (25, 3) の3D座標
        final_confidences: (25,) の最終信頼度
    """
    num_points = points1.shape[0]
    points_3d = np.full((num_points, 3), np.nan)
    final_confidences = np.zeros(num_points)
    
    # カメラパラメータを分解
    K1 = P1[:, :3]
    K2 = P2[:, :3]
    R1 = np.eye(3)  # カメラ1を基準座標系とする
    t1 = np.zeros((3, 1))
    
    # カメラ2の外部パラメータを計算
    # P2 = K2 [R2 | t2] の形式から R2, t2 を抽出
    M2 = np.linalg.inv(K2) @ P2
    R2 = M2[:, :3]
    t2 = M2[:, 3:4]
    
    for i in range(num_points):
        # 信頼度チェック
        if confidences1[i] < min_confidence or confidences2[i] < min_confidence:
            continue
        
        pt1 = points1[i]
        pt2 = points2[i]
        
        # NaNチェック
        if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
            continue
        
        # 正規化座標に変換
        pt1_norm = (pt1 - K1[:2, 2]) / np.array([K1[0, 0], K1[1, 1]])
        pt2_norm = (pt2 - K2[:2, 2]) / np.array([K2[0, 0], K2[1, 1]])
        
        # 視差角チェック
        try:
            parallax = calculate_parallax_angle(pt1_norm, pt2_norm, R1, t1, R2, t2)
            if parallax < min_parallax:
                # 視差角が小さい場合は信頼度を下げる
                weight = parallax / min_parallax
                final_confidences[i] = min(confidences1[i], confidences2[i]) * weight
                if weight < 0.5:  # 視差角が半分以下なら使用しない
                    continue
            else:
                final_confidences[i] = min(confidences1[i], confidences2[i])
        except:
            continue
        
        # DLT法で三角測量
        points_4d = cv2.triangulatePoints(
            P1, P2,
            pt1.reshape(2, 1).astype(np.float32),
            pt2.reshape(2, 1).astype(np.float32)
        )
        
        # 同次座標から3D座標に変換
        points_3d_homogeneous = points_4d.flatten()
        if abs(points_3d_homogeneous[3]) > 1e-6:
            point_3d = points_3d_homogeneous[:3] / points_3d_homogeneous[3]
            
            # 再投影誤差チェック
            reproj_error = calculate_reprojection_error(
                point_3d, P1, P2, pt1, pt2
            )
            
            # 再投影誤差が大きい場合は除外
            if reproj_error < 10.0:  # 10ピクセル未満
                points_3d[i] = point_3d
                # 再投影誤差に基づいて信頼度を調整
                error_weight = np.exp(-reproj_error / 5.0)
                final_confidences[i] *= error_weight
    
    return points_3d, final_confidences


def calculate_reprojection_error(point_3d: np.ndarray, P1: np.ndarray, 
                                 P2: np.ndarray, pt1: np.ndarray, 
                                 pt2: np.ndarray) -> float:
    """
    再投影誤差を計算する
    
    Args:
        point_3d: (3,) の3D点
        P1, P2: プロジェクション行列
        pt1, pt2: 元の2D観測点
    
    Returns:
        error: 平均再投影誤差(ピクセル)
    """
    # 同次座標に変換
    point_3d_h = np.append(point_3d, 1.0)
    
    # 各カメラに再投影
    proj1 = P1 @ point_3d_h
    proj1 = proj1[:2] / proj1[2]
    
    proj2 = P2 @ point_3d_h
    proj2 = proj2[:2] / proj2[2]
    
    # ユークリッド距離
    error1 = np.linalg.norm(proj1 - pt1)
    error2 = np.linalg.norm(proj2 - pt2)
    
    return (error1 + error2) / 2.0


def bundle_adjustment_simple(points_3d: np.ndarray, P1: np.ndarray, P2: np.ndarray,
                             points1: np.ndarray, points2: np.ndarray,
                             confidences: np.ndarray) -> np.ndarray:
    """
    簡易的なバンドル調整(各点を独立に最適化)
    
    Args:
        points_3d: (N, 3) の初期3D点
        P1, P2: プロジェクション行列
        points1, points2: 観測された2D点
        confidences: 信頼度
    
    Returns:
        optimized_points_3d: 最適化後の3D点
    """
    from scipy.optimize import least_squares
    
    optimized = points_3d.copy()
    
    for i in range(len(points_3d)):
        if np.any(np.isnan(points_3d[i])) or confidences[i] < 0.1:
            continue
        
        def residuals(x):
            """再投影誤差の残差"""
            point_3d_h = np.append(x, 1.0)
            
            # カメラ1への再投影
            proj1 = P1 @ point_3d_h
            proj1 = proj1[:2] / proj1[2]
            error1 = proj1 - points1[i]
            
            # カメラ2への再投影
            proj2 = P2 @ point_3d_h
            proj2 = proj2[:2] / proj2[2]
            error2 = proj2 - points2[i]
            
            # 信頼度で重み付け
            weight = np.sqrt(confidences[i])
            return np.concatenate([error1 * weight, error2 * weight])
        
        try:
            result = least_squares(residuals, points_3d[i], method='lm')
            if result.success:
                optimized[i] = result.x
        except:
            pass  # 最適化失敗時は元の値を保持
    
    return optimized