import numpy as np
from scipy.spatial.transform import Rotation as R

# 基準となる座標系aの定義
a_x = np.array([1, 0, 0])
a_y = np.array([0, 1, 0])
a_z = np.array([0, 0, 1])
base_a = np.array([a_x, a_y, a_z])
print(f"基準となる座標系base_a:\n{base_a}")

# 座標系aをz軸周りに30度回転させた座標系bの定義
theta1 = np.deg2rad(30)  # 30度をラジアンに変換
rot_1 = np.array([[np.cos(theta1), -np.sin(theta1), 0],
                             [np.sin(theta1), np.cos(theta1), 0],
                             [0, 0, 1]])
base_b = np.dot(base_a, rot_1)
print(f"base_aをx軸について30度回転させた座標系base_b:\n{base_b}")

# 座標系bをy軸周りに45度回転させた座標系cの定義
theta2 = np.deg2rad(45)  # 45度をラジアンに変換
rot_2 = np.array([[np.cos(theta2), 0, np.sin(theta2)],
                             [0, 1, 0],
                             [-np.sin(theta2), 0, np.cos(theta2)]])
base_c = np.dot(base_b, rot_2)
print(f"base_bをy軸について45度回転させた座標系base_c:\n{base_c}")

# 座標系cをx軸周りに60度回転させた座標系dの定義
theta3 = np.deg2rad(60)  # 60度をラジアンに変換
rot_3 = np.array([[1, 0, 0],
                             [0, np.cos(theta3), -np.sin(theta3)],
                             [0, np.sin(theta3), np.cos(theta3)]])
base_d = np.dot(base_c, rot_3)
print(f"base_cをx軸について60度回転させた座標系base_d:\n{base_d}")

def calculate_joint_angles(segment1, segment2, sequence="YZX", degrees=True):
    """
    2つのセグメントから関節角度を計算する。

    Args:
        segment1 (np.ndarray): 3x3のセグメント
        distal_frame (np.ndarray): 3x3のセグメント
        sequence (str): 'yzx'などのオイラー角の回転順序
        degrees (bool): 角度を度数法で返すか（True）、弧度法で返すか（False）

    Returns:
        np.ndarray: 3つの関節角度（例：[屈曲/伸展, 内転/外転, 内旋/外旋]）
    """
    # 1. 相対回転行列を計算
    # R_relative = R_proximal^T * R_distal
    relative_rotation_matrix = np.transpose(segment1) @ segment2

    # 2. Scipyを使って回転行列からオイラー角に分解
    r = R.from_matrix(relative_rotation_matrix)
    angles = r.as_euler(sequence, degrees=degrees)

    return angles

angles = calculate_joint_angles(base_a, base_d, degrees=True)
print(f"関節角度（屈曲/伸展, 内転/外転, 内旋/外旋）: {angles}")

angles2 = calculate_joint_angles(base_d, base_a, degrees=True)
print(f"関節角度（屈曲/伸展, 内転/外転, 内旋/外旋）: {angles2}")


print("\nオイラー角から直接計算した場合の角度[y,z,x][45, 30, 60]度を計算します。")
# YZXの順で45, 30, 60度回転させた姿勢を直接作る
r = R.from_euler('yzx', [45, 30, 60], degrees=True)
base_euler = r.as_matrix()

# 作成した回転行列を分解すれば、当然もとの角度が得られる
angles_euler = calculate_joint_angles(base_a, base_euler, sequence="YZX", degrees=True)
print(f"オイラー角から直接計算した場合の角度: {angles_euler}")

angels3 = calculate_joint_angles(base_a, base_euler, sequence="yzx", degrees=True)
print(f"オイラー角から直接計算した場合の角度: {angels3}")
