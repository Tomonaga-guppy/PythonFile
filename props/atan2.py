import numpy as np

def calculate_angle_3d(vector1, vector2):
    angle_list = []
    for frame in range(len(vector1)):
        dot_product = np.dot(vector1[frame, :], vector2[frame, :])
        cross_product = np.cross(vector1[frame, :], vector2[frame, :])
        cross_norm = np.linalg.norm(cross_product)  # クロス積の大きさ

        # atan2を使って符号付きの角度を計算
        angle = np.rad2deg(np.arctan2(cross_norm, dot_product))
        angle_list.append(angle)

    return angle_list

# 正負の方向を持つベクトルの例
vector1 = np.array([[1, 0, 0], [-1, 0, 0]])  # ベクトル1
vector2 = np.array([[0, 1, 0], [0, -1, 0]])  # ベクトル2

angles = calculate_angle_3d(vector1, vector2)
print("ベクトル間の角度: ", angles)
