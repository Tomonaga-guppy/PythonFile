import numpy as np

hip = np.array([7, 1])
knee = np.array([4, 4])
ankle = np.array([4, 8])

thigh = hip - knee
lower_leg = knee - ankle

def get_knee_angle(vector1, vector2):
    """
    arctan2を使用して、2つのベクトル間の角度を計算する。
    vector1を何度回転させるとvector2と一致するかを求める。
    戻り値の範囲は -180度から180度。
    """
    dot_product = np.dot(vector1, vector2)
    cross_product = np.cross(vector1, vector2)
    angle = np.rad2deg(np.arctan2(cross_product, dot_product))

    return angle

k_angle = get_knee_angle(thigh, lower_leg)
Knee_angle = -k_angle  #膝関節角度の正負の向きと合わせるため

if Knee_angle < 0:
    print(f"膝関節は{Knee_angle}度伸展しています。")
else:
    print(f"膝関節は{Knee_angle}度屈曲しています。")