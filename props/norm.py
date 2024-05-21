import math
import numpy as np

# def calculate_distance(x, y):
#     distance = math.sqrt(x**2 + y**2)
#     return distance


# print(calculate_distance(-29.7,31.8))

# #aの120フレームのシールの4点座標（おおよそ）
# bector1 = np.array([-11.3, -90.4, -2.3])
# bector2 = np.array([-5.7, -99.4, -4.0])
# bector3 = np.array([-13.7, -104.6, -4.7])
# bector4 = np.array([-20.5, -95.9, -5.4])
#a308フレームのシールの4点座標（おおよそ）
bector1 = np.array([-1.0, -15.8, -10.7])
bector2 = np.array([6.4, -22.0, -15.5])
bector3 = np.array([-0.6, -29.8, -18.6])
bector4 = np.array([-7.8, -22.5, -16.2])

norm1 = np.linalg.norm(bector3-bector1)
norm2 = np.linalg.norm(bector4-bector2)

print(f"norm1 = {norm1}, norm2 = {norm2}")

print(f"{norm1/15}, {norm2/15}")
