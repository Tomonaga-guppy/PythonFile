import numpy as np
import matplotlib.pyplot as plt


vector1 = np.array([-13.4,-75.4])  #27°回転でK7とほぼ一致
vector2 = np.array([-53.5,-91.9])
# vector1 = np.array([-9.8,-74.8])
# vector2 = np.array([-52.6,-93.7])
# vector1 = np.array([0,0])
# vector2 = np.array([-45.5,-23])

vec = vector2 - vector1
print(f"vec = {vec}")

angle = 27
rotation_matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                            [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])

rotated_vec = np.dot(rotation_matrix, vec)
print(f"rotated_vec = {rotated_vec}")
K7_vec = np.array([-28.2,-33])
K7_norm = np.linalg.norm(K7_vec)
print(f"K7_aspect = {K7_vec[1]/K7_vec[0]}, aspect = {rotated_vec[1]/rotated_vec[0]}")
print(f"K7_norm = {K7_norm}, norm = {np.linalg.norm(rotated_vec)}")

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(vector1[0], vector1[1], c='r', s=9)
ax.scatter(vector2[0], vector2[1], c='b', s=9)
ax.scatter(vec[0], vec[1], c='k', s=9)
ax.scatter(0,0,c='k',s=9)
ax.scatter(rotated_vec[0], rotated_vec[1], c='g', s=9)
ax.grid()
ax.set_aspect('equal', adjustable='box')
plt.show()
