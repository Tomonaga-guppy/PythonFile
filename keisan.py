import math
import numpy as np

mkg_ini = np.array([-70,-20])
seal_ini = np.array([-80,-30])
# mkg_ini = np.array([70,20])
# seal_ini = np.array([80,30])

rot_angle = np.deg2rad(60)
rotate_array = np.array([[np.cos(rot_angle), np.sin(rot_angle)],[-np.sin(rot_angle), np.cos(rot_angle)]]).T

mkg_rot = rotate_array @ mkg_ini
seal_rot = rotate_array @ seal_ini

print(f"seal_ini={seal_ini}, seal_rot={seal_rot}, mkg_ini={mkg_ini}, mkg_rot={mkg_rot}")

print(f"seal={seal_rot - seal_ini}, mkg={mkg_rot - mkg_ini}")


seal_displacemant = np.array(seal_rot) - np.array(seal_ini)
mkg_displacemant = np.array(mkg_rot) - np.array(mkg_ini)
print(f"seal_displacemant = {seal_displacemant}")
print(f"mkg_displacemant = {mkg_displacemant}")
print(f"y_diff = {seal_displacemant[1] - mkg_displacemant[1]}")
print(f"z_diff = {seal_displacemant[0] - mkg_displacemant[0]}")



#グラフで4点をプロット
import matplotlib.pyplot as plt
plt.scatter(seal_ini[0], seal_ini[1], c='red', alpha=0.5)
plt.scatter(seal_rot[0], seal_rot[1], c='red')
plt.scatter(mkg_ini[0], mkg_ini[1], c='blue',alpha=0.5)
plt.scatter(mkg_rot[0], mkg_rot[1], c='blue')
#sealとmkg,原点を結ぶ線をプロット
plt.plot([seal_ini[0],mkg_ini[0]],[seal_ini[1],mkg_ini[1]],c='black',alpha=0.5)
plt.plot([mkg_ini[0],0],[mkg_ini[1],0],c='black',alpha=0.5)
plt.plot([0,seal_ini[0]],[0,seal_ini[1]],c='black',alpha=0.5)

plt.plot([seal_rot[0],mkg_rot[0]],[seal_rot[1],mkg_rot[1]],c='black')
plt.plot([mkg_rot[0],0],[mkg_rot[1],0],c='black')
plt.plot([0,seal_rot[0]],[0,seal_rot[1]],c='black')
plt.grid()
lim = [-100, 100]
plt.xlim(lim[0], lim[1])
plt.ylim(lim[0], lim[1])
# plt.xlim(lim[1], lim[0])
# plt.ylim(lim[1], lim[0])
#縦横比を揃える
plt.gca().set_aspect('equal', adjustable='box')
#10度ごとに目盛りを表示
plt.xticks(np.arange(lim[0], lim[1], 20))
plt.yticks(np.arange(lim[0], lim[1], 20))
plt.show()
