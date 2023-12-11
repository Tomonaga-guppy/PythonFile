import math
import numpy as np

mkg_ini = np.array([-90,-20])
seal_ini = np.array([mkg_ini[0] + 10, mkg_ini[1] + -20])
rot_angle = np.deg2rad(30) #健常者の回転量平均おおよそ20°https://www.jstage.jst.go.jp/article/jjps1957/33/6/33_6_1359/_pdf

rotate_array = np.array([[np.cos(rot_angle), np.sin(rot_angle)],[-np.sin(rot_angle), np.cos(rot_angle)]]).T

mkg_rot = rotate_array @ mkg_ini
seal_rot = rotate_array @ seal_ini

seal_displacemant = np.array(seal_rot) - np.array(seal_ini)
mkg_displacemant = np.array(mkg_rot) - np.array(mkg_ini)

disp_diff = seal_displacemant - mkg_displacemant
print(f"seal_displacemant = {seal_displacemant}")
print(f"mkg_displacemant = {mkg_displacemant}")
print(f"z_diff = {disp_diff[0]}")
print(f"y_diff = {disp_diff[1]}")


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
plt.gca().set_aspect('equal', adjustable='box') #アスペクト比を1:1に
plt.xticks(np.arange(lim[0], lim[1], 20))#目盛りを20刻みに
plt.yticks(np.arange(lim[0], lim[1], 20))
plt.show()
