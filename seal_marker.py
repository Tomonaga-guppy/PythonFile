import math
import numpy as np

#補正角-8
seal_pos = np.array([8.7, -101.6])
marker_pos = np.array([11.7, -81.2])
rot_ori_pos = np.array([-56.3, -36.1])
#補正角なし
# seal_pos = np.array([-0.2, -101.9])
# marker_pos = np.array([4.3, -81.6])
# rot_ori_pos = np.array([-62.8, -38.3])

seal_ini = seal_pos - rot_ori_pos
marker_ini = marker_pos - rot_ori_pos

# seal_rot = np.array([-43.2, -133.6])
rot_angle = np.deg2rad(-30) #健常者の回転量平均おおよそ30°https://www.jstage.jst.go.jp/article/jjps1957/33/6/33_6_1359/_pdf

rotate_array = np.array([[np.cos(rot_angle), np.sin(rot_angle)],[-np.sin(rot_angle), np.cos(rot_angle)]]).T

mkg_rot = rotate_array @ marker_ini
seal_rot = rotate_array @ seal_ini

seal_displacemant = np.array(seal_rot) - np.array(seal_ini)
mkg_displacemant = np.array(mkg_rot) - np.array(marker_ini)

disp_diff = seal_displacemant - mkg_displacemant
print(f"seal_displacemant = {seal_displacemant}")
print(f"mkg_displacemant = {mkg_displacemant}")
print(f"z_diff = {disp_diff[0]}")
print(f"y_diff = {disp_diff[1]}")


#グラフで4点をプロット
import matplotlib.pyplot as plt
#回転中心をもう少し濃い色でプロット
plt.scatter(0, 0, c='darkorange', label='Origin' ,s=200)
plt.scatter(seal_ini[0], seal_ini[1], c='red', alpha=0.5, s=200)
plt.scatter(seal_rot[0], seal_rot[1], c='red', label='Seal', s=200)
plt.scatter(marker_ini[0], marker_ini[1], c='blue',alpha=0.5,s=200)
plt.scatter(mkg_rot[0], mkg_rot[1], c='blue', label='Marker',s=200)
plt.legend(fontsize=15)
#sealとmkg,原点を結ぶ線をプロット
plt.plot([seal_ini[0],marker_ini[0]],[seal_ini[1],marker_ini[1]],c='black',alpha=0.5)
plt.plot([marker_ini[0],0],[marker_ini[1],0],c='black',alpha=0.5)
plt.plot([0,seal_ini[0]],[0,seal_ini[1]],c='black',alpha=0.5)
plt.plot([seal_rot[0],mkg_rot[0]],[seal_rot[1],mkg_rot[1]],c='black')
plt.plot([mkg_rot[0],0],[mkg_rot[1],0],c='black')
plt.plot([0,seal_rot[0]],[0,seal_rot[1]],c='black')
plt.grid()
lim_x = [-20,100]
lim_y = [-100,20]
plt.xlim(lim_x[0], lim_x[1])
plt.ylim(lim_y[0], lim_y[1])
plt.gca().set_aspect('equal', adjustable='box')  #アスペクト比を1:1に
plt.xticks(np.arange(lim_x[0], lim_x[1], 20))  #目盛りを20刻みに
plt.yticks(np.arange(lim_y[0], lim_y[1], 20))
plt.xlabel('Z-axis', fontsize=20)
plt.ylabel('Y-axis', fontsize=20)
plt.tight_layout()
save_path = "./simulation.png"
plt.savefig(save_path)
plt.show()
