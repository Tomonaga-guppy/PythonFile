import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pathlib import Path
import mpl_toolkits.mplot3d.art3d as art3d
import os
import glob

root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_09_000"
npy_paths = glob.glob(os.path.join(root_dir,"*_B2*/XYZ_localdata.npy"))


for i, npy_path in enumerate(npy_paths):
    # ファイルから配列を読み込む dataは[axis][frame][id]の順で並んでいる
    data = np.load(npy_path,allow_pickle=True)
    save_dir = os.path.dirname(npy_path)
    # save_dir = Path('C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_09_000/20230606_J2')

    # 配列の形状を確認する
    print(data.shape)

    # matplotlibで3Dプロットを作成する
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    #view_initで視点 0,0でyz平面   0,90でxy平面
    # elev,azim = 0,90
    elev,azim = 0,0
    ax.view_init(elev=elev, azim=azim)
    ax.invert_zaxis()
    a = 100
    ax.set_zlim(-a,a)
    ax.set_xlim(-a,a)
    ax.set_ylim(a,-a)
    font_size = 8
    ax.set_xlabel("x", fontsize=font_size)
    ax.set_ylabel("z", fontsize=font_size)
    ax.set_zlabel("y", fontsize=font_size)
    scatter_seal = ax.scatter([], [], [], color='orange', marker = "*", s = 60)
    scatter = ax.scatter([], [], [], color='blue')
    scatter_red = ax.scatter([], [], [], color='red')
    scatter_green = ax.scatter([], [], [], color='green')


    line1= art3d.Line3D([-a,a],[0,0],[0,0], color='black')
    line2= art3d.Line3D([0,0],[-a,a],[0,0], color='black')
    line3= art3d.Line3D([0,0],[0,0],[-a,a], color='black')
    ax.add_line(line1)
    ax.add_line(line2)
    ax.add_line(line3)

    ax2 = fig.add_subplot(122, projection='3d')
    #view_initで視点 0,0でyz平面   0,90でxy平面
    # elev,azim = 0,90
    elev2,azim2 = 0,90
    ax2.view_init(elev=elev2, azim=azim2)
    ax2.invert_zaxis()
    a2 = 100
    ax2.set_zlim(-a2,a2)
    ax2.set_xlim(a2,-a2)
    ax2.set_ylim(-a2,a2)
    ax2.set_xlabel("x", fontsize=font_size)
    ax2.set_ylabel("z", fontsize=font_size)
    ax2.set_zlabel("y", fontsize=font_size)
    scatter2_seal = ax2.scatter([], [], [], color='orange',marker = "*", s = 60)
    scatter2 = ax2.scatter([], [], [], color='blue')
    scatter2_red = ax2.scatter([], [], [], color='red')
    scatter2_green = ax2.scatter([], [], [], color='green')

    line2_1= art3d.Line3D([-a2,a2],[0,0],[0,0], color='black')
    line2_2= art3d.Line3D([0,0],[-a2,a2],[0,0], color='black')
    line2_3= art3d.Line3D([0,0],[0,0],[-a2,a2], color='black')
    ax2.add_line(line2_1)
    ax2.add_line(line2_2)
    ax2.add_line(line2_3)

    # print('datashape[1]=',data.shape[1])

    # アニメーションの更新関数
    def update(frame):
        #print(data[frame,:,3], data[frame,:,2], data[frame,:,1])
        print(frame)
        x,y,z =[],[],[]
        x2,y2,z2 =[],[],[]
        x3,y3,z3 =[],[],[]
        x_seal,y_seal,z_seal =[],[],[]

        for i in range(data.shape[2]):
            if data[0,frame,i] is not None and data[1,frame,i] is not None and data[2,frame,i] is not None:
                if i==68:
                    x_seal.append(data[0,frame,i])
                    y_seal.append(data[2,frame,i])
                    z_seal.append(data[1,frame,i])
                elif i==45 or 9 <= i <= 16:
                    x2.append(data[0,frame,i])
                    y2.append(data[2,frame,i])
                    z2.append(data[1,frame,i])
                elif i==36 or 0 <= i <= 7:
                    x3.append(data[0,frame,i])
                    y3.append(data[2,frame,i])
                    z3.append(data[1,frame,i])
                else:
                    x.append(data[0,frame,i])
                    y.append(data[2,frame,i])
                    z.append(data[1,frame,i])

            else:
                print(i)
        scatter_seal._offsets3d = (x_seal,y_seal,z_seal)
        scatter._offsets3d = (x,y,z)
        scatter_red._offsets3d = (x2,y2,z2)
        scatter_green._offsets3d = (x3,y3,z3)

        scatter2_seal._offsets3d = (x_seal,y_seal,z_seal)
        scatter2._offsets3d = (x,y,z)
        scatter2_red._offsets3d = (x2,y2,z2)
        scatter2_green._offsets3d = (x3,y3,z3)

        ax.set_title(f"Frame {frame}\nYZyokogao")
        ax2.set_title(f"Frame {frame}\nXYsyomen")

    # アニメーションの作成
    anim = FuncAnimation(fig, update, frames=data.shape[1], interval=50, blit=False)

    # アニメーションを保存する
    anim.save(save_dir + f'/animation_local.mp4', fps=30)

    print(f"{i+1}/{len(npy_paths)} anim is saved in {save_dir + f'/animation_local.mp4'}")