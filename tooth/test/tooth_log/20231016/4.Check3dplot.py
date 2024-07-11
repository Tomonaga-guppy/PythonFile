import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pathlib import Path
import mpl_toolkits.mplot3d.art3d as art3d
import os
import glob

root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_09_000"
npy_paths = glob.glob(os.path.join(root_dir,"*_A2*/XYZ_localdata.npy"))


for npy_path in npy_paths:
    # ファイルから配列を読み込む dataは[axis][frame][id]の順で並んでいる
    data = np.load(npy_path,allow_pickle=True)
    save_dir = os.path.dirname(npy_path)
    # save_dir = Path('C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_09_000/20230606_J2')

    # 配列の形状を確認する
    print(data.shape)

    # matplotlibで3Dプロットを作成する
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #view_initで視点 0,0でyz平面   0,90でxy平面
    # elev,azim = 0,90
    elev,azim = 0,0
    ax.view_init(elev=elev, azim=azim)
    ax.invert_zaxis()
    a = 100
    ax.set_zlim(-a,a)
    ax.set_xlim(-a,a)
    ax.set_ylim(-a,a)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    scatter = ax.scatter(data[0,:,:], data[2,:,:], data[1,:,:], color='blue')
    scatterd = ax.scatter(data[0,:,68], data[2,:,68], data[1,:,68], color='red')

    line1= art3d.Line3D([-a,a],[0,0],[0,0], color='black')
    line2= art3d.Line3D([0,0],[-a,a],[0,0], color='black')
    line3= art3d.Line3D([0,0],[0,0],[-a,a], color='black')
    ax.add_line(line1)
    ax.add_line(line2)
    ax.add_line(line3)

    # print('datashape[1]=',data.shape[1])

    # アニメーションの更新関数
    def update(frame):
        #print(data[frame,:,3], data[frame,:,2], data[frame,:,1])
        print(frame)
        x,y,z,xd,yd,zd = [],[],[],[],[],[]
        for i in range(data.shape[2]):
            if data[0,frame,i] is not None and data[1,frame,i] is not None and data[2,frame,i] is not None:
                if i==68:
                    xd.append(data[0,frame,i])
                    yd.append(data[2,frame,i])
                    zd.append(data[1,frame,i])
                else:
                    x.append(data[0,frame,i])
                    y.append(data[2,frame,i])
                    z.append(data[1,frame,i])

            else:
                print(i)
        scatter._offsets3d = (x,y,z)
        scatterd._offsets3d = (xd,yd,zd)
        ax.set_title(f"Frame {frame}")

    # アニメーションの作成
    anim = FuncAnimation(fig, update, frames=data.shape[1], interval=50, blit=False)

    # アニメーションを保存する
    anim.save(save_dir + f'/animation_local[{elev},{azim}].mp4', fps=30)