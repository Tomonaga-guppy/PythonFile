import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pathlib import Path

# ファイルから配列を読み込む
data = np.load("C:/Users/Tomson/BRLAB/tooth/Temporomandibular_movement/movie/2023_06_05/result.npy",allow_pickle=True)


# 配列の形状を確認する
print(data.shape)

# matplotlibで3Dプロットを作成する
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.view_init(elev=0, azim=0)
ax.invert_zaxis()
ax.set_ylim(300,500)
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_zlabel("y")
scatter = ax.scatter(data[0,0:,1], data[0,:,4], data[0,:,2], color='blue')
scatterd = ax.scatter(data[0,68,1], data[0,68,4], data[0,68,2], color='red')

# アニメーションの更新関数
def update(frame):
    #print(data[frame,:,3], data[frame,:,2], data[frame,:,1])
    print(frame)
    x,y,z,xd,yd,zd=[],[],[],[],[],[]
    for i in range(data.shape[1]):
        if data[frame,i,1] is not None and data[frame,i,2] is not None and data[frame,i,3] is not None:
            #print(data[frame,i,4])
            #x.append(data[frame,i,1])
            #y.append(data[frame,i,4])
            #z.append(data[frame,i,2])
            #xd.append(data[frame,i,1])
            #yd.append(data[frame,i,4])
            #zd.append(data[frame,i,2])


            if i==68:
                xd.append(data[frame,i,1])
                yd.append(data[frame,i,4])
                zd.append(data[frame,i,2])
            else:
                x.append(data[frame,i,1])
                y.append(data[frame,i,4])
                z.append(data[frame,i,2])
        else:
            print(i)
    scatter._offsets3d = (x,y,z)
    scatterd._offsets3d = (xd,yd,zd)
    ax.set_title(f"Frame {frame}")

# アニメーションの作成
anim = FuncAnimation(fig, update, frames=data.shape[0], interval=50, blit=False)

# アニメーションを保存する
save_dir = Path('C:/Users/Tomson/BRLAB/tooth/Temporomandibular_movement/movie/2023_06_05')
anim.save(save_dir/'animation.mp4', fps=30)