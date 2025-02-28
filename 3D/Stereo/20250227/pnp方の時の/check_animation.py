import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np  # 必要に応じて

# --- 設定 ---
csv_file = r"G:\gait_pattern\20241126_br9g\gopro\3Dkeypoints_ngait_1_person0.csv"

keypoint_names = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow",
                "LWrist","MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle",
                "REye","LEye","REar","LEar","LBigToe","LSmallToe","LHeel","RBigToe",
                    "RSmallToe","RHeel"]
interval = 100  # アニメーションの更新間隔 (ミリ秒)

# --- データ読み込みと前処理 ---


df = pd.read_csv(csv_file)
print(f"df:{df}")


# --- 3Dプロットの準備 ---

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 軸ラベル
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Keypoint Animation')

# 軸範囲 (データ全体から計算)
x_cols = [f'{name}_Y' for name in keypoint_names]
y_cols = [f'{name}_Z' for name in keypoint_names]
z_cols = [f'{name}_X' for name in keypoint_names]

x_min, x_max = df[x_cols].min().min(), df[x_cols].max().max()
y_min, y_max = df[y_cols].min().min(), df[y_cols].max().max()
z_min, z_max = df[z_cols].min().min(), df[z_cols].max().max()

print(f"x_min: {x_min}, x_max: {x_max}")
print(f"y_min: {y_min}, y_max: {y_max}")
print(f"z_min: {z_min}, z_max: {z_max}")

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)


# フレーム数表示用のテキストオブジェクト
frame_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

# --- アニメーション更新関数 ---

def update(frame_num):
    ax.cla()  # 現在の軸をクリア (軸ラベル、タイトル、範囲は再設定)
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_zlabel('X')
    ax.set_title(f'3D Keypoint Animation (Frame: {frame_num})')
    ax.set_xlim(-10000, 10000)
    ax.set_ylim(-10000, 10000)
    ax.set_zlim(-10000, 10000)
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # ax.set_zlim(z_min, z_max)

    # 現在のフレームのデータを取得
    row = df[df['frame_num'] == frame_num].iloc[0]

    # X, Y, Z座標をリストに格納
    x_coords = [row[f'{name}_Y'] for name in keypoint_names]
    y_coords = [row[f'{name}_Z'] for name in keypoint_names]
    z_coords = [row[f'{name}_X'] for name in keypoint_names]

    # 3D散布図としてプロット
    ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')

    # フレーム数表示を更新
    frame_text.set_text(f'Frame: {frame_num}')

    return []

# --- アニメーション作成 ---

ani = animation.FuncAnimation(
    fig, update, frames=[1200], interval=interval, blit=False)
    # fig, update, frames=df['frame_num'].unique(), interval=interval, blit=False)


plt.tight_layout()
plt.show()