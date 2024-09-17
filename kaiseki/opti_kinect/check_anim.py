import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# データの読み込み
df_diagonal_right = pd.read_csv(r'C:\Users\Tomson\.vscode\PythonFile\df_diagonal_right_1.csv', index_col=0)
df_diagonal_left = pd.read_csv(r'C:\Users\Tomson\.vscode\PythonFile\df_diagonal_left_1.csv', index_col=0)

# インデックスの最小値と最大値を取得
min_index = min(df_diagonal_right.index.min(), df_diagonal_left.index.min())
max_index = max(df_diagonal_right.index.max(), df_diagonal_left.index.max())

# 新しいインデックスの範囲を作成
full_index_range = pd.Index(range(min_index, max_index + 1))

# 各データフレームに存在しないインデックスを追加し、その要素を0に設定
df_diagonal_right = df_diagonal_right.reindex(full_index_range, fill_value=0)
df_diagonal_left = df_diagonal_left.reindex(full_index_range, fill_value=0)

columns = df_diagonal_right.columns

# x, y, z座標を取得
x_right = [col for col in columns if col.endswith('_x')]
df_diagonal_right_x = df_diagonal_right[x_right]
y_right = [col for col in columns if col.endswith('_y')]
df_diagonal_right_y = df_diagonal_right[y_right]
z_right = [col for col in columns if col.endswith('_z')]
df_diagonal_right_z = df_diagonal_right[z_right]

x_left = [col for col in columns if col.endswith('_x')]
df_diagonal_left_x = df_diagonal_left[x_left]
y_left = [col for col in columns if col.endswith('_y')]
df_diagonal_left_y = df_diagonal_left[y_left]
z_left = [col for col in columns if col.endswith('_z')]
df_diagonal_left_z = df_diagonal_left[z_left]

# アニメーション作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

current_frame = 150  # 初期フレーム

def update(frame):
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([min(df_diagonal_right_x.values.min(), df_diagonal_left_x.values.min()),
                 max(df_diagonal_right_x.values.max(), df_diagonal_left_x.values.max())])
    ax.set_ylim([min(df_diagonal_right_y.values.min(), df_diagonal_left_y.values.min()),
                 max(df_diagonal_right_y.values.max(), df_diagonal_left_y.values.max())])
    ax.set_zlim([min(df_diagonal_right_z.values.min(), df_diagonal_left_z.values.min()),
                 max(df_diagonal_right_z.values.max(), df_diagonal_left_z.values.max())])
    ax.set_title(f'Frame {frame}')
    ax.set_aspect('equal')


    ax.scatter(df_diagonal_right_x.iloc[frame, :], df_diagonal_right_y.iloc[frame, :], df_diagonal_right_z.iloc[frame, :], c='r', label='Right')
    ax.scatter(df_diagonal_left_x.iloc[frame, :], df_diagonal_left_y.iloc[frame, :], df_diagonal_left_z.iloc[frame, :], c='b', label='Left')
    ax.legend()

def on_key(event):
    global current_frame
    if event.key == 'right':
        current_frame = min(current_frame + 1, len(df_diagonal_right) - 1)
    elif event.key == 'left':
        current_frame = max(current_frame - 1, 0)
    update(current_frame)
    fig.canvas.draw()

# キーボードイベントを登録
fig.canvas.mpl_connect('key_press_event', on_key)

# 初期フレームを表示
update(current_frame)
plt.show()
