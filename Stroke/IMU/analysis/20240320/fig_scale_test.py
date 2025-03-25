import matplotlib.pyplot as plt
import numpy as np

# データ作成
x = np.linspace(0, 10, 20)
y = np.sin(x)

# Figure と Axes を作成
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# (0, 0): デフォルト (自動スケーリング + マージン)
axes[0, 0].plot(x, y)
axes[0, 0].set_title('Default (Autoscaling + Margin)')

# (0, 1): 軸範囲を明示的に設定 + マージンなし
axes[0, 1].plot(x, y)
axes[0, 1].set_xlim(0, 10)
axes[0, 1].set_ylim(-1, 1)
axes[0, 1].margins(0)  # マージンをなくす
axes[0, 1].set_title('Explicit Limits + No Margin')

# (1, 0): 自動スケーリングを無効化
axes[1, 0].plot(x, y)
axes[1, 0].set_xlim(0, 10)
axes[1, 0].set_ylim(-1, 1)
axes[1, 0].autoscale(False)  # 自動スケーリングを無効化
axes[1, 0].set_title('Autoscaling Off')

# (1,1): use_sticky_edges
axes[1, 1].plot(x, y)
axes[1, 1].set_xlim(0, 10)
axes[1, 1].set_ylim(-1, 1)
print(axes[1,1].get_xbound())
print(axes[1,1].get_ybound())
axes[1,1].use_sticky_edges = [False, False] # デフォルトはTrue
print(axes[1,1].get_xbound())
print(axes[1,1].get_ybound())
axes[1, 1].set_title('use_sticky_edges')

plt.tight_layout()
plt.show()