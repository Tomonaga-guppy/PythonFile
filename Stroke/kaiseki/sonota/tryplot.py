import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline


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
index = df_diagonal_right.index

fig,ax = plt.subplots()