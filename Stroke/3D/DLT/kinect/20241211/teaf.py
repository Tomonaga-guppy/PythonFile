import pandas as pd
import numpy as np

# データフレームの作成
data = {'TEST': [321, 431, 431, 413, 524, 1541, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 62, 1514, 3421, 455, 134, 531, 6243, np.nan, np.nan, np.nan, 614, 246, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6225, 452]}
df = pd.DataFrame(data)

# 欠損値のインデックスを取得
is_missing = df['TEST'].isnull()

# 連続する欠損値に同じ番号を付けるためのグループIDを作成
group_id = (is_missing != is_missing.shift()).cumsum()

# グループごとに、連続する欠損値の数をカウント
consecutive_missing_counts = is_missing.groupby(group_id).cumsum()

# 連続欠損数が閾値以上のグループ番号を取得
long_missing_group_numbers = consecutive_missing_counts[consecutive_missing_counts >= 5].groupby(is_missing).first().index

# 連続欠損数が閾値以上のグループに属するインデックスを取得
long_missing_indices = df.groupby(is_missing).filter(lambda x: x.name in long_missing_group_numbers).index.tolist()

print(f"元のデータフレーム:\n{df}")
print(f"5つ以上連続して欠損している箇所のインデックス: {long_missing_indices}")