import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.stats.power as smp

# エクセルファイルからデータを読み込む
file_path = 'path_to_your_excel_file.xlsx'  # エクセルファイルのパスを指定
sheet_name = 'Sheet1'  # 読み込むシート名を指定

# 指定範囲でデータを読み込む
data = pd.read_excel(file_path, sheet_name=sheet_name, usecols="AE:AF", skiprows=185, nrows=16)

# 読み込んだデータを確認
print(data.head())

# データ列を指定
data1 = data.iloc[:, 0]  # AE列のデータを取得
data2 = data.iloc[:, 1]  # AF列のデータを取得

# スピアマンの相関係数とp値を計算
correlation, p_value = stats.spearmanr(data1, data2)
print(f"Spearman's correlation: {correlation}")
print(f"p-value: {p_value}")

# 効果量（スピアマンの相関係数が効果量とみなせます）
effect_size = correlation
print(f"Effect size: {effect_size}")

# 検出力を計算
n = len(data1)
alpha = 0.05  # 有意水準

# 検出力を計算（一般的な相関の場合）
power_analysis = smp.NormalIndPower()
power = power_analysis.solve_power(effect_size=effect_size, nobs=n, alpha=alpha)
print(f"Power: {power}")
