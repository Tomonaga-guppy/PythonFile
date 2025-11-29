import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. データの作成（振幅が増大する正弦波 + ノイズ + 誤検出スパイク）
np.random.seed(42)
t = np.linspace(0, 10, 200)
signal = t * np.sin(t * 2)  # 増大する信号
base_noise = np.random.normal(0, 0.5, size=len(t)) # 通常のノイズ

# 誤検出（スパイク）をランダムに混入させる
spikes = np.zeros_like(t)
spike_indices = np.random.choice(len(t), 10, replace=False)
spikes[spike_indices] = np.random.choice([-10, 10], 10) # 大きな外れ値

data = signal + base_noise + spikes

# 2. トレンド線の作成比較
window_size = 15

# A: 移動平均（Mean）- スパイクに弱い
trend_mean = pd.Series(data).rolling(window=window_size, center=True).mean()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 画像のような「階段状 ＋ スパイク ＋ ジッター」データの作成
np.random.seed(42)
frames = np.arange(250, 450)
n_points = len(frames)

# ベースの信号（階段状に下がる動きを模倣）
# 緩やかに下がりつつ、時々値を保持する(階段)挙動
true_signal = np.linspace(2200, 1300, n_points)
for i in range(1, n_points):
    if i % 40 < 20: # 定期的に値を「保持」させて階段状にする
        true_signal[i] = true_signal[i-1]

# ノイズの追加
# A: 全体的な細かいジッター
jitter = np.random.normal(0, 15, n_points)

# B: 特定区間の激しい振動 (Frame 330-350付近)
jitter[80:100] += np.random.normal(0, 50, 20)

# C: スパイク（外れ値）の追加 (Frame 380, 410付近の急激なドロップ)
spikes = np.zeros(n_points)
spikes[130] = -400 # Frame 380付近
spikes[160] = -300 # Frame 410付近
spikes[165] = 200  # 上へのスパイクも少し

# 合成データ
raw_data = true_signal + jitter + spikes

# 2. フィルタ処理の比較
window_size = 15 # スパイクの幅より十分大きく設定

# 移動平均 (Mean)
trend_mean = pd.Series(raw_data).rolling(window=window_size, center=True).mean()

# 移動中央値 (Median) - ★推奨手法
trend_median = pd.Series(raw_data).rolling(window=window_size, center=True).median()

# 3. 可視化
plt.figure(figsize=(12, 6))

# 元データ
plt.plot(frames, raw_data, 'b-', alpha=0.4, label='Raw Data (Blue Line mimic)')

# 移動平均
plt.plot(frames, trend_mean, 'r--', linewidth=1.5, label='Moving Mean (Sensitive to spikes)')

# メディアンフィルタ
plt.plot(frames, trend_median, 'g-', linewidth=2.5, label='Moving Median (Robust Trend)')

plt.title('Performance on "Stair-step + Spike" Data')
plt.xlabel('Frame')
plt.ylabel('Coordinate [px]')
plt.legend()
plt.grid(True)

plt.show()
# B: 移動中央値（Median）- スパイクに強い
trend_median = pd.Series(data).rolling(window=window_size, center=True).median()

# 3. プロット
plt.figure(figsize=(12, 6))
plt.plot(t, data, 'k.', label='Raw Data (with Spikes)', alpha=0.5)
plt.plot(t, signal, 'k--', label='True Signal', alpha=0.3)

# 移動平均の結果
plt.plot(t, trend_mean, 'r-', linewidth=2, label='Moving Mean (Affected by Spikes)')

# メディアンフィルタの結果
plt.plot(t, trend_median, 'g-', linewidth=2, label='Moving Median (Robust!)')

plt.title('Robustness Check: Mean vs Median Filter')
plt.legend()
plt.grid(True)
plt.show()