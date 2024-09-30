import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# モーキャプデータ
mocap_data = pd.DataFrame({
    'time': np.arange(0, 10, 0.01),
    'angle': np.sin(np.arange(0, 10, 0.01))  # サンプルの関節角度データ
})

# Kinectデータ（0.5秒のシフトを持つサンプルデータ）
kinect_data = pd.DataFrame({
    'time': np.arange(0, 10, 0.01),
    'angle': np.sin(np.arange(0, 10, 0.01) + 0.5)  # 0.5秒シフト
})

# 相互相関の計算関数
def compute_cross_correlation(mocap, kinect):
    mocap_angles = mocap['angle']
    kinect_angles = kinect['angle']

    # 相互相関を計算
    cross_corr = np.correlate(mocap_angles - mocap_angles.mean(),
                              kinect_angles - kinect_angles.mean(),
                              mode='full')

    # 最大相関を持つラグを見つける
    lag = cross_corr.argmax() - (len(mocap_angles) - 1)

    return lag

# ラグの計算
lag = compute_cross_correlation(mocap_data, kinect_data)
print(f"相互相関によるラグ: {lag} サンプル")

# ステップ 3: データの同期調整
kinect_data_synced = kinect_data.copy()
kinect_data_synced['angle'] = kinect_data['angle'].shift(lag).fillna(method='bfill')

# 同期後のデータを可視化
plt.figure(figsize=(10, 6))
plt.plot(mocap_data['time'], mocap_data['angle'], label='Mocap Data')
plt.plot(kinect_data['time'], kinect_data['angle'], label='Kinect Data (Before Sync)', linestyle='dashed')

# シフト後のデータを描画
plt.plot(kinect_data['time'], kinect_data_synced['angle'], label='Kinect Data (Synced)', linestyle='dashdot')

plt.legend()
plt.title('Mocap vs Kinect Data (Before and After Sync)')
plt.xlabel('Time [s]')
plt.ylabel('Joint Angle [degrees]')
plt.show()
