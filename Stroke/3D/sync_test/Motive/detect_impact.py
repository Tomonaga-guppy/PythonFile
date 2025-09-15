import pandas as pd
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample
import json

def butter_lowpass_filter(data, cutoff_freq, sampling_freq, order=4):
    """
    データに4次のバターワースローパスフィルタを適用する。
    """
    # pandasのSeriesやDataFrameの列をNumPy配列に変換
    data_array = data.to_numpy()
    
    nyquist_freq = sampling_freq / 2
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # filtfiltを適用してフィルタリング
    y = filtfilt(b, a, data_array)
    return y

def read_3d_optitrack(csv_path):
    """
    OptiTrackの3Dデータを読み込む
    """
    df = pd.read_csv(csv_path, skiprows=[0, 1, 2, 4], header=[0, 2])
    marker_set = ["Label"]
    marker_set_df = df[[col for col in df.columns if any(marker in col[0] for marker in marker_set)]].copy()
    return marker_set_df

def main():
    csv_path_dir = r"G:\gait_pattern\20250915_synctest\Motive\fault"
    csv_path = glob.glob(os.path.join(csv_path_dir, "*004.csv"))[0]
    df_original = read_3d_optitrack(csv_path)
    
    # Y列のデータを抽出し、DataFrameを作成
    y_series = df_original.loc[:, ('MarkerSet 01:Label', 'Y')]
    df_y = y_series.to_frame(name='Y')
    
    df_y['Y_filtered'] = butter_lowpass_filter(df_y['Y'], 
                                               cutoff_freq=6, 
                                               sampling_freq=100, 
                                               order=4)
    
    print("DataFrame after filtering:")
    print(df_y.head())
    
    df_y['Y_filtered_diff'] = df_y['Y_filtered'].diff().fillna(0)

    # ----------------------------------------------------------------
    # 2. グラフ描画
    # ----------------------------------------------------------------
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Butterworth Filter Effect on Y Position', fontsize=16)

    # 上のグラフ (ax1): 元のY座標とフィルタ後のY座標を比較
    ax1.plot(df_y['Y'], label='Original Y', color='silver', marker='o', markersize=4)
    ax1.plot(df_y['Y_filtered'], label='Filtered Y (6Hz Low-pass)', color='royalblue', marker='o', markersize=4)
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Original vs. Filtered Y Position')
    ax1.grid(True)
    ax1.legend()

    # 下のグラフ (ax2): "フィルタ後の" Y座標の差分
    ax2.plot(df_y['Y_filtered_diff'], label='Filtered Y Difference', color='orange', marker='o', markersize=4)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Y Position Difference (m)')
    ax2.set_title('Frame-to-Frame Difference of Filtered Data')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()