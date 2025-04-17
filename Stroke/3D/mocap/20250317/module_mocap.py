import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt


def read_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t', header=10)  #Qualisys
    df = df.apply(pd.to_numeric, errors='coerce')  # 数値に変換
    df.set_index("Frame", inplace=True)  # フレームをインデックスに設定
    df.index = df.index - 1  # フレームを0から始まるように調整(GoProと合わせやすくするため)
    df.dropna(axis=1, how='all', inplace=True)  # 全ての値がNaNの行を削除
    # print(f"df: {df.columns}")
    df = df.filter(regex='RASI|LASI|RPSI|LPSI|RKNE|LKNE|RANK|LANK|RTOE|LTOE|RHEE|LHEE')
    # print(f"filetered_df: {df.columns}")
    return df

def butterworth_filter(df, cutoff, order, fs):
    def _butter_lowpass_fillter(column_data, order, cutoff, fs):
        nyquist_freq = fs / 2
        normal_cutoff = cutoff / nyquist_freq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        data_to_filter = column_data
        filtered_data = filtfilt(b, a, data_to_filter)
        column_data = filtered_data
        return column_data

    butter_df = df.copy()
    butter_df = butter_df.apply(_butter_lowpass_fillter, args=(order, cutoff, fs))  #4次のバターワースローパスフィルタ
    return butter_df


def frame2percent(acc_frame_series):
    ori_idx = acc_frame_series.index.to_numpy()
    normalized_ori_idx = (ori_idx - ori_idx[0]) / (ori_idx[-1] - ori_idx[0]) * 100  # 横軸を0~100に正規化
    acc_data = acc_frame_series.to_numpy()
    # 0~99で1刻みになるようリサンプリング
    new_start_idx = 0
    new_end_idx = 100
    num_points = 100
    new_idx = np.linspace(new_start_idx, new_end_idx, num_points)
    new_acc_data = np.interp(new_idx, normalized_ori_idx, acc_data)
    # print(f"new_acc_data:{new_acc_data}")
    return new_acc_data