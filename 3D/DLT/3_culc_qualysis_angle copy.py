import pandas as pd
import numpy as np
import glob
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import json

down_hz = False
csv_path_dir = Path(r"F:\Tomson\gait_pattern\20240911\qualisys")
csv_paths = list(csv_path_dir.glob("sub2*.tsv"))

def read_3DMC(csv_path, down_hz):
    col_names = range(1, 100)  # データの形が汚い場合に対応するためあらかじめ列数(100:適当)を設定
    df = pd.read_csv(csv_path, names=col_names, sep='\t', skiprows=[0,1,2,3,4,5,6,7,8,10])  # Qualysis
    df.columns = df.iloc[0]  # 最初の行をヘッダーに
    df = df.drop(0).reset_index(drop=True)  # ヘッダーにした行をデータから削除し、インデックスをリセット

    if down_hz:
        df_down = df[::4].reset_index()
        sampling_freq = 30
    else:
        df_down = df
        sampling_freq = 120

    marker_set = ["RASI", "LASI", "RPSI", "LPSI", "RKNE", "LKNE", "RANK", "LANK", "RTOE", "LTOE", "RHEE", "LHEE", "RKNE2", "LKNE2", "RANK2", "LANK2"]
    marker_dict = dict()
    xyz_list = ['X', 'Y', 'Z']

    for marker in marker_set:
        for i, xyz in enumerate(xyz_list):
            key_index = df_down.columns.get_loc(f'{marker}')
            marker_rows = (key_index - 1) * 3 + i
            marker_dict[f"{marker}_{xyz}"] = marker_rows

    marker_set_df = pd.DataFrame(columns=marker_dict.keys())
    for column in marker_set_df.columns:
        marker_set_df[column] = df_down.iloc[:, marker_dict[column]].values

    marker_set_df = marker_set_df.apply(pd.to_numeric, errors='coerce')  # 文字列として読み込まれたデータを数値に変換
    marker_set_df.replace(0, np.nan, inplace=True)  # 0をnanに変換

    df_copy = marker_set_df.copy()
    valid_index_mask = df_copy.notna().all(axis=1)
    valid_index = df_copy[valid_index_mask].index
    valid_index = pd.Index(range(valid_index.min(), valid_index.max() + 1))

    interpolated_df = marker_set_df.interpolate(method='spline', order=3)  # 3次スプライン補間
    marker_set_fin_df = interpolated_df.apply(butter_lowpass_fillter, args=(4, 6, sampling_freq))  # 4次のバターワースローパスフィルタ

    # 保存のパス処理もpathlibで
    output_csv_path = csv_path.parent / f"marker_set_{csv_path.stem}.csv"
    marker_set_fin_df.to_csv(output_csv_path)

    return marker_set_fin_df, valid_index

def butter_lowpass_fillter(column_data, order, cutoff_freq, sampling_freq):  # 4次のバターワースローパスフィルタ
    nyquist_freq = sampling_freq / 2
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, column_data)
    return filtered_data

def main():
    for csv_path in csv_paths:
        marker_set_df, valid_index = read_3DMC(csv_path, down_hz)

        print(f"csv_path = {csv_path}")
        print(f"valid_index = {valid_index}")

        angle_list = []
        bector_list = []
        dist_list = []

        # 以下、省略された計算部分...

        # 保存のパス処理もpathlibで
        output_angle_csv_path = csv_path.parent / f"angle_120Hz_{csv_path.stem}.csv" if not down_hz else csv_path.parent / f"angle_30Hz_{csv_path.stem}.csv"
        df.to_csv(output_angle_csv_path)

        # npyファイルの保存もpathlibで
        output_ic_frame_npy_path = csv_path.parent / f"ic_frame_120Hz_{csv_path.stem}.npy" if not down_hz else csv_path.parent / f"ic_frame_30Hz_{csv_path.stem}.npy"
        np.save(output_ic_frame_npy_path, filtered_list)

if __name__ == "__main__":
    main()
