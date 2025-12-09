import os
import json
import pandas as pd
import glob
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import numpy as np


# --- 1. フォルダ選択 ---
selpath = r'G:\gait_pattern\BR9G_shuron\sub1\thera1-0\ICMST\fr_\openpose.json'
print(f"選択されたフォルダ: {selpath}")
current_folder_name = os.path.basename(selpath)

# jsonのファイル名を編集 朝長
title = current_folder_name.replace(".json", "")

# --- 2. 座標データを取得 ---
# ファイルリストを取得し、名前順にソートする
# これにより、フレームの順序が保証される
json_files = sorted(glob.glob(os.path.join(selpath, '*.json')))
count = len(json_files)

# 抽出したデータを保存するためのリストを初期化
df_data = []

# openposeのkeypointリスト
keypoint_names = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
                  "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
                  "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
                  "RBigToe", "RSmallToe", "RHeel"]

""" openpose keypoint index
//     {0,  "Nose"},
//     {1,  "Neck"},
//     {2,  "RShoulder"},
//     {3,  "RElbow"},
//     {4,  "RWrist"},
//     {5,  "LShoulder"},
//     {6,  "LElbow"},
//     {7,  "LWrist"},
//     {8,  "MidHip"},
//     {9,  "RHip"},
//     {10, "RKnee"},
//     {11, "RAnkle"},
//     {12, "LHip"},
//     {13, "LKnee"},
//     {14, "LAnkle"},
//     {15, "REye"},
//     {16, "LEye"},
//     {17, "REar"},
//     {18, "LEar"},
//     {19, "LBigToe"},
//     {20, "LSmallToe"},
//     {21, "LHeel"},
//     {22, "RBigToe"},
//     {23, "RSmallToe"},
//     {24, "RHeel"},
"""

# 複数のjsonを順番に読み込む
for i in range(count):
    filepath = json_files[i]
    with open(filepath, 'r') as f:
        val = json.load(f)

    # 座標を一時的に格納する辞書
    coords_dict = {}

    # 人物が検知されているか確認
    if val['people']:
        keypoints = val['people'][0]['pose_keypoints_2d']

        for index, keypoint_name in enumerate(keypoint_names):
            for i_type, type in enumerate(['x', 'y', 'p']):
                key = f"{keypoint_name}_{type}"
                coords_dict[key] = keypoints[index*3 + i_type]

    else:
        # 人物が検知されなかった場合、すべての座標を0にする
        coords_dict = {f"{keypoint_name}_{type}": 0 for keypoint_name in keypoint_names for type in ['x', 'y', 'p']}
        
    df_data.append(coords_dict)

# --- 3. excelに書き込み ---
# データをPandas DataFrameに変換
df = pd.DataFrame(df_data)

# ★ 保存先のパスを記入
output_path = os.path.dirname(selpath)
filename = os.path.join(output_path, f"{title}.csv")

df.to_csv(filename, index=True, header=True)

print(f"処理が完了しました。ファイルは {filename} に保存されました。")

df_cubic = df.copy()
# 座標が0のところはnanに変換して三次スプライン補間
df_cubic.replace(0, np.nan, inplace=True)

df_cubic.interpolate(method='spline', order=3, limit_direction='both', inplace=True)


df_butter = df_cubic.copy()

# --- 4. バターワースローパスフィルタを適用 ---
# フィルタパラメータ
fs = 60  # サンプリング周波数 (Hz) ※動画のフレームレートに合わせて調整してください
cutoff = 6  # カットオフ周波数 (Hz)
order = 4  # フィルタ次数

# バターワースフィルタの係数を計算
nyquist = fs / 2
normalized_cutoff = cutoff / nyquist
b, a = butter(order, normalized_cutoff, btype='low')

# 各列にフィルタを適用（'p'(確信度)列を除く座標データのみ）
for col in df_butter.columns:
    if not col.endswith('_p'):  # 確信度列はフィルタリングしない
        # 0でない値が十分にある場合のみフィルタを適用
        if (df_butter[col] != 0).sum() > order * 3:
            df_butter[col] = filtfilt(b, a, df_butter[col])

# フィルタ適用後のデータを保存
filename_butter = os.path.join(output_path, f"{title}_butter.csv")
df_butter.to_csv(filename_butter, index=True, header=True)

print(f"フィルタ適用後のファイルは {filename_butter} に保存されました。")

# --- 5. 元データとフィルタ適用後のデータを比較プロット ---
# プロット保存用フォルダを作成
plot_folder = os.path.join(output_path, f"{title}_plots")
os.makedirs(plot_folder, exist_ok=True)

# フレーム数（時間軸）
frames = range(len(df))

# 各キーポイントごとにプロット
for keypoint_name in keypoint_names:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # X座標のプロット
    x_col = f"{keypoint_name}_x"
    axes[0].plot(frames, df[x_col], label='Original', alpha=0.7, linewidth=1)
    axes[0].plot(frames, df_cubic[x_col], label='Cubic Spline', alpha=0.7, linewidth=1)
    axes[0].plot(frames, df_butter[x_col], label='Butterworth', alpha=0.9, linewidth=1.5)
    axes[0].set_ylabel('X coordinate (px)')
    axes[0].set_ylim(0, 3840)
    axes[0].set_title(f'{keypoint_name} - X coordinate')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Y座標のプロット
    y_col = f"{keypoint_name}_y"
    axes[1].plot(frames, df[y_col], label='Original', alpha=0.7, linewidth=1)
    axes[1].plot(frames, df_cubic[y_col], label='Cubic Spline', alpha=0.7, linewidth=1)
    axes[1].plot(frames, df_butter[y_col], label='Butterworth', alpha=0.9, linewidth=1.5)
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Y coordinate (px)')
    axes[1].set_ylim(0, 2160)
    axes[1].set_title(f'{keypoint_name} - Y coordinate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # プロットを保存
    plot_filename = os.path.join(plot_folder, f"{keypoint_name}.png")
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    
    print(f"  {keypoint_name} のプロットを保存しました。")

print(f"\nすべてのプロットは {plot_folder} に保存されました。")