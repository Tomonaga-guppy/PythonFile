import os
import json
import pandas as pd
import glob

# --- 1. フォルダ選択 ---
selpath = r'G:\gait_pattern\20250811_br\sub1\thera1-0\fr_yoloseg\openpose.json'
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
output_path = r'G:\gait_pattern\20250811_br\sub1\thera1-0\fr_yoloseg'
os.makedirs(output_path, exist_ok=True) # 保存先フォルダがなければ作成
filename = os.path.join(output_path, f"{title}.csv")

df.to_csv(filename, index=True, header=True)

print(f"処理が完了しました。ファイルは {filename} に保存されました。")