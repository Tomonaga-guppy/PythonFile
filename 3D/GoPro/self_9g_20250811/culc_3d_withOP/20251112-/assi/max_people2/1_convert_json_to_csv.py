import os
import json
import pandas as pd
import glob

# --- 1. フォルダ選択 ---
selpath = r'G:\gait_pattern\20250811_br\sub1\thera1-0\fr\openpose.json'
print(f"選択されたフォルダ: {selpath}")
current_folder_name = os.path.basename(selpath)

# jsonのファイル名を編集
title = current_folder_name.replace(".json", "")

# --- 2. 座標データを取得 ---
# ファイルリストを取得し、名前順にソートする
json_files = sorted(glob.glob(os.path.join(selpath, '*.json')))
count = len(json_files)

# openposeのkeypointリスト
keypoint_names = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
                  "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
                  "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
                  "RBigToe", "RSmallToe", "RHeel"]

# --- 2-1. まず全JSONファイルをスキャンして最大人数を把握 ---
print("全JSONファイルをスキャン中...")
max_people_detected = 0

for json_file in json_files:
    with open(json_file, 'r') as f:
        val = json.load(f)
    num_people = len(val['people'])
    max_people_detected = max(max_people_detected, num_people)

print(f"検出された最大人数: {max_people_detected}")

# --- 2-2. 各人物ごとにデータを格納する辞書を初期化 ---
# all_people_data[person_idx] = [frame0のデータ, frame1のデータ, ...]
all_people_data = {person_idx: [] for person_idx in range(max_people_detected)}

# --- 2-3. 各JSONファイルを順番に処理 ---
print("座標データを抽出中...")
for frame_idx, json_file in enumerate(json_files):
    with open(json_file, 'r') as f:
        val = json.load(f)

    num_people = len(val['people'])

    # 各人物インデックスについて処理（0からmax_people_detected-1まで）
    for person_idx in range(max_people_detected):
        if person_idx < num_people:
            # この人物が検出された場合
            keypoints = val['people'][person_idx]['pose_keypoints_2d']
            
            # 座標を一時的に格納する辞書
            coords_dict = {}
            for index, keypoint_name in enumerate(keypoint_names):
                for i_type, coord_type in enumerate(['x', 'y', 'p']):
                    key = f"{keypoint_name}_{coord_type}"
                    coords_dict[key] = keypoints[index*3 + i_type]
        else:
            # この人物が検出されなかった場合、すべての座標を0にする
            coords_dict = {f"{keypoint_name}_{coord_type}": 0 
                          for keypoint_name in keypoint_names 
                          for coord_type in ['x', 'y', 'p']}
        
        all_people_data[person_idx].append(coords_dict)

# --- 3. 複数のCSVファイルに書き込み ---
output_path = r'G:\gait_pattern\20250811_br\sub1\thera1-0\fr'
os.makedirs(output_path, exist_ok=True)

print("\nCSVファイルを保存中...")
saved_files = []
for person_idx in sorted(all_people_data.keys()):
    df = pd.DataFrame(all_people_data[person_idx])
    filename = os.path.join(output_path, f"{title}_person{person_idx}.csv")
    df.to_csv(filename, index=True, header=True)
    saved_files.append(filename)
    
    # データが0でないフレーム数をカウント（実際に検出されたフレーム数）
    detected_frames = 0
    for row_data in all_people_data[person_idx]:
        # Nose_xが0でない場合、検出されたとみなす
        if row_data['Nose_x'] != 0:
            detected_frames += 1
    
    print(f"Person {person_idx}: 総フレーム数={len(df)}, 検出フレーム数={detected_frames}")

print(f"\n処理が完了しました。")
print(f"総JSONファイル数（総フレーム数）: {count}")
print(f"検出された最大人数: {max_people_detected}")
print(f"\n保存されたファイル:")
for file in saved_files:
    print(f"  - {file}")

print(f"\n注意: すべてのCSVファイルは {count} 行のデータを持ちます。")
print(f"     検出されなかったフレームは座標が0で埋められています。")