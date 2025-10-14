import os
import json
import pandas as pd
import glob
from tkinter import Tk, filedialog

# =============================================================================
# %% メインスクリプト
# =============================================================================

print("処理するJSONファイルが含まれるフォルダを選択してください...")

# --- 1. 処理するjsonフォルダの選択 ---
# Tkinterを使用してフォルダ選択ダイアログを表示
root = Tk()
root.withdraw()  # 小さなTkウィンドウが表示されるのを防ぐ
# ★もしダイアログが不要な場合は、以下の行をコメントアウトし、
#   selpath に直接パスをハードコーディングしてください。
#   例: selpath = r'D:\...\json\your_folder'
selpath = filedialog.askdirectory(
    initialdir=r'G:\gait_pattern\20251014_nakazaki_pythontest\sub4_com_nfpa_2_op.json' # ←←←←←←←←jsonファイルがあるフォルダの親パスを記入★
)

if not selpath:
    print("フォルダが選択されませんでした。処理を終了します。")
else:
    print(f"選択されたフォルダ: {selpath}")
    current_folder_name = os.path.basename(selpath)
    # excelのファイル名を編集
    title = current_folder_name.replace("use_", "")
    
    # jsonのファイル名を編集 朝長
    title = title.replace("_op.json", "")

    # --- 2. 座標データを取得 ---
    # ファイルリストを取得し、名前順にソートする
    # これにより、フレームの順序が保証される
    json_files = sorted(glob.glob(os.path.join(selpath, '*.json')))
    count = len(json_files)

    # 抽出したデータを保存するためのリストを初期化
    right_leg_data = []
    left_leg_data = []

    # 複数のjsonを順番に読み込む
    for i in range(count):
        filepath = json_files[i]
        with open(filepath, 'r') as f:
            val = json.load(f)

        # 座標を一時的に格納する辞書
        r_coords = {}
        l_coords = {}

        # 人物が検知されているか確認
        if val['people']:
            keypoints = val['people'][0]['pose_keypoints_2d']

            # 座標の取得 (Pythonのインデックスは0から始まるため-1する)
            # 右脚
            r_coords['hip_x'] = keypoints[9*3 + 0]
            r_coords['hip_y'] = keypoints[9*3 + 1]
            r_coords['knee_x'] = keypoints[10*3 + 0]
            r_coords['knee_y'] = keypoints[10*3 + 1]
            r_coords['ankle_x'] = keypoints[11*3 + 0]
            r_coords['ankle_y'] = keypoints[11*3 + 1]
            r_coords['bigtoe_x'] = keypoints[22*3 + 0]
            r_coords['bigtoe_y'] = keypoints[22*3 + 1]
            r_coords['heel_x'] = keypoints[24*3 + 0]
            r_coords['heel_y'] = keypoints[24*3 + 1]
            
            # 左脚
            l_coords['hip_x'] = keypoints[12*3 + 0]
            l_coords['hip_y'] = keypoints[12*3 + 1]
            l_coords['knee_x'] = keypoints[13*3 + 0]
            l_coords['knee_y'] = keypoints[13*3 + 1]
            l_coords['ankle_x'] = keypoints[14*3 + 0]
            l_coords['ankle_y'] = keypoints[14*3 + 1]
            l_coords['bigtoe_x'] = keypoints[19*3 + 0]
            l_coords['bigtoe_y'] = keypoints[19*3 + 1]
            l_coords['heel_x'] = keypoints[21*3 + 0]
            l_coords['heel_y'] = keypoints[21*3 + 1]

            # いずれかの座標が0の場合、その脚のすべての座標を0にする
            if any(v == 0 for v in r_coords.values()):
                r_coords = {key: 0 for key in r_coords}
            
            if any(v == 0 for v in l_coords.values()):
                l_coords = {key: 0 for key in l_coords}
        else:
            # 人物が検知されなかった場合、すべての座標を0にする
            r_coords = {'hip_x': 0, 'hip_y': 0, 'knee_x': 0, 'knee_y': 0, 'ankle_x': 0, 'ankle_y': 0, 'bigtoe_x': 0, 'bigtoe_y': 0, 'heel_x': 0, 'heel_y': 0}
            l_coords = {'hip_x': 0, 'hip_y': 0, 'knee_x': 0, 'knee_y': 0, 'ankle_x': 0, 'ankle_y': 0, 'bigtoe_x': 0, 'bigtoe_y': 0, 'heel_x': 0, 'heel_y': 0}

        right_leg_data.append(r_coords)
        left_leg_data.append(l_coords)

    # --- 3. excelに書き込み ---
    # データをPandas DataFrameに変換
    df_right = pd.DataFrame(right_leg_data)
    df_right = df_right[['hip_x', 'hip_y', 'knee_x', 'knee_y', 'ankle_x', 'ankle_y', 'bigtoe_x', 'bigtoe_y', 'heel_x', 'heel_y']] # 列の順序を定義

    df_left = pd.DataFrame(left_leg_data)
    df_left = df_left[['hip_x', 'hip_y', 'knee_x', 'knee_y', 'ankle_x', 'ankle_y', 'bigtoe_x', 'bigtoe_y', 'heel_x', 'heel_y']] # 列の順序を定義

    # ★ 保存先のパスを記入
    output_path = r'G:\gait_pattern\20251014_nakazaki_pythontest\json_excel'
    os.makedirs(output_path, exist_ok=True) # 保存先フォルダがなければ作成
    filename = os.path.join(output_path, f"{title}.xlsx")

    # Excelファイルに書き込み
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df_right.to_excel(writer, sheet_name='Sheet1', index=False, header=False)
        df_left.to_excel(writer, sheet_name='Sheet2', index=False, header=False)

    print(f"処理が完了しました。ファイルは {filename} に保存されました。")