import json
import csv
from tqdm import tqdm
import pickle
import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

keypoints_name = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow",
                "LWrist","MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle",
                "REye","LEye","REar","LEar","LBigToe","LSmallToe","LHeel","RBigToe",
                    "RSmallToe","RHeel"]

def mkCSVOpenposeData(openpose_dir, start_frame, overwrite=True):
    # print(f"openpose_dir:{openpose_dir}")
    condition = (openpose_dir.stem).split("_op")[0]


    json_files = list(openpose_dir.glob("*.json"))
    if len(json_files) == 0:
        print(f"jsonファイルが見つかりませんでした。")
        return

    people_num = []
    for json_file in json_files:
        with open(json_file, "r") as f_json:
            json_data = json.load(f_json)
        people_num.append(len(json_data["people"]))
    all_people_num = np.unique(people_num).max()
    print(f"all_people_num:{all_people_num}")

    # すでにcsvファイルがあれば処理を終了する
    output_csvs = [openpose_dir.with_name("keypoints2d_"  + condition  + f"_{i}.csv") for i in range(all_people_num)]
    if overwrite:
        for output_csv in output_csvs:
            if output_csv.exists():
                print(f"csvファイル{output_csvs}を上書きします。")
                output_csv.unlink()
    elif overwrite == False and output_csvs[0].exists() and output_csvs[1].exists():
        print(f"以前の{output_csvs}を再利用します。")
        return output_csvs

    csv_header = []

    for keypoint in keypoints_name:
        for n in ("x","y","p"):
            csv_header.append(f"{keypoint}_{n}")
    csv_header.insert(0, "frame_num")

    header_write_list = [False, False]
    for ijson, json_file in tqdm(enumerate(json_files), total=len(json_files)):
        frame = ijson + start_frame
        with open(json_file, "r") as f_json:
            json_data = json.load(f_json)
        all_people_data = json_data["people"]
        for ipeople, data in enumerate(all_people_data):
            # person_id = people["person_id"]  #これを使いたいがなぜかすべて-1になる
            # person_id = str(ipeople)
            output_csv = output_csvs[ipeople]
            with open(output_csv, "a", newline="") as f:
                writer = csv.writer(f)
                if not header_write_list[ipeople]:
                    writer.writerow(csv_header)
                    header_write_list[ipeople] = True
                pose_keypoints_2d = data["pose_keypoints_2d"]
                pose_keypoints_2d_str = [str(value) for value in pose_keypoints_2d]
                pose_keypoints_2d_str.insert(0, str(frame))
                writer.writerow(pose_keypoints_2d_str)
    print(f"OpenPose結果をcsvファイルに保存しました。")
    print(f"csvファイル:{output_csvs}")
    # print(f"all_people_data:{len(all_people_data)}")
    return output_csvs

def undistordOpenposeData(openpose_df, CamPrams_dict):
    for keypoint_name in keypoints_name:
        points = np.array([openpose_df[keypoint_name+"_x"], openpose_df[keypoint_name+"_y"]]).T
        # undistort_points.shape = (frame, 1, 2(xy))
        undistort_points =cv2.undistortPoints(points, CamPrams_dict["intrinsicMat"], CamPrams_dict["distortion"], P=CamPrams_dict["intrinsicMat"])
        openpose_df[keypoint_name+"_x"] = undistort_points[:,:,0]
        openpose_df[keypoint_name+"_y"] = undistort_points[:,:,1]
    return openpose_df

def fillindex(df):
    # frame_num を整数としてリセット（インデックスになっている場合）
    df = df.reset_index()
    # 連続するフレーム番号の範囲を作成
    all_frames = pd.DataFrame({'frame_num': np.arange(df['frame_num'].min(), df['frame_num'].max() + 1)})
    # 欠損フレームを 0 で埋める
    df_filled = all_frames.merge(df, on='frame_num', how='left').fillna(0)
    # frame_num を再びインデックスに
    df_filled.set_index('frame_num', inplace=True)
    return df_filled

def loadCameraParameters(filename):
    with open(filename
              , "rb") as f:
        CameraParams_dict = pickle.load(f)
    return CameraParams_dict

def mkFrameCheckCSV(frame_ch_csv, condition_list):
    header = []
    for c in condition_list:
        for check in ["Start", "End"]:
            header.append(f"{c}_{check}")
    print(f"header:{header}")
    with open(frame_ch_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow(condition_list)
        writer.writerow(header)
    print(f"{frame_ch_csv}を作成しました。")

def convert_nan(df, threshhold):
    for name in keypoints_name:
        #座標が0で信頼度が閾値よりも低い部分をnanに変換、nanの位置も記録しておく
        nan_mask = (df[name+"_x"] == 0) & (df[name+"_y"] == 0) & (df[name+"_p"] < threshhold)
        df.loc[nan_mask, name+"_x"] = np.nan
        df.loc[nan_mask, name+"_y"] = np.nan
    return df, nan_mask

def spline_interpolation(openpose_df_dicf):
    openpose_df_dict_spline = {}
    for iPeople in range(len(openpose_df_dicf)):
        iPeople = str(iPeople)
        # 座標が0で信頼度が閾値よりも低い部分をnanに変換、nanの位置も記録しておく
        df_dict_couvert_nan, nan_mask = convert_nan(openpose_df_dicf[iPeople], threshhold=0.2)
        df = pd.DataFrame()
        for name in openpose_df_dicf[iPeople].columns:
            data_not_nan = df_dict_couvert_nan[name].dropna()
            if data_not_nan.empty:  #nanしかない場合はそのまま返す
                # openpose_df_dict_spline[name] = openpose_df_dicf.copy()[iPeople][name]
                series = openpose_df_dicf[iPeople][name]
                df[name] = series
            else:
                # print(f"name{name}, data_not_nan:{data_not_nan}")
                spline = CubicSpline(data_not_nan.index, data_not_nan.values)
                # openpose_df_dict_spline[name] = spline(df_dict_couvert_nan.index)
                series = pd.Series(spline(df_dict_couvert_nan.index), index=df_dict_couvert_nan.index)
                df[name] = series
        openpose_df_dict_spline[iPeople] = df
    # print(f"openpose_df_dict_spline:{openpose_df_dict_spline}")
    return openpose_df_dict_spline

def butter_lowpass_fillter(openpose_df_dict, sampling_freq, order, cutoff_freq):
    # print(f"openpose_df_dict:{openpose_df_dict}")
    for iPeople in range(len(openpose_df_dict)):
        iPeople = str(iPeople)
        for name in openpose_df_dict[iPeople].columns:
            nyquist_freq = sampling_freq / 2
            normal_cutoff = cutoff_freq / nyquist_freq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            y = filtfilt(b, a, openpose_df_dict[iPeople][name])
            openpose_df_dict[iPeople].loc[:, name] = y
    return openpose_df_dict

def save_as_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"{filename}を保存しました。")

def checkSwithing(openpose_df_dict):
    # 2人以上いる場合は腰のキーポイントのx座標が前の方を1人目のデータとした
    # print(f"openpose_df_dict:{openpose_df_dict}")
    keys = list(openpose_df_dict.keys())  #人のidをリストで取得
    people_num = len(keys)
    # 格納されているdfのフレーム数を取得
    if people_num > 0:  # 辞書が空でないことを確認
        first_key = list(openpose_df_dict.keys())[0]  # 最初のキーを取得
        frames = openpose_df_dict[first_key].index #最初のdfのindexを取得
        # print(f"frames:{frames}")
    else:
        return openpose_df_dict  #人が0人の場合はそのまま返す

    for frame in frames:
        #midhip_x座標を比較して小さい法から順に並べ替え
        mid_hip_x_list = []
        for key in keys:
            openpose_df = openpose_df_dict[key]
            mid_hip_x = openpose_df.loc[frame, "MidHip_x"]
            mid_hip_x_list.append([mid_hip_x, openpose_df.loc[frame, :]])
        # print(f"mid_hip_x_list:{mid_hip_x_list}\n")
        mid_hip_x_list.sort(key=lambda x: x[0])  #昇順に並べ替え
        # print(f"mid_hip_x_list:{mid_hip_x_list}")
        for key in keys:
            openpose_df_dict[key].loc[frame, :] = mid_hip_x_list[int(key)][1]
    return openpose_df_dict


def animate_keypoints(data_dict, condition, save_path, all_check):
    """
    二人のキーポイント座標をアニメーションで表示する関数。

    Args:
        data_dict: キーポイントデータを含む辞書。
                   'sub*_df_filter_dict' の形式で、
                   各値は pandas DataFrame。
    """

    keys = list(data_dict.keys())
    people_num = len(keys)
    frames = data_dict[list(data_dict.keys())[0]].index

    # FigureとAxesオブジェクトを作成
    fig, ax = plt.subplots(figsize=(10, 8))  # サイズを調整
    ax.set_aspect('equal')  # アスペクト比を保持

    # 描画範囲の計算
    x_min, x_max = 0, 3840
    y_max, y_min = 0, 2160  #わかりやすくするためy軸を反転
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    #信頼度(p)が低い場合は描画しない
    threshold = 0.2

    # 関節間の接続関係
    connections = [
        (0, 15), (15, 17), (0, 16),(16,18),       # 顔
        (0, 1), (1, 8),             # 胴体
        (1, 2), (2, 3), (3, 4),         # 右腕
        (1, 5), (5, 6), (6, 7),          # 左腕
        (8, 9), (9, 10), (10, 11),(11,22), (22,23), (11, 24),   # 右足
        (8, 12), (12, 13), (13, 14), (14, 19),(19, 20), (14, 21),   # 左足
    ]

    colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']
    lines = []
    scatters = []

    for ipeople in range(people_num):
        person_lines = [ax.plot([], [], c=colors[ipeople], lw=2, alpha=0.7)[0] for _ in connections]
        lines.append(person_lines)
    scatters = [ax.scatter([], [], s=50, label=f"Person {ipeople+1}") for ipeople in range(people_num)]
    title_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=12) #タイトル

    def update(frame_index):
        frame = frames[frame_index]
        artists = []  # 更新が必要なアーティストを格納

        for i, (key, df) in enumerate(data_dict.items()):
            x = df.loc[frame].filter(like='_x').values
            y = df.loc[frame].filter(like='_y').values
            p = df.loc[frame].filter(like='_p').values

            x_filtered = np.where(p > threshold, x, np.nan)
            y_filtered = np.where(p > threshold, y, np.nan)

            valid_indices = np.where(~np.isnan(x_filtered))[0]  # NaN でないインデックス
            scatters[i].set_offsets(np.c_[x_filtered, y_filtered])
            if all_check:
                scatters[i].set_color(colors[i % len(colors)])
            else:
                colors_indices = np.arange(len(x_filtered)) % len(colors)
                colors_array = np.array([colors[idx] for idx in colors_indices])
                full_colors = np.full(len(x_filtered), 'gray', dtype=object)  # 灰色で初期化
                full_colors[valid_indices] = colors_array[valid_indices]
                scatters[i].set_color(full_colors)
            artists.append(scatters[i])

            for j, (start, end) in enumerate(connections):
                if p[start] > threshold and p[end] > threshold:
                    lines[i][j].set_data([x[start], x[end]], [y[start], y[end]])
                else:
                    lines[i][j].set_data([], [])
                artists.append(lines[i][j])


        title_text.set_text(f'{condition} Frame: {frame}')
        artists.append(title_text) #タイトル
        return artists

    # アニメーションを作成
    ani = animation.FuncAnimation(
        fig, update, frames=range(len(frames)), interval=30, blit=False  # interval: フレーム間の時間(ミリ秒)
    )

    ax.legend()  # 凡例を表示
    plt.tight_layout()  # レイアウトの調整
    ani.save(save_path, writer="ffmpeg", fps=60)  # アニメーションを保存
    # plt.show()
    plt.close(fig)  # プロットを閉じる