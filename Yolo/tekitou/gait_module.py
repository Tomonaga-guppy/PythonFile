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
    # print(f"all_people_num:{all_people_num}")

    # すでにcsvファイルがあれば処理を終了する
    output_csvs = [openpose_dir.with_name("keypoints2d_"  + condition  + f"_{i}.csv") for i in range(all_people_num)]
    if overwrite:
        for output_csv in output_csvs:
            if output_csv.exists():
                # print(f"csvファイル{output_csvs}を上書きします。")
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
    # print(f"csvファイル:{output_csvs}")
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
    # print(f"header:{header}")
    with open(frame_ch_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow(condition_list)
        writer.writerow(header)
    # print(f"{frame_ch_csv}を作成しました。")

def checkSwithing(openpose_df_dict):
    # print(f"openpose_df_dict:{openpose_df_dict}")
    # 2人以上いる場合は腰のキーポイントのx座標が前の方を1人目のデータとした
    # print(f"openpose_df_dict:{openpose_df_dict}")
    keys = list(openpose_df_dict.keys())  #人のidをリストで取得
    people_num = len(keys)
    # 格納されているdfのフレーム数を取得
    if people_num > 0: #人が1人以上いる場合
        first_key = list(openpose_df_dict.keys())[0]  # 最初のキーを取得
        frames = openpose_df_dict[first_key].index #最初のdfのindexを取得
        # print(f"frames:{frames}")
    else:
        return openpose_df_dict  #人が0人の場合はそのまま返す

    new_openpose_df_dict = {key: {} for key in keys}
    for key in keys:
        new_openpose_df_dict[key] = openpose_df_dict[key].copy()


    for frame in frames:
        # print(f"frame:{frame}")
        #midhip_x座標を比較して小さい法から順に並べ替え
        skip = False
        for key in keys:
            # print(f"key:{key}")
            try:
                a = new_openpose_df_dict[key].loc[frame, "MidHip_x"]
            except:
                if key == "0": target = "1"
                else: target = "0"
                new_openpose_df_dict[target].drop(index=frame, inplace=True)
                # print(f"{frame}フレームには2人のデータがありません。")
                skip = True
        if skip:
            continue


        mid_hip_x_list = []
        for key in keys:
            # print(f"key:{key}")
            openpose_df = openpose_df_dict[key]
            mid_hip_x = openpose_df.loc[frame, "MidHip_x"]
            mid_hip_x_list.append([mid_hip_x, openpose_df.loc[frame, :]])
        # if frame == 1645:
        #     print(f"mid_hip_x_list 元々:{mid_hip_x_list}")
        mid_hip_x_list.sort(key=lambda x: x[0])  #昇順に並べ替え
        # if frame == 1645:
        #     print(f"mid_hip_x_list 昇順:{mid_hip_x_list}\n")

        for key in keys:
            # if frame == 1645:
            #     print(f"key:{key}")
            #     print(f"mid_hip_x_list[int(key)][1]:{mid_hip_x_list[int(key)][1]}")
            # openpose_df_dict[key].loc[frame, :] = mid_hip_x_list[int(key)][1]
            new_openpose_df_dict[key].loc[frame, :] = mid_hip_x_list[int(key)][1]
    return new_openpose_df_dict
    # return openpose_df_dict

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
        # print(f"spline処理中: iPeople:{iPeople}")
        iPeople = str(iPeople)
        # 座標が0で信頼度が閾値よりも低い部分をnanに変換、nanの位置も記録しておく
        df_dict_couvert_nan, nan_mask = convert_nan(openpose_df_dicf[iPeople], threshhold=0.2)
        df = pd.DataFrame()
        for name in openpose_df_dicf[iPeople].columns:
            data_not_nan = df_dict_couvert_nan[name].dropna()
            # print(f"name:{name}, data_not_nan:{data_not_nan}")
            if data_not_nan.empty:  #nanしかない場合はそのまま返す
                # openpose_df_dict_spline[name] = openpose_df_dicf.copy()[iPeople][name]
                series = openpose_df_dicf[iPeople][name]
                df[name] = series

            elif len(data_not_nan) == 1:  #1つだけ値がある場合はそのまま返す
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

def calc_dist_pel2heel(df):
    dict_dist_pel2heel = {}
    for iPeople in range(len(df)):
        iPeople = str(iPeople)
        # かかとと骨盤の座標を取得
        RHeel_x = df[iPeople].loc[:, "RHeel_x"]
        LHeel_x = df[iPeople].loc[:, "LHeel_x"]
        MidHip_x = df[iPeople].loc[:, "MidHip_x"]
        dist_Rx = RHeel_x - MidHip_x
        dist_Lx = LHeel_x - MidHip_x
        dict_dist_pel2heel[iPeople] = {"dist_Rx": dist_Rx, "dist_Lx": dist_Lx}
    return dict_dist_pel2heel

def find_initial_contact(dict_dist_pel2heel, condition, root_dir):
    ic_frame_dict = {}
    to_frame_dict = {}
    for iPeople in range(len(dict_dist_pel2heel)):
        iPeople = str(iPeople)
        dist_Rx = dict_dist_pel2heel[iPeople]["dist_Rx"].copy()
        dist_Lx = dict_dist_pel2heel[iPeople]["dist_Lx"].copy()
        start_frame = dist_Rx.index[0]
        print(f"start_frame:{start_frame}")

        # かかとが地面に接地するタイミングを初期接地として記録
        for dist_df, side in zip([dist_Rx, dist_Lx], ["R", "L"]):
            dist_df_asc = dist_df.sort_values()
            ic_check_list = dist_df_asc.index[:120].values
            filler_list = []
            skip_list = []
            for check_frame in ic_check_list:
                if check_frame in skip_list:
                    continue
                filler_list.append(check_frame)
                [skip_list.append(sk_frame) for sk_frame in range(check_frame-30, check_frame+30)]
            filler_list.sort()

            #ここけす！
            # filler_list = [icframe - start_frame for icframe in filler_list]

            if iPeople not in ic_frame_dict:
                ic_frame_dict[iPeople] = {}
            ic_frame_dict[iPeople][f"IC_{side}"] = filler_list
            # print(f"filler_list_{side}:{filler_list}")

            #つま先接地のタイミングを記録
            dist_df_desc = dist_df.sort_values(ascending=False)
            to_check_list = dist_df_desc.index[:120].values
            filler_list_to = []
            skip_list_to = []
            for check_frame in to_check_list:
                if check_frame in skip_list_to:
                    continue
                filler_list_to.append(check_frame)
                [skip_list_to.append(sk_frame) for sk_frame in range(check_frame-30, check_frame+30)]
            filler_list_to.sort()

            #ここけす！
            # filler_list_to = [icframe - start_frame for icframe in filler_list_to]

            if iPeople not in to_frame_dict:
                to_frame_dict[iPeople] = {}
            to_frame_dict[iPeople][f"TO_{side}"] = filler_list_to
            # print(f"kari_{side}:{kari}")

    # print(f"ic_frame_dict:{ic_frame_dict}")

    for iPeople in range(len(dict_dist_pel2heel)):
        iPeople = str(iPeople)
        dist_Rx = dict_dist_pel2heel[iPeople]["dist_Rx"].copy()
        dist_Lx = dict_dist_pel2heel[iPeople]["dist_Lx"].copy()
        plt.plot(dist_Rx.reset_index(drop=True), label="R")
        plt.plot(dist_Lx.reset_index(drop=True), label="L")
        plt.legend()
        save_path = root_dir / f"{condition}_{iPeople}_IC.png"
        plt.title(f"{condition}_{iPeople}")
        plt.savefig(save_path)
        # plt.show()
        plt.cla()

    return ic_frame_dict, to_frame_dict

def check_edge_frame(event_frame_list):
    # フレームリストの端の値が異常値の場合は削除
    diffs = np.diff(event_frame_list)
    median_diff = np.median(abs(diffs))
    threshold = median_diff * 1.2
    if abs(diffs[0]) > threshold:
        event_frame_list = event_frame_list[1:]
    if abs(diffs[-1]) > threshold:
        event_frame_list = event_frame_list[:-1]
    event_frame_list_filtered = event_frame_list
    return event_frame_list_filtered

def calc_stride_time(ic_frame_list, side, fps):
    """指定された側の初期接地フレームリストから歩行周期の平均時間と標準偏差を計算"""
    list = ic_frame_list[side]
    #リスト内の要素間の差を計算
    stride_time_frame = [list[i+1] - list[i] for i in range(len(list)-1)]
    avg_stride_frame = np.mean(stride_time_frame)
    std_frame = np.std(stride_time_frame)
    avg_stride_time = avg_stride_frame / fps
    std_time = std_frame / fps
    return avg_stride_time, std_time

def calc_walk_params(stride_time, pixpermm, ic_frame_dict, df_ft):
    Rcycle = ic_frame_dict["IC_R"]
    Lcycle = ic_frame_dict["IC_L"]
    Rcycle_block = [[Rcycle[i], Rcycle[i+1]] for i in range(len(Rcycle)-1)]
    Lcycle_block = [[Lcycle[i], Lcycle[i+1]] for i in range(len(Lcycle)-1)]
    if Rcycle_block[0][0] > Lcycle_block[0][0]:
        print("左足から接地開始")
        start_ic_left = True
    else:
        start_ic_left = False
        print("右足から接地開始")

    walk_speed_list = []
    stride_length_list_l = []
    step_length_list_l = []
    stride_length_list_l = []
    step_length_list_r = []

    #左足の歩行パラメータを計算
    for i, block in enumerate(Lcycle_block):
        mid_hip_start_x, mid_hip_start_y = df_ft.loc[block[0], "MidHip_x"], df_ft.loc[block[0], "MidHip_y"]
        mid_hip_end_x, mid_hip_end_y = df_ft.loc[block[1], "MidHip_x"], df_ft.loc[block[1], "MidHip_y"]
        norm_mid_hip = np.sqrt((mid_hip_end_x - mid_hip_start_x)**2 + (mid_hip_end_y - mid_hip_start_y)**2)
        print(f"i:{i}, block:{block}")
        print(f"norm_mid_hip:{norm_mid_hip}")
        print(f"stride_time:{stride_time}")
        Lheel_start_x, Lheel_start_y = df_ft.loc[block[0], "LHeel_x"], df_ft.loc[block[0], "LHeel_y"]
        Lheel_end_x, Lheel_end_y = df_ft.loc[block[1], "LHeel_x"], df_ft.loc[block[1], "LHeel_y"]
        walk_speed = (norm_mid_hip * pixpermm / 1000) / stride_time #どちらもミリ単位
        print(f"walk_speed:{walk_speed}")
        stride_length_l = np.sqrt((Lheel_end_x - Lheel_start_x)**2 + (Lheel_end_y - Lheel_start_y)**2) * pixpermm / 1000
        if start_ic_left:  #左足から接地開始
            try:
                ic_right_frame = Rcycle_block[i][0]
            except:
                print("右足のサイクルがなくなったので計算を終了します。")
                continue
        else:  #右足から接地開始
            try:
                ic_right_frame = Rcycle_block[i][1]
            except:
                print("左足のサイクルがなくなったので計算を終了します。")
                continue
        # ステップの計算が逆
        step_length_l = np.sqrt((df_ft.loc[ic_right_frame, "RHeel_x"] - df_ft.loc[block[0], "LHeel_x"])**2 + (df_ft.loc[ic_right_frame, "RHeel_y"] - df_ft.loc[block[0], "LHeel_y"])**2) * pixpermm / 1000
        walk_speed_list.append(walk_speed)
        stride_length_list_l.append(stride_length_l)
        step_length_list_l.append(step_length_l)
    print(f"Lcycle_block:{Lcycle_block}")
    print(f"Rcycle_block:{Rcycle_block}")
    #右足の歩行パラメータを計算
    for i, block in enumerate(Rcycle_block):
        Rheel_start_x, Rheel_start_y = df_ft.loc[block[0], "RHeel_x"], df_ft.loc[block[0], "RHeel_y"]
        Rheel_end_x, Rheel_end_y = df_ft.loc[block[1], "RHeel_x"], df_ft.loc[block[1], "RHeel_y"]
        stride_length_r = np.sqrt((Rheel_end_x - Rheel_start_x)**2 + (Rheel_end_y - Rheel_start_y)**2) * pixpermm / 1000
        if start_ic_left:  #左足から接地開始
            try:
                ic_left_frame = Lcycle_block[i][1]
            except:
                print("左足のサイクルがなくなったので計算を終了します。1")
                continue
        else:  #右足から接地開始
            try:
                ic_left_frame = Lcycle_block[i][0]
            except:
                print("左足のサイクルがなくなったので計算を終了します。2")
                continue
        # ステップの計算が逆
        step_length_r = np.sqrt((df_ft.loc[ic_left_frame, "LHeel_x"] - df_ft.loc[block[0], "RHeel_x"])**2 + (df_ft.loc[ic_left_frame, "LHeel_y"] - df_ft.loc[block[0], "RHeel_y"])**2) * pixpermm / 1000
        step_length_list_r.append(step_length_r)

    walk_speed = np.mean(walk_speed_list)
    stride_length_l = np.mean(stride_length_list_l)
    step_length_l = np.mean(step_length_list_r)  #ステップの計算が逆なので一時的に反対に入れる（要修正）
    stride_length_r = np.mean(stride_length_r)
    step_length_r = np.mean(step_length_list_l)  #ステップの計算が逆なので一時的に反対に入れる（要修正）
    std_walk_speed = np.std(walk_speed_list)
    std_stride_length_l = np.std(stride_length_list_l)
    std_step_length_l = np.std(step_length_list_r)
    std_stride_length_r = np.std(stride_length_r)
    std_step_length_r = np.std(step_length_list_l)
    return walk_speed, stride_length_l, stride_length_r, step_length_l, step_length_r, std_walk_speed, std_stride_length_l, std_stride_length_r, std_step_length_l, std_step_length_r

def calc_stance_phase_ratio(ic_frame_dict, to_frame_dict):
    stance_phase_ratio_list_r = []
    stance_phase_ratio_list_l = []
    for side in ("R", "L"):
        ic_list = ic_frame_dict[f"IC_{side}"]
        to_list = to_frame_dict[f"TO_{side}"]
        print(f"{side} ic_list:{ic_list}")
        print(f"{side} to_list:{to_list}")
        loop_num = min(len(ic_list), len(to_list))
        cycle_frame = np.mean([ic_list[i+1] - ic_list[i] for i in range(loop_num-1)])
        if ic_list[0] > to_list[0]:
            stance_phase_frame = np.mean([to_list[i+1] - ic_list[i] for i in range(loop_num-1)])
        else:
            stance_phase_frame = np.mean([to_list[i] - ic_list[i] for i in range(loop_num-1)])
        stance_phase_ratio = (stance_phase_frame / cycle_frame * 100)
        if side == "R":
            stance_phase_ratio_list_r.append(stance_phase_ratio)
        else:
            stance_phase_ratio_list_l.append(stance_phase_ratio)
    stance_phase_ratio_r = np.mean(stance_phase_ratio_list_r)
    stance_phase_ratio_l = np.mean(stance_phase_ratio_list_l)
    return stance_phase_ratio_r, stance_phase_ratio_l


def calGaitPhase(ic_frame_dict_ori, to_frame_dict_ori):
    phase_dict = {key : [] for key in ic_frame_dict_ori.keys()}
    for iPeople in range(len(ic_frame_dict_ori)):
        ic_frame_dict = ic_frame_dict_ori[f"{iPeople}"]
        to_frame_dict = to_frame_dict_ori[f"{iPeople}"]
        phase_frame_list = []
        for i in range(len(ic_frame_dict["IC_L"])):
            try:
                IC_l_side_frame = ic_frame_dict["IC_L"][i]
                TO_r_side_frame = [to_frame for to_frame in to_frame_dict["TO_R"] if to_frame > IC_l_side_frame][0]
                IC_r_side_frame = [ic_frame for ic_frame in ic_frame_dict["IC_R"] if ic_frame > TO_r_side_frame][0]
                To_l_side_frame = [to_frame for to_frame in to_frame_dict["TO_L"] if to_frame > IC_r_side_frame][0]
                Next_IC_l_side_frame = [ic_frame for ic_frame in ic_frame_dict["IC_L"] if ic_frame > To_l_side_frame][0]

                if Next_IC_l_side_frame > [ic_frame for ic_frame in ic_frame_dict["IC_L"] if ic_frame > IC_l_side_frame][0]:
                    continue
                # phase_frame_list.append([IC_l_side_frame, TO_r_side_frame, IC_r_side_frame, To_l_side_frame, Next_IC_l_side_frame])
                phase_frame_list.append([IC_l_side_frame, TO_r_side_frame, To_l_side_frame, Next_IC_l_side_frame])
            except:
                continue
        print(f"phase_frame_list:{phase_frame_list}")
        phase_dict[f"{iPeople}"] = phase_frame_list
    return phase_dict

def calGaitPhasePercent(phase_frame_list_dict):
    phase_percent_list_dict = {key : [] for key in phase_frame_list_dict.keys()}
    for iPeople in range(len(phase_frame_list_dict)):
        phase_frame_list = phase_frame_list_dict[f"{iPeople}"]
        phase_percent_list_res = []
        for i, phase_frames in enumerate(phase_frame_list):
            phase_percent_list = []
            for i in range(len(phase_frames)):
                phase_percent = (phase_frames[i] - phase_frames[0]) / (phase_frames[-1] - phase_frames[0]) * 100
                phase_percent_list.append(phase_percent)
            phase_percent_list_res.append(phase_percent_list)
        phase_percent_list_dict[f"{iPeople}"] = phase_percent_list_res
    return phase_percent_list_dict

def extract_keypoints(dict_ft):
    """患者側の使用するキーポイントを抽出し、辞書型で格納"""
    keypoints_dict = {}
    keypoints_dict["Neck"] = dict_ft["0"].loc[:, ["Neck_x", "Neck_y"]].copy()
    keypoints_dict["MidHip"] = dict_ft["0"].loc[:, ["MidHip_x", "MidHip_y"]].copy()
    keypoints_dict["RHip"] = dict_ft["0"].loc[:, ["RHip_x", "RHip_y"]].copy()
    keypoints_dict["LHip"] = dict_ft["0"].loc[:, ["LHip_x", "LHip_y"]].copy()
    keypoints_dict["RKnee"] = dict_ft["0"].loc[:, ["RKnee_x", "RKnee_y"]].copy()
    keypoints_dict["LKnee"] = dict_ft["0"].loc[:, ["LKnee_x", "LKnee_y"]].copy()
    keypoints_dict["RAnkle"] = dict_ft["0"].loc[:, ["RAnkle_x", "RAnkle_y"]].copy()
    keypoints_dict["LAnkle"] = dict_ft["0"].loc[:, ["LAnkle_x", "LAnkle_y"]].copy()
    keypoints_dict["RBigToe"] = dict_ft["0"].loc[:, ["RBigToe_x", "RBigToe_y"]].copy()
    keypoints_dict["LBigToe"] = dict_ft["0"].loc[:, ["LBigToe_x", "LBigToe_y"]].copy()
    keypoints_dict["RSmallToe"] = dict_ft["0"].loc[:, ["RSmallToe_x", "RSmallToe_y"]].copy()
    keypoints_dict["LSmallToe"] = dict_ft["0"].loc[:, ["LSmallToe_x", "LSmallToe_y"]].copy()
    keypoints_dict["RHeel"] = dict_ft["0"].loc[:, ["RHeel_x", "RHeel_y"]].copy()
    keypoints_dict["LHeel"] = dict_ft["0"].loc[:, ["LHeel_x", "LHeel_y"]].copy()

    print(f"keypoints_dict:{keypoints_dict.keys()}")

    return keypoints_dict

def calc_joint_angle(keypoints_dict):
    """関節角度を計算して辞書型で格納"""
    # 使用するキーポイントの抽出  valuesのためフレーム数の情報が飛んでいる
    trunk_bec = keypoints_dict["Neck"].values - keypoints_dict["MidHip"].values
    thigh_r_bec = keypoints_dict["RKnee"].values - keypoints_dict["RHip"].values
    thigh_l_bec = keypoints_dict["LKnee"].values - keypoints_dict["LHip"].values
    lower_leg_r_bec = keypoints_dict["RKnee"].values - keypoints_dict["RAnkle"].values
    lower_leg_l_bec = keypoints_dict["LKnee"].values - keypoints_dict["LAnkle"].values
    foot_r_bec = (keypoints_dict["RBigToe"].values + keypoints_dict["RSmallToe"].values) /2 - keypoints_dict["RHeel"].values
    foot_l_bec = (keypoints_dict["LBigToe"].values + keypoints_dict["LSmallToe"].values) /2 - keypoints_dict["LHeel"].values

    # 関節角度の計算
    joint_angle_dict = {}
    def _calc_angle(v1, v2):
        """ベクトルv1とv2のなす角を計算 フレーム数は元データと対応していないので注意"""
        angle_array = np.zeros(v1.shape[0])
        for false_frame in range(v1.shape[0]):
            if np.linalg.norm(v1[false_frame]) == 0 or np.linalg.norm(v2[false_frame]) == 0:
                angle = np.nan
            dot_product = np.dot(v1[false_frame], v2[false_frame])
            cross_product = np.cross(v1[false_frame], v2[false_frame])
            angle = np.rad2deg(np.arctan2(cross_product, dot_product))
            angle_array[false_frame] = angle
        # print(f"angle_array_shape:{angle_array.shape}")
        # print(f"angle_array:{angle_array}")
        return angle_array

    Hip_r_angle = _calc_angle(thigh_r_bec, trunk_bec)
    Hip_l_angle = _calc_angle(thigh_l_bec, trunk_bec)
    Knee_r_angle = _calc_angle(thigh_r_bec, lower_leg_r_bec)
    Knee_l_angle = _calc_angle(thigh_l_bec, lower_leg_l_bec)
    Ankle_r_angle = _calc_angle(foot_r_bec, lower_leg_r_bec)
    Ankle_l_angle = _calc_angle(foot_l_bec, lower_leg_l_bec)

    Hip_r_angle = np.where(Hip_r_angle < 0, Hip_r_angle + 360, Hip_r_angle)
    Hip_l_angle = np.where(Hip_l_angle < 0, Hip_l_angle + 360, Hip_l_angle)
    Knee_r_angle = np.where(Knee_r_angle < 0, Knee_r_angle + 360, Knee_r_angle)
    Knee_l_angle = np.where(Knee_l_angle < 0, Knee_l_angle + 360, Knee_l_angle)
    Ankle_r_angle = np.where(Ankle_r_angle < 0, Ankle_r_angle + 360, Ankle_r_angle)
    Ankle_l_angle = np.where(Ankle_l_angle < 0, Ankle_l_angle + 360, Ankle_l_angle)
    joint_angle_dict["Hip_r"] = Hip_r_angle - 180
    joint_angle_dict["Hip_l"] = 180 - Hip_l_angle
    # joint_angle_dict["Knee_r"] = Knee_r_angle
    # joint_angle_dict["Knee_l"] = Knee_l_angle
    joint_angle_dict["Knee_r"] = 180 - Knee_r_angle
    joint_angle_dict["Knee_l"] = 180 - Knee_l_angle
    joint_angle_dict["Ankle_r"] = Ankle_r_angle - 90
    joint_angle_dict["Ankle_l"] = 90 - Ankle_l_angle
    return joint_angle_dict

def changedict2df(joint_angle_dict, to_index):
    """辞書型の関節角度データをDataFrameに変換する関数"""
    df = pd.DataFrame()
    for key, value in joint_angle_dict.items():
        # print(f"key:{key}, value:{value}")
        df[key] = value
    df.index = to_index
    # print(f"df:{df}")
    return df

def plot_angle(angle, frame, ylim, title, save_path):
    """
    関節角度をプロットする関数
    Args:
        angle: 関節角度の配列
        ylim: y軸の範囲
        title: グラフのタイトル
        save_path: 保存先のパス
    """
    # print(f"angle:{angle}")

    plt.figure(figsize=(10, 6))
    plt.plot(frame, angle, label=title)
    plt.ylim(ylim)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.savefig(save_path)
    plt.close()  # プロットを閉じる

def joint_agnle_devided_by_cycle(joint_angle_df, phase_frame_list):
    """
    関節角度を歩行サイクルごとに分割して保存する関数
    Args:
        joint_angle_df: 関節角度のDataFrame
        phase_frame_list_dict: 歩行サイクルのフレームリスト辞書
        save_dir: 保存先のディレクトリ
    """

    def _frame2percent(df):
        """フレームをパーセントに変換する関数"""
        angle_percent_array = np.zeros((len(df.columns), 100))
        ori_idx = df.index
        normalized_idx = (ori_idx - ori_idx[0]) / (ori_idx[-1] - ori_idx[0]) * 100
        for i, name in enumerate(df.columns):
            angle_data = df[name].values
            new_start_idx = 0
            new_end_idx = 100
            num_points = 100
            new_idx = np.linspace(new_start_idx, new_end_idx, num_points)
            interpolated_data = np.interp(new_idx, normalized_idx, angle_data).T
            # print(f"angle_percent_array.shape:{angle_percent_array.shape}")
            # print(f"interpolated_data.shape:{interpolated_data.shape}")
            angle_percent_array[i][:] = interpolated_data
        # print(f"angle_percent_array:{angle_percent_array}")
        # print(f"angle_percent_array_last.shape:{angle_percent_array.shape}")
        return angle_percent_array

    #結果を格納するnp配列を用意 形状：サイクル数 x 角度の種類 x ポイント数
    all_angle_array = np.zeros((len(phase_frame_list), len(joint_angle_df.columns), 100))
    for icycle, cycle_frames in enumerate(phase_frame_list):
        print(f"icycle:{icycle}")
        # if icycle == len(phase_frame_list) - 1:
        #     continue
        start_frame = cycle_frames[0]
        end_frame = cycle_frames[-1]
        joint_angle_df_cycle = joint_angle_df.loc[start_frame:end_frame].copy()
        joint_angle_percent_array = _frame2percent(joint_angle_df_cycle)
        all_angle_array[icycle, :, :] = joint_angle_percent_array
    # print(f"all_angle_array:{all_angle_array}")
    # print(f"all_angle_array.shape:{all_angle_array.shape}")
    return all_angle_array

def plot_angle_mean_std(all_angle_mean, all_angle_std, idx, angle_name, save_path):
    """
    関節角度の平均と標準偏差をプロットする関数
    Args:
        all_angle_mean: 関節角度の平均値
        all_angle_std: 関節角度の標準偏差
        angle_name: 関節名
        save_path: 保存先のパス
    """
    # print(f"all_angle_mean:{all_angle_mean}")
    # print(f"all_angle_std:{all_angle_std}")

    plt.figure(figsize=(10, 6))
    plt.ylim(-35, 75)  # y軸の範囲を設定
    plt.plot(range(0,100), all_angle_mean, label=f"{angle_name}", color='blue')
    plt.fill_between(range(len(all_angle_mean)), all_angle_mean - all_angle_std, all_angle_mean + all_angle_std, color='blue', alpha=0.2, label=f"{angle_name} Std")
    plt.title(f"{angle_name} Angle", fontsize=25)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Gait Cycle (%)", fontsize=20)
    plt.ylabel("Angle (degrees)", fontsize=20)
    # plt.legend()
    plt.savefig(save_path)
    # plt.show()  # グラフを表示
    plt.close()  # プロットを閉じる

