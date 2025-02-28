import json
import csv
from tqdm import tqdm
import pickle
import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline

keypoints_name = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow",
                "LWrist","MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle",
                "REye","LEye","REar","LEar","LBigToe","LSmallToe","LHeel","RBigToe",
                    "RSmallToe","RHeel"]

def mkCSVOpenposeData(openpose_dir, overwrite=True):
    # print(f"openpose_dir:{openpose_dir}")
    condition = (openpose_dir.stem).split("_op")[0]
    # すでにcsvファイルがあれば処理を終了する
    output_csvs = [openpose_dir.with_name("keypoints2d_"  + condition  + f"_{i}.csv") for i in range(2)]
    if overwrite and (output_csvs[0].exists() or output_csvs[1].exists()):
            print(f"csvファイル{output_csvs}を上書きします。")
            for output_csv in output_csvs:
                output_csv.unlink()
    elif overwrite == False and output_csvs[0].exists() and output_csvs[1].exists():
        print(f"以前の{output_csvs}を再利用します。")
        return output_csvs

    json_files = list(openpose_dir.glob("*.json"))
    if len(json_files) == 0:
        print(f"jsonファイルが見つかりませんでした。")
        return

    csv_header = []

    for keypoint in keypoints_name:
        for n in ("x","y","p"):
            csv_header.append(f"{keypoint}_{n}")
    csv_header.insert(0, "frame_num")

    header_write_list = [False, False]
    for ijson, json_file in tqdm(enumerate(json_files), total=len(json_files)):

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
                pose_keypoints_2d_str.insert(0, str(ijson))
                writer.writerow(pose_keypoints_2d_str)
    print(f"OpenPose結果をcsvファイルに保存しました。")
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
    print(f"condition_list:{condition_list}")
    with open(frame_ch_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(condition_list)
    print(f"{frame_ch_csv}を作成しました。")









def load_keypoints_for_frame(json_file_path):
    # jsonファイルを読み込んで[25,3]のnumpy配列を返す
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
        if len(json_data['people']) < 1:  #人が検出されなかった場合はnanで埋める
            # keypoints_data = np.zeros((25, 3))
            keypoints_data = np.full((25, 3), np.nan)
        else:
            json_data = np.array(json_data['people'][0]['pose_keypoints_2d'])  #[75,]
            keypoints_data = json_data.reshape((25, 3))
    return keypoints_data

def butter_lowpass_fillter(data, order, cutoff_freq, frame_list):  #4次のバターワースローパスフィルタ
    sampling_freq = 30
    nyquist_freq = sampling_freq / 2
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data[frame_list])
    data_fillter = np.copy(data)
    data_fillter[frame_list] = y
    return data_fillter

def cubic_spline_interpolation(keypoints_set, frame_range):
    # 新しい配列を作成して補間結果を保持
    interpolated_keypoints = np.copy(keypoints_set)

    for axis in range(keypoints_set.shape[1]):
        # 指定されたフレーム範囲のデータを取り出す
        frames = frame_range
        values = np.nan_to_num(keypoints_set[frames, axis])

        # フレーム範囲のフレームを基準に3次スプラインを構築
        spline = CubicSpline(frames, values)

        # 補間した値をその範囲のフレームに再適用
        interpolated_values = spline(frames)
        interpolated_keypoints[frames, axis] = interpolated_values

    return interpolated_keypoints

def linear_interpolation(x, x0, x1, y0, y1):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def read_2d_openpose(json_folder, camParams_dict):
    all_keypoints_2d = []  # 各フレームの2Dキーポイントを保持するリスト
    check_openpose_list = [1, 8, 12, 13, 14, 19, 20, 21]
    valid_frames = []
    for i, json_file in enumerate(json_folder.glob("*.json")):
        keypoints_data = load_keypoints_for_frame(json_file)
        undistort_points = cv2.undistortPoints(np.array([keypoints_data[:, 0], keypoints_data[:, 1]]).T, camParams_dict["intrinsicMat"], camParams_dict["distortion"])
        # print(f"undistort_points.shape:{undistort_points.shape}")  #(25, 1, 2)
        keypoints_data[:, 0] = undistort_points[:, 0, 0]
        keypoints_data[:, 1] = undistort_points[:, 0, 1]
        p = keypoints_data[:, 2]  #openposeが算出した信頼度

        # キーポイント抽出が出来ているフレームを記録
        if all(not np.all(np.isnan(keypoints_data[point, :])) for point in check_openpose_list):
            valid_frames.append(i)

        #確率が0.5未満のキーポイントをnanに変換
        threshold = 0.
        for j in range(len(p)):
            if p[j] < threshold:
                keypoints_data[j] = [np.nan, np.nan]
        all_keypoints_2d.append(keypoints_data)
        #keypoints_dataの0をnanに変換
        keypoints_data[keypoints_data == 0] = np.nan
        all_keypoints_2d.append(keypoints_data)
    keypoints_2d_openpose = np.array(all_keypoints_2d)
    print(f"keypoints_2d_openpose.shape:{keypoints_2d_openpose.shape}")

    return keypoints_2d_openpose, valid_frames

def calculate_angle(vector1, vector2):  #(frame, xyz)または(frame, xy)の配列を入力)
    angle_list = []
    for frame in range(len(vector1)):
        dot_product = np.dot(vector1[frame], vector2[frame])
        cross_product = np.cross(vector1[frame], vector2[frame])
        angle = np.arctan2(np.linalg.norm(cross_product), dot_product)
        angle = angle * 180 / np.pi
        angle_list.append(angle)

    return angle_list