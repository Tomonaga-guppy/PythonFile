# 2台のカメラそれぞれのOpenPose座標とカメラパラメータから3D座標を求める

import numpy as np
import cv2
from pathlib import Path
import _4_modules as m4
import sys
import pandas as pd

def main():
    target_facility = "tkrzk_9g"  #解析対象施設：tkrzk_9g か ota
    # target_facility = "ota"  #解析対象施設：tkrzk_9g か ota

    if target_facility == "ota":  dir = "20241114_ota_test"
    elif target_facility == "tkrzk_9g":  dir = "20241126_br9g"

    # int_cali_dir = root_dir / "int_cali" / target_facility  #内部キャリブレーション結果を保存するフォルダ
    target_cameras = ["fl","fr"]
    src_dirs = [Path(fr"G:\gait_pattern\{dir}\gopro\{target_camera}") for target_camera in target_cameras]
    condition = "ngait_1"

    # 辞書としてよく使う
    camera_dict = {"fl":[],"fr":[]}

    # 手の振り上げおよび歩行開始のフレーム番号を記録用のcsvファイルを作成
    # プログラムとは別で自分で確認して記録する(かなり定性的。startwalkは目安に使用する程度)
    frame_ch_csvs = [src_dir / ("FrameCheck_"+condition+"_"+src_dir.stem +".csv") for src_dir in src_dirs]
    [m4.mkFrameCheckCSV(frame_ch_csv) for frame_ch_csv in frame_ch_csvs if not frame_ch_csv.exists()]
    # 記録用csvファイルを読み込んでDataFrameに変換
    frame_ch_dfs = [pd.read_csv(frame_ch_csv) for frame_ch_csv in frame_ch_csvs]
    #カメラ間のフレーム差とおおよその歩行開始フレームを求める
    frame_diff = frame_ch_dfs[0].loc[0,"RiseHandFrame"] - frame_ch_dfs[1].loc[0,"RiseHandFrame"]

    # OpenPoseの結果を読み込んでCSVファイルに変換
    openpose_csv_dict = {key:[] for key in camera_dict}
    for src_dir in src_dirs:
        openpose_dir = src_dir / (condition+"_op.json")
        csv_files = m4.mkCSVOpenposeData(openpose_dir, overwrite=False)
        openpose_csv_dict[src_dir.stem] = csv_files
    # print(f"openpose_csv_dict:{openpose_csv_dict}")

    # OpenPoseの結果をDataFrameに変換(frame_diffを考慮)
    openpose_df_dict = {key:[] for key in camera_dict}
    for src_dir in src_dirs:
        for i, csv_file in enumerate(openpose_csv_dict[src_dir.stem]):
            read_df = pd.read_csv(csv_file, index_col=0)
            if frame_diff >= 0 and src_dir.stem == "fr":
                read_df = read_df.shift(periods=frame_diff)
                walk_start_frame = frame_ch_dfs[0].loc[0,"StartWalkFrame"]
            elif frame_diff < 0 and src_dir.stem == "fl":
                read_df = read_df.shift(periods=-frame_diff)
                walk_start_frame = frame_ch_dfs[1].loc[0,"StartWalkFrame"]
            openpose_df_dict[src_dir.stem].append(read_df)

    keyoiints2d_dict = m4.adjustOpenposeDF(openpose_df_dict, walk_start_frame)
    frame_range = keyoiints2d_dict["fl"][0].index

    # 前までの処理で求めたカメラパラメータを読み込む
    CamPramsPaths = [frame_ch_csvs[i].with_name(f"cameraIntrinsicsExtrinsics_soln0.pickle") for i in range(2)]
    CamPrams_dict = {key:[] for key in camera_dict}
    if CamPramsPaths[0].exists() and CamPramsPaths[1].exists():
        for i, CamPramsPath in enumerate(CamPramsPaths):
            camera_params_dict = m4.loadCameraParameters(CamPramsPath)
            CamPrams_dict[target_cameras[i]] = camera_params_dict
    else:
        print(f"カメラパラメータファイルが見つかりません。")
        sys.exit()

    keypoints3d_dict = m4.cul_3DKeyPoints(keyoiints2d_dict, CamPrams_dict)
    for iPeople in range(len(keypoints3d_dict)):
        df = pd.DataFrame.from_dict(keypoints3d_dict[f"person{iPeople}"])
        df.index = frame_range
        df.to_csv((src_dirs[0]).with_name(f"3Dkeypoints_{condition}_person{iPeople}.csv"))
    print(f"3Dkeypoints_{condition}を作成しました。")







if __name__ == "__main__":
    main()