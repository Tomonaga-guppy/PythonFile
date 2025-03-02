from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import sys
from tqdm import tqdm
import pandas as pd
import gait_module as sggait
import copy

root_dir = Path(r"G:\gait_pattern\20241114_ota_test\gopro\sagi")
condition = "sub0_asgait_2"
make_ani = False

def main():
    #2d上でのキーポイントを取得"
    openpose_dir = root_dir / (condition + "_udCropped_op.json")
    # OpenPose処理をした開始フレームを取得
    frame_ch_csv = root_dir / "FrameCheck.csv"
    frame_check_df = pd.read_csv(frame_ch_csv)
    start_frame = frame_check_df[f"{condition}_Start"].values[0]
    end_frame = frame_check_df[f"{condition}_End"].values[0]
    csv_files = sggait.mkCSVOpenposeData(openpose_dir, start_frame, overwrite=True)
    df_dict = {}
    for ipeople, csv_file in enumerate(csv_files):
        # 検出した2次元座標に対して歪み補正
        read_df = pd.read_csv(csv_file, index_col=0)
        df_dict[f"{ipeople}"] = read_df
    # print(f"df_dict:{df_dict}")  #jsonを読み込んだ生データを格納した辞書

    # 前後の人のスイッチング処理
    df_dict_chsw = sggait.checkSwithing(copy.deepcopy(df_dict))
    # 3次スプライン補間
    df_dict_ip = sggait.spline_interpolation(copy.deepcopy(df_dict_chsw))
    # 4次のバターワースフィルター（カットオフ周波数6Hz）処理
    df_dict_ft = sggait.butter_lowpass_fillter(copy.deepcopy(df_dict_ip), sampling_freq=60, order=4,cutoff_freq=6)  #姿勢角度の計算

    if make_ani:
        print(f"アニメーション作成を開始します。")
        process = ["raw", "switching", "interpolated", "filtered"]
        save_paths = [root_dir / f"{condition}_{pro}.mp4" for pro in process]
        dicts = [df_dict, df_dict_chsw, df_dict_ip, df_dict_ft]
        i = 0
        for save_path , keypoint_dict in tqdm(zip(save_paths, dicts), total=len(save_paths)):
            if  (i == 0 or i ==1 )and save_path.exists():
                print(f"    {save_path}は既に存在します。")
                i += 1
                continue
            sggait.animate_keypoints(keypoint_dict, condition, str(save_path), all_check=True)
            i += 1
        print(f"    {condition}のアニメーション作成が完了しました。")

        save_path2 = root_dir / f"{condition}_Checkleg.mp4"
        sggait.animate_keypoints(df_dict_ft, condition, save_path2, all_check=False)

        print(f"すべてのアニメーション作成が完了しました。")






    # keypoint = "RWrist_x"
    # print(f"df_dict_ft:{df_dict_ft}")
    # x = df_dict["0"].loc[:, keypoint]
    # x_filter = df_dict_ft["0"].loc[:, keypoint]
    # fig, ax = plt.subplots()
    # ax.plot(x, label="raw")
    # ax.plot(x_filter, label="filtered")
    # ax.legend()
    # plt.show()



if __name__ == "__main__":
    main()