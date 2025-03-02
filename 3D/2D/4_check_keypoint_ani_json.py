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
condition_list = ["sub0_abngait", "sub0_asgait_2"]
make_ani = False

def main():
    for condition in condition_list:
        print(f"\n{condition}の処理を開始します。")
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

        # 初期接地のタイミングを記録
        # 骨盤を基準とした踵のx座標を計算
        dict_dist_pel2heel = sggait.calc_dist_pel2heel(df_dict_ft)
        # 最小のピークを初期接地として記録
        ic_frame_dict = sggait.find_initial_contact(dict_dist_pel2heel, condition, root_dir)
        """
        左麻痺患者 左側で計算
        ic_frame_dict:{'0': {'IC_R': [1377, 1448, 1522, 1595, 1666], 'IC_L': [1421, 1492, 1564, 1638, 1704]}, '1': {'IC_R': [1383, 1458, 1530, 1604, 1678], 'IC_L': [1421, 1494, 1567, 1641, 1704]}}
            kari_R:[5, 76, 150, 223, 294]
        kari_L:[49, 120, 192, 266, 332]
            kari_R:[11, 86, 158, 232, 306]
        kari_L:[49, 122, 195, 269, 332]
        """
        #マニュアルで確認した歪みの少ないピクセル範囲,、pix/mm を取得
        mesureParams_path = root_dir / "mesure_params.pickle"
        with open(mesureParams_path, "rb") as f:
            mesure_params = pickle.load(f)
        picpermm = mesure_params["mmperpix"]

        # rheel_x = df_dict_ft["0"].loc[:, "RHeel_x"]
        # plt.plot(rheel_x)
        # plt.show()

        #踵の位置が有効範囲の外にある場合は初期接地のタイミングを削除
        new_ic_frame_dict = {"0": {"IC_R": [], "IC_L": []}, "1": {"IC_R": [], "IC_L": []}}
        for ipeople in range(len(df_dict_ft)):
            for ICside, heel_name in zip(["IC_R", "IC_L"], ["RHeel_x", "LHeel_x"]):
                ic_check_list = ic_frame_dict[f"{ipeople}"][ICside]
                for index, i in enumerate(ic_check_list):
                    heel_x = df_dict_ft[f"{ipeople}"].loc[i, heel_name]
                    if heel_x < mesure_params["m1"][0] and heel_x > mesure_params["p1"][0]:
                        new_ic_frame_dict[f"{ipeople}"][ICside].append(i)
        # print(f"new_ic_frame_dict:{new_ic_frame_dict}")

        walk_param_names = ["walk_speed", "stride_time", "stride_length", "step_length"]

        for ipeople in range(len(df_dict_ft)):
            ipeople = str(ipeople)
            distR_pre = ic_frame_dict[f"{ipeople}"]["IC_R"]
            distL_pre = ic_frame_dict[f"{ipeople}"]["IC_L"]
            distR = new_ic_frame_dict[f"{ipeople}"]["IC_R"]
            distL = new_ic_frame_dict[f"{ipeople}"]["IC_L"]
            print(f"{ipeople}:distR_pre:{distR_pre} ")
            print(f"{ipeople}:distR:{distR} ")
            print(f"{ipeople}:distL_pre:{distL_pre} ")
            print(f"{ipeople}:distL:{distL} ")

        gait_params_dict = {key : [] for key in walk_param_names}
        for ipeople in range(len(df_dict_ft)):
            gait_params_dict[f"{ipeople}"] = {key : [] for key in walk_param_names}
            stride_time = sggait.calc_stride_time(new_ic_frame_dict[f"{ipeople}"], "IC_L", fps = 60)
            walk_speed, stride_length, step_length = sggait.calc_walk_params(stride_time, picpermm, new_ic_frame_dict[f"{ipeople}"], df_dict_ft[f"{ipeople}"])
            print(f"ストライド時間：{stride_time:.2f} s, 歩行速度:{walk_speed:.2f} m/s, ストライド:{stride_length:.2f} m, ステップ:{step_length:.2f} m")
            gait_params_dict[f"{ipeople}"]["walk_speed"] = walk_speed
            gait_params_dict[f"{ipeople}"]["stride_time"] = stride_time
            gait_params_dict[f"{ipeople}"]["stride_length"] = stride_length
            gait_params_dict[f"{ipeople}"]["step_length"] = step_length

        # パラメータを保存
        gait_params_save_path = root_dir / f"{condition}_gait_params.pickle"
        sggait.save_as_pickle(gait_params_dict, gait_params_save_path)




if __name__ == "__main__":
    main()