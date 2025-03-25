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
# condition_list = ["sub0_abngait", "sub0_asgait_1","sub0_asgait_2"]
condition_list = ["sub0_abngait", "sub0_asgait_1"]
make_ani = False

def main():
    for condition in condition_list:
        # if not condition == "sub0_asgait_1":
        #     continue
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
        # 最小のピークをHeel Strike(初期接地IC)として記録, 最大のピークをToe Offとして記録
        ic_frame_dict, to_frame_dict = sggait.find_initial_contact(dict_dist_pel2heel, condition, root_dir)
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

        print(f"ic_frame_dict:{ic_frame_dict}")
        print(f"to_frame_dict:{to_frame_dict}")

        #踵の位置が有効範囲の外にある場合は初期接地のタイミングを削除
        new_ic_frame_dict = {key : {"IC_R": [], "IC_L": []} for key in df_dict_ft.keys()}
        for ipeople in range(len(df_dict_ft)):
            for ICside, heel_name in zip(["IC_R", "IC_L"], ["RHeel_x", "LHeel_x"]):
                ic_check_list = ic_frame_dict[f"{ipeople}"][ICside]
                for index, i in enumerate(ic_check_list):
                    heel_x = df_dict_ft[f"{ipeople}"].loc[i, heel_name]
                    if heel_x < mesure_params["m1"][0] and heel_x > mesure_params["p1"][0]:
                        new_ic_frame_dict[f"{ipeople}"][ICside].append(i)
        ic_frame_dict_path = root_dir / f"{condition}_ic_frame.pickle"
        sggait.save_as_pickle(new_ic_frame_dict, ic_frame_dict_path)

        new_to_frame_dict = {key : {"TO_R": [], "TO_L": []} for key in df_dict_ft.keys()}
        for ipeople in range(len(df_dict_ft)):
            for TOside, heel_name in zip(["TO_R", "TO_L"], ["RHeel_x", "LHeel_x"]):
                to_check_list = to_frame_dict[f"{ipeople}"][TOside]
                for index, i in enumerate(to_check_list):
                    heel_x = df_dict_ft[f"{ipeople}"].loc[i, heel_name]
                    if heel_x < mesure_params["m1"][0] and heel_x > mesure_params["p1"][0]:
                        new_to_frame_dict[f"{ipeople}"][TOside].append(i)
        to_frame_dict_path = root_dir / f"{condition}_to_frame.pickle"
        sggait.save_as_pickle(new_to_frame_dict, to_frame_dict_path)

        print(F"new_ic_frame_dict:{new_ic_frame_dict}")
        print(F"new_to_frame_dict:{new_to_frame_dict}")

        # 歩行の3つの層を決めるフレームや割合を計算
        phase_frame_dict = sggait.calGaitPhase(ic_frame_dict, to_frame_dict)
        new_phase_frame_dict = {key: [] for key in df_dict_ft.keys()}
        for iPeople in range(len(df_dict_ft)):
            for ic_frame_list in phase_frame_dict[f"{iPeople}"]:
                for index in range(len(new_ic_frame_dict[f"{iPeople}"]["IC_L"])):
                    if ic_frame_list[0] == new_ic_frame_dict[f"{iPeople}"]["IC_L"][index]:
                        new_phase_frame_dict[f"{iPeople}"].append(ic_frame_list)
        new_phase_percent_dict = sggait.calGaitPhasePercent(new_phase_frame_dict)
        sggait.save_as_pickle(new_phase_frame_dict, root_dir / f"{condition}_phase_frame.pickle")
        sggait.save_as_pickle(new_phase_percent_dict, root_dir / f"{condition}_phase_percent.pickle")

        """
        ew_phase_frame_dict:{'0': [[1349, 1356, 1392, 1421], [1421, 1428, 1461, 1489]], '1': [[1348, 1358, 1393, 1418], [1418, 1429, 1464, 1489], [1489, 1500, 1535, 1559]]}
        new_phase_percent_dict:{'0': [[0.0, 9.722222222222223, 59.72222222222222, 100.0], [0.0, 10.294117647058822, 58.82352941176471, 100.0]], '1': [[0.0, 14.285714285714285, 64.28571428571429, 100.0], [0.0, 15.492957746478872, 64.7887323943662, 100.0], [0.0, 15.714285714285714, 65.71428571428571, 100.0]]}
        """


        # walk_param_names = ["walk_speed", "stride_time", "stride_length", "step_length"]
        # gait_params_dict = {key : [] for key in walk_param_names}
        gait_params_dict = {}
        for ipeople in range(len(df_dict_ft)):
            gait_params_dict[f"{ipeople}"] = {}  #格納するパラメータの数だけ初期化
            stride_time = sggait.calc_stride_time(new_ic_frame_dict[f"{ipeople}"], "IC_L", fps = 60)
            walk_speed, stride_length_l, stride_length_r, step_length_l, step_length_r = sggait.calc_walk_params(stride_time, picpermm, new_ic_frame_dict[f"{ipeople}"], df_dict_ft[f"{ipeople}"])
            step_avg = (step_length_l + step_length_r) / 2
            stance_phase_ratio_r, stance_phase_ratio_l = sggait.calc_stance_phase_ratio(new_ic_frame_dict[f"{ipeople}"], new_to_frame_dict[f"{ipeople}"])
            print(f"{condition}の歩行パラメータを計算しました。")
            # print(f"ストライド時間：{stride_time:.3f} s, 歩行速度:{walk_speed:.3f} m/s, ストライド(左):{stride_length_l:.3f} m, ステップ（左）:{step_length_l:.3f} m, 立脚相割合（左）:{stance_phase_ratio_l:.3f} %, ストライド（右）:{stride_length_r:.3f} m, ステップ（右）:{step_length_r:.3f} m, 立脚期割合（右）：{stance_phase_ratio_r:.3f} %, 平均ステップ:{step_avg:.3f} m\n")
            print(f"歩行速度:{walk_speed:.3f} m/s, ステップ（左）:{step_length_l:.3f} m, ステップ（右）:{step_length_r:.3f} m\n")
            gait_params_dict[f"{ipeople}"]["walk_speed"] = walk_speed
            gait_params_dict[f"{ipeople}"]["stride_time"] = stride_time
            gait_params_dict[f"{ipeople}"]["stride_length_l"] = stride_length_l
            gait_params_dict[f"{ipeople}"]["step_length_l"] = step_length_l
            gait_params_dict[f"{ipeople}"]["stance_phase_ratio_l"] = stance_phase_ratio_l
            gait_params_dict[f"{ipeople}"]["stride_length_r"] = stride_length_r
            gait_params_dict[f"{ipeople}"]["step_length_r"] = step_length_r
            gait_params_dict[f"{ipeople}"]["stance_phase_ratio_r"] = stance_phase_ratio_r
            gait_params_dict[f"{ipeople}"]["step_avg"] = step_avg

        # パラメータを保存
        gait_params_save_path = root_dir / f"{condition}_gait_params.pickle"
        sggait.save_as_pickle(gait_params_dict, gait_params_save_path)



if __name__ == "__main__":
    main()