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

root_dir = Path(r"G:\gait_pattern\20250228_ota\data\20250221\sub0")
condition_list = ["thera0-3", "thera1-1", "thera2-1"]
# condition_list = ["thera0-3"]
# condition_list = ["thera1-1", "thera2-1"]
make_ani = True
# cali = "ota_20250228"
cali = "ota_20250228_custom"

def main():
    for condition in condition_list:
        # if not condition == "sub0_asgait_1":
        #     continue
        print(f"\n{condition}の処理を開始します。")
        condition_dir = root_dir / condition
        #2d上でのキーポイントを取得"
        openpose_dir = condition_dir / "sagi" / f"Undistort_op.json"
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

        print(f"df_dict:{df_dict}")
        # ch_dict0 = {0: df_dict["0"].loc[:, "MidHip_x"], 1: df_dict["1"].loc[:, "MidHip_x"]}
        # print(f"ch_dict0:{ch_dict0}")

        #### 独歩の場合は0のみを使用（要修正！）
        if condition == "thera0-3":
            solo = True
        else:
            solo = False
        # 前後の人のスイッチング処理
        df_dict_chsw = sggait.checkSwithing(copy.deepcopy(df_dict))
        frame_range = df_dict_chsw["0"].index

        df_dict2 = {key: [] for key in df_dict.keys()}
        for key in df_dict_chsw.keys():
            df_dict2[key] = df_dict.copy()[key].loc[frame_range]
        # ch_dfct2 = {0: df_dict_chsw["0"].loc[:, "MidHip_x"], 1: df_dict_chsw["1"].loc[:, "MidHip_x"]}
        # print(f"ch_dfct2:{ch_dfct2}")
        # print(f"df_dict_chsw:{df_dict_chsw}")
        # sys.exit()
        # print(f"df_dict_chsw:{df_dict_chsw}")

        # 3次スプライン補間
        df_dict_ip = sggait.spline_interpolation(copy.deepcopy(df_dict_chsw))
        # print(f"df_dict_ip:{df_dict_ip}")
        # 4次のバターワースフィルター（カットオフ周波数6Hz）処理
        df_dict_ft = sggait.butter_lowpass_fillter(copy.deepcopy(df_dict_ip), sampling_freq=60, order=4,cutoff_freq=6)  #姿勢角度の計算
        # print(f"df_dict_ft:{df_dict_ft}")

        if make_ani:
            print(f"アニメーション作成を開始します。")

            process = ["raw", "filtered"]
            save_paths = [openpose_dir.with_name(openpose_dir.stem + f"_{pro}.mp4") for pro in process]
            dicts = [df_dict2, df_dict_ft]
            # process = ["raw", "switching", "interpolated", "filtered"]
            # save_paths = [openpose_dir.with_name(openpose_dir.stem + f"_{pro}.mp4") for pro in process]
            # dicts = [df_dict2, df_dict_chsw, df_dict_ip, df_dict_ft]
            i = 0
            for save_path , keypoint_dict in tqdm(zip(save_paths, dicts), total=len(save_paths)):
                if  (i == 0 or i ==1 )and save_path.exists():
                    print(f"    {save_path}は既に存在します。")
                    i += 1
                    continue
                sggait.animate_keypoints(keypoint_dict, condition, str(save_path), all_check=True)
                i += 1
            print(f"    {condition}のアニメーション作成が完了しました。")

            # save_path2 = root_dir / f"{condition}_Checkleg.mp4"
            save_path2 = openpose_dir.with_name(openpose_dir.stem + "_ChKeyPoints.mp4")
            if save_path2.exists():
                print(f"    {save_path2}は既に存在します。")
                pass
            else:
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
        mesureParams_path = root_dir / f"sagi_mesure_params_{cali}.pickle"
        with open(mesureParams_path, "rb") as f:
            mesure_params = pickle.load(f)
        picpermm = mesure_params["mmperpix"]

        print(f"ic_frame_dict:{ic_frame_dict}")
        print(f"to_frame_dict:{to_frame_dict}")

        #踵の位置が有効範囲の外にある場合は初期接地のタイミングを削除
        new_ic_frame_dict = {key : {"IC_R": [], "IC_L": []} for key in df_dict_ft.keys()}
        for ipeople in range(len(df_dict_ft)):
            for ICside, heel_name in zip(["IC_R", "IC_L"], ["RHeel_x", "LHeel_x"]):
                new_ic_frame_list = []
                ic_check_list = ic_frame_dict[f"{ipeople}"][ICside]
                for index, i in enumerate(ic_check_list):
                    heel_x = df_dict_ft[f"{ipeople}"].loc[i, heel_name]
                    if heel_x < mesure_params["m"][0] and heel_x > mesure_params["p"][0]:
                        new_ic_frame_list.append(i)
                new_ic_frame_list = sggait.check_edge_frame(new_ic_frame_list)
                new_ic_frame_dict[f"{ipeople}"][ICside].extend(new_ic_frame_list)
                # new_ic_frame_dict[f"{ipeople}"][ICside].append(new_ic_frame_list[i] for i in range(len(new_ic_frame_list)))
        ic_frame_dict_path = condition_dir / "IC_frame.pickle"
        sggait.save_as_pickle(new_ic_frame_dict, ic_frame_dict_path)

        new_to_frame_dict = {key : {"TO_R": [], "TO_L": []} for key in df_dict_ft.keys()}
        for ipeople in range(len(df_dict_ft)):
            for TOside, heel_name in zip(["TO_R", "TO_L"], ["RHeel_x", "LHeel_x"]):
                new_to_frame_list = []
                to_check_list = to_frame_dict[f"{ipeople}"][TOside]
                for index, i in enumerate(to_check_list):
                    heel_x = df_dict_ft[f"{ipeople}"].loc[i, heel_name]
                    if heel_x < mesure_params["m"][0] and heel_x > mesure_params["p"][0]:
                        new_to_frame_list.append(i)
                new_to_frame_list = sggait.check_edge_frame(new_to_frame_list)
                new_to_frame_dict[f"{ipeople}"][TOside].extend(new_to_frame_list)
        to_frame_dict_path = condition_dir / "TO_frame.pickle"
        sggait.save_as_pickle(new_to_frame_dict, to_frame_dict_path)

        print(F"new_ic_frame_dict:{new_ic_frame_dict}")
        print(F"new_to_frame_dict:{new_to_frame_dict}")

        # 歩行の3つの層を決めるフレームや割合を計算
        phase_frame_dict = sggait.calGaitPhase(new_ic_frame_dict, new_to_frame_dict)
        new_phase_frame_dict = {key: [] for key in df_dict_ft.keys()}
        for iPeople in range(len(df_dict_ft)):
            for ic_frame_list in phase_frame_dict[f"{iPeople}"]:
                for index in range(len(new_ic_frame_dict[f"{iPeople}"]["IC_L"])):
                    if ic_frame_list[0] == new_ic_frame_dict[f"{iPeople}"]["IC_L"][index]:
                        new_phase_frame_dict[f"{iPeople}"].append(ic_frame_list)
        new_phase_percent_dict = sggait.calGaitPhasePercent(new_phase_frame_dict)
        sggait.save_as_pickle(new_phase_frame_dict, condition_dir / "phase_frame.pickle")
        sggait.save_as_pickle(new_phase_percent_dict, condition_dir / "phase_percent.pickle")

        print(f"phase_frame_dict:{new_phase_frame_dict}")
        print(f"phase_percent_dict:{new_phase_percent_dict}")

        """
        new_phase_frame_dict:{'0': [[1349, 1356, 1392, 1421], [1421, 1428, 1461, 1489]], '1': [[1348, 1358, 1393, 1418], [1418, 1429, 1464, 1489], [1489, 1500, 1535, 1559]]}
        new_phase_percent_dict:{'0': [[0.0, 9.722222222222223, 59.72222222222222, 100.0], [0.0, 10.294117647058822, 58.82352941176471, 100.0]], '1': [[0.0, 14.285714285714285, 64.28571428571429, 100.0], [0.0, 15.492957746478872, 64.7887323943662, 100.0], [0.0, 15.714285714285714, 65.71428571428571, 100.0]]}
        """


        # walk_param_names = ["walk_speed", "stride_time", "stride_length", "step_length"]
        # gait_params_dict = {key : [] for key in walk_param_names}
        gait_params_dict = {}
        for ipeople in range(len(df_dict_ft)):
            if ipeople == 1:
                continue
            gait_params_dict[f"{ipeople}"] = {}  #格納するパラメータの数だけ初期化
            stride_time, std_stride_time = sggait.calc_stride_time(new_ic_frame_dict[f"{ipeople}"], "IC_L", fps = 60)
            walk_speed, stride_length_l, stride_length_r, step_length_l, step_length_r, std_walk_speed, std_stride_length_l, std_stride_length_r, std_step_length_l, std_step_length_r = sggait.calc_walk_params(stride_time, picpermm, new_ic_frame_dict[f"{ipeople}"], df_dict_ft[f"{ipeople}"])
            step_avg = (step_length_l + step_length_r) / 2
            std_step = (std_step_length_l + std_step_length_r) / 2
            # stance_phase_ratio_r, stance_phase_ratio_l = sggait.calc_stance_phase_ratio(new_ic_frame_dict[f"{ipeople}"], new_to_frame_dict[f"{ipeople}"])
            print(f"{condition}の{ipeople+1}人目：歩行パラメータを計算しました。")
            # print(f"ストライド時間：{stride_time:.3f} s, 歩行速度:{walk_speed:.3f} m/s, ストライド(左):{stride_length_l:.3f} m, ステップ（左）:{step_length_l:.3f} m, 立脚相割合（左）:{stance_phase_ratio_l:.3f} %, ストライド（右）:{stride_length_r:.3f} m, ステップ（右）:{step_length_r:.3f} m, 立脚期割合（右）：{stance_phase_ratio_r:.3f} %, 平均ステップ:{step_avg:.3f} m\n")
            print(f"歩行速度:{walk_speed:.3f}+-{std_walk_speed:.3f} m/s, 歩幅(左):{step_length_l:.3f}+-{std_step_length_l:.3f} m, 歩幅(右):{step_length_r:.3f}+-{std_step_length_r:.3f} m\n")
            # print(f"歩行速度:{walk_speed:.3f} m/s, ステップ（左）:{step_length_l:.3f} m, ステップ（右）:{step_length_r:.3f} m\n")
            gait_params_dict[f"{ipeople}"]["walk_speed"] = walk_speed
            gait_params_dict[f"{ipeople}"]["stride_time"] = stride_time
            gait_params_dict[f"{ipeople}"]["stride_length_l"] = stride_length_l
            gait_params_dict[f"{ipeople}"]["step_length_l"] = step_length_l
            # gait_params_dict[f"{ipeople}"]["stance_phase_ratio_l"] = stance_phase_ratio_l
            gait_params_dict[f"{ipeople}"]["stride_length_r"] = stride_length_r
            gait_params_dict[f"{ipeople}"]["step_length_r"] = step_length_r
            # gait_params_dict[f"{ipeople}"]["stance_phase_ratio_r"] = stance_phase_ratio_r
            gait_params_dict[f"{ipeople}"]["step_avg"] = step_avg
            gait_params_dict[f"{ipeople}"]["std_walk_speed"] = std_walk_speed
            gait_params_dict[f"{ipeople}"]["std_stride_time"] = std_stride_time
            gait_params_dict[f"{ipeople}"]["std_stride_length_l"] = std_stride_length_l
            gait_params_dict[f"{ipeople}"]["std_step_length_l"] = std_step_length_l
            gait_params_dict[f"{ipeople}"]["std_stride_length_r"] = std_stride_length_r
            gait_params_dict[f"{ipeople}"]["std_step_length_r"] = std_step_length_r
            gait_params_dict[f"{ipeople}"]["std_step"] = std_step

        # パラメータを保存
        gait_params_save_path = condition_dir / f"gait_params.pickle"
        sggait.save_as_pickle(gait_params_dict, gait_params_save_path)



if __name__ == "__main__":
    main()