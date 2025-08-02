from pathlib import Path
import pandas as pd
import imu_module as imu
import copy
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import numpy as np

root_dir = Path(r"G:\gait_pattern\20250228_ota\data\20250221\IMU")
target_imus = ["sync", "sub", "thera", "thera_rhand", "thera_lhand"]
# target_dir = root_dir / target_imus[0]

root_dir_op = Path(r"G:\gait_pattern\20250228_ota\data\20250221\sub0")
condition_list = ["thera0-3", "thera1-1", "thera2-1"]
op_IC_frame_paths = [root_dir_op / condition / "IC_frame.pickle" for condition in condition_list]
op_TO_frame_paths = [root_dir_op / condition / "TO_frame.pickle" for condition in condition_list]
op_gait_params_paths = [root_dir_op / condition / "gait_params.pickle" for condition in condition_list]

#手を降り始めた瞬間のフレーム（目視）"thera0-3", "thera1-1", "thera2-1"の順
op_sync_frame = [165, 146, 280]
# #手を降り始めた瞬間のフレーム（目視）sub0_abngait, sub0_asgait_1の順
# op_sync_frame = [345, 461]
#imu同士のフレーム調整用
sync_start_time_list = []

# 各条件での比較用にリストを用意
thera_r_hand_rms_list = []
thera_l_hand_rms_list = []

imu_result_dict = {key:[] for key in target_imus}
for target_imu in target_imus:
    target_dir = root_dir / target_imu
    #解析対象のimuファイルを取得
    csv_list = [file for file in target_dir.glob("*.csv")
        if file.name.startswith("1") or file.name.startswith("2") or file.name.startswith("3") ]

    # 各条件での比較用にリストを用意
    sub_lumb_rms_list = []


    for i, csv_path in enumerate(csv_list):
        print(f"\ncsv_path:{csv_path}")
        print(f"condition_list[i]:{condition_list[i]}")
        # OpenPose結果を読み込む
        openpose_res_path = root_dir_op / condition_list[i] / "gait_params.pickle"
        openpose_res_dict = imu.loadPickle(openpose_res_path)
        # openpose_res_dict:{'0': {'walk_speed': 0.6850662125743122, 'stride_time': 1.2833333333333334, 'stride_length_l': 0.8662926063741502, 'step_length_l': 0.44607708502794957, 'stance_phase_ratio_l': 62.33766233766234, 'stride_length_r': 0.8554579614359269, 'step_length_r': 0.4279558456969168, 'stance_phase_ratio_r': 70.12987012987013, 'step_avg': 0.4370164653624332}}

        #初期接地データを読み込む
        op_condition_dir = root_dir_op / condition_list[i]
        op_IC_frame_dict = imu.loadPickle(op_IC_frame_paths[i])
        print(f"op_IC_frame_dict:{op_IC_frame_dict}")
        # op_IC_frame_dict:{'0': {'IC_R': [1309, 1380, 1450], 'IC_L': [1349, 1421]}, '1': {'IC_R': [1380, 1453], 'IC_L': [1348, 1418, 1489]}}
        #踵離地のフレームを取得
        op_TO_frame_dict = imu.loadPickle(op_TO_frame_paths[i])
        phase_frame_dict = imu.loadPickle(root_dir_op / condition_list[i] / "phase_frame.pickle")
        phase_percent_list_dict = imu.loadPickle(root_dir_op / condition_list[i] / "phase_percent.pickle")
        # print(f"phase_frame_dict:{phase_frame_dict}")
        # print(f"phase_percent_list_dict:{phase_percent_list_dict}")

        #歩行パラメータを取得
        gait_params = imu.loadPickle(op_gait_params_paths[i])
        walk_speed = gait_params["0"]["walk_speed"] #歩行速度(RMSの正規化に使用)
        step = gait_params["0"]["step_avg"] #左右平均した歩幅(RMSの正規化に使用)


        imu_df_100hz = imu.read_ags_dataframe(csv_path)
        if target_imu == "sync":
            sync_start_time_list.append(imu_df_100hz["time"].iloc[0])
        else:
            imu_df_100hz = imu.arrange_imu_data(imu_df_100hz, sync_start_time_list[i])

        """
            time  acc_x  acc_y  acc_z
        0  73338560  10131   -673    997
        1  73338570  10070   -726   1024
        """
        # 100Hzのデータを60Hzにダウンサンプリングする前に、バターワースフィルタを適用
        imu_df_butter_100Hz = imu.butter_lowpass_fillter(copy.copy(imu_df_100hz), sampling_freq=60, order=4, cutoff_freq=10)
        # 100Hzのデータを60Hzにダウンサンプリング
        imu_df = imu.resampling(copy.copy(imu_df_butter_100Hz), pre_Hz = 100, post_Hz = 60)



        #加速度の単位を0.1mGから1m/s^2へ変換    https://www.atr-p.com/support/TSND-QA.html
        for col in list (imu_df.columns):
            if col.startswith("acc_"):
                imu_df[col] = imu_df[col] * 9.81 /1000 * 0.1
        """
        imu_df_単位変換後:              time     acc_x     acc_y      acc_z
        0     7.355353e+07 -1.004544 -7.907841   7.224084
        1     7.355355e+07 -1.052613 -7.922065   7.098026
        """
        #三軸の合計加速度を追加
        imu_df["acc"] = (imu_df["acc_x"]**2 + imu_df["acc_y"]**2 + imu_df["acc_z"]**2) **0.5
        # imu_df["acc_xyrms"] = (imu_df["acc_x"]**2 + imu_df["acc_y"]**2) / 3 **0.5

        if target_imu == "sub":
            plt.plot(imu_df["acc_x"], label="S-I", color="red")
            plt.plot(imu_df["acc_y"], label="L-R", color="green")
            plt.plot(imu_df["acc_z"], label="A-P", color="blue")
            plt.legend()
            # plt.show()
            plt.savefig(csv_path.with_name(f"check1_{condition_list[i]}.png"))
            plt.close()

        # print(f"imu_df:{imu_df.head()}")

        #同期用IMuなら同期タイミングの確認
        if target_imu == "sync":
            #手を降り始めた瞬間のフレーム xの変化が大きいとしてacc_xから同期フレームを求める（yは一応）
            imu_sync_frame_x = imu.find_sync_frame(imu_df, "acc_x")
            imu_sync_frame_y = imu.find_sync_frame(imu_df, "acc_y")
            print(f"imu_sync_frame_x:{imu_sync_frame_x}, imu_sync_frame_y:{imu_sync_frame_y}")

            #加速度を確認用にプロット
            plt.plot(imu_df["acc_x"], label="acc_x", color="red")
            plt.plot(imu_df["acc_y"], label="acc_y", color="green")
            plt.plot(imu_df["acc_z"], label="acc_z", color="blue")
            plt.legend()
            plt.savefig(csv_path.with_name(f"check_{condition_list[i]}.png"))
            # plt.show()
            plt.cla()


            ### 同期フレームを手動で設定
            sync_frame_x_memo = [599, 419, 493]
            ###
            imu_sync_frame_x = sync_frame_x_memo[i]

            #IMUデータを同期フレームからの値に調整してIMUメモリにおける開始時刻を取得
            frame_diff = imu_sync_frame_x - op_sync_frame[i]
            imu_df.index = imu_df.index - frame_diff
            imu_df = imu_df[imu_df.index >= 0]
            continue

        # openposeと開始フレームを合わせる
        imu_df.index = imu_df.index - frame_diff
        imu_df = imu_df[imu_df.index >= 0]

        imu_data_dict = {key : [] for key in target_imus}

        figsize_xyz = (11, 11)

        sub_IC_frame_l = op_IC_frame_dict["0"]["IC_L"]
        sub_IC_frame_r = op_IC_frame_dict["0"]["IC_R"]
        sub_TO_frame_l = op_TO_frame_dict["0"]["TO_L"]
        sub_TO_frame_r = op_TO_frame_dict["0"]["TO_R"]
        sub_IC_frame_r = imu.adjust_gait_event(target_frame_list=sub_IC_frame_r, base_frame_list=sub_IC_frame_l)
        sub_TO_frame_l = imu.adjust_gait_event(target_frame_list=sub_TO_frame_l, base_frame_list=sub_IC_frame_l)
        sub_TO_frame_r = imu.adjust_gait_event(target_frame_list=sub_TO_frame_r, base_frame_list=sub_IC_frame_l)
        # print(f"調整後のsub_IC_frame_l:{sub_IC_frame_l}")
        # print(f"調整後のsub_IC_frame_r:{sub_IC_frame_r}")
        # print(f"調整後のsub_TO_frame_l:{sub_TO_frame_l}")
        # print(f"調整後のsub_TO_frame_r:{sub_TO_frame_r}")
        gait_event_frame_array = imu.get_gait_event_block(sub_IC_frame_l, sub_IC_frame_r, sub_TO_frame_l, sub_TO_frame_r)
        gait_event_percent_array = imu.get_event_percent_block(gait_event_frame_array)
        """
        gait_event_percent_array:
        [[  0.          13.04347826  44.92753623  56.52173913 100.        ]
        [  0.          11.5942029   44.92753623  55.07246377 100.        ]]"""
        print(f"gait_event_percent_array.shape:{gait_event_percent_array.shape}")
        print(f"gait_event_percent_array:{gait_event_percent_array}")

        gait_event_percent_avg = np.mean(gait_event_percent_array, axis=0)
        print(f"gait_event_percent_avg:{gait_event_percent_avg}")
        # shape = (ブロック数, 5)


        if target_imu == "sub":
            plt.plot(imu_df["acc_x"], label="S-I", color="red")
            plt.plot(imu_df["acc_y"], label="L-R", color="green")
            plt.plot(imu_df["acc_z"], label="A-P", color="blue")
            plt.legend()
            [plt.axvline(sub_IC_frame_l[i], color="black", alpha=0.9, linestyle="-") for i in range(len(sub_IC_frame_l))]
            [plt.axvline(sub_IC_frame_r[i], color="black", alpha=0.9, linestyle="--") for i in range(len(sub_IC_frame_r))]
            # plt.show()
            plt.savefig(csv_path.with_name(f"check2_{condition_list[i]}.png"))
            plt.close()



            sub_ic_block_r = [[sub_IC_frame_r[i], sub_IC_frame_r[i+1]] for i in range(len(sub_IC_frame_r)-1)]
            sub_ic_block_l = [[sub_IC_frame_l[i], sub_IC_frame_l[i+1]] for i in range(len(sub_IC_frame_l)-1)]
            ### このあたりできたら次はfor分で回す
            imu_df_sublumb = imu_df.loc[sub_ic_block_l[0][0]:sub_ic_block_l[-1][1]] #とりあえず左足基準で最後まで
            ###
            imu_df_sublumb = imu_df_sublumb - imu_df_sublumb.loc[sub_IC_frame_l[0]]

            # plt.figure(figsize=(11, 11))
            plt.plot(imu_df_sublumb["acc_x"], label="S-I", color="red")
            plt.plot(imu_df_sublumb["acc_y"], label="L-R", color="green")
            plt.plot(imu_df_sublumb["acc_z"], label="A-P", color="blue")
            plt.legend()
            [plt.axvline(sub_IC_frame_l[i], color="black", alpha=0.9, linestyle="-") for i in range(len(sub_IC_frame_l))]
            [plt.axvline(sub_IC_frame_r[i], color="black", alpha=0.9, linestyle="--") for i in range(len(sub_IC_frame_r))]
            # plt.show()
            plt.ylim(-7.5, 15)
            plt.savefig(csv_path.with_name(f"check3_{condition_list[i]}.png"))
            plt.close()

            # 各歩行周期で各相のRMSを計算
            rms_dict = {key : [] for key in ["rms_x", "rms_y", "rms_z", "std_x", "std_y", "std_z", "rms_x_n", "rms_y_n", "rms_z_n", "std_x_n", "std_y_n", "std_z_n"]}
            rms_x_list , rms_y_list, rms_z_list = [], [], []
            for idx in range(gait_event_frame_array.shape[0]):
                rms_x_list_cycle, rms_y_list_cycle, rms_z_list_cycle = [], [], []
                for event_idx in range(gait_event_frame_array.shape[1]-1):
                    start_frame = gait_event_frame_array[idx][event_idx]
                    end_frame = gait_event_frame_array[idx][event_idx+1]
                    # print(f"start_frame:{start_frame}, end_frame:{end_frame}")
                    imu_df_sublumb_in_phase = imu_df_sublumb.loc[start_frame:end_frame]
                    # print(f'imu_df_sublumb_in_phase["acc_x"]:{imu_df_sublumb_in_phase["acc_x"]}')
                    rms_x = np.sqrt(np.mean(imu_df_sublumb_in_phase["acc_x"]**2))
                    rms_y = np.sqrt(np.mean(imu_df_sublumb_in_phase["acc_y"]**2))
                    rms_z = np.sqrt(np.mean(imu_df_sublumb_in_phase["acc_z"]**2))
                    rms_x_list_cycle.append(rms_x)
                    rms_y_list_cycle.append(rms_y)
                    rms_z_list_cycle.append(rms_z)
                rms_x_list.append(rms_x_list_cycle)
                rms_y_list.append(rms_y_list_cycle)
                rms_z_list.append(rms_z_list_cycle)
            # 複数歩行周期の平均、標準偏差を取る
            rms_x = np.mean(rms_x_list, axis=0)
            rms_y = np.mean(rms_y_list, axis=0)
            rms_z = np.mean(rms_z_list, axis=0)
            std_x = np.std(rms_x_list, axis=0)
            std_y = np.std(rms_y_list, axis=0)
            std_z = np.std(rms_z_list, axis=0)
            rms_dict["rms_x"] = rms_x
            rms_dict["rms_y"] = rms_y
            rms_dict["rms_z"] = rms_z
            rms_dict["std_x"] = std_x
            rms_dict["std_y"] = std_y
            rms_dict["std_z"] = std_z

            # 歩行周期全体でのRMS（歩行速度の2乗で除算して歩幅を乗算して正規化：水池さんの論文参考）を計算
            rms_x_n_list, rms_y_n_list, rms_z_n_list = [], [], []
            for cycle_num in range(gait_event_frame_array.shape[0]):
                print(f"サイクルの番号:{cycle_num}")
                start_frame = gait_event_frame_array[cycle_num][0]
                end_frame = gait_event_frame_array[cycle_num][-1]
                imu_df_sublumb_rms_n = imu_df_sublumb.loc[start_frame:end_frame]
                rms_x_n = np.sqrt(np.mean(imu_df_sublumb_rms_n["acc_x"]**2)) / (walk_speed**2)
                rms_y_n = np.sqrt(np.mean(imu_df_sublumb_rms_n["acc_y"]**2)) / (walk_speed**2)
                rms_z_n = np.sqrt(np.mean(imu_df_sublumb_rms_n["acc_z"]**2)) / (walk_speed**2)
                rms_x_n_list.append(rms_x_n)
                rms_y_n_list.append(rms_y_n)
                rms_z_n_list.append(rms_z_n)
            print(f"rms_x_n_list:{rms_x_n_list}")
            rms_x_n = np.mean(rms_x_n_list, axis=0)
            rms_y_n = np.mean(rms_y_n_list, axis=0)
            rms_z_n = np.mean(rms_z_n_list, axis=0)
            std_x_n = np.std(rms_x_n_list, axis=0)
            std_y_n = np.std(rms_y_n_list, axis=0)
            std_z_n = np.std(rms_z_n_list, axis=0)
            print(f"rms_x_n:{rms_x_n}")
            print(f"std_x_n:{std_x_n}")
            rms_dict["rms_x_n"] = rms_x_n
            rms_dict["rms_y_n"] = rms_y_n
            rms_dict["rms_z_n"] = rms_z_n
            rms_dict["std_x_n"] = std_x_n
            rms_dict["std_y_n"] = std_y_n
            rms_dict["std_z_n"] = std_z_n

            sub_lumb_rms_list.append(rms_dict)  #各条件でのRMSを比較するためにリストに格納

            # 複数歩行周期の平均を取る
            acc_x_list, acc_y_list, acc_z_list = [], [], []
            for event_frame_array in gait_event_frame_array:
                print(f"sub_IC_frame_l:{sub_IC_frame_l}")
                start_frame = event_frame_array[0]
                end_frame = event_frame_array[-1]
                imu_df_sublumb_x = imu_df_sublumb["acc_x"].loc[start_frame:end_frame]
                imu_df_sublumb_y = imu_df_sublumb["acc_y"].loc[start_frame:end_frame]
                imu_df_sublumb_z = imu_df_sublumb["acc_z"].loc[start_frame:end_frame]
                #各歩行周期のデータを正規化して1パーセントごとにリストに格納
                imu_df_sublumb_x = imu.frame2percent(imu_df_sublumb_x)
                imu_df_sublumb_y = imu.frame2percent(imu_df_sublumb_y)
                imu_df_sublumb_z = imu.frame2percent(imu_df_sublumb_z)
                acc_x_list.append(imu_df_sublumb_x)
                acc_y_list.append(imu_df_sublumb_y)
                acc_z_list.append(imu_df_sublumb_z)
            acc_x_mean = np.mean(acc_x_list, axis=0)
            acc_x_std = np.std(acc_x_list, axis=0)
            acc_y_mean = np.mean(acc_y_list, axis=0)
            acc_y_std = np.std(acc_y_list, axis=0)
            acc_z_mean = np.mean(acc_z_list, axis=0)
            acc_z_std = np.std(acc_z_list, axis=0)

            plot_range = range(100)
            plt.figure(figsize=(12, 8))
            plt.plot(acc_x_mean, label="S-I", color="red")
            plt.plot(acc_y_mean, label="L-R", color="green")
            plt.plot(acc_z_mean, label="A-P", color="blue")
            # plt.plot(acc_x_mean, label="Superior-Inferior", color="red")
            # plt.plot(acc_y_mean, label="Left-Right", color="green")
            # plt.plot(acc_z_mean, label="Anterior-Posterior", color="blue")
            plt.fill_between(plot_range, acc_x_mean - acc_x_std, acc_x_mean + acc_x_std, alpha=0.3, color="red")
            plt.fill_between(plot_range, acc_y_mean - acc_y_std, acc_y_mean + acc_y_std, alpha=0.3, color="green")
            plt.fill_between(plot_range, acc_z_mean - acc_z_std, acc_z_mean + acc_z_std, alpha=0.3, color="blue")
            plt.axvline(gait_event_percent_avg[0], color="black", alpha=0.5, linestyle="-")
            plt.axvline(gait_event_percent_avg[1], color="black", alpha=0.5, linestyle="--")
            plt.axvline(gait_event_percent_avg[2], color="black", alpha=0.5, linestyle="-.")
            plt.axvline(gait_event_percent_avg[3], color="black", alpha=0.5, linestyle=":")
            plt.axvline(gait_event_percent_avg[4], color="black", alpha=0.5, linestyle="-")
            plt.xlabel("Gait cycle [%]", fontsize=25)
            plt.ylabel("Acc [m/$s^2$]", fontsize=25)
            # plt.legend(fontsize=25)
            plt.tick_params(labelsize=25)
            plt.ylim(-7.5, 15)
            plt.savefig(csv_path.with_name(f"acc_multicycle_{condition_list[i]}.png"))
            plt.close()

        if target_imu == "thera":
            if condition_list[i] == "thera0-3":
                pass
            else:
                thera_ic_frame_r = op_IC_frame_dict["1"]["IC_R"]
                thera_ic_frame_l = op_IC_frame_dict["1"]["IC_L"]

        if target_imu == "thera_rhand":
            if condition_list[i] != "thera0-3":
                imu_df_therarhand = imu_df.loc[sub_IC_frame_l[0]:sub_IC_frame_l[-1]]  #患者の腰IMUと比較するため患者左足の初期接地を範囲に

                # 有効範囲内の左足初期接地における腰IMUのデータをプロット
                plt.plot(imu_df_therarhand["acc_x"], label="S-I", color="red")
                plt.plot(imu_df_therarhand["acc_y"], label="L-R", color="green")
                plt.plot(imu_df_therarhand["acc_z"], label="A-P", color="blue")
                plt.plot(imu_df_therarhand["acc"], label="RMS", color="black")
                [plt.axvline(sub_IC_frame_l[i], color="black", alpha=0.5, linestyle="-") for i in range(len(sub_IC_frame_l))]
                [plt.axvline(sub_IC_frame_r[i], color="black", alpha=0.5, linestyle="--") for i in range(len(sub_IC_frame_r))]
                plt.legend()
                plt.ylim(-17.5, 17.5)
                plt.xlabel("Frames [-]")
                plt.ylabel("Acc [m/$s^2$]")
                plt.savefig(csv_path.with_name(f"acc_{condition_list[i]}.png"))
                # plt.show()
                plt.close()

                # 開始時点の加速度を0とする
                imu_df_therarhand = imu_df_therarhand - imu_df_therarhand.loc[sub_IC_frame_l[0]]
                #各層でのRMSを計算
                rms_dict_rhand = {key : [] for key in ["rms", "std"]}
                rms_rhand_list = []
                for idx in range(gait_event_frame_array.shape[0]): #各歩行周期でのRMSを計算
                    rms_cycle = []
                    rms_n_cycle = []
                    for event_idx in range(gait_event_frame_array.shape[1]-1): #5つのイベントごとにRMSを計算
                        start_frame = gait_event_frame_array[idx][event_idx]
                        end_frame = gait_event_frame_array[idx][event_idx+1]
                        imu_df_therarhand_in_phase = imu_df_therarhand.loc[start_frame:end_frame]
                        rms = np.sqrt(np.mean(imu_df_therarhand_in_phase["acc"]**2))
                        rms_cycle.append(rms)
                    rms_rhand_list.append(rms_cycle)

                print(f"rms_rhand_list:{rms_rhand_list}")

                rms = np.mean(rms_rhand_list, axis=0)
                std = np.std(rms_rhand_list, axis=0)
                print(f"rms:{rms}")
                print(f"std:{std}")
                rms_dict_rhand["rms"] = rms
                rms_dict_rhand["std"] = std
                thera_r_hand_rms_list.append(rms_dict_rhand)
            else:
                pass





        if target_imu == "thera_lhand":
            if condition_list[i] != "thera0-3":
                imu_df_theralhand = imu_df.loc[sub_IC_frame_l[0]:sub_IC_frame_l[-1]]

                # 有効範囲内の左足初期接地における腰IMUのデータをプロット
                plt.plot(imu_df_theralhand["acc_x"], label="S-I", color="red")
                plt.plot(imu_df_theralhand["acc_y"], label="L-R", color="green")
                plt.plot(imu_df_theralhand["acc_z"], label="A-P", color="blue")
                plt.plot(imu_df_theralhand["acc"], label="RMS", color="black")
                [plt.axvline(sub_IC_frame_l[i], color="black", alpha=0.5, linestyle="-") for i in range(len(sub_IC_frame_l))]
                [plt.axvline(sub_IC_frame_r[i], color="black", alpha=0.5, linestyle="--") for i in range(len(sub_IC_frame_r))]
                plt.legend()
                plt.ylim(-17.5, 17.5)
                plt.xlabel("Frames [-]")
                plt.ylabel("Acc [m/$s^2$]")
                plt.savefig(csv_path.with_name(f"acc_{condition_list[i]}.png"))
                # plt.show()
                plt.close()

                # 開始時点の加速度を0とする
                imu_df_theralhand = imu_df_theralhand - imu_df_theralhand.loc[sub_IC_frame_l[0]]
                #各層でのRMSを計算
                rms_dict_lhand = {key : [] for key in ["rms", "std"]}
                rms_lhand_list = []
                for idx in range(gait_event_frame_array.shape[0]): #各歩行周期でのRMSを計算
                    rms_cycle = []
                    for event_idx in range(gait_event_frame_array.shape[1]-1): #5つのイベントごとにRMSを計算
                        start_frame = gait_event_frame_array[idx][event_idx]
                        end_frame = gait_event_frame_array[idx][event_idx+1]
                        imu_df_theralhand_in_phase = imu_df_theralhand.loc[start_frame:end_frame]
                        rms = np.sqrt(np.mean(imu_df_theralhand_in_phase["acc"]**2))
                        rms_cycle.append(rms)
                    rms_lhand_list.append(rms_cycle)

                rms = np.mean(rms_lhand_list, axis=0)
                std = np.std(rms_lhand_list, axis=0)
                rms_dict_lhand["rms"] = rms
                rms_dict_lhand["std"] = std
                thera_l_hand_rms_list.append(rms_dict_lhand)
            else:
                pass



    if target_imu == "sub":
        direction = ["Superior-Inferior", "Left-Right", "Anterior-Posterior"]

        # 各方向で各条件ごとのRMSの比較
        rms_n_list = sub_lumb_rms_list
        tittle_labels = ["S-I", "L-R", "A-P"]
        print(f"rms_n_list:{rms_n_list}")
        unasi_rms_n = [rms_n_list[0]["rms_x_n"], rms_n_list[0]["rms_y_n"], rms_n_list[0]["rms_z_n"]]
        asi_rms1_n = [rms_n_list[1]["rms_x_n"], rms_n_list[1]["rms_y_n"], rms_n_list[1]["rms_z_n"]]
        asi_rms2_n = [rms_n_list[2]["rms_x_n"], rms_n_list[2]["rms_y_n"], rms_n_list[2]["rms_z_n"]]

        # --- 標準偏差のデータをエラーバー用に抽出 ---
        unasi_errors_std = [rms_n_list[0]["std_x_n"], rms_n_list[0]["std_y_n"], rms_n_list[0]["std_z_n"]]
        asi1_errors_std = [rms_n_list[1]["std_x_n"], rms_n_list[1]["std_y_n"], rms_n_list[1]["std_z_n"]]
        asi2_errors_std = [rms_n_list[2]["std_x_n"], rms_n_list[2]["std_y_n"], rms_n_list[2]["std_z_n"]]
        # ------------------------------------------

        x = np.arange(len(tittle_labels))  # 各棒の位置 (0, 1, 2)
        width = 0.25  # 棒の幅

        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(x - width, unasi_rms_n, width, label='Unassisted', color="#c0c0c0", yerr=unasi_errors_std, capsize=5)
        rects2 = ax.bar(x, asi_rms1_n, width, label='Assisted PT1', color = "tab:blue", yerr=asi1_errors_std, capsize=5)
        rects3 = ax.bar(x + width, asi_rms2_n, width, label='Assisted PT2', color = "tab:red", yerr=asi2_errors_std, capsize=5)

        # ラベル、タイトル、凡例の設定
        ax.set_ylabel("RMS [1/m]", fontsize=30)
        ax.set_xlabel("Direction", fontsize=30)
        # ax.set_title(title)
        ax.set_ylim(0, 6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(tittle_labels, fontsize=30)
        ax.tick_params(labelsize=30)
        ax.legend(fontsize=25)
        ax.margins(y=0.1)

        # 各棒の上に値を表示 (オプション)
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=25)

        # autolabel(rects1)
        # autolabel(rects2)
        # autolabel(rects3)

        fig.tight_layout()
        plt.savefig(csv_path.with_name(f"rms_normalized.png"))
        # plt.show()
        plt.close()


        # 各方向ごとに各相における各条件RMSの比較
        for i, target in enumerate(["rms_x", "rms_y", "rms_z"]):
            # print(f"sub_lumb_rms_list:{sub_lumb_rms_list}")
            rms_list = sub_lumb_rms_list
            tittle_labels = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
            labels = ["rms_x", "rms_y", "rms_z"]
            # 処理前後のデータの最初の要素を取り出す
            unasi_rms = rms_list[0][target]  #介助無し
            asi_rms1 = rms_list[1][target]  #介助あり 療法士1
            asi_rms2 = rms_list[2][target]  #介助あり 療法士2

            x = np.arange(len(tittle_labels))  # 各棒の位置 (0, 1, 2, 3)
            width = 0.2  # 棒の幅

            # print(f"unasi_rms:{unasi_rms}")
            # print(f"asi_rms1:{asi_rms1}")
            # print(f"asi_rms2:{asi_rms2}")

            fig, ax = plt.subplots(figsize=(12, 8))
            rects1 = ax.bar(x - width, unasi_rms, width, label='Unassisted')
            rects2 = ax.bar(x, asi_rms1, width, label='Assisted PT1')
            rects3 = ax.bar(x + width, asi_rms2, width, label='Assisted PT2')

            # ラベル、タイトル、凡例の設定
            ax.set_ylabel("RMS [m/$s^2$]", fontsize=30)
            ax.set_xlabel(f"{direction[i]}", fontsize=30)
            # ax.set_title(title)
            ax.set_ylim(0, 5)
            ax.set_xticks(x)
            ax.set_xticklabels(tittle_labels, fontsize=30)
            ax.tick_params(labelsize=30)
            ax.legend(fontsize=25)
            ax.margins(y=0.1)

            # 各棒の上に値を表示 (オプション)
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=20)

            autolabel(rects1)
            autolabel(rects2)
            autolabel(rects3)

            fig.tight_layout()
            plt.savefig(csv_path.with_name(f"rms_phase_{target}.png"))
            # plt.show()
            plt.close()


# 左右間でのRMSの比較
for ithera in range(3):
    if ithera == 0:
        continue

    condition = condition_list[ithera]  # "thera1-1", "thera2-1"
    # 各相でrms_Rhand, rms_Lhandの比較(棒グラフ)
    print(f"thera_r_hand_rms_list:{thera_r_hand_rms_list}")
    print(f"thera_l_hand_rms_list:{thera_l_hand_rms_list}")
    """
    # thera_r_hand_rms_list:
    # [{'rms': array([ 9.15172206, 10.34387119,  9.29644235, 10.33462626]),  #療法士1
    # 'std': array([0.10585933, 0.09781776, 0.52143535, 0.1499961 ])},
    # {'rms': array([10.55574518, 10.76164765,  9.56614682, 10.10350017]),  #療法士2
    # 'std': array([0.21175673, 0.06003275, 0.09850981, 0.04442505])}]
    """

    print("ithear:", ithera)
    Rhand_values = thera_r_hand_rms_list[ithera-1]["rms"]
    Lhand_values = thera_l_hand_rms_list[ithera-1]["rms"]

    Rhand_std = thera_r_hand_rms_list[ithera-1]["std"]
    Lhand_std = thera_l_hand_rms_list[ithera-1]["std"]

    x = np.arange(4)  # 各棒の位置 (0, 1, 2)
    width = 0.35  # 棒の幅

    fig, ax = plt.subplots(figsize=(10, 8))
    rects1 = ax.bar(x + width/2, Rhand_values, width, label='Right hand', color="#90ee90", yerr=Rhand_std, capsize=5)
    rects2 = ax.bar(x - width/2, Lhand_values, width, label='Left hand', color="#deb887", yerr=Lhand_std, capsize=5)

    # ラベル、タイトル、凡例の設定
    ax.set_ylabel("RMS [m/$s^2$]", fontsize=30)
    # ax.set_xlabel("Phase")
    # ax.set_title(title)
    ax.set_ylim(0, 5.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Phase 1", "Phase 2", "Phase 3", "Phase 4"], fontsize=30)
    ax.tick_params(labelsize=30)
    # ax.legend(fontsize=25)
    ax.margins(y=0.1)

    # 各棒の上に値を表示 (オプション)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=20)

    # autolabel(rects1)
    # autolabel(rects2)

    fig.tight_layout()
    plt.savefig(csv_path.parent.with_name(f"hand_rms_{condition}.png"))
    # plt.show()
    plt.close()