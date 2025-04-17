from pathlib import Path
import pandas as pd
import imu_module as imu
import copy
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import numpy as np

root_dir = Path(r"G:\gait_pattern\20241114_ota_test\IMU\20241114")
target_imus = ["sync", "subLumb", "theraLumb", "theraRhand", "theraLhand"]
# target_dir = root_dir / target_imus[0]

root_dir_op = Path(r"G:\gait_pattern\20241114_ota_test\gopro\sagi")
condition_list = ["sub0_abngait", "sub0_asgait_1"]
op_ic_frame_paths = [root_dir_op / f"{condition}_ic_frame.pickle" for condition in condition_list]
op_toeoff_frame_paths = [root_dir_op / f"{condition}_to_frame.pickle" for condition in condition_list]

#手を降り始めた瞬間のフレーム（目視）sub0_abngait, sub0_asgait_1の順
op_sync_frame = [345, 461]
#imu同士のフレーム調整用
sync_start_time_list = []


imu_result_dict = {key:[] for key in target_imus}
for target_imu in target_imus:
    target_dir = root_dir / target_imu
    csv_list = [file for file in target_dir.glob("*.csv")
        if file.name.startswith("2t2") or file.name.startswith("3") ]
    sub_lumb_rms_list = []
    for i, csv_path in enumerate(csv_list):
        print(f"\ncsv_path:{csv_path}")
        # OpenPose結果を読み込む
        openpose_res_path = root_dir_op / f"{condition_list[i]}_gait_params.pickle"
        openpose_res_dict = imu.loadPickle(openpose_res_path)
        # openpose_res_dict:{'0': {'walk_speed': 0.6850662125743122, 'stride_time': 1.2833333333333334, 'stride_length_l': 0.8662926063741502, 'step_length_l': 0.44607708502794957, 'stance_phase_ratio_l': 62.33766233766234, 'stride_length_r': 0.8554579614359269, 'step_length_r': 0.4279558456969168, 'stance_phase_ratio_r': 70.12987012987013, 'step_avg': 0.4370164653624332}}

        #初期接地データを読み込む
        op_ic_frame_dict = imu.loadPickle(op_ic_frame_paths[i])
        print(f"op_ic_frame_dict:{op_ic_frame_dict}")
        # op_ic_frame_dict:{'0': {'IC_R': [1309, 1380, 1450], 'IC_L': [1349, 1421]}, '1': {'IC_R': [1380, 1453], 'IC_L': [1348, 1418, 1489]}}
        #踵離地のフレームを取得
        op_toeoff_frame_dict = imu.loadPickle(op_toeoff_frame_paths[i])
        phase_frame_dict = imu.loadPickle(root_dir_op / f"{condition_list[i]}_phase_frame.pickle")
        phase_percent_list_dict = imu.loadPickle(root_dir_op / f"{condition_list[i]}_phase_percent.pickle")
        print(f"phase_frame_dict:{phase_frame_dict}")
        print(f"phase_percent_list_dict:{phase_percent_list_dict}")




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
        #RMSを追加
        imu_df["acc"] = (imu_df["acc_x"]**2 + imu_df["acc_y"]**2 + imu_df["acc_z"]**2) **0.5
        # imu_df["acc_xyrms"] = (imu_df["acc_x"]**2 + imu_df["acc_y"]**2) / 3 **0.5

        # print(f"imu_df:{imu_df.head()}")

        #同期用IMuなら同期タイミングの確認
        if target_imu == "sync":
            #手を降り始めた瞬間のフレーム xの変化が大きいとしてacc_xから同期フレームを求める（yは一応）
            imu_sync_frame_x = imu.find_sync_frame(imu_df, "acc_x")
            imu_sync_frame_y = imu.find_sync_frame(imu_df, "acc_y")

            #IMUデータを同期フレームからの値に調整してIMUメモリにおける開始時刻を取得
            frame_diff = imu_sync_frame_x - op_sync_frame[i]
            imu_df.index = imu_df.index - frame_diff
            imu_df = imu_df[imu_df.index >= 0]
            # print(f"imu_df:{imu_df.head()}")

            plt.rcParams["font.size"] = 20

            # imu.create_plot(imu_df.index, [imu_df["acc_x"], imu_df["acc_y"]], title= f"{condition_list[i]}_accxy", xlabel="frames [-]", ylabel="acc [m/s^2]",
            #                 labels=["acc_x", "acc_y"], colors=["red", "blue"], save_path=csv_path.with_name(f"{condition_list[i]}_acc_XY.png"))

            # imu.create_plot(imu_df.index, [imu_df["acc_x"], imu_df["acc_y"], imu_df["acc_xyrms"]], title=None,
            #                 xlabel="frames [-]", ylabel="acc [m/s^2]", labels=["acc_x", "acc_y", "acc_z"], colors=["red", "blue", "green"], save_path=csv_path.with_name(f"{condition_list[i]}_acc_xyrms.png"))

            # imu.create_plot(imu_df.index, [imu_df["acc_x"], imu_df["acc_y"], imu_df["acc_z"]], title=None,
            #                 xlabel="frames [-]", ylabel="acc [m/s^2]", labels=["acc_x", "acc_y", "acc_z"], colors=["red", "blue", "green"], save_path=csv_path.with_name(f"{condition_list[i]}_acc_XYZ.png"))

            # imu.create_plot(imu_df.index, [imu_df["acc_rms"]], title=None,
            #                 xlabel="frames [-]", ylabel="acc [m/s^2]", labels=["acc_rms"], colors=["black"], save_path=csv_path.with_name(f"{condition_list[i]}_acc_rms.png"))
            # plt.close()

            continue

        # openposeと開始フレームを合わせる
        imu_df.index = imu_df.index - frame_diff
        imu_df = imu_df[imu_df.index >= 0]

        imu_data_dict = {key : [] for key in target_imus}

        figsize_xyz = (11, 11)

        if target_imu == "subLumb":
            #
            sub_ic_frame_r = op_ic_frame_dict["0"]["IC_R"]
            sub_ic_frame_l = op_ic_frame_dict["0"]["IC_L"]
            sub_ic_block_r = [[sub_ic_frame_r[i], sub_ic_frame_r[i+1]] for i in range(len(sub_ic_frame_r)-1)]
            sub_ic_block_l = [[sub_ic_frame_l[i], sub_ic_frame_l[i+1]] for i in range(len(sub_ic_frame_l)-1)]
            imu_df_sublumb = imu_df.loc[sub_ic_frame_l[0]:sub_ic_frame_l[-1]]
            imu_df_sublumb = imu_df_sublumb - imu_df_sublumb.loc[sub_ic_frame_l[0]]

            # ratio_x = abs(imu_df_sublumb["acc_x"].max() / imu_df_sublumb["acc_x"].min())
            # ratio_y = abs(imu_df_sublumb["acc_y"].max() / imu_df_sublumb["acc_y"].min())
            # ratio_z = abs(imu_df_sublumb["acc_z"].max() / imu_df_sublumb["acc_z"].min())
            # print(f"ratio_x:{ratio_x:.2f}, ratio_y:{ratio_y:.2f}, ratio_z:{ratio_z:.2f}")
            # print(f"x_max = {imu_df_sublumb['acc_x'].max():.2f}, x_min = {imu_df_sublumb['acc_x'].min():.2f}")
            # print(f"y_max = {imu_df_sublumb['acc_y'].max():.2f}, y_min = {imu_df_sublumb['acc_y'].min():.2f}")
            # print(f"z_max = {imu_df_sublumb['acc_z'].max():.2f}, z_min = {imu_df_sublumb['acc_z'].min():.2f}")

            # 相が切り替わるフレームと割合を取得
            print(f"phase_percent_list_dict:{phase_percent_list_dict['0']}")
            check_phase_frame_dict = {key: [] for key in op_ic_frame_dict.keys()}
            for i in range(len(phase_percent_list_dict["0"])):
                if phase_frame_dict["0"][i][0] == op_ic_frame_dict["0"]["IC_L"][0]:
                    check_phase_frame_dict["0"] = phase_frame_dict["0"][i]
            print(f"check_phase_frame_dict:{check_phase_frame_dict}")

            check_phase_frame_list = check_phase_frame_dict["0"]
            check_phase_percent_list = []
            for idx in range(len(check_phase_frame_list)):
                persent = (check_phase_frame_list[idx]-check_phase_frame_list[0]) / (check_phase_frame_list[-1] - check_phase_frame_list[0]) * 100
                check_phase_percent_list.append(persent)
            print(f"check_phase_percent_list:{check_phase_percent_list}")

            rms_dict = {key : [] for key in ["rms_x", "rms_y", "rms_z"]}
            rms_x_list , rms_y_list, rms_z_list = [], [], []
            for idx in range(len(check_phase_frame_list)-1):
                start_frame = check_phase_frame_list[idx]
                end_frame = check_phase_frame_list[idx+1]
                imu_df_sublumb_in_phase = imu_df_sublumb.loc[start_frame:end_frame]
                # print(f'imu_df_sublumb_in_phase["acc_x"]:{imu_df_sublumb_in_phase["acc_x"]}')
                rms_x = np.sqrt(np.mean(imu_df_sublumb_in_phase["acc_x"]**2))
                rms_y = np.sqrt(np.mean(imu_df_sublumb_in_phase["acc_y"]**2))
                rms_z = np.sqrt(np.mean(imu_df_sublumb_in_phase["acc_z"]**2))
                rms_x_list.append(rms_x)
                rms_y_list.append(rms_y)
                rms_z_list.append(rms_z)
            rms_dict["rms_x"] = rms_x_list
            rms_dict["rms_y"] = rms_y_list
            rms_dict["rms_z"] = rms_z_list
            print(f"rms_dict:{rms_dict}")
            sub_lumb_rms_list.append(rms_dict)


            # # 加速度を平面ごとにプロット
            # plt.figure(figsize=figsize_xyz)
            # plt.plot(imu_df_sublumb["acc_x"], imu_df_sublumb["acc_y"])
            # plt.scatter(imu_df_sublumb.iloc[0,1], imu_df_sublumb.iloc[0,2], color="red", s=100)
            # # plt.title(f"{condition_list[i]}_XY", fontsize=20)
            # plt.xlabel("Acc Superior-Inferior [m/$s^2$]", fontsize=30)
            # plt.ylabel("Acc Left-Right [m/$s^2$]", fontsize=30)
            # plt.tick_params(labelsize=30)
            # plt.axvline(0, color="black", alpha=0.5)
            # plt.axhline(0, color="black", alpha=0.5)
            # plt.xlim(-5, 15)
            # plt.ylim(-13, 7)
            # plt.gca().set_aspect('equal', adjustable='box')
            # plt.grid()
            # plt.tight_layout()
            # # plt.show()
            # plt.savefig(csv_path.with_name(f"{condition_list[i]}_acc_XY.png"))
            # plt.close()

            # plt.figure(figsize=figsize_xyz)
            # plt.plot(imu_df_sublumb["acc_y"], imu_df_sublumb["acc_z"])
            # plt.scatter(imu_df_sublumb.iloc[0,2], imu_df_sublumb.iloc[0,3], color="red", s=100)
            # # plt.title(f"{condition_list[i]}_YZ", fontsize=20)
            # plt.xlabel("Acc Left-Right [m/$s^2$]", fontsize=30)
            # plt.ylabel("Acc Posterior-Anterior [m/$s^2$]", fontsize=30)
            # plt.tick_params(labelsize=30)
            # plt.axvline(0, color="black", alpha=0.5)
            # plt.axhline(0, color="black", alpha=0.5)
            # # plt.xlim(-10, 10)
            # # plt.ylim(-10, 10)
            # plt.axis("equal")
            # plt.grid()
            # plt.tight_layout()
            # # plt.show()
            # plt.savefig(csv_path.with_name(f"{condition_list[i]}_acc_YZ.png"))
            # plt.close()

            # plt.figure(figsize=figsize_xyz)
            # plt.plot(imu_df_sublumb["acc_z"], imu_df_sublumb["acc_x"])
            # plt.scatter(imu_df_sublumb.iloc[0,3], imu_df_sublumb.iloc[0,1], color="red", s=100)
            # # plt.title(f"{condition_list[i]}_ZX", fontsize=20)
            # plt.xlabel("Acc Anterior-Posterior [m/$s^2$]", fontsize=30)
            # plt.ylabel("Acc Superior-Inferior [m/$s^2$]", fontsize=30)
            # plt.tick_params(labelsize=30)
            # plt.axvline(0, color="black", alpha=0.5)
            # plt.axhline(0, color="black", alpha=0.5)
            # # plt.xlim(-10, 10)
            # # plt.ylim(-10, 10)
            # plt.axis("equal")
            # plt.grid()
            # plt.tight_layout()
            # # plt.show()
            # plt.savefig(csv_path.with_name(f"{condition_list[i]}_acc_ZX.png"))
            # plt.close()

            imu_df_sublumb.index = (imu_df_sublumb.index - imu_df_sublumb.index[0]) / (imu_df_sublumb.index[-1] - imu_df_sublumb.index[0]) * 100

            # 時間軸をとって加速度をプロット abngaitのときになぜかylimが適用されない（showから保存した）
            plt.figure(figsize=(10,8))
            plt.plot(imu_df_sublumb.index, imu_df_sublumb["acc_x"], label="鉛直方向（上向き＋）", color="red")
            plt.plot(imu_df_sublumb.index, imu_df_sublumb["acc_y"], label="左右方向（左向き＋）", color="green")
            plt.plot(imu_df_sublumb.index, imu_df_sublumb["acc_z"], label="前後方向（前向き＋）", color="blue")
            # plt.plot(imu_df_sublumb.index, imu_df_sublumb["acc_x"], label="S-I", color="red")
            # plt.plot(imu_df_sublumb.index, imu_df_sublumb["acc_y"], label="L-R", color="green")
            # plt.plot(imu_df_sublumb.index, imu_df_sublumb["acc_z"], label="A-P", color="blue")
            plt.legend(fontsize=30)
            plt.xlabel("Gait cycle [%]", fontsize=30)
            plt.ylabel("Acc [m/$s^2$]", fontsize=30)
            plt.tick_params(labelsize=30)
            plt.ylim(-15, 15)
            plt.xlim(imu_df_sublumb.index[0], imu_df_sublumb.index[-1])
            # plt.title(f"{condition_list[i]}_acc_XYZ", fontsize=20)
            # [plt.axvline(frame, color="black", alpha=0.8) for frame in phase_percent_list_dict["0"][0]]
            print(f"phase_percent_list_dict[0][0]:{phase_percent_list_dict['0'][0]}")
            plt.axvline(phase_percent_list_dict["0"][0][1], color="black", alpha=0.9, linestyle="-")
            plt.axvline(phase_percent_list_dict["0"][0][2], color="black", alpha=0.9, linestyle="--")
            plt.grid()
            plt.tight_layout()
            plt.savefig(csv_path.with_name(f"{condition_list[i]}_acc_XYZ.png"))
            plt.show()
            plt.close()

        if target_imu == "theraLumb":
            if condition_list[i] == "sub0_abngait":
                pass
            else:
                thera_ic_frame_r = op_ic_frame_dict["1"]["IC_R"]
                thera_ic_frame_l = op_ic_frame_dict["1"]["IC_L"]

                # imu_df_theralumb = imu_df.loc[thera_ic_frame_l[0]:thera_ic_frame_l[1]]
                # imu_df_theralumb = imu_df_theralumb - imu_df_theralumb.loc[thera_ic_frame_l[0]]

                # plt.figure(figsize=figsize_xyz)
                # plt.plot(imu_df_theralumb["acc_x"], imu_df_theralumb["acc_y"])
                # plt.scatter(imu_df_theralumb.iloc[0,1], imu_df_theralumb.iloc[0,2], color="red", s=100)
                # # plt.title(f"{condition_list[i]}_XY")
                # plt.xlabel("acc_x [m/s^2]")
                # plt.ylabel("acc_y [m/s^2]")
                # plt.axvline(0, color="black")
                # plt.axhline(0, color="black")
                # plt.grid()
                # plt.tight_layout()
                # # plt.show()
                # plt.savefig(csv_path.with_name(f"{condition_list[i]}_acc_XY.png"))
                # plt.cla()

                # plt.figure(figsize=figsize_xyz)
                # plt.plot(imu_df_theralumb["acc_y"], imu_df_theralumb["acc_z"])
                # plt.scatter(imu_df_theralumb.iloc[0,2], imu_df_theralumb.iloc[0,3], color="red", s=100)
                # # plt.title(f"{condition_list[i]}_YZ")
                # plt.xlabel("acc_y [m/s^2]")
                # plt.ylabel("acc_z [m/s^2]")
                # plt.axvline(0, color="black")
                # plt.axhline(0, color="black")
                # plt.grid()
                # plt.tight_layout()
                # # plt.show()
                # plt.savefig(csv_path.with_name(f"{condition_list[i]}_acc_YZ.png"))
                # plt.cla()

                # plt.figure(figsize=figsize_xyz)
                # plt.plot(imu_df_theralumb["acc_z"], imu_df_theralumb["acc_x"])
                # plt.scatter(imu_df_theralumb.iloc[0,3], imu_df_theralumb.iloc[0,1], color="red", s=100)
                # # plt.title(f"{condition_list[i]}_ZX")
                # plt.xlabel("acc_z [m/s^2]")
                # plt.ylabel("acc_x [m/s^2]")
                # plt.axvline(0, color="black")
                # plt.axhline(0, color="black")
                # plt.grid()
                # plt.tight_layout()
                # # plt.show()
                # plt.savefig(csv_path.with_name(f"{condition_list[i]}_acc_ZX.png"))
                # plt.cla()

                # # print(f"imu_df_theralumb.index:{imu_df_theralumb.index}")
                # imu_df_theralumb.index = (imu_df_theralumb.index - imu_df_theralumb.index[0]) / (imu_df_theralumb.index[-1] - imu_df_theralumb.index[0]) * 100
                # # print(f"imu_df_theralumb.index:{imu_df_theralumb.index}")
                # plt.plot(imu_df_theralumb.index, imu_df_theralumb["acc_x"], label="acc_x", color="red")
                # plt.plot(imu_df_theralumb.index, imu_df_theralumb["acc_y"], label="acc_y", color="green")
                # plt.plot(imu_df_theralumb.index, imu_df_theralumb["acc_z"], label="acc_z", color="blue")
                # plt.plot(imu_df_theralumb.index, imu_df_theralumb["acc_rms"], label="acc_rms", color="black")
                # plt.legend()
                # plt.xlabel("Gait cycle [%]")
                # plt.ylabel("acc [m/s^2]")
                # plt.ylim(-10, 10)
                # plt.xlim(imu_df_theralumb.index[0], imu_df_theralumb.index[-1])
                # # plt.title(f"{condition_list[i]}_acc_XYZ")
                # [plt.axvline(frame, color="black", alpha=0.5) for frame in thera_ic_frame_l]
                # plt.tight_layout()
                # plt.savefig(csv_path.with_name(f"{condition_list[i]}_acc_XYZ.png"))
                # plt.cla()

        if target_imu == "theraRhand":
            if condition_list[i] == "sub0_abngait":
                pass
            else:
                # imu_df_therarhand = imu_df.loc[sub_ic_frame_l[0]:sub_ic_frame_l[1]]  #患者の腰IMUと比較するため患者左足の初期接地を範囲に
                # imu_df_therarhand.index = (imu_df_therarhand.index - imu_df_therarhand.index[0]) / (imu_df_therarhand.index[-1] - imu_df_therarhand.index[0]) * 100
                imu_df_therarhand = imu_df.copy() - imu_df.copy().loc[sub_ic_frame_l[0]]
                rms_Rhand = []
                for idx in range(len(check_phase_frame_list)-1):
                    start_frame = check_phase_frame_list[idx]
                    end_frame = check_phase_frame_list[idx+1]
                    imu_df_therarhand_in_phase = imu_df_therarhand.loc[start_frame:end_frame]
                    # print(f"imu_df_therarhand_in_phase:{imu_df_therarhand_in_phase}")
                    rms_Rhand.append(np.sqrt(np.mean(imu_df_therarhand_in_phase["acc"]**2)))

                print(f"rms_Rhand:{rms_Rhand}")

        #         imu.create_plot(imu_df_therarhand.index, [imu_df_therarhand["acc_x"], imu_df_therarhand["acc_y"], imu_df_therarhand["acc_z"]], title=None,
        #                         xlabel="Gait cycle [%]", ylabel="Acc [m/$s^2$]", labels=["acc_x", "acc_y", "acc_z"], colors=["red", "green", "blue"], save_path=csv_path.with_name(f"{condition_list[i]}_acc_XYZ.png"))

        #         imu.create_plot(imu_df_therarhand.index, [imu_df_therarhand["acc_x"], imu_df_therarhand["acc_y"], imu_df_therarhand["acc_z"], imu_df_therarhand["acc_rms"]], title=None,
        #                         xlabel="Gait cycle [%]", ylabel="Acc [m/$s^2$]", labels=["acc_x", "acc_y", "acc_z", "acc_rms"], colors=["red", "green", "blue", "black"],
        #                         save_path=csv_path.with_name(f"{condition_list[i]}_acc_rms.png"), plt_close=False, plt_save=False, legend_outside=True)
        #         plt.axvline(openpose_res_dict["0"]["stance_phase_ratio_l"], color="black", alpha=0.5, linestyle="--")
        #         plt.savefig(csv_path.with_name(f"{condition_list[i]}_acc_rms.png"))



        if target_imu == "theraLhand":
            if condition_list[i] == "sub0_abngait":
                pass
            else:
                # imu_df_theralhand = imu_df.loc[sub_ic_frame_l[0]:sub_ic_frame_l[1]]
                # imu_df_theralhand.index = (imu_df_theralhand.index - imu_df_theralhand.index[0]) / (imu_df_theralhand.index[-1] - imu_df_theralhand.index[0]) * 100
                imu_df_theralhand = imu_df.copy() - imu_df.copy().loc[sub_ic_frame_l[0]]
                rms_Lhand = []
                for idx in range(len(check_phase_frame_list)-1):
                    start_frame = check_phase_frame_list[idx]
                    end_frame = check_phase_frame_list[idx+1]
                    imu_df_theralhand_in_phase = imu_df_theralhand.loc[start_frame:end_frame]
                    rms_Lhand.append(np.sqrt(np.mean(imu_df_theralhand_in_phase["acc"]**2)))

                print(f"rms_Lhand:{rms_Lhand}")
                # imu.create_plot(imu_df_theralhand.index, [imu_df_theralhand["acc_x"], imu_df_theralhand["acc_y"], imu_df_theralhand["acc_z"]], title=None,
                #                 xlabel="Gait cycle [%]", ylabel="Acc [m/$s^2$]", labels=["acc_x", "acc_y", "acc_z"], colors=["red", "green", "blue"], save_path=csv_path.with_name(f"{condition_list[i]}_acc_XYZ.png"))

                # imu.create_plot(imu_df_theralhand.index, [imu_df_theralhand["acc_x"], imu_df_theralhand["acc_y"], imu_df_theralhand["acc_z"], imu_df_theralhand["acc_rms"]], title=None,
                #                 xlabel="Gait cycle [%]", ylabel="Acc [m/$s^2$]", labels=["acc_x", "acc_y", "acc_z", "acc_rms"], colors=["red", "green", "blue", "black"],
                #                 save_path=csv_path.with_name(f"{condition_list[i]}_acc_rms.png"), plt_close=False, plt_save=False, legend_outside=True)
                # plt.axvline(openpose_res_dict["0"]["stance_phase_ratio_l"], color="black", alpha=0.5, linestyle="--")
                # plt.savefig(csv_path.with_name(f"{condition_list[i]}_acc_rms.png"))

                # titjle_labels = []


    if target_imu == "subLumb":
        direction = ["Superior-Inferior", "Left-Right", "Anterior-Posterior"]
        for i, target in enumerate(["rms_x", "rms_y", "rms_z"]):
            print(f"sub_lumb_rms_list:{sub_lumb_rms_list}")
            rms_list = sub_lumb_rms_list
            tittle_labels = ["Phase 1", "Phase 2", "Phase 3"]
            labels = ["rms_x", "rms_y", "rms_z"]
            # 処理前後のデータの最初の要素を取り出す
            before_values = rms_list[0][target][:]
            after_values = rms_list[1][target][:]

            x = np.arange(len(labels))  # 各棒の位置 (0, 1, 2)
            width = 0.3  # 棒の幅

            fig, ax = plt.subplots(figsize=(10, 8))
            rects1 = ax.bar(x - width/2, before_values, width, label='Unassisted')
            rects2 = ax.bar(x + width/2, after_values, width, label='Assisted')

            # ラベル、タイトル、凡例の設定
            ax.set_ylabel("RMS [m/$s^2$]", fontsize=30)
            ax.set_xlabel(f"{direction[i]}", fontsize=30)
            # ax.set_title(title)
            # ax.set_ylim(0, 5)
            ax.set_xticks(x)
            ax.set_xticklabels(tittle_labels, fontsize=30)
            ax.tick_params(labelsize=30)
            # ax.legend(fontsize=30)
            ax.margins(y=0.1)

            # 各棒の上に値を表示 (オプション)
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=30)

            autolabel(rects1)
            autolabel(rects2)

            fig.tight_layout()
            plt.savefig(csv_path.with_name(f"subLumb_{target}.png"))
            # plt.show()
            plt.close()

# Phase1,2,3でrms_Rhand, rms_Lhandの比較(棒グラフ)
Rhand_values = [rms_Rhand[0], rms_Rhand[1], rms_Rhand[2]]
Lhand_values = [rms_Lhand[0], rms_Lhand[1], rms_Lhand[2]]

x = np.arange(3)  # 各棒の位置 (0, 1, 2)
width = 0.35  # 棒の幅

fig, ax = plt.subplots(figsize=(10, 8))
rects1 = ax.bar(x - width/2, Rhand_values, width, label='Right hand')
rects2 = ax.bar(x + width/2, Lhand_values, width, label='Left hand')

# ラベル、タイトル、凡例の設定
ax.set_ylabel("RMS [m/$s^2$]", fontsize=30)
# ax.set_xlabel("Phase")
# ax.set_title(title)
# ax.set_ylim(0, 5)
ax.set_xticks(x)
ax.set_xticklabels(["Phase 1", "Phase 2", "Phase 3"], fontsize=30)
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
                    ha='center', va='bottom', fontsize=30)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig(csv_path.with_name(f"hand_rms.png"))
# plt.show()
plt.close()











