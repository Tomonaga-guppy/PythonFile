import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import calu_saggital_gait_module as sggait
import sys

root_dir = Path(r"G:\gait_pattern\20241114_ota_test\gopro\sagi")
# condition_list = ["sub0_abngait", "sub0_asgait_2"]
# condition_list = ["sub0_abngait"]
condition_list = ["sub0_asgait_2"]

def main():
    # 動画を切り出す際に行っているはずなので削除
    # #各条件ごとに歩行開始目安のフレームを目視で確認してcsvファイルに保存
    # frame_ch_csv = root_dir / f"FrameCheck.csv"
    # if not frame_ch_csv.exists():
    #     sggait.mkFrameCheckCSV(frame_ch_csv, condition_list)

    for condition in condition_list:
        print(f"condition = {condition}")
        #歩行開始目安のフレームを取得
        frame_ch_csv = root_dir / "FrameCheck.csv"
        frame_check_df = pd.read_csv(frame_ch_csv)
        start_frame = frame_check_df[f"{condition}_Start"].values[0]
        #2d上でのキーポイントを取得"
        openpose_dir = root_dir / (condition + "_udCropped_op.json")
        csv_files = sggait.mkCSVOpenposeData(openpose_dir, start_frame, overwrite=True)
        # OpenPose処理をした開始フレームを取得
        frame_ch_csv = root_dir / "FrameCheck.csv"
        frame_check_df = pd.read_csv(frame_ch_csv)
        start_frame = frame_check_df[f"{condition}_Start"].values[0]
        end_frame = frame_check_df[f"{condition}_End"].values[0]

        print(f"csv_files = {csv_files}")


        openpose_df_dict = {key:[] for key in range(len(csv_files))}
        for iPeople, csv_file in enumerate(csv_files):
            read_df = pd.read_csv(csv_file, index_col=0)
            read_df = sggait.fillindex(read_df)
            openpose_df_dict[iPeople] = read_df.loc[start_frame:end_frame]

        """
        openpose_df_dict = {0:                 Nose_x      Nose_y    Nose_p       Neck_x      Neck_y    Neck_p  RShoulder_x  RShoulder_y  RShoulder_p  ...    RBigToe_x    RBigToe_y  RBigToe_p  RSmallToe_x  RSmallToe_y  RSmallToe_p      RHeel_x
        RHeel_y   RHeel_p
        frame_num                                                                                                               ...
        1118       2665.214118  753.510564  0.893658  2778.941622  817.995691  0.844580  2749.118959   818.192262     0.848053  ...  2679.672847  1594.613102   0.900727  2679.432080  1576.886164     0.816999  2775.668392  1547.994006  0.751330
        ...                ...         ...       ...          ...         ...       ...          ...          ...          ...  ...          ...          ...        ...          ...          ...          ...          ...          ...       ...
        1397        809.060745  728.311302  0.877542   913.061947  752.938364  0.791799   925.059367   753.035688     0.711717  ...   934.296681  1536.595916   0.765661   952.717256  1536.127353     0.694959  1018.697317  1565.385644  0.769688
        [280 rows x 75 columns], 1:                  Nose_x       Nose_y    Nose_p       Neck_x       Neck_y    Neck_p  RShoulder_x  RShoulder_y  RShoulder_p  ...    RBigToe_x    RBigToe_y  RBigToe_p  RSmallToe_x  RSmallToe_y  RSmallToe_p      RHeel_x
        RHeel_y   RHeel_p
        frame_num                                                                                                                  ...
        1118       1.182388e+03  1166.856617  0.052723  1188.314411  1172.961306  0.195757  1188.657481  1172.860319     0.167680  ...  1176.309025  1291.229718   0.205865  1176.298257  1296.804564     0.180736  1176.248332  1291.260055  0.144976
        ...                 ...          ...       ...          ...          ...       ...          ...          ...          ...  ...          ...          ...        ...          ...          ...          ...          ...          ...       ...
        1397       0.000000e+00     0.000000  0.000000     0.000000     0.000000  0.000000     0.000000     0.000000     0.000000  ...     0.000000     0.000000   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000  0.000000
        [280 rows x 75 columns]}
        """

        #スプライン補間
        openpose_df_post_dict = sggait.splineAndFillter(openpose_df_dict, sampling_freq=60, order=4, cutoff_freq=6)
        #openpose_dictをpickleで保存
        openpose_df_dict_path = root_dir / f"{condition}_2d_dict.pickle"
        openpose_df_post_dict_path = root_dir / f"{condition}_2d_post_dict.pickle"
        sggait.save_as_pickle(openpose_df_dict, openpose_df_dict_path)
        sggait.save_as_pickle(openpose_df_post_dict, openpose_df_post_dict_path)

        openpose_df_dict_chSw = sggait.checkSwithing(openpose_df_dict)
        openpose_df_dict_chSw_path = root_dir / f"{condition}_2d_dict_chSw.pickle"
        sggait.save_as_pickle(openpose_df_dict_chSw, openpose_df_dict_chSw_path)





        continue
        sys.exit()

        mid_hip_sagttal_2d = sggait.cubic_spline_interpolation(keypoints_sagittal_2d[:, 8, :], sagi_frame_2d) #[frame, 2]
        neck_sagittal_2d = sggait.cubic_spline_interpolation(keypoints_sagittal_2d[:, 1, :], sagi_frame_2d)
        lhip_sagittal_2d = sggait.cubic_spline_interpolation(keypoints_sagittal_2d[:, 12, :], sagi_frame_2d)
        lknee_sagittal_2d = sggait.cubic_spline_interpolation(keypoints_sagittal_2d[:, 13, :], sagi_frame_2d)
        lankle_sagittal_2d = sggait.cubic_spline_interpolation(keypoints_sagittal_2d[:, 14, :], sagi_frame_2d)
        lbigtoe_sagittal_2d = sggait.cubic_spline_interpolation(keypoints_sagittal_2d[:, 19, :], sagi_frame_2d)
        lsmalltoe_sagittal_2d = sggait.cubic_spline_interpolation(keypoints_sagittal_2d[:, 20, :], sagi_frame_2d)
        lheel_sagittal_2d = sggait.cubic_spline_interpolation(keypoints_sagittal_2d[:, 21, :], sagi_frame_2d)

        mid_hip_sagttal_2d_filltered = np.array([sggait.butter_lowpass_fillter(mid_hip_sagttal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        neck_sagittal_2d_filltered = np.array([sggait.butter_lowpass_fillter(neck_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lhip_sagittal_2d_filltered = np.array([sggait.butter_lowpass_fillter(lhip_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lknee_sagittal_2d_filltered = np.array([sggait.butter_lowpass_fillter(lknee_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lankle_sagittal_2d_filltered = np.array([sggait.butter_lowpass_fillter(lankle_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lbigtoe_sagittal_2d_filltered = np.array([sggait.butter_lowpass_fillter(lbigtoe_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lsmalltoe_sagittal_2d_filltered = np.array([sggait.butter_lowpass_fillter(lsmalltoe_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T
        lheel_sagittal_2d_filltered = np.array([sggait.butter_lowpass_fillter(lheel_sagittal_2d[:, x], order = 4, cutoff_freq = 6, frame_list = sagi_frame_2d) for x in range(2)]).T

        trunk_vector_sagittal_2d = neck_sagittal_2d - mid_hip_sagttal_2d
        thigh_vector_l_sagittal_2d = lknee_sagittal_2d - lhip_sagittal_2d
        lower_leg_vector_l_sagittal_2d = lknee_sagittal_2d - lankle_sagittal_2d
        foot_vector_l_sagittal_2d = (lbigtoe_sagittal_2d + lsmalltoe_sagittal_2d) / 2 - lheel_sagittal_2d

        trunk_vector_sagittal_2d_filtered = neck_sagittal_2d_filltered - mid_hip_sagttal_2d_filltered
        thigh_vector_l_sagittal_2d_filtered = lknee_sagittal_2d_filltered - lhip_sagittal_2d_filltered
        lower_leg_vector_l_sagittal_2d_filtered = lknee_sagittal_2d_filltered - lankle_sagittal_2d_filltered
        foot_vector_l_sagittal_2d_filtered = (lbigtoe_sagittal_2d_filltered + lsmalltoe_sagittal_2d_filltered) / 2 - lheel_sagittal_2d_filltered

        #すべてで記録できているフレームを抽出
        print(f"sagi_frame_2d = {sagi_frame_2d}")

        #ICのフレームを取得
        ic_frame_sg = [sagi_frame_2d[0], sagi_frame_2d[-1]]

        hip_angle_sagittal_2d = pd.DataFrame(sggait.calculate_angle(trunk_vector_sagittal_2d, thigh_vector_l_sagittal_2d))
        knee_angle_sagittal_2d = pd.DataFrame(sggait.calculate_angle(thigh_vector_l_sagittal_2d, lower_leg_vector_l_sagittal_2d))
        ankle_angle_sagittal_2d = pd.DataFrame(sggait.calculate_angle(lower_leg_vector_l_sagittal_2d, foot_vector_l_sagittal_2d))

        hip_angle_sagittal_2d = 180 - hip_angle_sagittal_2d
        knee_angle_sagittal_2d = 180 - knee_angle_sagittal_2d
        ankle_angle_sagittal_2d =  90 - ankle_angle_sagittal_2d

        hip_angle_sagittal_2d_filtered = pd.DataFrame(sggait.calculate_angle(trunk_vector_sagittal_2d_filtered, thigh_vector_l_sagittal_2d_filtered))
        knee_angle_sagittal_2d_filtered = pd.DataFrame(sggait.calculate_angle(thigh_vector_l_sagittal_2d_filtered, lower_leg_vector_l_sagittal_2d_filtered))
        ankle_angle_sagittal_2d_filtered = pd.DataFrame(sggait.calculate_angle(lower_leg_vector_l_sagittal_2d_filtered, foot_vector_l_sagittal_2d_filtered))

        hip_angle_sagittal_2d_filtered = 180 - hip_angle_sagittal_2d_filtered
        knee_angle_sagittal_2d_filtered = 180 - knee_angle_sagittal_2d_filtered
        ankle_angle_sagittal_2d_filtered =  90 - ankle_angle_sagittal_2d_filtered

        for ic_frame in ic_frame_sg:
            plt.axvline(x=ic_frame, color='gray', linestyle='--')
        plt.plot(sagi_frame_2d, hip_angle_sagittal_2d_filtered.loc[sagi_frame_2d], label="2D sagittal", color='#1f77b4')
        plt.plot(sagi_frame_2d, hip_angle_sagittal_2d.loc[sagi_frame_2d], color='#1f77b4', alpha=0.5)
        plt.title("Hip Angle")
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition}_hip_angle.png"))
        # plt.show()  #5frame
        plt.cla()

        for ic_frame in ic_frame_sg:
            plt.axvline(x=ic_frame, color='gray', linestyle='--')
        plt.plot(sagi_frame_2d, knee_angle_sagittal_2d_filtered.loc[sagi_frame_2d], label="2D sagittal", color='#1f77b4')
        plt.plot(sagi_frame_2d, knee_angle_sagittal_2d.loc[sagi_frame_2d], color='#1f77b4', alpha=0.5)
        plt.title("Knee Angle")
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition}_knee_angle.png"))
        # plt.show() #3frame
        plt.cla()

        for ic_frame in ic_frame_sg:
            plt.axvline(x=ic_frame, color='gray', linestyle='--')
        plt.plot(sagi_frame_2d, ankle_angle_sagittal_2d_filtered.loc[sagi_frame_2d], label="2D sagittal", color='#1f77b4')
        plt.plot(sagi_frame_2d, ankle_angle_sagittal_2d.loc[sagi_frame_2d], color='#1f77b4', alpha=0.5)
        plt.title("Ankle Angle")
        plt.xlabel("frame [-]")
        plt.ylabel("angle [°]")
        plt.savefig(os.path.join(root_dir, f"{condition}_ankle_angle.png"))
        # plt.show() #5frame
        plt.cla()

if __name__ == "__main__":
    main()
