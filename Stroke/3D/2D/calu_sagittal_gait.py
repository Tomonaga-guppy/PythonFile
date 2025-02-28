import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import calu_saggital_gait_module as sggait
import sys

root_dir = Path(r"G:\gait_pattern\20241114_ota_test\gopro\sagi")
condition_list = ["sub0_abngait", "sub0_asgait_1", "sub0_asgait_2"]
# condition_list = ["sub0_asgait_1"]
# condition_list = ["sub0_ngait"]

def main():
    #各条件ごとに歩行開始目安のフレームを目視で確認してcsvファイルに保存
    frame_ch_csv = root_dir / f"FrameCheck.csv"
    if not frame_ch_csv.exists():
        sggait.mkFrameCheckCSV(frame_ch_csv, condition_list)

    for condition in condition_list:
        print(f"condition = {condition}")
        # 歪みを補正するためのカメラパラメータを読み込む
        camParames_path = root_dir.parent.parent.parent / "int_cali" / "ota" / "Intrinsic_sg.pickle"
        CameraParams_dict = sggait.loadCameraParameters(camParames_path)
        #2d上でのキーポイントを取得"
        openpose_dir = root_dir / (condition + "_op.json")
        csv_files = sggait.mkCSVOpenposeData(openpose_dir, overwrite=True)

        openpose_df_dict = {key:[] for key in range(len(csv_files))}
        for iPeople, csv_file in enumerate(csv_files):
            read_df_distort = pd.read_csv(csv_file, index_col=0)
            read_df = sggait.undistordOpenposeData(read_df_distort, CameraParams_dict)
            read_df = sggait.fillindex(read_df)
            openpose_df_dict[iPeople] = read_df

        """
        openpose_df_dict = {0:                 Nose_x       Nose_y   Nose_p       Neck_x       Neck_y    Neck_p  RShoulder_x  RShoulder_y  RShoulder_p  ...    RBigToe_x    RBigToe_y  RBigToe_p  RSmallToe_x  RSmallToe_y  RSmallToe_p      RHeel_x      RHeel_y   RHeel_p
        frame_num                                                                                                                ...
        33         -916.878974  -538.951577  0.00000  1189.444310  1171.589859  0.175547  1194.981554  1171.577124     0.151104  ...  1177.883606  1289.556565   0.196314  1183.616948  1295.080439     0.171203  1177.752887  1289.536505  0.132779
        ...                ...          ...      ...          ...          ...       ...          ...          ...          ...  ...          ...          ...        ...          ...          ...          ...          ...          ...       ...
        1753       -916.878974  -538.951577  0.00000  1189.635504  1159.922769  0.222892  1195.383603  1159.790705     0.208536  ...  1177.815929  1295.118300   0.223432  1183.586822  1295.150446     0.198983  1171.973680  1295.176230  0.168741
        [1721 rows x 75 columns], 1:                 Nose_x       Nose_y    Nose_p       Neck_x       Neck_y    Neck_p  RShoulder_x  RShoulder_y  RShoulder_p  ...    RBigToe_x    RBigToe_y  RBigToe_p  RSmallToe_x  RSmallToe_y  RSmallToe_p      RHeel_x        RHeel_y   RHeel_p
        frame_num                                                                                                                 ...
        157        -916.878974  -538.951577  0.000000  1189.484033  1177.483313  0.207803  1195.071455  1177.450401     0.185503  ...  1165.919190  1295.304196   0.219265  1165.929265  1295.344216     0.189380  1165.929252  1295.314204  0.164109
        ...                ...          ...       ...          ...          ...       ...          ...          ...          ...  ...          ...          ...        ...          ...          ...          ...          ...          ...       ...
        1749       -916.878974  -538.951577  0.000000  -916.878974  -538.951577  0.000000  -916.878974  -538.951577     0.000000  ...  1590.591011  1295.656777   0.191401  1590.610857  1295.477066     0.192436  1625.872953  1295.458659  0.192937
        [1593 rows x 75 columns]}
        """

        walk_start_frame = sggait.get_walk_start_frame(frame_ch_csv, condition)
        #歩行開始フレームを取得
        frame_adjust_df = sggait.adjust_frame(openpose_df_dict)
        #矢状面2d用の処理
        openpose_df_spline = sggait.spline_interpolation(openpose_df_dict)




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
