from pathlib import Path
import imu_module as imu
# import matplotlib
import matplotlib.pyplot as plt

#元の出力から確認するよう

root_dir = Path(r"G:\gait_pattern\20250228_ota\data\20250221\IMU")
target_imus = ["sync", "sub", "thera", "thera_lhand", "thera_rhand"]

condition_list = ["thera0-3", "thera1-1", "thera2-1"]

for target_imu in target_imus:
    target_dir = root_dir / target_imu
    csv_list = [file for file in target_dir.glob("*.csv")
        if file.name.startswith("1") or file.name.startswith("2") or file.name.startswith("3") ]
    for i, csv_path in enumerate(csv_list):
        imu_df = imu.read_ags_dataframe(csv_path)
        # print(imu_df.head())
        # imu_df = imu.resampling(imu_df)
        # print(imu_df.head())
        # imu_df = imu.butter_lowpass_fillter(imu_df, 100, 3, 10)
        # print(imu_df.head())

        if target_imu == "sync" or target_imu == "sub":
            condition = condition_list[i]

            plt.plot(imu_df["time"], imu_df["acc_x"], label="acc_x")
            plt.plot(imu_df["time"], imu_df["acc_y"], label="acc_y")
            plt.plot(imu_df["time"], imu_df["acc_z"], label="acc_z")
            plt.legend()

            # plt.show()
            plt.savefig(csv_path.with_name(f"check_{condition}.png"))
            plt.close()


