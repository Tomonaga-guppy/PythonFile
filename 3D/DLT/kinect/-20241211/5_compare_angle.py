import pandas as pd
import numpy as np
from pathlib import Path
from pyk4a import PyK4A, PyK4APlayback, CalibrationType
import json
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline

root_dir = Path(r"G:\gait_pattern\20240912")
condition = "*sub3_normal*"
target_names_set = set()
for f in root_dir.glob(f"{condition}"):
    if f.is_dir():
        parent_name = f.name.split('_dev')[0]  # "_dev" 以降を削除して親ディレクトリの名前を取得
        target_names_set.add(root_dir / parent_name)  # ユニークなパスとしてセットに追加

# リストに変換してソート
target_names = list(target_names_set)
target_names.sort()

def main():

    check_side = "right"
    # check_side = "left"

    for target_name in target_names:
        print(f"target_name: {target_name}")

        # mocapデータの読み込み
        target_base_name = target_name.name
        print(f"target_base_name: {target_base_name}")
        condition = target_base_name.split("_f")[0]
        number_part = f"f{int(target_base_name.split('_')[-1]):04d}"
        condition_mocap = f"{condition}_{number_part}"
        print(f"condition_mocap: {condition_mocap}")
        print(f'root_dir.joinpath("qualisys") :{root_dir.joinpath("qualisys")}')
        angle_csv_file = list(root_dir.joinpath("qualisys").glob(f"angle_30Hz_{condition_mocap}*.csv"))[0]
        df_mocap = pd.read_csv(angle_csv_file, index_col=0)

        # cameraデータの読み込み(2d)
        camera_sagittal_csv_file = list(root_dir.glob(f"{target_base_name}_sagittal_angle*.csv"))[0]
        df_sagittal_2d = pd.read_csv(camera_sagittal_csv_file, index_col=0)

        # cameraデータの読み込み(3d)
        camera_sagittal_csv_file = list(root_dir.glob(f"{target_base_name}_3d_angle*.csv"))[0]
        df_frontal_3d = pd.read_csv(camera_sagittal_csv_file, index_col=0)

        # print(f"df_mocap: {df_mocap}")
        # print(f"df_sagittal_2d: {df_sagittal_2d}")
        # print(f"df_frontal_3d: {df_frontal_3d}")

        frame_range = sorted(list(set(df_mocap.index) & set(df_sagittal_2d.index) & set(df_frontal_3d.index)))
        frame_range = range(frame_range[0], frame_range[-1])

        if not Path(f"{root_dir}/compare_angle").exists():
            Path(f"{root_dir}/compare_angle").mkdir(parents=True, exist_ok=True)

        # 接地しているフレームを取得
        ic_frame_mocap_path = list(root_dir.joinpath('qualisys').glob(f"ic_frame_30Hz_{condition_mocap}*.npy"))[0]
        ic_frame_mocap = np.load(ic_frame_mocap_path)

        frame_range = range(ic_frame_mocap[3], ic_frame_mocap[9])


        if check_side == "right":
            [plt.axvline(x, color='gray', linestyle='--') for x in ic_frame_mocap]
            plt.plot(frame_range, df_mocap.loc[frame_range, "r_hip_angle_flex_ext"], label="Mocap", color = 'tab:orange')
            plt.plot(frame_range, df_sagittal_2d.loc[frame_range, "hip_angle_right"], label="2D sagittal", color = 'tab:blue')
            plt.plot(frame_range, df_frontal_3d.loc[frame_range, "hip_angle_right"], label="3D frontal", color = 'tab:green')
            plt.xlim(frame_range[0], frame_range[-1])
            plt.ylim(-30, 70)
            plt.title("Right Hip Angle")
            plt.legend()
            plt.xlabel("Frame [-]")
            plt.ylabel("Angle [deg]")
            plt.savefig(f"{root_dir}/compare_angle/{condition_mocap}_right_hip_angle.png")
            plt.show()
            plt.close()

            [plt.axvline(x, color='gray', linestyle='--') for x in ic_frame_mocap]
            plt.plot(frame_range, df_mocap.loc[frame_range, "r_knee_angle_flex_ext"], label="Mocap", color = 'tab:orange')
            plt.plot(frame_range, df_sagittal_2d.loc[frame_range, "knee_angle_right"], label="2D sagittal", color = 'tab:blue')
            plt.plot(frame_range, df_frontal_3d.loc[frame_range, "knee_angle_right"], label="3D frontal", color = 'tab:green')
            plt.xlim(frame_range[0], frame_range[-1])
            plt.ylim(-30, 70)
            plt.title("Right Knee Angle")
            plt.legend()
            plt.xlabel("Frame [-]")
            plt.ylabel("Angle [deg]")
            plt.savefig(f"{root_dir}/compare_angle/{condition_mocap}_right_knee_angle.png")
            plt.show()
            plt.close()

            [plt.axvline(x, color='gray', linestyle='--') for x in ic_frame_mocap]
            plt.plot(frame_range, df_mocap.loc[frame_range, "r_ankle_angle_flex_ext"], label="Mocap", color = 'tab:orange')
            plt.plot(frame_range, df_sagittal_2d.loc[frame_range, "ankle_angle_right"], label="2D sagittal", color = 'tab:blue')
            plt.plot(frame_range, df_frontal_3d.loc[frame_range, "ankle_angle_right"], label="3D frontal", color = 'tab:green')
            plt.xlim(frame_range[0], frame_range[-1])
            plt.ylim(-30, 70)
            plt.title("Right Ankle Angle")
            plt.legend()
            plt.xlabel("Frame [-]")
            plt.ylabel("Angle [deg]")
            plt.savefig(f"{root_dir}/compare_angle/{condition_mocap}_right_ankle_angle.png")
            plt.show()
            plt.close()

            hip_abs_error_2d = abs(df_sagittal_2d.loc[frame_range, "hip_angle_right"] - df_mocap.loc[frame_range, "r_hip_angle_flex_ext"])
            knee_abs_error_2d = abs(df_sagittal_2d.loc[frame_range, "knee_angle_right"] - df_mocap.loc[frame_range, "r_knee_angle_flex_ext"])
            ankle_abs_error_2d = abs(df_sagittal_2d.loc[frame_range, "ankle_angle_right"] - df_mocap.loc[frame_range, "r_ankle_angle_flex_ext"])

            mae_hip_2d = np.nanmean(hip_abs_error_2d)
            mae_knee_2d = np.nanmean(knee_abs_error_2d)
            mae_ankle_2d = np.nanmean(ankle_abs_error_2d)
            print(f"MAE Hip 2D: {mae_hip_2d}")
            print(f"MAE Knee 2D: {mae_knee_2d}")
            print(f"MAE Ankle 2D: {mae_ankle_2d}")

            hip_abs_error_3d = abs(df_frontal_3d.loc[frame_range, "hip_angle_right"] - df_mocap.loc[frame_range, "r_hip_angle_flex_ext"])
            knee_abs_error_3d = abs(df_frontal_3d.loc[frame_range, "knee_angle_right"] - df_mocap.loc[frame_range, "r_knee_angle_flex_ext"])
            ankle_abs_error_3d = abs(df_frontal_3d.loc[frame_range, "ankle_angle_right"] - df_mocap.loc[frame_range, "r_ankle_angle_flex_ext"])

            mae_hip_3d = np.nanmean(hip_abs_error_3d)
            mae_knee_3d = np.nanmean(knee_abs_error_3d)
            mae_ankle_3d = np.nanmean(ankle_abs_error_3d)
            print(f"MAE Hip 3D: {mae_hip_3d}")
            print(f"MAE Knee 3D: {mae_knee_3d}")
            print(f"MAE Ankle 3D: {mae_ankle_3d}")

            [plt.axvline(x, color='gray', linestyle='--') for x in ic_frame_mocap]
            plt.plot(frame_range, df_mocap.loc[frame_range, "r_hip_angle_abd_add"], label="Mocap", color = 'tab:orange')
            plt.plot(frame_range, df_frontal_3d.loc[frame_range, "hip_angle_right"], label="3D frontal", color = 'tab:green')
            plt.xlim(frame_range[0], frame_range[-1])
            plt.ylim(-10, 30)
            plt.title("Right Hip Angle Abduction/Adduction")
            plt.legend()
            plt.xlabel("Frame [-]")
            plt.ylabel("Angle [deg]")
            plt.savefig(f"{root_dir}/compare_angle/{condition_mocap}_right_hip_angle_abd_add.png")
            plt.show()
            plt.close()

            hip_abd_add_abs_error_3d = abs(df_frontal_3d.loc[frame_range, "hip_angle_right"] - df_mocap.loc[frame_range, "r_hip_angle_abd_add"])
            mae_hip_abd_add_3d = np.nanmean(hip_abd_add_abs_error_3d)
            print(f"MAE Hip Abduction/Adduction 3D: {mae_hip_abd_add_3d}")
            nmae_hip_abd_add_3d = mae_hip_abd_add_3d / (np.max(df_mocap.loc[frame_range, "r_hip_angle_abd_add"]) - np.min(df_mocap.loc[frame_range, "r_hip_angle_abd_add"]))
            print(f"NMAE Hip Abduction/Adduction 3D: {nmae_hip_abd_add_3d}")

        elif check_side == "left":
            [plt.axvline(x, color='gray', linestyle='--') for x in ic_frame_mocap]
            plt.plot(frame_range, df_mocap.loc[frame_range, "l_hip_angle_flex_ext"], label="Mocap", color = 'tab:orange')
            plt.plot(frame_range, df_sagittal_2d.loc[frame_range, "hip_angle_left"], label="2D sagittal", color = 'tab:blue')
            plt.plot(frame_range, df_frontal_3d.loc[frame_range, "hip_angle_left"], label="3D frontal", color = 'tab:green')
            plt.xlim(frame_range[0], frame_range[-1])
            plt.ylim(-30, 70)
            plt.title("Left Hip Angle")
            plt.legend()
            plt.xlabel("Frame [-]")
            plt.ylabel("Angle [deg]")
            plt.savefig(f"{root_dir}/compare_angle/{condition_mocap}_left_hip_angle.png")
            plt.show()
            plt.close()

            [plt.axvline(x, color='gray', linestyle='--') for x in ic_frame_mocap]
            plt.plot(frame_range, df_mocap.loc[frame_range, "l_knee_angle_flex_ext"], label="Mocap", color = 'tab:orange')
            plt.plot(frame_range, df_sagittal_2d.loc[frame_range, "knee_angle_left"], label="2D sagittal", color = 'tab:blue')
            plt.plot(frame_range, df_frontal_3d.loc[frame_range, "knee_angle_left"], label="3D frontal", color = 'tab:green')
            plt.xlim(frame_range[0], frame_range[-1])
            plt.ylim(-30, 70)
            plt.title("Left Knee Angle")
            plt.legend()
            plt.xlabel("Frame [-]")
            plt.ylabel("Angle [deg]")
            plt.savefig(f"{root_dir}/compare_angle/{condition_mocap}_left_knee_angle.png")
            plt.show()
            plt.close()

            [plt.axvline(x, color='gray', linestyle='--') for x in ic_frame_mocap]
            plt.plot(frame_range, df_mocap.loc[frame_range, "l_ankle_angle_flex_ext"], label="Mocap", color = 'tab:orange')
            plt.plot(frame_range, df_sagittal_2d.loc[frame_range, "ankle_angle_left"], label="2D sagittal", color = 'tab:blue')
            plt.plot(frame_range, df_frontal_3d.loc[frame_range, "ankle_angle_left"], label="3D frontal", color = 'tab:green')
            plt.xlim(frame_range[0], frame_range[-1])
            plt.ylim(-30, 70)
            plt.title("Left Ankle Angle")
            plt.legend()
            plt.xlabel("Frame [-]")
            plt.ylabel("Angle [deg]")
            plt.savefig(f"{root_dir}/compare_angle/{condition_mocap}_left_ankle_angle.png")
            plt.show()
            plt.close()

            hip_abs_error_2d = abs(df_sagittal_2d.loc[frame_range, "hip_angle_left"] - df_mocap.loc[frame_range, "l_hip_angle_flex_ext"])
            knee_abs_error_2d = abs(df_sagittal_2d.loc[frame_range, "knee_angle_left"] - df_mocap.loc[frame_range, "l_knee_angle_flex_ext"])
            ankle_abs_error_2d = abs(df_sagittal_2d.loc[frame_range, "ankle_angle_left"] - df_mocap.loc[frame_range, "l_ankle_angle_flex_ext"])

            mae_hip_2d = np.nanmean(hip_abs_error_2d)
            mae_knee_2d = np.nanmean(knee_abs_error_2d)
            mae_ankle_2d = np.nanmean(ankle_abs_error_2d)

            print(f"MAE Hip 2D: {mae_hip_2d}")
            print(f"MAE Knee 2D: {mae_knee_2d}")
            print(f"MAE Ankle 2D: {mae_ankle_2d}")

            hip_abs_error_3d = abs(df_frontal_3d.loc[frame_range, "hip_angle_left"] - df_mocap.loc[frame_range, "l_hip_angle_flex_ext"])
            knee_abs_error_3d = abs(df_frontal_3d.loc[frame_range, "knee_angle_left"] - df_mocap.loc[frame_range, "l_knee_angle_flex_ext"])
            ankle_abs_error_3d = abs(df_frontal_3d.loc[frame_range, "ankle_angle_left"] - df_mocap.loc[frame_range, "l_ankle_angle_flex_ext"])

            mae_hip_3d = np.nanmean(hip_abs_error_3d)
            mae_knee_3d = np.nanmean(knee_abs_error_3d)
            mae_ankle_3d = np.nanmean(ankle_abs_error_3d)

            print(f"MAE Hip 3D: {mae_hip_3d}")
            print(f"MAE Knee 3D: {mae_knee_3d}")
            print(f"MAE Ankle 3D: {mae_ankle_3d}")

            [plt.axvline(x, color='gray', linestyle='--') for x in ic_frame_mocap]
            plt.plot(frame_range, df_mocap.loc[frame_range, "l_hip_angle_abd_add"], label="Mocap", color = 'tab:orange')
            plt.plot(frame_range, df_frontal_3d.loc[frame_range, "hip_angle_left"], label="3D frontal", color = 'tab:green')
            plt.xlim(frame_range[0], frame_range[-1])
            plt.ylim(-10, 30)
            plt.title("Left Hip Angle Abduction/Adduction")
            plt.legend()
            plt.xlabel("Frame [-]")
            plt.ylabel("Angle [deg]")
            plt.savefig(f"{root_dir}/compare_angle/{condition_mocap}_left_hip_angle_abd_add.png")
            plt.show()
            plt.close()

            hip_abd_add_abs_error_3d = abs(df_frontal_3d.loc[frame_range, "hip_angle_left"] - df_mocap.loc[frame_range, "l_hip_angle_abd_add"])
            mae_hip_abd_add_3d = np.nanmean(hip_abd_add_abs_error_3d)
            print(f"MAE Hip Abduction/Adduction 3D: {mae_hip_abd_add_3d}")


if __name__ == "__main__":
    main()
