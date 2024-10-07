import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

root_dir = r"F:\Tomson\gait_pattern\20240912"
keywords = "normalgait_f0001"  #解析したい条件番号を指定
# keywords = "tpose"  #解析したい条件番号を指定
angle_csv = os.path.join(root_dir, 'qualisys', 'angle*120Hz*.csv')
self_angle_paths = glob.glob(os.path.join(root_dir, 'qualisys', f'angle_120Hz*{keywords}*.csv'))
# print(f"self_angle_paths = {self_angle_paths}")

for self_angle_path in self_angle_paths:
    print(f"self_angle_path = {self_angle_path}")
    skycom_file_name = os.path.basename(self_angle_path).replace('angle_120Hz_', '')
    skycom_path = os.path.join(os.path.dirname(self_angle_path), "SKYCOM", skycom_file_name)
    print(f"skycom_path = {skycom_path}")
    df_self = pd.read_csv(self_angle_path, index_col=0)
    df_skycom = pd.read_csv(skycom_path, index_col=0, skiprows=11, encoding='ISO-8859-1')

    ic_frame = np.load(os.path.join(root_dir, 'qualisys', f'ic_frame_120Hz_{skycom_file_name.split(".")[0]}.npy'))
    frame_range = range(ic_frame[0], ic_frame[-1])

    if ic_frame is not None:
        for ic_f in ic_frame:
            plt.axvline(x=ic_f, color='gray', linestyle='--')
    plt.plot(frame_range, df_self['l_hip_angle'].loc[frame_range], label='self', color='tab:orange')#color='#2ca02c' 緑 color='#ff7f0e' オレンジ  color='#1f77b4' 青
    plt.plot(frame_range, df_skycom.iloc[frame_range,4], label='SKYCOM', color='tab:green')
    plt.title(f'Hip Angle "{os.path.basename(skycom_file_name).split(".")[0]}"')
    plt.xlabel('frame [Hz]')
    plt.ylabel('angle [°]')
    plt.legend()
    plt.savefig(os.path.join(root_dir, 'qualisys', f'{os.path.basename(skycom_file_name).split(".")[0]}_hip_angle.png'))
    plt.show()

    if ic_frame is not None:
        for ic_f in ic_frame:
            plt.axvline(x=ic_f, color='gray', linestyle='--')
    plt.plot(frame_range, df_self['l_knee_angle'].loc[frame_range], label='self', color='tab:orange')
    plt.plot(frame_range, df_skycom.iloc[frame_range,5], label='SKYCOM', color='tab:green')
    plt.title(f'Knee Angle "{os.path.basename(skycom_file_name).split(".")[0]}"')
    plt.xlabel('frame [Hz]')
    plt.ylabel('angle [°]')
    plt.legend()
    plt.savefig(os.path.join(root_dir, 'qualisys', f'{os.path.basename(skycom_file_name).split(".")[0]}_knee_angle.png'))
    plt.show()

    if ic_frame is not None:
        for ic_f in ic_frame:
            plt.axvline(x=ic_f, color='gray', linestyle='--')
    plt.plot(frame_range, df_self['l_ankle_angle'].loc[frame_range], label='self', color='tab:orange')
    plt.plot(frame_range, -df_skycom.iloc[frame_range,6], label='SKYCOM', color='tab:green') #底背屈分の符号を反転
    plt.title(f'Ankle Angle "{os.path.basename(skycom_file_name).split(".")[0]}"')
    plt.xlabel('frame [Hz]')
    plt.ylabel('angle [°]')
    plt.legend()
    plt.savefig(os.path.join(root_dir, 'qualisys', f'{os.path.basename(skycom_file_name).split(".")[0]}_ankle_angle.png'))
    plt.show()

    hip_self = df_self['l_hip_angle']
    hip_skycom = df_skycom.iloc[:,4].loc[df_self.index]

    knee_self = df_self['l_knee_angle']
    knee_skycom = df_skycom.iloc[:,5].loc[df_self.index]

    ankle_self = df_self['l_ankle_angle']
    ankle_skycom = -df_skycom.iloc[:,6].loc[df_self.index]

    mae_hip = (hip_self - hip_skycom).abs().mean()
    mae_knee = (knee_self - knee_skycom).abs().mean()
    mae_ankle = (ankle_self - ankle_skycom).abs().mean()

    print(f"hip: {mae_hip:.2f}, knee: {mae_knee:.2f}, ankle: {mae_ankle:.2f}")