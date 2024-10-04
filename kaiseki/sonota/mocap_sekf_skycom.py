import glob
import os
import matplotlib.pyplot as plt
import pandas as pd

root_dir = r"F:\Tomson\gait_pattern\20240808"
check_num = 2  #解析したい条件番号を指定
angle_csv = os.path.join(root_dir, 'Motive', 'angle*120Hz*.csv')
self_angle_paths = glob.glob(os.path.join(root_dir, 'Motive', f'angle_120Hz*_{check_num}_*2-0*.csv'))
# self_angle_paths = glob.glob(os.path.join(root_dir, 'Motive', f'angle_120Hz*_{check_num}_*.csv'))
# print(self_angle_paths)

# skycom_path = os.path.join(root_dir, "Motive", "SKYCOM","SKYCOM_1_Take 2024-08-08 04.03.20 PM_angle.csv")
skycom_path = os.path.join(root_dir, "Motive", "SKYCOM","SKYCOM_2_Take 2024-08-08 04.04.08 PM_angle.csv")
# skycom_path = os.path.join(root_dir, "Motive", "SKYCOM","SKYCOM_4_Take 2024-08-08 04.06.53 PM_lt limb angle.csv")

for self_angle_path in self_angle_paths:
    print(self_angle_path)
    print(skycom_path)
    df_self = pd.read_csv(self_angle_path, index_col=0)
    print(f"self = {df_self}")
    df_skycom = pd.read_csv(skycom_path, index_col=0, skiprows=11, encoding='ISO-8859-1')
    print(f"df_skycom = {df_skycom}")

    print(f"df_skycom.index = {df_skycom.index}")
    print(f"df_skycom.iloc[:,4] = {df_skycom.iloc[:,4]}")

    plt.plot(df_self.index, df_self['l_hip_angle'], label='self', color='tab:orange')#color='#2ca02c' 緑 color='#ff7f0e' オレンジ  color='#1f77b4' 青
    # plt.plot(df_skycom.index, df_skycom['lt hip'], label='SKYCOM', color='tab:green')
    plt.plot(df_skycom.index, df_skycom.iloc[:,4], label='SKYCOM', color='tab:green')
    plt.title('l_hip_angle')
    plt.xlabel('frame [Hz]')
    plt.ylabel('angle [°]')
    plt.legend()
    plt.savefig(os.path.join(root_dir, 'Motive', f'hip_angle_{check_num}.png'))
    plt.show()

    plt.plot(df_self.index, df_self['l_knee_angle'], label='self', color='tab:orange')
    # plt.plot(df_skycom.index, df_skycom['lt knee'], label='SKYCOM', color='tab:green')
    plt.plot(df_skycom.index, df_skycom.iloc[:,5], label='SKYCOM', color='tab:green')
    plt.title('l_knee_angle')
    plt.xlabel('frame [Hz]')
    plt.ylabel('angle [°]')
    plt.legend()
    plt.savefig(os.path.join(root_dir, 'Motive', f'knee_angle_{check_num}.png'))
    plt.show()

    plt.plot(df_self.index, df_self['l_ankle_angle'], label='self', color='tab:orange')
    # plt.plot(df_skycom.index, -df_skycom['lt ankle'], label='SKYCOM', color='tab:green') #底背屈分の符号を反転
    plt.plot(df_skycom.index, -df_skycom.iloc[:,6], label='SKYCOM', color='tab:green') #底背屈分の符号を反転
    plt.title('l_ankle_angle')
    plt.xlabel('frame [Hz]')
    plt.ylabel('angle [°]')
    plt.legend()
    plt.savefig(os.path.join(root_dir, 'Motive', f'ankle_angle_{check_num}.png'))
    plt.show()

    hip_self = df_self['l_hip_angle']
    # hip_skycom = df_skycom['lt hip'].loc[df_self.index]
    hip_skycom = df_skycom.iloc[:,4].loc[df_self.index]

    knee_self = df_self['l_knee_angle']
    # knee_skycom = df_skycom['lt knee'].loc[df_self.index]
    knee_skycom = df_skycom.iloc[:,5].loc[df_self.index]

    ankle_self = df_self['l_ankle_angle']
    # ankle_skycom = -df_skycom['lt ankle'].loc[df_self.index]
    ankle_skycom = -df_skycom.iloc[:,6].loc[df_self.index]

    mae_hip = (hip_self - hip_skycom).abs().mean()
    mae_knee = (knee_self - knee_skycom).abs().mean()
    mae_ankle = (ankle_self - ankle_skycom).abs().mean()

    print(f"hip: {mae_hip}, knee: {mae_knee}, ankle: {mae_ankle}")