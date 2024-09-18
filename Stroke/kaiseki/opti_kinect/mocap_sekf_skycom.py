import glob
import os
import matplotlib.pyplot as plt
import pandas as pd

root_dir = r"F:\Tomson\gait_pattern\20240808"
angle_csv = os.path.join(root_dir, 'Motive', 'angle*120Hz*.csv')
self_angle_paths = glob.glob(os.path.join(root_dir, 'Motive', 'angle_120Hz*_4*.csv'))
print(self_angle_paths)

skycom_path = os.path.join(root_dir, "Motive", "SKYCOM","SKYCOM_4_Take 2024-08-08 04.06.53 PM_lt limb angle.csv")

for self_angle_path in self_angle_paths:
    print(self_angle_path)
    df_self = pd.read_csv(self_angle_path, index_col=0)
    df_skycom = pd.read_csv(skycom_path, index_col=0, skiprows=11, encoding='ISO-8859-1')
    print(df_skycom)

    plt.plot(df_self.index, df_self['l_hip_angle'], label='self', color='#ff7f0e')#color='#2ca02c' 緑 color='#ff7f0e' オレンジ  color='#1f77b4' 青
    plt.plot(df_skycom.index, df_skycom['lt hip'], label='SKYCOM', color='#2ca02c')
    plt.title('l_hip_angle')
    plt.xlabel('frame [Hz]')
    plt.ylabel('angle [°]')
    plt.legend()
    plt.savefig(os.path.join(root_dir, 'Motive', 'hip_angle.png'))
    plt.show()

    plt.plot(df_self.index, df_self['l_knee_angle'], label='self', color='#ff7f0e')
    plt.plot(df_skycom.index, df_skycom['lt knee'], label='SKYCOM', color='#2ca02c')
    plt.title('l_knee_angle')
    plt.xlabel('frame [Hz]')
    plt.ylabel('angle [°]')
    plt.legend()
    plt.savefig(os.path.join(root_dir, 'Motive', 'knee_angle.png'))
    plt.show()

    plt.plot(df_self.index, df_self['l_ankle_angle'], label='self', color='#ff7f0e')
    plt.plot(df_skycom.index, -df_skycom['lt ankle'], label='SKYCOM', color='#2ca02c')
    plt.title('l_ankle_angle')
    plt.xlabel('frame [Hz]')
    plt.ylabel('angle [°]')
    plt.legend()
    plt.savefig(os.path.join(root_dir, 'Motive', 'ankle_angle.png'))
    plt.show()

    hip_self = df_self['l_hip_angle']
    hip_skycom = df_skycom['lt hip'].loc[df_self.index]

    knee_self = df_self['l_knee_angle']
    knee_skycom = df_skycom['lt knee'].loc[df_self.index]

    ankle_self = df_self['l_ankle_angle']
    ankle_skycom = -df_skycom['lt ankle'].loc[df_self.index]

    mae_hip = (hip_self - hip_skycom).abs().mean()
    mae_knee = (knee_self - knee_skycom).abs().mean()
    mae_ankle = (ankle_self - ankle_skycom).abs().mean()

    print(f"hip: {mae_hip}, knee: {mae_knee}, ankle: {mae_ankle}")