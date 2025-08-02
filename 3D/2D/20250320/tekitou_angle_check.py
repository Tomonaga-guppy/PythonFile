# 2つのcsvから関節角度を比較

import pandas as pd
from pathlib import Path

normal_csv = Path(r"G:\gait_pattern\20250228_ota\data\20250221\sub0\thera0-3\sagi\joint_angle_mean_std_thera0-3_Undistort_op.csv")
yolo_csv = Path(r"G:\gait_pattern\20250228_ota\data\20250221\sub0\thera0-3\sagi\joint_angle_mean_std_thera0-3_Undistort_black_fill_black_op.csv")

normal_df = pd.read_csv(normal_csv, index_col=0)
yolo_df = pd.read_csv(yolo_csv, index_col=0)

print(f"normal_df:\n{normal_df}")
print(f"yolo_df:\n{yolo_df}")

# 左足の股関節角度をプロット
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(normal_df['Hip_l_mean'], label='Normal', color='blue')
plt.plot(yolo_df['Hip_l_mean'], label='Masked', color='orange')
plt.fill_between(range(len(normal_df)), normal_df['Hip_l_mean'] - normal_df['Hip_l_std'], normal_df['Hip_l_mean'] + normal_df['Hip_l_std'], color='blue', alpha=0.1)
plt.fill_between(range(len(yolo_df)), yolo_df['Hip_l_mean'] - yolo_df['Hip_l_std'], yolo_df['Hip_l_mean'] + yolo_df['Hip_l_std'], color='orange', alpha=0.1)
plt.title('Hip Angle Comparison')
plt.xlabel('Gait Cycle [%]')
plt.ylabel('Angle [deg]')
plt.legend()
plt.grid()
plt.savefig(normal_csv.parent / 'HipAngle_l_comparison.png')


# 左足の膝関節角度をプロット
plt.figure(figsize=(10, 5))
plt.plot(normal_df['Knee_l_mean'], label='Normal', color='blue')
plt.plot(yolo_df['Knee_l_mean'], label='Masked', color='orange')
plt.fill_between(range(len(normal_df)), normal_df['Knee_l_mean'] - normal_df['Knee_l_std'], normal_df['Knee_l_mean'] + normal_df['Knee_l_std'], color='blue', alpha=0.1)
plt.fill_between(range(len(yolo_df)), yolo_df['Knee_l_mean'] -
yolo_df['Knee_l_std'], yolo_df['Knee_l_mean'] + yolo_df['Knee_l_std'], color='orange', alpha=0.1)
plt.title('Knee Angle Comparison')
plt.xlabel('Gait Cycle [%]')
plt.ylabel('Angle [deg]')
plt.legend()
plt.grid()
plt.savefig(normal_csv.parent / 'KneeAngle_l_comparison.png')

# 左足の足関節角度をプロット
plt.figure(figsize=(10, 5))
plt.plot(normal_df['Ankle_l_mean'], label='Normal', color='blue')
plt.plot(yolo_df['Ankle_l_mean'], label='Masked', color='orange')
plt.fill_between(range(len(normal_df)), normal_df['Ankle_l_mean'] - normal_df['Ankle_l_std'], normal_df['Ankle_l_mean'] + normal_df['Ankle_l_std'], color='blue', alpha=0.1)
plt.fill_between(range(len(yolo_df)), yolo_df['Ankle_l_mean'] -
yolo_df['Ankle_l_std'], yolo_df['Ankle_l_mean'] + yolo_df['Ankle_l_std'], color='orange', alpha=0.1)
plt.title('Ankle Angle Comparison')
plt.xlabel('Gait Cycle [%]')
plt.ylabel('Angle [deg]')
plt.legend()
plt.grid()
plt.savefig(normal_csv.parent / 'AnkleAngle_l_comparison.png')
