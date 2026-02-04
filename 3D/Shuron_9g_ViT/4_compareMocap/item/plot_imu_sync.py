from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

imu_path = Path(r"G:\gait_pattern\2025_shuron_BR9G\sub1\thera1-0\IMU\imu_sync_extdata.csv")
imu_df = pd.read_csv(imu_path, sep=",", header=None)

print(f"imu_df: {imu_df}")

imu_df["time"] = imu_df[1] - imu_df[1].iloc[0]  #ms 換算
print(f"imu_df with time: {imu_df}")

plt.figure(figsize=(10, 6))
plt.plot(imu_df["time"], imu_df[2], label="Port0", color="tab:red")
plt.plot(imu_df["time"], imu_df[3], label="Port1", color="tab:blue")
plt.xlim(0, 3000)
plt.yticks([0, 1])  # Y軸の目盛りを0と1のみに設定
plt.legend()
plt.xlabel("IMU elapsed Time [ms]")
plt.ylabel("Signal Value [-]")
plt.title("IMU Signal Synchronization")
plt.savefig(imu_path.parent / "imu_sync_signals.png", dpi=200)
plt.show()