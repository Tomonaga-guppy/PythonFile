from pathlib import Path
import pandas as pd

lumber_dir = Path(r"G:\gait_pattern\20241114_ota_test\IMU\20241114\患者腰用")
csv_list = list(lumber_dir.glob("**/*.csv"))
print(f"csv_list: {csv_list}")


for csv_path in csv_list:
    read_csv = pd.read_csv(csv_path, na_values=[""], header=None)
    print(f"read_csv: {read_csv}")




