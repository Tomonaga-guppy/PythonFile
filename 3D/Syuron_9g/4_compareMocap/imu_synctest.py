"""
IMUの同期信号から、フレームずれ量を計算
お試し用
"""

import pandas as pd
from pathlib import Path

csv_path = Path(r"G:\gait_pattern\BR9G_shuron\sub5\thera5-0\IMU\mem-TSND151_AP04212533_SYNC-20251127-154620291.csv")

df = pd.read_csv(csv_path, sep=",", header=None)

# ext data 行のみ抽出
ext_df = df[df[0] == "ext data"].reset_index(drop=True)
ext_df.to_csv(csv_path.parent / "imu_sync_extdata.csv", index=False, header=False)

# 念のため int 化
ext_df[2] = ext_df[2].astype(int)
ext_df[3] = ext_df[3].astype(int)

# =========================
# 変化点検出
# =========================

# 2列目: 1 → 0
col2_fall = ext_df.index[
    (ext_df[2].shift(1) == 1) & (ext_df[2] == 0)
]

# 3列目: 0 → 1
col3_rise = ext_df.index[
    (ext_df[3].shift(1) == 0) & (ext_df[3] == 1)
]

print("col2 (1→0) frames:", col2_fall.tolist())
print("col3 (0→1) frames:", col3_rise.tolist())

# =========================
# フレームずれ量の計算
# =========================

if len(col2_fall) > 0 and len(col3_rise) > 0:
    # 最初のイベント同士で比較（一般的）
    frame_diff = col3_rise[0] - col2_fall[0]
    print(f"Frame difference (col3 - col2): {frame_diff}")
else:
    print("変化点が検出できませんでした")
