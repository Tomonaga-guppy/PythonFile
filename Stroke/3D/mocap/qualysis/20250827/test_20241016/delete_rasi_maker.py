"""
骨盤の補間が妥当かどうか確認するためにRASIを60%程削除してみてテスト
"""

from pathlib import Path
import module_mocap as moc
import pandas as pd

test_dir = Path(r"G:\gait_pattern\20250827_fukuyama\qualisys\psub_label\qtm\test")
ori_tsv_path = test_dir / "sub4_com_nfpa0001.tsv"


ori_df = moc.read_tsv(ori_tsv_path)
