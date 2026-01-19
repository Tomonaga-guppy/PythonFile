"""
ViTPose結果とMocap結果をアニメーションで表示比較
表示するのはPAのみ
"""
from pathlib import Path

root_dir = Path(r"G:\gait_pattern\2025_shuron_BR9G")

# 余裕なくなって中断
def main():
    for sub_i in range(1, 11):
        sub_dir = next(root_dir.glob(f"sub_{sub_i}_*"), None)
        if sub_dir is None:
            print(f"[SKIP] missing: {sub_i}")
            continue
        thera_dir = sub_dir / "thera{sub_i}-0"
        if not thera_dir.exists():
            print(f"[SKIP] missing: {sub_i} thera folder")
            continue
        

if __name__ == "__main__":
    main()