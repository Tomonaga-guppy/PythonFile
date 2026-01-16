"""
IMU CSV 振り分けスクリプト（実行前確認付き）

- 実行前に「コピー元 → コピー先」の対応一覧をすべて表示
- ユーザ確認 (y/n) 後にのみコピー／移動を実行
"""

import shutil
from pathlib import Path
from tkinter import Tk, filedialog

# =========================
# 設定
# =========================
IMU_INDICES = [10, 11, 12, 13, 14, 15, 16, 17]   # 1始まり
KEYWORDS = ["SYNC", "PAW", "PTW", "PTLH", "PTRH"]
FILTER_BY_KEYWORDS = True

# =========================
# ユーティリティ
# =========================
def natural_sorted(files):
    return sorted(files, key=lambda p: p.name.lower())

def pick_csvs(folder):
    csvs = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".csv"]
    csvs = natural_sorted(csvs)
    if FILTER_BY_KEYWORDS:
        csvs = [f for f in csvs if any(k in f.name.upper() for k in KEYWORDS)]
    return csvs

def safe_copy_or_move(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        stem, suf = dst.stem, dst.suffix
        i = 1
        while True:
            cand = dst.with_name(f"{stem}_dup{i}{suf}")
            if not cand.exists():
                dst = cand
                break
            i += 1
    shutil.copy2(src, dst)

# =========================
# メイン処理
# =========================
def main(src_root, dst_root):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    src_folders = natural_sorted([d for d in src_root.iterdir() if d.is_dir()])
    dst_folders = natural_sorted([
        d for d in dst_root.iterdir()
        if d.is_dir() and d.name.lower() != "cali"
    ])

    if len(dst_folders) < len(IMU_INDICES):
        print("エラー: コピー先フォルダ数が不足しています")
        return

    dst_base_is_imu = dst_root.name.lower() == "imu"

    # =========================
    # プレビュー作成
    # =========================
    preview = []

    for idx, dst_trial in zip(IMU_INDICES, dst_folders):
        target_dir = dst_trial if dst_base_is_imu else dst_trial / "IMU"
        for sf in src_folders:
            csvs = pick_csvs(sf)
            if idx < 1 or idx > len(csvs):
                preview.append(("ERROR", sf, None, target_dir))
            else:
                preview.append(("OK", sf, csvs[idx - 1], target_dir))

    # =========================
    # 実行前確認表示
    # =========================
    print("\n==============================")
    print(" 実行前確認（コピー／移動予定）")
    print("==============================")
    print(f"\nコピー元: {src_root}")
    print(f"コピー先: {dst_root}")
    print("\n--- 対象ファイル一覧 ---")

    for status, sf, csv, tgt in preview:
        if status == "OK":
            print(f"[OK] {sf.name} / {csv.name} -> {tgt}")
        else:
            print(f"[NG] {sf.name} / インデックス範囲外")

    print("\n==============================")

    resp = input("この内容で実行しますか？ (y/n): ")
    if resp.lower() != "y":
        print("処理を中止しました")
        return

    # =========================
    # 実行
    # =========================
    print("\n=== 実行開始 ===")
    for status, sf, csv, tgt in preview:
        if status != "OK":
            continue
        safe_copy_or_move(csv, tgt / csv.name)
        print(f"処理: {csv.name}")

    print("\n=== 完了 ===")

# =========================
# GUI フォルダ選択
# =========================
if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    print("コピー元 IMU フォルダを選択してください")
    src = filedialog.askdirectory()
    if not src:
        raise SystemExit("コピー元未選択")

    print("コピー先ベースフォルダを選択してください")
    dst = filedialog.askdirectory()
    if not dst:
        raise SystemExit("コピー先未選択")

    main(src, dst)
