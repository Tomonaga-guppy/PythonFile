"""
pythonファイルを書いて削除しますか？という確認をしてから以下を消す
G:\gait_pattern\BR9G_shuron\sub*\thera*\gopro\*\*\*_plots
G:\gait_pattern\BR9G_shuron\sub*\thera*\gopro\*\*\*_all*.csv 
G:\gait_pattern\BR9G_shuron\sub*\thera*\gopro\*\*\*_pa**.csv
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(r"G:\gait_pattern\BR9G_shuron")

# =========================
# パターン定義
# =========================
# フォルダ削除対象（*_plots）
PLOT_DIR_PATTERN = "sub*/thera*/gopro/*/*/*_plots"

# csv（globで可能）
ALLCSV_PATTERN = "sub*/thera*/gopro/*/*/*_all*.csv"

# csv（*_pa**.csv は glob できない）
CSV_SEARCH_PATTERN = "sub*/thera*/gopro/*/*/*.csv"

# =========================
# 収集
# =========================
delete_dirs = set()
delete_files = set()

# --- (1) *_plots フォルダ ---
for d in ROOT.glob(PLOT_DIR_PATTERN):
    if d.is_dir():
        delete_dirs.add(d)

# --- (2) *_all*.csv ---
for f in ROOT.glob(ALLCSV_PATTERN):
    if f.is_file():
        delete_files.add(f)

# --- (3) *_pa**.csv ---
for f in ROOT.glob(CSV_SEARCH_PATTERN):
    if not f.is_file():
        continue
    stem = f.stem  # ファイル名（拡張子なし）
    idx = stem.rfind("_pa")
    if idx == -1:
        continue
    tail = stem[idx + 3 :]  # "_pa" の後の部分（3文字後から）
    # tail が空文字列（_pa.csv）、数字で始まる（_pa0.csv）、またはアンダースコアで始まる（_pa_butter.csv）
    if not tail or tail[0].isdigit() or tail[0] == '_':
        delete_files.add(f)

# =========================
# 確認表示
# =========================
if not delete_dirs and not delete_files:
    print("削除対象は見つかりませんでした。")
    sys.exit(0)

print("=" * 100)
print(f"[削除フォルダ] {len(delete_dirs)} 件")
for d in sorted(delete_dirs):
    print(d)

print("-" * 100)
print(f"[削除ファイル] {len(delete_files)} 件")
for f in sorted(delete_files):
    print(f)
print("=" * 100)

ans = input("上記をすべて削除しますか？ [y/N]: ").strip().lower()
if ans not in ("y", "yes"):
    print("削除をキャンセルしました。")
    sys.exit(0)

# =========================
# 削除実行
# =========================
dir_ok = file_ok = dir_err = file_err = 0

for d in delete_dirs:
    try:
        shutil.rmtree(d)
        dir_ok += 1
    except Exception as e:
        dir_err += 1
        print(f"[ERROR] フォルダ削除失敗: {d} ({e})")

for f in delete_files:
    try:
        f.unlink()
        file_ok += 1
    except Exception as e:
        file_err += 1
        print(f"[ERROR] ファイル削除失敗: {f} ({e})")

print("=" * 100)
print(f"フォルダ削除成功: {dir_ok} 件 / 失敗: {dir_err} 件") 
print(f"ファイル削除成功: {file_ok} 件 / 失敗: {file_err} 件")
print("処理終了")