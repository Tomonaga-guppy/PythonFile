from pathlib import Path
import shutil

root_dir = Path(r"C:\Users\Tomson\Pictures")
img_paths = list(root_dir.glob("*.png"))

folder_path = root_dir / "Intrinsic_fr_15m"
if not folder_path.exists():
    folder_path.mkdir()

# PNGファイルをint_caliフォルダに移動
for img_path in img_paths:
    shutil.move(str(img_path), folder_path / img_path.name)

print(f"{len(img_paths)} 個のファイルを {folder_path} に移動しました。")