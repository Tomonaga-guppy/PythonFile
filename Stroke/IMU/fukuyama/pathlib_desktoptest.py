from pathlib import Path

# 1. ユーザーのホームディレクトリのパスを取得します
home_directory = Path.home()

# 2. デスクトップのパスを組み立てます
#    osモジュールの例と同様に、日本語環境も考慮します
desktop_path = home_directory / 'Desktop'
if not desktop_path.is_dir():
    desktop_path = home_directory / 'デスクトップ'

# 3. デスクトップ上に作成したいフォルダのパスを組み立てます
folder_to_create = desktop_path / '新しいフォルダ_pathlib'

# 4. フォルダを作成します
#    parents=True: 中間ディレクトリも作成します
#    exist_ok=True: フォルダが既に存在していてもエラーになりません
folder_to_create.mkdir(parents=True, exist_ok=True)

print(f"フォルダを作成しました: {folder_to_create}")