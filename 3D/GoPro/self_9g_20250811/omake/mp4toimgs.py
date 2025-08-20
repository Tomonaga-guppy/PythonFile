import cv2
from pathlib import Path
from tqdm import tqdm

def video_to_frames_pro(video_path, output_dir):
    """
    動画ファイルをフレームごとのPNG画像に変換して保存する関数 (pathlib, tqdm対応版)

    Args:
        video_path (str or Path): 入力する動画ファイルのパス
        output_dir (str or Path): 画像を保存するディレクトリのパス
    """
    # 出力ディレクトリが存在しない場合は作成 (親ディレクトリもまとめて作成)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 動画を読み込む (OpenCVの関数には文字列としてパスを渡すのが安全)
    cap = cv2.VideoCapture(str(video_path))

    # 動画が正常に開けたか確認
    if not cap.isOpened():
        print(f"エラー: 動画ファイルが開けませんでした: {video_path}")
        return

    # 動画の総フレーム数を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 総フレーム数が0の場合はエラー処理
    if total_frames == 0:
        print(f"エラー: フレームが読み込めませんでした。ファイルが空か、破損している可能性があります。")
        cap.release()
        return

    print(f"動画の処理を開始します: {video_path.name}")

    # tqdmでプログレスバーを表示しながらフレームを処理
    for frame_count in tqdm(range(total_frames), desc=f"🖼️  '{video_path.name}' を抽出中"):
        ret, frame = cap.read()

        # フレームが正しく読み込めなかった場合はループを抜ける
        if not ret:
            print(f"\n警告: {frame_count}フレーム目で読み込みが予期せず終了しました。")
            break

        # pathlibを使って出力パスを生成
        output_path = output_dir / f"frame_{frame_count:05d}.png"

        # フレームをPNG画像として保存
        cv2.imwrite(str(output_path), frame)

    # リソースを解放
    cap.release()
    print(f"\n処理が完了しました {output_dir.resolve()} に画像を保存しました。")

# --- ここから実行部分 ---
if __name__ == '__main__':
    # 1. 入力動画のパスを指定
    input_video_file = Path(r'G:\gait_pattern\20250807_br\ngait\fr\trim.mp4')  # <<< ここに動画ファイルのパスを入力してください

    # 2. 出力先ディレクトリの名前を指定
    output_image_dir = Path(r'G:\gait_pattern\20250807_br\ngait\fr\distorted')  # <<< ここに保存先フォルダ名を入力してください

    # 関数を実行
    video_to_frames_pro(input_video_file, output_image_dir)