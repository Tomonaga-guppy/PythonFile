"""
動画をフレーム画像群に分割保存するスクリプト
"""

from pathlib import Path
import cv2
from tqdm import tqdm

# =========================
# 設定
# =========================
# 動画ファイルのパス（ここを変更）
VIDEO_PATH = Path(r"G:\gait_pattern\2025_shuron_BR9G\sub1\thera1-0\gopro\fl\trimed.mp4")

# 出力先ディレクトリ（動画と同じ場所に作成）
OUTPUT_DIR = VIDEO_PATH.parent / "trim_frames"


def extract_frames(video_path: Path, output_dir: Path):
    """動画からフレームを抽出してPNG画像として保存"""
    
    # 動画の存在確認
    if not video_path.exists():
        print(f"[ERROR] 動画が見つかりません: {video_path}")
        return
    
    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 動画を開く
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 動画を開けません: {video_path}")
        return
    
    # 動画情報取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n動画: {video_path.name}")
    print(f"総フレーム数: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"出力先: {output_dir}\n")
    
    # フレーム処理
    frame_count = 0
    with tqdm(total=total_frames, desc="フレーム抽出", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # PNG保存
            out_path = output_dir / f"frame_{frame_count:05d}.png"
            cv2.imwrite(str(out_path), frame)
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    print(f"\n完了: {frame_count} フレームを保存しました")


if __name__ == "__main__":
    extract_frames(VIDEO_PATH, OUTPUT_DIR)