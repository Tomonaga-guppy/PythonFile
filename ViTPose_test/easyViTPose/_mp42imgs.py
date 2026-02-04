"""
動画を連続画像に変換するスクリプト
"""

from pathlib import Path
import cv2
from tqdm import tqdm

# =========================
# 設定
# =========================
# 動画ファイルのパス（ここを変更）
VIDEO_PATH = Path(r"C:\Users\Tomson\Desktop\vitpose_kasa/pa5_pt1_cali.mp4")

# 出力先ディレクトリ（動画と同じ場所に作成される）
OUTPUT_DIR = VIDEO_PATH.parent / f"{VIDEO_PATH.stem}_frames"

# 画像フォーマット（'png' or 'jpg'）
IMAGE_FORMAT = "png"

# フレームをスキップする場合（1なら全フレーム、2なら1フレームおき）
FRAME_SKIP = 1


def extract_frames(video_path: Path, output_dir: Path, 
                   image_format: str = "png", frame_skip: int = 1):
    """
    動画からフレームを抽出して画像として保存
    
    Parameters
    ----------
    video_path : Path
        入力動画ファイルのパス
    output_dir : Path
        出力先ディレクトリ
    image_format : str
        画像フォーマット ('png' or 'jpg')
    frame_skip : int
        フレームスキップ数（1=全フレーム, 2=1フレームおき）
    """
    
    # 動画の存在確認
    if not video_path.exists():
        print(f"[ERROR] 動画が見つかりません: {video_path}")
        return False
    
    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 動画を開く
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 動画を開けません: {video_path}")
        return False
    
    # 動画情報取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n{'='*70}")
    print(f"動画ファイル: {video_path.name}")
    print(f"解像度: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"総フレーム数: {total_frames}")
    print(f"再生時間: {total_frames/fps:.2f} 秒")
    print(f"出力先: {output_dir}")
    print(f"画像形式: {image_format.upper()}")
    if frame_skip > 1:
        print(f"フレームスキップ: {frame_skip} (出力数: {total_frames//frame_skip})")
    print(f"{'='*70}\n")
    
    # フレーム処理
    frame_count = 0
    saved_count = 0
    ext = image_format if image_format in ["png", "jpg"] else "png"
    
    with tqdm(total=total_frames, desc="フレーム抽出", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # フレームスキップ判定
            if frame_count % frame_skip == 0:
                # ファイル名生成（保存したフレーム番号でナンバリング）
                out_path = output_dir / f"frame_{saved_count:05d}.{ext}"
                cv2.imwrite(str(out_path), frame)
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n[SUCCESS] {saved_count} フレームを保存しました")
    print(f"出力先: {output_dir}")
    return True


def main():
    """メイン処理"""
    print("=" * 70)
    print("動画フレーム抽出ツール")
    print("=" * 70)
    
    success = extract_frames(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        image_format=IMAGE_FORMAT,
        frame_skip=FRAME_SKIP
    )
    
    if success:
        print("\n処理が正常に完了しました。")
    else:
        print("\n処理が失敗しました。")


if __name__ == "__main__":
    main()