# 動画を切り出してキャリブレーション用画像を作成する

from pathlib import Path
import cv2
import numpy as np

root_dir = Path(r"G:\gait_pattern\stereo_cali\9g_20250806")
directions = ["fl", "fr"]

def extract_frames(video_path, output_dir, num_frames, video_name):
    """
    動画から指定された枚数のフレームを均等に抽出して保存する

    Args:
        video_path: 動画ファイルのパス
        output_dir: 出力ディレクトリ
        num_frames: 抽出するフレーム数
        video_name: 動画名（ファイル名に使用）
    """
    # 動画ファイルが存在するかチェック
    if not video_path.exists():
        print(f"警告: {video_path} が見つかりません")
        return

    # 動画を開く
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"エラー: {video_path} を開けません")
        return

    # 総フレーム数を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{video_name}: 総フレーム数 = {total_frames}")

    if total_frames == 0:
        print(f"エラー: {video_name} にフレームがありません")
        cap.release()
        return

    # 抽出するフレーム番号を計算（均等に分散）
    if num_frames >= total_frames:
        # 要求フレーム数が総フレーム数以上の場合は全フレームを使用
        frame_indices = list(range(total_frames))
    else:
        # 均等に分散させるためのインデックスを計算
        step = total_frames / num_frames
        frame_indices = [int(i * step) for i in range(num_frames)]

    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0

    for i, frame_idx in enumerate(frame_indices):
        # 指定フレームに移動
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # フレームを読み取り
        ret, frame = cap.read()

        if ret:
            # ファイル名を作成（元動画名_抽出番号_フレーム番号_総フレーム数.jpg）
            filename = f"{video_name}_{i+1:02d}of{len(frame_indices):02d}_frame{frame_idx:06d}_total{total_frames:06d}.png"
            output_path = output_dir / filename

            # 画像を保存
            cv2.imwrite(str(output_path), frame)
            extracted_count += 1
            print(f"保存: {filename}")
        else:
            print(f"警告: フレーム {frame_idx} を読み取れませんでした")

    cap.release()
    print(f"{video_name}: {extracted_count}枚の画像を抽出しました\n")

# メイン処理
for direction in directions:
    print(f"=== {direction} ディレクトリの処理開始 ===")

    # 動画ファイルのパス
    video1_path = root_dir / direction / "cali.MP4"
    video2_path = root_dir / direction / "near.MP4"

    # 出力ディレクトリ
    output_dir = root_dir / direction / "cali_imgs"

    # video1から40枚抽出
    print(f"video1 (cali.MP4) から40枚抽出中...")
    extract_frames(video1_path, output_dir, 40, "cali")

    # # video2から20枚抽出
    # print(f"video2 (near.MP4) から20枚抽出中...")
    # extract_frames(video2_path, output_dir, 20, "near")

    print(f"=== {direction} ディレクトリの処理完了 ===\n")

print("全ての処理が完了しました！")