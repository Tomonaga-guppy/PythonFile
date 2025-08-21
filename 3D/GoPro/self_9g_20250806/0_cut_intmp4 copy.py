# 左右カメラの同期フレームを抽出してキャリブレーション用画像を作成する

from pathlib import Path
import cv2
import numpy as np

root_dir = Path(r"G:\gait_pattern\int_cali\9g_20250807_6x5_35")
directions = ["fl", "fr"]

def get_video_info(video_path):
    """
    動画の基本情報を取得する

    Returns:
        tuple: (総フレーム数, フレームレート, 成功フラグ)
    """
    if not video_path.exists():
        print(f"警告: {video_path} が見つかりません")
        return 0, 0, False

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"エラー: {video_path} を開けません")
        return 0, 0, False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()
    return total_frames, fps, True

def extract_synchronized_frames(video_paths, output_dirs, num_frames, video_names):
    """
    複数の動画から同期したフレームを抽出する

    Args:
        video_paths: 動画ファイルのパスのリスト
        output_dirs: 出力ディレクトリのリスト
        num_frames: 抽出するフレーム数
        video_names: 動画名のリスト
    """
    # 全ての動画の情報を取得
    video_info = []
    caps = []

    for i, video_path in enumerate(video_paths):
        total_frames, fps, success = get_video_info(video_path)

        if not success:
            print(f"エラー: {video_names[i]} の情報取得に失敗")
            return

        video_info.append((total_frames, fps))
        print(f"{video_names[i]}: 総フレーム数={total_frames}, FPS={fps:.2f}")

        # 動画キャプチャを開く
        cap = cv2.VideoCapture(str(video_path))
        caps.append(cap)

    # 最小フレーム数を基準にする（短い方の動画に合わせる）
    min_frames = min([info[0] for info in video_info])
    print(f"基準フレーム数: {min_frames} (最短動画に合わせる)")

    if min_frames == 0:
        print("エラー: 有効なフレームがありません")
        for cap in caps:
            cap.release()
        return

    # 抽出するフレーム番号を計算（均等に分散）
    if num_frames >= min_frames:
        frame_indices = list(range(min_frames))
    else:
        step = min_frames / num_frames
        frame_indices = [int(i * step) for i in range(num_frames)]

    print(f"抽出フレーム数: {len(frame_indices)}")

    # 出力ディレクトリを作成
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0

    # 各フレームインデックスについて処理
    for i, frame_idx in enumerate(frame_indices):
        all_frames_read = True
        frames = []

        # 全ての動画から同じフレーム番号の画像を読み取り
        for j, cap in enumerate(caps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frames.append(frame)
            else:
                print(f"警告: {video_names[j]} のフレーム {frame_idx} を読み取れませんでした")
                all_frames_read = False
                break

        # 全ての動画からフレームが正常に読み取れた場合のみ保存
        if all_frames_read:
            for j, frame in enumerate(frames):
                # タイムスタンプを計算（秒）
                timestamp = frame_idx / video_info[j][1]

                # ファイル名を作成
                filename = f"{video_names[j]}_{i+1:02d}of{len(frame_indices):02d}_frame{frame_idx:06d}_time{timestamp:.3f}s.png"
                output_path = output_dirs[j] / filename

                # 画像を保存
                cv2.imwrite(str(output_path), frame)

            extracted_count += 1
            print(f"同期フレーム {i+1}/{len(frame_indices)} 保存完了 (フレーム番号: {frame_idx})")

    # リソースを解放
    for cap in caps:
        cap.release()

    print(f"同期フレーム抽出完了: {extracted_count}枚\n")

# メイン処理
def main():
    print("=== 同期フレーム抽出開始 ===")

    # cali.MP4から60枚の同期フレームを抽出
    print("cali.MP4 から60枚の同期フレーム抽出中...")

    cali_video_paths = []
    cali_output_dirs = []
    cali_video_names = []

    for direction in directions:
        video_path = root_dir / direction / "cali.mp4"
        output_dir = root_dir / direction / "cali_imgs"

        cali_video_paths.append(video_path)
        cali_output_dirs.append(output_dir)
        cali_video_names.append(f"cali")

    extract_synchronized_frames(cali_video_paths, cali_output_dirs, 60, cali_video_names)

    # near.MP4から30枚の同期フレームを抽出
    print("near.MP4 から30枚の同期フレーム抽出中...")

    near_video_paths = []
    near_output_dirs = []
    near_video_names = []

    for direction in directions:
        video_path = root_dir / direction / "near.MP4"
        output_dir = root_dir / direction / "cali_imgs"

        near_video_paths.append(video_path)
        near_output_dirs.append(output_dir)
        near_video_names.append(f"near_{direction}")

    extract_synchronized_frames(near_video_paths, near_output_dirs, 30, near_video_names)

    print("=== 全ての処理が完了しました！ ===")

if __name__ == "__main__":
    main()