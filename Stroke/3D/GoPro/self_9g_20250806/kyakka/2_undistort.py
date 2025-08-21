from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm  # tqdmをインポート

# 基本のパス設定
root_dir = Path(r"G:\gait_pattern")
video_dir = root_dir / "20250807_br" / "ngait"
directions = ["fl", "fr"]

for direction in directions:
    print(f"\n{'='*60}")
    print(f"処理を開始します: {direction}")
    print(f"{'='*60}")

    work_dir = video_dir / direction
    video_path = work_dir / "trim.mp4"

    # 歪み補正後の動画と画像の保存先
    output_video_path = work_dir / "undistorted_49d5.mp4"
    output_img_dir = work_dir / "undistorted_49d5"

    # 歪みパラメータのパス
    camera_params_path = root_dir / "int_cali" / "9g_20250807_6x5_49d5" / direction / "camera_params.json"

    # --- 入力ファイルの存在確認 ---
    if not video_path.exists():
        print(f"警告: 動画ファイルが見つかりません。スキップします。\n -> {video_path}")
        continue
    if not camera_params_path.exists():
        print(f"エラー: カメラパラメータファイルが見つかりません。スキップします。\n -> {camera_params_path}")
        continue

    # --- 必要な処理の開始 ---
    try:
        # 1. カメラパラメータをJSONファイルから読み込む
        print(f"カメラパラメータを読み込み中: {camera_params_path.name}")
        with open(camera_params_path, 'r') as f:
            camera_params = json.load(f)

        if 'intrinsics' in camera_params:
            mtx = np.array(camera_params['intrinsics'])
            dist = np.array(camera_params['distortion'])
        elif 'camera_matrix' in camera_params:
            mtx = np.array(camera_params['camera_matrix'])
            dist = np.array(camera_params['distortion_coefficients'])
        else:
            print("エラー: JSONファイル内にカメラ行列が見つかりません。")
            continue

        # 2. 出力用ディレクトリを作成
        output_img_dir.mkdir(parents=True, exist_ok=True)
        print(f"画像出力先: {output_img_dir}")

        # 3. 動画ファイルを読み込む準備
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"エラー: 動画ファイルを開けませんでした: {video_path}")
            continue

        # 4. 出力動画の仕様を設定
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # ★総フレーム数を取得
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        print(f"動画仕様: {width}x{height}, {fps:.2f} FPS, {total_frames} フレーム")
        print(f"動画出力先: {output_video_path}")
        print("\n歪み補正処理を開始します...")

        # ★★★ ここからが修正箇所 ★★★
        # 5. tqdmを使って1フレームずつ処理
        for frame_count in tqdm(range(total_frames), desc=f"歪み補正中 ({direction})"):
            ret, frame = cap.read()
            if not ret:
                print(f"警告: フレーム {frame_count} で動画の読み込みが終了しました。")
                break

            # 歪みを補正
            undistorted_frame = cv2.undistort(frame, mtx, dist, None, mtx)

            # 補正後のフレームを動画ファイルに書き込む
            out.write(undistorted_frame)

            # 補正後のフレームを画像ファイルとして保存
            img_filename = f"frame_{frame_count:05d}.png"
            cv2.imwrite(str(output_img_dir / img_filename), undistorted_frame)
        # ★★★ ここまでが修正箇所 ★★★

        print(f"\n処理が完了しました。")

    except Exception as e:
        print(f"エラーが発生しました: {e}")

    finally:
        # 6. 後片付け
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals() and out.isOpened():
            out.release()
        cv2.destroyAllWindows()