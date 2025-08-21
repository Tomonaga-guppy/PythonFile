from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm

# 基本のパス設定
root_dir = Path(r"G:\gait_pattern\20250811_br")
subject_dir_list = [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("sub")]
print(f"対象のPAディレクトリ: {[d.name for d in subject_dir_list]}")

for subject_dir in subject_dir_list:
    therapist_dir_list = [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("thera")]
    print(f"対象のPTディレクトリ: {[d.name for d in therapist_dir_list]}")
    for thera_dir in therapist_dir_list:
        ##################################################
        if thera_dir.name == "thera0-15":
            print(f"Mocap課題用の動画で今は使用しないのでスキップ: {thera_dir.name}")
            continue
        ###################################################
        directions = ["fl", "fr", "sagi"]
        for direction in directions:
            video_dir = thera_dir / direction
            if not video_dir.exists():
                continue

            mp4_files = sorted(video_dir.glob("trimed*.mp4"))
            if not mp4_files:
                print(f"警告: trimedから始まるmp4ファイルが見つかりません: {video_dir}")
                continue

            # trimedから始まるmp4ファイルが1つの場合はそれを使用
            video_path = mp4_files[0]
            print(f"使用する動画ファイル: {video_path}")

            print(f"\n{'='*60}")
            print(f"処理を開始します: {direction}")
            print(f"{'='*60}")

            # 歪み補正後の動画と画像の保存先
            output_video_path = video_dir / "undistorted.mp4"
            output_img_dir = video_dir / "undistorted"
            
            if output_img_dir.exists():
                print(f"出力画像ディレクトリがすでに存在するため処理をスキップします: {output_img_dir}")
                continue

            # 歪みパラメータのパス
            camera_params_path = root_dir.parent / "int_cali" / "9g_20250807_6x5" / direction / "camera_params.json"

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

                # 2. 動画ファイルを読み込む準備
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"エラー: 動画ファイルを開けませんでした: {video_path}")
                    continue

                # 3. 動画の仕様を取得
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                print(f"動画仕様: {width}x{height}, {fps:.2f} FPS, {total_frames} フレーム")

                # undistortマップを事前作成(毎フレームでcv2.undistortを使うよりも効率的)
                print("undistortマップを作成中...")
                mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (width, height), cv2.CV_16SC2)
                print("undistortマップの作成が完了しました。")

                # 4. 出力用ディレクトリを作成
                output_img_dir.mkdir(parents=True, exist_ok=True)
                print(f"画像出力先: {output_img_dir}")

                # 5. 出力動画の設定
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

                print(f"動画出力先: {output_video_path}")
                print("\n歪み補正処理を開始します...")

                # 6. 各フレームを高速処理
                for frame_count in tqdm(range(total_frames), desc=f"歪み補正中 ({direction})"):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"警告: フレーム {frame_count} で動画の読み込みが終了しました。")
                        break

                    # ★★★ 高速化: cv2.remapを使用（事前作成したマップを利用） ★★★
                    # cv2.undistortの代わりにremapを使用することで大幅に高速化
                    undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

                    # 補正後のフレームを動画ファイルに書き込む
                    out.write(undistorted_frame)

                    # 補正後のフレームを画像ファイルとして保存
                    img_filename = f"frame_{frame_count:05d}.png"
                    cv2.imwrite(str(output_img_dir / img_filename), undistorted_frame)

                print(f"\n処理が完了しました。")

            except Exception as e:
                print(f"エラーが発生しました: {e}")

            finally:
                # 7. 後片付け
                if 'cap' in locals() and cap.isOpened():
                    cap.release()
                if 'out' in locals() and out.isOpened():
                    out.release()
                cv2.destroyAllWindows()

print(f"\n{'='*60}")
print("すべての処理が完了しました。")
print(f"{'='*60}")