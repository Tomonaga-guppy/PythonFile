"""
歪み補正行うためだが保存処理が遅いので使用中止
undistorted_fast.pyが改良版
"""


from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm

# 基本のパス設定
root_dir = Path(r"G:\gait_pattern\BR9G_shuron")
subject_dir_list = [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("sub")]
print(f"対象のPAディレクトリ: {[d.name for d in subject_dir_list]}")

directions = ["fl", "fr", "sagi"]

# ----------------------------
# 1) 処理対象（動画タスク）を全部リストアップ
# ----------------------------
tasks = []
for subject_dir in subject_dir_list:
    therapist_dir_list = [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("thera")]
    for thera_dir in therapist_dir_list:
        for direction in directions:
            video_dir = thera_dir / "gopro" / direction
            if not video_dir.exists():
                continue

            mp4_files = sorted(video_dir.glob("trimed.mp4"))
            if not mp4_files:
                # 後でスキップとして数えたいなら tasks に入れても良いが、ここでは対象外にする
                continue

            video_path = mp4_files[0]
            output_img_dir = video_dir / "undistorted"
            camera_params_path = root_dir.parent / "int_cali" / "9g_20250807_6x5" / direction / "camera_params.json"

            tasks.append({
                "subject_dir": subject_dir,
                "thera_dir": thera_dir,
                "direction": direction,
                "video_dir": video_dir,
                "video_path": video_path,
                "output_img_dir": output_img_dir,
                "camera_params_path": camera_params_path
            })

print(f"検出した動画タスク数: {len(tasks)}")

# ----------------------------
# 2) 全体進捗（動画タスク単位）
# ----------------------------
done = 0
skipped = 0
failed = 0

with tqdm(total=len(tasks), desc="全体進捗(動画)", unit="video") as pbar_total:
    for t in tasks:
        subject_dir = t["subject_dir"]
        thera_dir = t["thera_dir"]
        direction = t["direction"]
        video_path = t["video_path"]
        output_img_dir = t["output_img_dir"]
        camera_params_path = t["camera_params_path"]

        # ターミナル上部に「いま何を処理してるか」出す
        pbar_total.set_postfix_str(f"{subject_dir.name}/{thera_dir.name}/{direction}")

        # --- スキップ条件 ---
        if output_img_dir.exists():
            tqdm.write(f"[SKIP] 出力画像dirが存在: {output_img_dir}")
            skipped += 1
            pbar_total.update(1)
            continue

        if not video_path.exists():
            tqdm.write(f"[SKIP] 動画なし: {video_path}")
            skipped += 1
            pbar_total.update(1)
            continue

        if not camera_params_path.exists():
            tqdm.write(f"[FAIL] カメラパラメータなし: {camera_params_path}")
            failed += 1
            pbar_total.update(1)
            continue

        cap = None
        try:
            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"開始: {subject_dir.name} / {thera_dir.name} / {direction}")
            tqdm.write(f"動画: {video_path}")
            tqdm.write(f"params: {camera_params_path}")
            tqdm.write(f"{'='*60}")

            # 1. カメラパラメータ読み込み
            with open(camera_params_path, "r") as f:
                camera_params = json.load(f)

            if "intrinsics" in camera_params:
                mtx = np.array(camera_params["intrinsics"])
                dist = np.array(camera_params["distortion"])
            elif "camera_matrix" in camera_params:
                mtx = np.array(camera_params["camera_matrix"])
                dist = np.array(camera_params["distortion_coefficients"])
            else:
                tqdm.write("[FAIL] JSON内にカメラ行列が見つからない")
                failed += 1
                pbar_total.update(1)
                continue

            # 2. 動画オープン
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                tqdm.write(f"[FAIL] 動画を開けない: {video_path}")
                failed += 1
                pbar_total.update(1)
                continue

            # 3. 動画仕様
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            tqdm.write(f"動画仕様: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

            # undistortマップ
            tqdm.write("undistortマップ作成中...")
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (width, height), cv2.CV_16SC2)
            tqdm.write("undistortマップ作成完了")

            # 出力dir
            output_img_dir.mkdir(parents=True, exist_ok=True)
            tqdm.write(f"画像出力先: {output_img_dir}")

            # 4. フレーム進捗（内側 tqdm）
            for frame_count in tqdm(
                range(total_frames),
                desc=f"フレーム({direction})",
                unit="frame",
                leave=False
            ):
                ret, frame = cap.read()
                if not ret:
                    tqdm.write(f"[WARN] frame {frame_count} で読み込み終了")
                    break

                undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

                img_filename = f"frame_{frame_count:05d}.png"
                cv2.imwrite(str(output_img_dir / img_filename), undistorted_frame)

            done += 1
            tqdm.write(f"[OK] 完了: {subject_dir.name}/{thera_dir.name}/{direction}")

        except Exception as e:
            failed += 1
            tqdm.write(f"[FAIL] 例外: {subject_dir.name}/{thera_dir.name}/{direction} -> {e}")

        finally:
            if cap is not None and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()

        # 全体バー更新 + 状態表示
        pbar_total.update(1)
        pbar_total.set_postfix_str(f"OK:{done} SKIP:{skipped} FAIL:{failed}")

print(f"\n{'='*60}")
print("すべての処理が完了しました。")
print(f"OK={done}, SKIP={skipped}, FAIL={failed}")
print(f"{'='*60}")
