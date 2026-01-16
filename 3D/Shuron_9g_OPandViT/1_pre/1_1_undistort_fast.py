"""
歪み補正実行用スクリプト
歪み補正と保存処理を分けて高速化
"""


from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os

# =========================
# 設定
# =========================
NUM_WORKERS = min(8, max(2, (os.cpu_count() or 8) // 2))   # 保存スレッド数
MAX_INFLIGHT = NUM_WORKERS * 4  # 同時に溜める保存ジョブ上限（メモリ暴走防止）
USE_COPY_FOR_THREAD = True      # 参照事故防止。基本True推奨

# =========================
# 基本のパス設定
# =========================
root_dir = Path(r"G:\gait_pattern\2025_shuron_BR9G")
subject_dir_list = [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("sub")]

# 時間足りないのでsub7以降を除外 ###################################################################################################################
subject_dir_list = [d for d in subject_dir_list if int(d.name.replace("sub", "")) < 11]

print(f"対象のPAディレクトリ: {[d.name for d in subject_dir_list]}")
# directions = ["fl", "fr"]#####################################################################################################################
directions = ["sagi"]


# ----------------------------
# 1) まず処理対象タスク（動画）を全部リストアップ
# ----------------------------
tasks = []
for subject_dir in subject_dir_list:
    therapist_dir_list = [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("thera")]
    
    # thera0-1で始まるものを除外 #################################################################################################################
    therapist_dir_list = [d for d in therapist_dir_list if not d.name.startswith("thera0-1")]
    
    for thera_dir in therapist_dir_list:
        for direction in directions:
            video_dir = thera_dir / "gopro" / direction
            if not video_dir.exists():
                continue

            mp4_files = sorted(video_dir.glob("trimed.mp4"))
            if not mp4_files:
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

print(f"検出したタスク: {tasks}")
print(f"検出した動画タスク数: {len(tasks)}")
print(f"保存スレッド: {NUM_WORKERS}, MAX_INFLIGHT: {MAX_INFLIGHT}")


def save_png(path: Path, img: np.ndarray):
    cv2.imwrite(str(path), img)

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
        executor = None
        inflight = set()

        try:
            tqdm.write(f"\n{'='*70}")
            tqdm.write(f"開始: {subject_dir.name} / {thera_dir.name} / {direction}")
            tqdm.write(f"動画: {video_path}")
            tqdm.write(f"params: {camera_params_path}")
            tqdm.write(f"{'='*70}")

            # 1) カメラパラメータ読み込み
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

            # 2) 動画オープン
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                tqdm.write(f"[FAIL] 動画を開けない: {video_path}")
                failed += 1
                pbar_total.update(1)
                continue

            # 3) 動画仕様
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            tqdm.write(f"動画仕様: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

            # 4) undistortマップ作成（高速化の要）
            tqdm.write("undistortマップ作成中...")
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (width, height), cv2.CV_16SC2)
            tqdm.write("undistortマップ作成完了")

            # 5) 出力dir作成
            output_img_dir.mkdir(parents=True, exist_ok=True)
            tqdm.write(f"画像出力先: {output_img_dir}")

            # 6) 保存用スレッドプール開始
            executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)

            # 7) フレーム処理（内側進捗バー）
            frame_count = 0
            pbar = tqdm(total=total_frames, desc=f"フレーム({direction})", unit="frame", leave=False)

            while True:
                ret, frame = cap.read()
                if not ret:
                    tqdm.write(f"[WARN] frame {frame_count} で読み込み終了")
                    break

                # 歪み補正（高速）
                undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

                # ファイル名
                out_path = output_img_dir / f"frame_{frame_count:05d}.png"

                # スレッドへ保存依頼（参照事故防止でcopy）
                img_for_save = undistorted.copy() if USE_COPY_FOR_THREAD else undistorted
                fut = executor.submit(save_png, out_path, img_for_save)
                inflight.add(fut)

                # メモリ暴走防止：溜まり過ぎたらどれか終わるまで待つ
                if len(inflight) >= MAX_INFLIGHT:
                    done_set, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    # 例外があったらここで拾える
                    for d in done_set:
                        _ = d.result()

                frame_count += 1
                pbar.update(1)

            pbar.close()

            # 8) 残りの保存完了待ち（進捗つき）
            if inflight:
                tqdm.write(f"保存完了待ち: 残 {len(inflight)}")
                # 進捗バーで待つ
                with tqdm(total=len(inflight), desc="書き込み待ち", unit="file", leave=False) as pbar_wait:
                    while inflight:
                        done_set, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                        for d in done_set:
                            _ = d.result()  # 例外があればここで上がる
                            pbar_wait.update(1)

            done += 1
            tqdm.write(f"[OK] 完了: {subject_dir.name}/{thera_dir.name}/{direction}")

        except Exception as e:
            failed += 1
            tqdm.write(f"[FAIL] 例外: {subject_dir.name}/{thera_dir.name}/{direction} -> {e}")

        finally:
            # 解放
            if cap is not None and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()

            if executor is not None:
                executor.shutdown(wait=True)

        pbar_total.update(1)
        pbar_total.set_postfix_str(f"OK:{done} SKIP:{skipped} FAIL:{failed}")

print(f"\n{'='*70}")
print("すべての処理が完了しました。")
print(f"OK={done}, SKIP={skipped}, FAIL={failed}")
print(f"{'='*70}")
