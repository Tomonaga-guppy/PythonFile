import pickle
from pathlib import Path
import cv2
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import gait_module as sggait
import sys

# 複数画像から内部パラメータを求める
root_dir = Path(r"G:\gait_pattern")

##################### 毎回チェック！
# target_facility = "ota"  #解析対象施設：tkrzk_9g か ota
target_facility = "ota_20250228"   #このプログラム内ではどのキャリブレーション結果を使うか
#####################

mov = "20250228_ota"
date = "20250221"
sub = "sub0"

int_cali_dir = root_dir / "int_cali" / target_facility  #内部キャリブレーション結果を保存するフォルダ
pickle_path = int_cali_dir/f"Intrinsic_sg_custom.pickle"
with open(pickle_path, "rb") as f:
    CameraParams = pickle.load(f)
condition_list= ["thera0-3", "thera1-1", "thera2-1"]

for condition in condition_list:
    # mp4_path = int_cali_dir/f"20250227_ota_test/gopro/sa/gi/{condition}.MP4"
    sub_dir = root_dir/ mov / "data" / date / sub
    mp4_dir = sub_dir / condition / "sagi"
    print(f"mp4_dir: {mp4_dir}")
    mp4_path = list(mp4_dir.glob("*GX*.MP4"))[0]
    # print(f"pickle_path: {pickle_path}")
    # print(f"rms: {CameraParams['rms']}")
    # print(f"焦点距離(fx, fy): {CameraParams['intrinsicMat'][0,0],CameraParams['intrinsicMat'][1,1]}")
    # print(f"光学中心(cx, cy): {CameraParams['intrinsicMat'][0,2],CameraParams['intrinsicMat'][1,2]}")
    # print(f"歪み係数: {CameraParams['distortion']}")

    #FrameCheck.csvを読み込み
    frame_check_csv_path = (sub_dir / condition).with_name(f"FrameCheck.csv")
    if not frame_check_csv_path.exists():
        print(f"FrameCheck.csvが見つかりません。")
        #これ実行されると書いていたフレームとか消されちゃうので注意！（要修正）
        sggait.mkFrameCheckCSV(frame_check_csv_path, condition_list)
        sys.exit()

    frame_check_df = pd.read_csv(frame_check_csv_path)
    start_frame = frame_check_df[f"{condition}_Start"].values[0]
    end_frame = frame_check_df[f"{condition}_End"].values[0]
    print(f"start_frame: {start_frame}")

    cap = cv2.VideoCapture(str(mp4_path))
    ud_mov_path =mp4_path.with_name(f"Undistort_Custom.MP4")
    all_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    if not ret:
        print(f"開始のフレームが取得できませんでした")
        continue

    ori_folder = mp4_path.parent / "Ori"
    ori_folder.mkdir(exist_ok=True)
    undistort_folder = mp4_path.parent / "Undistort"
    undistort_folder.mkdir(exist_ok=True)

    frame_count = start_frame

    with tqdm(total=end_frame - start_frame + 1, desc = f"Undistorting {condition}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            undistort_img = cv2.undistort(frame, CameraParams['intrinsicMat'], CameraParams['distortion'])
            frame_count += 1
            pbar.update(1)

            img_shape = frame.shape[:2]
            mini_img = cv2.resize(undistort_img, (int(img_shape[1]/4), int(img_shape[0]/4)))
            cv2.imshow(f"undistort_img frame", mini_img)
            cv2.waitKey(1)

            ori_img_path = ori_folder / f"frame_{frame_count:04d}.png"
            undistort_img_path = undistort_folder / f"frame_{frame_count:04d}.png"
            cv2.imwrite(str(ori_img_path), frame)
            cv2.imwrite(str(undistort_img_path), undistort_img)

            if frame_count == end_frame:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"歪み補正を終了しました。{ud_mov_path}を確認してください。")




