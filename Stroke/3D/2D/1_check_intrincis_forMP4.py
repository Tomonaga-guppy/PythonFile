import pickle
from pathlib import Path
import cv2
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import calu_saggital_gait_module as sggait
import sys

# 複数画像から内部パラメータを求める
root_dir = Path(r"G:\gait_pattern")

target_facility = "ota"  #解析対象施設：tkrzk_9g か ota

int_cali_dir = root_dir / "int_cali" / target_facility  #内部キャリブレーション結果を保存するフォルダ
pickle_path = int_cali_dir/f"Intrinsic_sg.pickle"
with open(pickle_path, "rb") as f:
    CameraParams = pickle.load(f)
condition_list= ["sub0_abngait", "sub0_asgait_2"]

for condition in condition_list:
    # mp4_path = int_cali_dir/f"20250227_ota_test/gopro/sa/gi/{condition}.MP4"
    mp4_path = root_dir/ "20241114_ota_test" / "gopro" / "sagi" /f"{condition}.MP4"
    # print(f"pickle_path: {pickle_path}")
    # print(f"rms: {CameraParams['rms']}")
    # print(f"焦点距離(fx, fy): {CameraParams['intrinsicMat'][0,0],CameraParams['intrinsicMat'][1,1]}")
    # print(f"光学中心(cx, cy): {CameraParams['intrinsicMat'][0,2],CameraParams['intrinsicMat'][1,2]}")
    # print(f"歪み係数: {CameraParams['distortion']}")



    #FrameCheck.csvを読み込み
    frame_check_csv_path = mp4_path.with_name(f"FrameCheck.csv")
    if not frame_check_csv_path.exists():
        print(f"FrameCheck.csvが見つかりません。")
        sggait.mkFrameCheckCSV(frame_check_csv_path, condition_list)
        sys.exit()

    frame_check_df = pd.read_csv(frame_check_csv_path)
    start_frame = frame_check_df[f"{condition}_Start"].values[0]
    end_frame = frame_check_df[f"{condition}_End"].values[0]
    print(f"start_frame: {start_frame}")

    cap = cv2.VideoCapture(str(mp4_path))
    ud_mov_path =mp4_path.with_name(mp4_path.stem+"_udCropped.MP4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    all_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    if not ret:
        print(f"開始のフレームが取得できませんでした")
        continue
    ud_img_sample = cv2.undistort(frame, CameraParams['intrinsicMat'], CameraParams['distortion'])
    ud_height, ud_width = ud_img_sample.shape[:2]

    writer = cv2.VideoWriter(str(ud_mov_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (ud_width, ud_height))

    frame_count = start_frame

    with tqdm(total=end_frame - start_frame + 1, desc = f"Undistorting {condition}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            undistort_img = cv2.undistort(frame, CameraParams['intrinsicMat'], CameraParams['distortion'])
            frame_count += 1
            pbar.update(1)

            mini_img = cv2.resize(undistort_img, (int(ud_width/4), int(ud_height/4)))
            cv2.imshow("undistort_img", mini_img)
            cv2.waitKey(1)

            writer.write(undistort_img)

            if frame_count == end_frame:
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"歪み補正を終了しました。{ud_mov_path}を確認してください。")




