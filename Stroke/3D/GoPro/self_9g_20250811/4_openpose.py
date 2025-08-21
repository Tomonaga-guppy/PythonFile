# 歪み補正した画像に対してOpenPoseを実行

import os
import subprocess
from pathlib import Path
import time

os.chdir(r"C:\Users\Tomson\openpose")###OpenPoseのあるところにカレントディレクトリを変更
i = 1  #処理する動画のカウンター

root_dir = Path(r"G:\gait_pattern\20250811_br")
subject_dir_list = [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("sub")]
print(f"対象のPAディレクトリ: {[d.name for d in subject_dir_list]}")
for subject_dir in subject_dir_list:  # 各被験者ディレクトリに対して
    therapist_dir_list = [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("thera")]
    print(f"対象のPTディレクトリ: {[d.name for d in therapist_dir_list]}")
    for thera_dir in therapist_dir_list:  # 各PTディレクトリに対して
        if thera_dir.name == "thera0-15":
            print(f"Mocap課題用の動画で今は使用しないのでスキップ: {thera_dir.name}")
            continue
        if thera_dir.name.startswith("thera0"):
            max_people = 1
        else:
            max_people = 2
        directions = ["fl", "fr", "sagi"]
        for direction in directions:  # 各方向に対して
            if i != 1:
                time.sleep(60)  #少しでもPC負荷を減らすために1分待つ
            ori_img_dir = thera_dir / direction / "undistorted"
            print(f"{i}/{len(subject_dir_list) * len(therapist_dir_list) * len(directions)}: {ori_img_dir}の処理を開始します")
            
            stem_name = f"openpose"  # 出力ファイルの名前のベース
            if Path(ori_img_dir.with_name(stem_name+'.avi')).exists():
                print(f"{stem_name}.avi はすでに存在します") #すでに推定済みの場合はスキップ
                i += 1
                continue

            ######OpenPoseへの命令作成 windowsの場合
            program= r".\build\x64\Release\OpenPoseDemo.exe"
            pre_img_dir = f" --image_dir " + str(ori_img_dir)  #画像の場所
            fps = f" --write_video_fps 60"
            after_video_place= f" --write_video {ori_img_dir.with_name(stem_name+'.avi')}"  #動画で出力
            after_images_place= f" --write_images {ori_img_dir.with_name(stem_name)}"  #画像で出力
            images_format = f" --write_images_format jpg"  #画像出力のフォーマットを指定
            after_json_place= f" --write_json {ori_img_dir.with_name(stem_name+'.json')}"  #各キーポイントの座標をjsonで出力
            other_order= f" --number_people_max {max_people} --scale_number 2 --scale_gap 0.2"
            cmd =program + pre_img_dir + fps + after_video_place + after_images_place + images_format + after_json_place + other_order
            ######OpenPoseへの命令作成終了


            subprocess.run(cmd)###作成したコマンドをターミナルに渡して実行
            i += 1