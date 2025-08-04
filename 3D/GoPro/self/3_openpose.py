# 歪み補正した画像に対してOpenPoseを実行

import os
import subprocess
from pathlib import Path
import sys

root_dir = Path(r"G:\gait_pattern\20250717_br\Tpose")
directions = ["fl", "fr"]
max_people = 1

os.chdir(r"C:\Users\Tomson\openpose")###OpenPoseのあるところにカレントディレクトリを変更

for i, direction in enumerate(directions):
    print(f"{i+1}/{len(directions)}: {direction}")
    ori_img_dir = root_dir / direction / "undistorted"

    stem_name = f"openpose"
    if Path(ori_img_dir.with_name(stem_name+'.avi')).exists():
        print(f"{stem_name}.avi はすでに存在します") #すでに推定済みの場合はスキップ
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