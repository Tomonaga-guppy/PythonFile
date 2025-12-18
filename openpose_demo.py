# 歪み補正した画像に対してOpenPoseを実行

import os
import subprocess
from pathlib import Path
import time

os.chdir(r"C:\Users\Tomson\openpose")###OpenPoseのあるところにカレントディレクトリを変更
i = 1  #処理する動画のカウンター

root_dir = Path(r"C:\Users\Tomson\openpose\examples\media")

# 介助歩行でも最大検出人数を1人にしてみる
max_people = 1


ori_video_path = root_dir / "video.avi"


print(f"{ori_video_path}の処理を開始します")


stem_name = f"openpose"  # 出力ファイルの名前のベース
    
######OpenPoseへの命令作成 windowsの場合
program= r".\build\x64\Release\OpenPoseDemo.exe"
pre_video_dir = f" --video " + str(ori_video_path)  #画像の場所
fps = f" --write_video_fps 60"
after_video_place= f" --write_video {ori_video_path.with_name(stem_name+'.avi')}"  #動画で出力
after_images_place= f" --write_images {ori_video_path.with_name(stem_name)}"  #画像で出力
images_format = f" --write_images_format jpg"  #画像出力のフォーマットを指定
after_json_place= f" --write_json {ori_video_path.with_name(stem_name+'.json')}"  #各キーポイントの座標をjsonで出力
other_order= f" --number_people_max {max_people} --scale_number 2 --scale_gap 0.2"
cmd =program + pre_video_dir + fps + after_video_place + after_images_place + images_format + after_json_place + other_order
######OpenPoseへの命令作成終了


subprocess.run(cmd)###作成したコマンドをターミナルに渡して実行