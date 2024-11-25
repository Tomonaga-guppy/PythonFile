import os
import subprocess
import glob
from pathlib import Path

# root_dir = r"G:\gait_pattern\20240912"
root_dir = Path(r"G:\gait_pattern\20241114_ota_test\gopro")

ori_mov_paths = root_dir.glob(f"**/*sub*.MP4")
print(f"ori_mov_paths: {ori_mov_paths}")

os.chdir(r"c:\Users\Tomson\openpose")###OpenPoseのあるところにカレントディレクトリを変更

for i, ori_mov_path in enumerate(ori_mov_paths):
    dir_name = ori_mov_path.parent
    print(f"{i+1}/{len(ori_mov_paths)}: {ori_mov_path}")

    # #すでに推定済みの場合はスキップ
    # if os.path.exists(dir_name + "/estimated.avi") and os.path.exists(dir_name + "/estimated.json"):
    #     print(f"{os.path.basename(os.path.dirname(ori_mov_path))} already estimated")
    #     continue

    stem_name = ori_mov_path.stem

    ######OpenPoseへの命令作成 windowsの場合
    program= r".\build\x64\Release\OpenPoseDemo.exe"
    pre_video_place= " --video " + dir_name + "/" + stem_name + ".MP4"  #動画の場所
    after_video_place= " --write_video " + dir_name + "/" + stem_name +"_op.avi"  #動画で出力
    after_json_place= " --write_json " + dir_name + "/" + stem_name + "_op.json"  #各キーポイントの座標をjsonで出力
    other_order= " --number_people_max 1 --scale_number 2 --scale_gap 0.2"
    cmd =program + pre_video_place + after_video_place + after_json_place + other_order
    ######OpenPoseへの命令作成終了


    subprocess.run(cmd)###作成したコマンドをターミナルに渡して実行