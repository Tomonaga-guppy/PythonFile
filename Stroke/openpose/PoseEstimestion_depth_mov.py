import os
import subprocess
import glob

root_dir = r"F:\Tomson\gait_pattern\first_test\recorded_data\realsense\two_dev"
keyward = "test_[7-8]"
ori_mov_paths = glob.glob(os.path.join(root_dir, ("*" + keyward + "*"), 'original_depth.mp4'))

os.chdir(r"c:\Users\Tomson\openpose")###OpenPoseのあるところにカレントディレクトリを変更

for i, ori_mov_path in enumerate(ori_mov_paths):
    dir_name = os.path.dirname(ori_mov_path)
    print(f"{i+1}/{len(ori_mov_paths)}: {ori_mov_path}")

    # #すでに推定済みの場合はスキップ
    # if os.path.exists(dir_name + "/estimated.avi") and os.path.exists(dir_name + "/estimated.json"):
    #     print(f"{os.path.basename(os.path.dirname(ori_mov_path))} already estimated")
    #     continue

    ######OpenPoseへの命令作成 windowsの場合
    program= r".\build\x64\Release\OpenPoseDemo.exe"
    pre_video_place= " --video " + dir_name +"/original_depth.mp4"
    after_video_place= " --write_video " + dir_name + "/estimated_depth_color.avi"  #動画で出力
    after_json_place= " --write_json " + dir_name + "/estimated_depth_color.json"  #各キーポイントの座標をjsonで出力
    other_order= " --number_people_max 2 --scale_number 2 --scale_gap 0.2"
    cmd =program + pre_video_place + after_video_place + after_json_place + other_order
    ######OpenPoseへの命令作成終了

    print(cmd)
    subprocess.run(cmd)###作成したコマンドをターミナルに渡して実行