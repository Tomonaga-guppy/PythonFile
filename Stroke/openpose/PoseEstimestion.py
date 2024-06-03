import os
import subprocess

root_dir = r"c:\Users\Tomson\BRLAB\gait_pattern\first_test\recorded_data\realsense\two_dev"
id = "output_device1_test_8"
os.chdir(r"c:\Users\Tomson\openpose")###OpenPoseのあるところにカレントディレクトリを変更


######OpenPoseへの命令作成 windowsの場合
program= r".\build\x64\Release\OpenPoseDemo.exe"
pre_video_place=" --video " + root_dir+"/"+id +"/original.mp4"
after_video_place=" --write_video " + root_dir + "/"+id + "/estimated.avi"
after_json_place=" --write_json " + root_dir + "/"+id + "/estimated.json"
# other_order=' --number_people_max 2  --num_gpu 0 --scale_number 2 --scale_gap 0.2
cmd =program+pre_video_place+after_video_place+after_json_place
######OpenPoseへの命令作成終了


subprocess.run(cmd)###作成したコマンドをターミナルに渡す
print(f"cmd = {cmd}")