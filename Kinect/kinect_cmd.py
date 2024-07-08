import subprocess

# 実行するコマンド
commands = [
    r'"C:\Program Files\Azure Kinect SDK v1.4.1\tools\k4arecorder.exe" --device 0 --external-sync sub --imu OFF --exposure-control -8 --rate 30 --record-length 10 --color-mode 1080p out_put0.mkv',
    r'"C:\Program Files\Azure Kinect SDK v1.4.1\tools\k4arecorder.exe" --device 1 --external-sync sub --imu OFF --exposure-control -8 --rate 30 --record-length 10 --color-mode 1080p out_put1.mkv',
    r'"C:\Program Files\Azure Kinect SDK v1.4.1\tools\k4arecorder.exe" --device 2 --external-sync sub --imu OFF --exposure-control -8 --rate 30 --record-length 10 --color-mode 1080p out_put2.mkv'
]

# 各コマンドを新しいコマンドプロンプトウィンドウで実行
for command in commands:
    subprocess.Popen(f'start cmd /k "{command}"', shell=True)

