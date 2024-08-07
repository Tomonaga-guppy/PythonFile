import subprocess

name = input("Enter name: ")

# 実行するコマンド
commands = [
    rf'"C:\Program Files\Azure Kinect SDK v1.4.1\tools\k4arecorder.exe" --device 0 --external-sync master --imu OFF --exposure-control -8 --rate 30 --record-length 10 --color-mode 1080p {name}_0.mkv',
    rf'"C:\Program Files\Azure Kinect SDK v1.4.1\tools\k4arecorder.exe" --device 1 --external-sync sub --imu OFF --exposure-control -8 --rate 30 --record-length 10 --color-mode 1080p {name}_1.mkv',
    rf'"C:\Program Files\Azure Kinect SDK v1.4.1\tools\k4arecorder.exe" --device 2 --external-sync sub --imu OFF --exposure-control -8 --rate 30 --record-length 10 --color-mode 1080p {name}_2.mkv'
]

# 各コマンドを新しいコマンドプロンプトウィンドウで実行
for command in commands:
    subprocess.Popen(f'start cmd /k "{command}"', shell=True)
