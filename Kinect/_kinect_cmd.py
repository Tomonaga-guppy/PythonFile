import subprocess

name = input("Enter name: ")
save_dir = r"C:\Users\tus\Desktop\record\recorded_data\kinect"

# 実行するコマンド
commands = [
    rf'"C:\Program Files\Azure Kinect SDK v1.4.1\tools\k4arecorder.exe" --device 0 --external-sync sub --depth-delay 160 --imu OFF --exposure-control -6 --rate 30 --color-mode 1080p {save_dir}\{name}_0.mkv',
    rf'"C:\Program Files\Azure Kinect SDK v1.4.1\tools\k4arecorder.exe" --device 1 --external-sync sub --depth-delay 320 --imu OFF --exposure-control -6 --rate 30 --color-mode 1080p {save_dir}\{name}_1.mkv',
    rf'"C:\Program Files\Azure Kinect SDK v1.4.1\tools\k4arecorder.exe" --device 2 --external-sync sub --depth-delay 480 --imu OFF --exposure-control -6 --rate 30 --color-mode 1080p {save_dir}\{name}_2.mkv'
]

# 各コマンドを新しいコマンドプロンプトウィンドウで実行
for command in commands:
    subprocess.Popen(f'start cmd /k "{command}"', shell=True)