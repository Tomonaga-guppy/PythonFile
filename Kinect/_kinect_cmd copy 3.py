import subprocess
import signal
import os
import time
import threading

name = input("Enter name: ")

# 実行するコマンド
commands = [
    rf'"C:\Program Files\Azure Kinect SDK v1.4.1\tools\k4arecorder.exe" --device 0 --external-sync sub --depth-delay 160 --imu OFF --exposure-control -8 --rate 30 --color-mode 1080p {name}_0.mkv',
    rf'"C:\Program Files\Azure Kinect SDK v1.4.1\tools\k4arecorder.exe" --device 1 --external-sync sub --depth-delay 320 --imu OFF --exposure-control -8 --rate 30 --color-mode 1080p {name}_1.mkv',
    rf'"C:\Program Files\Azure Kinect SDK v1.4.1\tools\k4arecorder.exe" --device 2 --external-sync sub --depth-delay 480 --imu OFF --exposure-control -8 --rate 30 --color-mode 1080p {name}_2.mkv'
]

# 子プロセスを追跡するリスト
procs = []

# 各コマンドを新しいコマンドプロンプトウィンドウで実行
for command in commands:
    proc = subprocess.Popen(f'start /wait cmd /c "{command} & pause"', shell=True)
    procs.append(proc)

def signal_handler(sig, frame):
    for proc in procs:
            os.kill(proc.pid, signal.CTRL_C_EVENT) 
            print(f"プロセス {proc} 終了")
    print(f"python終了")
    os._exit(0)

# SIGINTシグナル（CTRL-C）をキャッチする
signal.signal(signal.SIGINT, signal_handler)

while True:
    pass
