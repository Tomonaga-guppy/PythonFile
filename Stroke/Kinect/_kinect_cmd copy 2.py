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

# ロックオブジェクトを作成
print_lock = threading.Lock()

# 各コマンドを新しいコマンドプロンプトウィンドウで実行
for command in commands:
    proc = subprocess.Popen(f'start /wait cmd /c "{command} & pause"', shell=True)
    procs.append(proc)

def signal_handler(sig, frame):
    with print_lock:
        print("CTRL-C pressed. Terminating all processes...")
    for proc in procs:
        try:
            os.kill(proc.pid, signal.CTRL_BREAK_EVENT)  # CTRL-BREAKシグナルを送信
        except Exception as e:
            with print_lock:
                print(f"Failed to terminate process {proc.pid}: {e}")
    with print_lock:
        print("All processes terminated.")
    print(f"python終了")
    os._exit(0)

# SIGINTシグナル（CTRL-C）をキャッチする
signal.signal(signal.SIGINT, signal_handler)

while True:
    pass
