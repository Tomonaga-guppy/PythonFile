import os
import sys
import subprocess
from pathlib import Path

OPENPOSE_DIR = Path(r"C:\Users\Tomson\openpose")
OPENPOSE_EXECUTABLE_PATH = Path(r"C:\Users\Tomson\openpose\build\x64\Release\OpenPoseDemo.exe")
VIDEO_FL_PATH = Path(r"G:\gait_pattern\20250717_br\Tpose\fl\trim.mp4")
VIDEO_FR_PATH = Path(r"G:\gait_pattern\20250717_br\Tpose\fr\trim.mp4")
CAMERA_PARAMETER_PATH = Path(r"G:\gait_pattern\20250717_br\camera_parameters")

def run_command(command_list):
    """コマンドをリストとして受け取り、出力をリアルタイムで表示しながら実行する"""

    print("--- [EXECUTING COMMAND] ---")
    print(f"Command: {subprocess.list2cmdline(command_list)}")
    print("---------------------------")

    # Popenを使ってプロセスを開始し、出力をパイプで受け取る
    process = subprocess.Popen(
        command_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1, # 1行ずつバッファリングする
        cwd=OPENPOSE_DIR
    )

    # プロセスの出力をリアルタイムで一行ずつ読み取って表示
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line) # print()より高速なsys.stdout.writeを使用
            sys.stdout.flush()

    # プロセスの終了を待つ
    return_code = process.wait()

    # 終了コードを確認して結果を表示
    if return_code == 0:
        print("\n[SUCCESS] コマンドが正常に実行されました。\n")
    else:
        print(f"\n[ERROR] コマンドの実行に失敗しました (終了コード: {return_code})\n")

output_video_path = VIDEO_FL_PATH.parent.with_name("3d.avi")
output_json_path = VIDEO_FL_PATH.parent.with_name("3d_json")

command_list = [
    OPENPOSE_EXECUTABLE_PATH,
    '--video', str(VIDEO_FL_PATH), str(VIDEO_FR_PATH),
    '--camera_parameter_path', str(CAMERA_PARAMETER_PATH),
    '--3d',
    '--write_video', str(output_video_path),
    '--write_json', str(output_json_path),
    '--number_people_max', '1'
]
print(f"実行コマンド: {subprocess.list2cmdline(command_list)}")
run_command(command_list)