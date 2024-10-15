import subprocess

name = input("保存するファイル名を入力してください: ")
save_dir = r"C:\Users\tus\Desktop\record\recorded_data\kinect\20241011"

# 実行するコマンド
command = rf'"C:\Program Files\Azure Kinect SDK v1.4.1\tools\k4arecorder.exe" --record-length 5 --imu OFF --exposure-control -6 --rate 30 --color-mode 1080p {save_dir}\{name}_.mkv'
# 各コマンドを新しいコマンドプロンプトウィンドウで実行
subprocess.Popen(f'start cmd /k "{command}"', shell=True)
    

"""" command引数の意味
k4arecorder [options] output.mkv
/
 Options:
  -h, --help              Prints this help
  --list                  List the currently connected K4A devices
  --device                Specify the device index to use (default: 0)
  -l, --record-length     Limit the recording to N seconds (default: infinite)
  -c, --color-mode        Set the color sensor mode (default: 1080p), Available options:
                            3072p, 2160p, 1536p, 1440p, 1080p, 720p, 720p_NV12, 720p_YUY2, OFF
  -d, --depth-mode        Set the depth sensor mode (default: NFOV_UNBINNED), Available options:
                            NFOV_2X2BINNED, NFOV_UNBINNED, WFOV_2X2BINNED, WFOV_UNBINNED, PASSIVE_IR, OFF
  --depth-delay           Set the time offset between color and depth frames in microseconds (default: 0)
                            A negative value means depth frames will arrive before color frames.
                            The delay must be less than 1 frame period.
  -r, --rate              Set the camera frame rate in Frames per Second
                            Default is the maximum rate supported by the camera modes.
                            Available options: 30, 15, 5
  --imu                   Set the IMU recording mode (ON, OFF, default: ON)
  --external-sync         Set the external sync mode (Master, Subordinate, Standalone default: Standalone)
  --sync-delay            Set the external sync delay off the master camera in microseconds (default: 0)
                            This setting is only valid if the camera is in Subordinate mode.
  -e, --exposure-control  Set manual exposure value (-11 to 1) for the RGB camera (default: auto exposure)
"""