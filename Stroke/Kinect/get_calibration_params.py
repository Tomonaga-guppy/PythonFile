from pyk4a import PyK4APlayback, CalibrationType
import glob
import os

# MKVファイルのパス
root_dir = r"F:\Tomson\gait_pattern\20240912"
mkv_file_paths = glob.glob(os.path.join(root_dir, "*.mkv"))

for mkv_file_path in mkv_file_paths:
    # MKVファイルを開く
    playback = PyK4APlayback(mkv_file_path)

    # 再生を開始
    playback.open()

    # キャリブレーションデータ（内部パラメータ）を取得
    calibration = playback.calibration
    if calibration is not None:
        try:
            # カラーカメラの内部パラメータを取得
            color_camera_calibration = calibration.get_camera_matrix(CalibrationType.COLOR)
            print(f"Color Camera Calibration Matrix: {mkv_file_path}")
            print(color_camera_calibration)

            # # 深度カメラの内部パラメータを取得
            # depth_camera_calibration = calibration.get_camera_matrix(CalibrationType.DEPTH)
            # print("\nDepth Camera Calibration Matrix:")
            # print(depth_camera_calibration)

        except ValueError as e:
            print(f"Error occurred: {e}")
    else:
        print("Calibration data is not available.")

    # 再生を終了
    playback.close()