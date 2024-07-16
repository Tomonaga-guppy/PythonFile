import pyrealsense2 as rs
import time

root_dir = r"C:\Users\zutom\BRLAB\gait pattern\sync\test"

# ストリーミング設定を行う関数
def setup_config(camera_id, file_path):
    config = rs.config()
    config.enable_record_to_file(file_path)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_device(camera_id)
    return config

# カメラIDのリスト (カメラごとのシリアル番号を事前に確認してください)
camera_ids = ["SerialNumber1", "SerialNumber2"]
configs = []

# 各カメラの設定を初期化
for index, camera_id in enumerate(camera_ids):
    file_path = f"{root_dir}/output_camera_{index+1}.bag"
    config = setup_config(camera_id, file_path)
    configs.append(config)
    print(f"Setup complete for camera {index+1}")

# パイプラインの作成
pipeline1 = rs.pipeline()
pipeline2 = rs.pipeline()

try:
    # 両カメラのスタートを試みる
    pipeline1.start(configs[0])
    pipeline2.start(configs[1])
    print("Both cameras have started capturing.")

    # 撮影時間（例えば5秒間）
    time.sleep(5)

finally:
    # 各パイプラインを停止
    pipeline1.stop()
    pipeline2.stop()
    print("Capture completed for both cameras")
