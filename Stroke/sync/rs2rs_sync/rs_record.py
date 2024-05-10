import pyrealsense2 as rs
import time


root_dir = r"C:\Users\zutom\BRLAB\gait pattern\sync\test"

# ストリーミング設定
pipeline = rs.pipeline()
config = rs.config()

#input関数でターミナルに入力された番号を取得
blinking_interval = input("点灯間隔を入力してください: ")


# .bagファイルに保存する設定
file_path = root_dir+f'/output_solo_{blinking_interval}.bag'
config.enable_record_to_file(file_path)

# カメラの設定: 解像度とフレームレート
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

try:
    # ストリーミング開始30
    pipeline.start(config)

    # カメラのセットアップ
    profile = pipeline.get_active_profile()
    color_sensor = profile.get_device().first_color_sensor()
    color_sensor.set_option(rs.option.enable_auto_exposure, 0)  # 自動露光を無効化

    # RGBカメラの露光時間設定
    color_sensor.set_option(rs.option.exposure, 156)  # 1000マイクロ秒 = 1ミリ秒

    # 撮影時間（例えば5秒間）
    print("Capturing data to", file_path)
    time.sleep(5)

finally:
    # ストリーミング停止
    pipeline.stop()
    print("Capture completed, data saved to", file_path)

