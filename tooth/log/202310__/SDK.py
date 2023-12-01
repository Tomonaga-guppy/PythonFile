#参考 https://qiita.com/tom_eng_ltd/items/635414ff0b43e1c506f6

import pyrealsense2 as rs
import numpy as np
import cv2
import keyboard
import time

dir = "C:/Users/Tomson/Desktop/test/0922/"
mp4file = dir + 'original.mp4'

# save_dir = "F:/tooth/Temporomandibular_movement/movie/2023_09"
# file_naem = input("ファイル名を入力してください >> ")

# ストリーム(Depth/Color)の設定
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)

# Alignオブジェクト生成
align_to = rs.stream.color
align = rs.align(align_to)

# device = profile.get_device()
# playback = device.as_playback()

frame_rate = 30
size = (1280, 720)
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
writer = cv2.VideoWriter(mp4file, fmt, frame_rate, size) # ライター作成

# hole_filling_filterのパラメータ
hole_filling = rs.hole_filling_filter(2)

print("データ取得開始")
start = time.time()
fps_count = 1

try:
    while True:
        # フレーム待ち(Color & Depth)
        frames = pipeline.wait_for_frames()
        # print(frames)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        # color_frame = frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # if not depth_frame or not color_frame:
        if not color_frame:
            continue

        #imageをnumpy arrayに
        color_image = np.asanyarray(color_frame.get_data())
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        writer.write(color_image)

        filter_frame = hole_filling.process(depth_frame)
        result_frame = filter_frame.as_depth_frame()
        depth_scale = depth_frame.get_units()
        # print(fps_count, "depth_data *",depth_scale,"= meter")
        #depth imageをカラーマップに変換
        depth_image = np.asanyarray(result_frame.get_data())
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
        cv2.imwrite('{}{}.png'.format(dir, str(fps_count).zfill(4)), depth_image)

        #画像表示サイズは適当に設定
        # color_image_s = cv2.resize(color_image, (640, 360))
        # depth_colormap_s = cv2.resize(depth_colormap, (640, 360))
        # images = np.hstack((color_image_s, depth_colormap_s))
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        # cv2.waitKey(1)

        if cv2.waitKey(1) &keyboard.is_pressed("escape"):  #ESCで終了
            print("測定終了")
            break

        fps_count += 1

finally:
    end = time.time()
    elasped_time = end - start
    print(f"elapsep_time = ",{elasped_time})
    print(f"ideal_fps_count = ",{elasped_time * 30})
    print(f"fps_count = ",{fps_count})

    #ストリーミング停止
    writer.release()
    pipeline.stop()


