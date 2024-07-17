import pyrealsense2 as rs
import numpy as np
import cv2
import glob
import os
import math
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

def linear_interpolation(x, x0, x1, y0, y1):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def get_3d_coordinates(pixel, depth_frame, intrinsics):
    pixel_x, pixel_y = pixel[0], pixel[1]
    # 近傍のピクセル座標を計算(切り捨て、切り上げ)
    x0, x1 = int(math.floor(pixel_x)), int(math.ceil(pixel_x))
    y0, y1 = int(math.floor(pixel_y)), int(math.ceil(pixel_y))

    # 近傍の深度を取得
    depth_x0_y0 = depth_frame.get_distance(x0, y0)
    depth_x1_y0 = depth_frame.get_distance(x1, y0)
    depth_x0_y1 = depth_frame.get_distance(x0, y1)
    depth_x1_y1 = depth_frame.get_distance(x1, y1)

    # 近傍のピクセル座標の3D位置を計算
    point_x0_y0 = rs.rs2_deproject_pixel_to_point(intrinsics, [x0, y0], depth_x0_y0)
    point_x1_y0 = rs.rs2_deproject_pixel_to_point(intrinsics, [x1, y0], depth_x1_y0)
    point_x0_y1 = rs.rs2_deproject_pixel_to_point(intrinsics, [x0, y1], depth_x0_y1)
    point_x1_y1 = rs.rs2_deproject_pixel_to_point(intrinsics, [x1, y1], depth_x1_y1)

    # x方向の線形補間
    point_y0 = [linear_interpolation(pixel_x, x0, x1, point_x0_y0[i], point_x1_y0[i]) for i in range(3)]
    point_y1 = [linear_interpolation(pixel_x, x0, x1, point_x0_y1[i], point_x1_y1[i]) for i in range(3)]

    # y方向の線形補間
    point = [linear_interpolation(pixel_y, y0, y1, point_y0[i], point_y1[i]) for i in range(3)]

    return point

def load_keypoints_for_frame(frame_number, json_folder_path):
    json_file_name = f"original_{frame_number:012d}_keypoints.json"
    json_file_path = os.path.join(json_folder_path, json_file_name)

    if not os.path.exists(json_file_path):
        return None

    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
        keypoints_data = np.array(json_data['people'][0]['pose_keypoints_2d']).reshape((25, 3))

    return keypoints_data

def is_keypoint_missing(keypoint):
    return keypoint[2] == 0.0 or np.isnan(keypoint[0]) or np.isnan(keypoint[1])

def interpolate_missing_keypoints(current_frame, previous_frame, next_frame):
    interpolated_frame = current_frame.copy()
    for i, keypoint in enumerate(current_frame):
        if is_keypoint_missing(keypoint):
            if not is_keypoint_missing(previous_frame[i]) and not is_keypoint_missing(next_frame[i]):
                interpolated_frame[i][:2] = (previous_frame[i][:2] + next_frame[i][:2]) / 2
                interpolated_frame[i][2] = (previous_frame[i][2] + next_frame[i][2]) / 2
            elif not is_keypoint_missing(previous_frame[i]):
                interpolated_frame[i] = previous_frame[i]
            elif not is_keypoint_missing(next_frame[i]):
                interpolated_frame[i] = next_frame[i]
    return interpolated_frame

def animate_3d_keypoints(all_keypoints_3d, save_path):
    # BODY_25モデルのペア定義
    pairs = [
        (0, 15), (15, 17), (0, 16), (16, 18),
        (0, 1),
        (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (1, 8),
        (8, 9), (9, 10), (10, 11), (11, 24), (11,22), (22, 23),
        (8, 12), (12, 13), (13, 14), (14, 21), (14, 19), (19, 20)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        print(frame)
        ax.clear()
        keypoints = all_keypoints_3d[frame]
        x_coords = [point[0] for point in keypoints]
        y_coords = [point[1] for point in keypoints]
        z_coords = [point[2] for point in keypoints]
        ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')

        for pair in pairs:
            point1 = keypoints[pair[0]]
            point2 = keypoints[pair[1]]
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], 'b')

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.invert_yaxis()
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(f"Frame {frame}")

    ani = FuncAnimation(fig, update, frames=len(all_keypoints_3d), repeat=False)
    # writer = FFMpegWriter(fps=30)
    # ani.save(save_path, writer=writer)
    # print(f"{save_path}を保存しました")
    plt.show()


def main():
    bag_file_path = r"F:\Tomson\gait_pattern\20240607\output_device1_test_9.bag"
    foloder_path = os.path.dirname(bag_file_path) + '/' + os.path.basename(bag_file_path).split('.')[0]

    config = rs.config()
    config.enable_device_from_file(bag_file_path, repeat_playback=False)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # create Align Object
    align_to = rs.stream.color
    align = rs.align(align_to)

    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)  #リアルタイム再生をオフにするとフレームごとに処理が終わるまで待機してくれる

    # hole_filling_filterのパラメータ
    hole_filling = rs.hole_filling_filter(2)

    #内部パラメータの取得(RGBセンサー)
    color_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()

    try:
        frame_count = 0
        pre_time = 0
        json_foloder_path = os.path.join(foloder_path, 'estimated.json')
        all_keypoints_3d = []  # 各フレームの3Dキーポイントを保持するリスト

        while True:
            if frame_count < 60:
                frame_count +=1

            if frame_count >= 60:
                frames = pipeline.wait_for_frames()

                aligned_frames = align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                filter_frame = hole_filling.process(depth_frame)
                result_frame = filter_frame.as_depth_frame()

                # 対応するフレームのJSONファイルを読み込む
                keypoints_data = load_keypoints_for_frame(frame_count, json_foloder_path)
                if keypoints_data is None:
                    print(f"Frame {frame_count}: JSON file not found, exiting loop.")
                    break

                if any(is_keypoint_missing(keypoint) for keypoint in keypoints_data):
                    # 前後のフレームのキーポイントを読み込む
                    previous_keypoints_data = load_keypoints_for_frame(frame_count - 1, json_foloder_path) if frame_count > 0 else keypoints_data
                    if previous_keypoints_data is None:
                        print(f"Previous frame {frame_count - 1}: JSON file not found, exiting loop.")
                        break

                    next_keypoints_data = load_keypoints_for_frame(frame_count + 1, json_foloder_path)
                    if next_keypoints_data is None:
                        print(f"Next frame {frame_count + 1}: JSON file not found, exiting loop.")
                        break

                    # キーポイントの補完を行う
                    keypoints_data = interpolate_missing_keypoints(keypoints_data, previous_keypoints_data, next_keypoints_data)

                frame_keypoints_3d = []

                for keypoint in keypoints_data:
                    pixel = np.array(keypoint[:2])
                    coordinates = get_3d_coordinates(pixel, result_frame, color_intrinsics)
                    frame_keypoints_3d.append(coordinates)

                all_keypoints_3d.append(frame_keypoints_3d)

                print(f"Frame {frame_count}")
                cur_time = playback.get_position()  #再生時間の取得 単位はナノ秒
                if cur_time < pre_time:
                    break

                pre_time = cur_time
                frame_count += 1

    finally:
        pipeline.stop()

    all_keypoints_3d = np.array(all_keypoints_3d)
    corrected_all_keypoints_3d = all_keypoints_3d - all_keypoints_3d[0,8,:]  #0フレーム面の8番目のキーポイント(MidHip)を原点とする
    corrected_all_keypoints_3d.tolist()

    animate_3d_keypoints(corrected_all_keypoints_3d, save_path=os.path.join(foloder_path, 'animation.mp4'))

if __name__ == "__main__":
    main()
