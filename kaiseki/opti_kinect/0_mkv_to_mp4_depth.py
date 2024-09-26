from pyk4a import PyK4APlayback
import cv2
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

helpers_dir = r"C:\Users\pyk4a\example"
os.chdir(helpers_dir)
sys.path.append(helpers_dir)
from helpers import convert_to_bgra_if_required

def main():
    mkv_folder = r"F:\Tomson\gait_pattern\20240808"
    mkv_files = glob.glob(os.path.join(mkv_folder, '0_*.mkv'))
    mkv_files = glob.glob(os.path.join(mkv_folder, '1_walk*.mkv'))
    # mkv_files = glob.glob(os.path.join(mkv_folder, '2_walk_slow*.mkv'))
    print(f"mkv_files = {mkv_files}")

    for i, mkv_file_path in enumerate(mkv_files):
        folder_path = os.path.dirname(mkv_file_path) + '/' + os.path.basename(mkv_file_path).split('.')[0]
        if os.path.exists(folder_path) == False:
            os.mkdir(folder_path)

        # MKVファイルの再生
        playback = PyK4APlayback(mkv_file_path)
        playback.open()
        calibration = playback.calibration

        frame_count = 1

        """
        # mp4ファイルの作成
        mp4file = folder_path + "/original.mp4"
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
        fps = 30.0
        size = (1920,1080)
        writer = cv2.VideoWriter(mp4file, fmt, fps, size) # ライター作成

        while True:
            try:
                # 画像をキャプチャ
                capture = playback.get_next_capture()
            except:
                print("再生を終了します")
                break

            # キャプチャが有効でない場合（ファイルの終わり）ループを抜ける
            if capture is None:
                break

            if capture.color is None:
                print(f"Frame {frame_count} has no RGB image data.")
                continue

            # RGB画像を取得
            rgb_image = convert_to_bgra_if_required(playback.configuration["color_format"], capture.color)
            depth_image = capture.transformed_depth

            rgb_image_mini = cv2.resize(rgb_image, (1080,720))
            cv2.imshow("RGB Image", rgb_image_mini)

            # キーが押されるまで待機
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(f"{i}/{len(mkv_files)} RGB frame_count = {frame_count}")
            writer.write(rgb_image)
            frame_count += 1

        # クリーンアップ
        playback.close()
        cv2.destroyAllWindows()
        writer.release()
        """


        # depth画像の保存
        original_depth_image_folder_path = folder_path + "/depth_image_original"
        if not os.path.exists(original_depth_image_folder_path):
            os.mkdir(original_depth_image_folder_path)

        depth_image_folder_path = folder_path + "/filled_depth_image"
        if not os.path.exists(depth_image_folder_path):
            os.mkdir(depth_image_folder_path)



        while True:
            try:
                # 画像をキャプチャ
                capture = playback.get_next_capture()
            except:
                print("再生を終了します")
                break

            # キャプチャが有効でない場合（ファイルの終わり）ループを抜ける
            if capture is None:
                break

            if capture.transformed_depth is None:
                print(f"Frame {frame_count} has no depth image data.")
                continue

            # Depht画像を取得
            depth_image = capture.transformed_depth

            # 深度画像の欠損値を近傍の最小値で補間
            mask = depth_image == 0
            interpolated_depth_image = depth_image.copy()
            shifted_list = []
            kernel_size = 7
            # 行方向と列方向にシフトした画像を作成
            for x_shift in range(-int((kernel_size-1)/2),  int((kernel_size-1)/2 + 1)):
                for y_shift in range(int((kernel_size-1)/2), -int((kernel_size-1)/2 + 1), -1):
                    if x_shift == 0 and y_shift == 0:
                        continue
                    shifted = np.roll(depth_image, (x_shift, y_shift), axis=(0, 1))
                    shifted_list.append(shifted)
            # シフトした画像の中で0以外の最小値を取得
            min_values = np.full(depth_image.shape, np.inf)  # 初期値を無限大に設定
            for shifted in shifted_list:
                non_zero_mask = shifted != 0
                min_values[non_zero_mask] = np.minimum(min_values[non_zero_mask], shifted[non_zero_mask])
            # 無限大が残っている場所はすべて0（欠損値）だった場所なので、0に置き換える
            min_values[min_values == np.inf] = 0
            # 欠損値の位置に最小値を代入
            interpolated_depth_image[mask] = min_values[mask]

            # 画像の保存
            original_depth_image_path = os.path.join(original_depth_image_folder_path, f"{str(frame_count).zfill(4)}.png")
            depth_image_path = os.path.join(depth_image_folder_path, f"{str(frame_count).zfill(4)}.png")
            # cv2.imwrite(original_depth_image_path, depth_image)
            # cv2.imwrite(depth_image_path, interpolated_depth_image)

            plt.imshow(depth_image)
            plt.colorbar()
            plt.show()

            plt.imshow(interpolated_depth_image)
            plt.colorbar()
            plt.show()

            # ax, fig = plt.subplots(1, 2)
            # fig[0].imshow(depth_image)
            # fig[1].imshow(interpolated_depth_image)

            # # 各ピクセルの値を表示
            # for image in [depth_image, interpolated_depth_image]:
            #     for i in range(image.shape[0]):
            #         for j in range(image.shape[1]):
            #             plt.text(j, i, str(image[i, j]), ha='center', va='center', color='red')


            plt.show()


            # # 一度保存した画像を読み込んで深度が同じかどうか確認
            # check_val = interpolated_depth_image[400, 1000] #y,xの順
            # print(f"check_val = {check_val}")

            # check_depth_img = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
            # check_val2 = check_depth_img[400, 1000]
            # print(f"check_val2 = {check_val2}")

            print(f"{i}/{len(mkv_files)} Depth frame_count = {frame_count}")
            frame_count += 1


if __name__ == '__main__':
    main()