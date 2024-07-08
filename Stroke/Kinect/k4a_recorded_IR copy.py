import cv2
import numpy as np
from pyk4a import PyK4APlayback, CalibrationType

def find_high_brightness_regions(ir_image, threshold=200):
    # 16ビットのIR画像を8ビットに変換
    ir_image_8bit = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 輝度が高い部分をバイナリマスクで抽出
    _, binary = cv2.threshold(ir_image_8bit, threshold, 255, cv2.THRESH_BINARY)

    # 高輝度部分の重心を計算
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []

    # print(f"")
    for i, contour in enumerate(contours):
        print(f"Contour {i+1}: {len(contour)}")
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])  #(輝度値*x座標)の和/輝度値の和
            cy = int(M['m01'] / M['m00'])  #(輝度値*y座標)の和/輝度値の和
            regions.append((cx, cy))

    return regions, binary

def get_3d_positions(calibration, depth_image, regions):
    positions_3d = []

    for region in regions:
        x, y = region
        depth_value = depth_image[y, x]

        if depth_value == 0:
            print("Depth value is 0. Skipping...")
            continue

        point3d = calibration.convert_2d_to_3d([x, y], depth_value, CalibrationType.DEPTH)
        print(f"point3d = {point3d}")
        positions_3d.append(point3d)

    return positions_3d

def main():
    # 録画したMKVファイルのパス
    mkv_file_path = r"C:\Users\zutom\output_test.mkv"

    # MKVファイルの再生
    playback = PyK4APlayback(mkv_file_path)
    playback.open()

    frame_count = 0

    while True:
        # 画像をキャプチャ
        capture = playback.get_next_capture()

        # キャプチャが有効でない場合（ファイルの終わり）ループを抜ける
        if capture is None:
            break

        # IR画像と深度画像を取得
        ir_image = capture.ir
        depth_image = capture.depth

        # 高輝度部分を検出
        high_brightness_regions, binary_image = find_high_brightness_regions(ir_image)

        # 高輝度部分に番号を表示
        for i, (cx, cy) in enumerate(high_brightness_regions):
            cv2.putText(binary_image, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 高輝度部分の3次元位置を取得
        positions_3d = get_3d_positions(playback.calibration, depth_image, high_brightness_regions)

        print(f"Frame {frame_count}: 3D Positions of High Brightness Regions: {positions_3d}")

        cv2.imshow('Binary Image', binary_image)

        # # 画像を保存
        # cv2.imwrite(f"ir_image_frame_{frame_count}.png", ir_image_8bit)

        # キーが押されるまで待機
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # クリーンアップ
    playback.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
