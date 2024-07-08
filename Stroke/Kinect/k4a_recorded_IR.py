import cv2
import numpy as np
from pyk4a import PyK4APlayback

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

        # IR画像を取得
        ir_image = capture.ir

        # 16ビットのIR画像の最大輝度を取得
        max_intensity = np.max(ir_image)
        print(f"Frame {frame_count}: The highest intensity value in the IR image is: {max_intensity}")

        # 16ビットのIR画像を表示（16ビットのまま）
        cv2.imshow('IR Image 16-bit', ir_image)

        # # 画像を保存
        # cv2.imwrite(f"ir_image_frame_{frame_count}.png", ir_image)

        # キーが押されるまで待機
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # クリーンアップ
    playback.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
