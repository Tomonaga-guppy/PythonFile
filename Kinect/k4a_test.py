import cv2
# from pyk4a import PyK4APlayback
import pyk4a

# MKVファイルのパス
mkv_file_path = r"c:\Users\zutom\test.mkv"

# 再生オブジェクトの作成
playback = pyk4a.PyK4APlayback(mkv_file_path)
playback.open()

while True:
    try:
        # フレームを読み込む
        capture = playback.get_next_capture()
        if capture is None:
            print("Reached end of file.")
            break

        # カラー画像を取得
        color_image = capture.color
        if color_image is not None:
            # OpenCVで表示
            cv2.imshow('Color Image', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
        break

# 終了処理
playback.close()
cv2.destroyAllWindows()
