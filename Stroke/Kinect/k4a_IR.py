import open3d as o3d
import cv2
import numpy as np
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS

def get_image_from_k4a(k4a):
    # 画像をキャプチャ
    capture = k4a.get_capture()
    ir_image = capture.ir

    max_val = np.max(ir_image)
    print(f"Max value: {max_val}")

    cv2.imshow("IR Image", ir_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ir_image

def main():
    # Azure Kinectの設定
    k4a = PyK4A(Config(
        color_resolution=ColorResolution.RES_1080P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        camera_fps=FPS.FPS_30,
        synchronized_images_only=True  # 全てのセンサーを同期して取得
    ))

    k4a.start()

    while True:
        try:
            ir_image = get_image_from_k4a(k4a)
        except Exception as e:
            print(f"An error occurred: {e}")
            break
        except KeyboardInterrupt:
            break

    # クリーンアップ
    k4a.stop()


if __name__ == "__main__":
    main()