from argparse import ArgumentParser

import cv2

from pyk4a import PyK4APlayback

kinect2coco = [27, 3, 12, 13, 14, 5, 6, 7, 22, 23, 24, 18, 19, 20, 30, 28, 31, 29]


def play(playback: PyK4APlayback):
    while True:
        # 1フレームのデータを取得
        capture = playback.get_next_capture()

        # 骨格の取得
        pose = capture.body_skeleton

        # カラー画像，デプス画像の撮影タイムスタンプ（マイクロ秒）
        ctime = capture.color_timestamp_usec
        dtime = capture.depth_timestamp_usec
        print(ctime, dtime)

        # 骨格が検出されたら表示
        if pose is not None:
            for i in range(pose.shape[0]):
                pts = pose[i, :, :3].reshape(-1, 3)
                # Kinectの形式
                print("Kinect Format:", pts)
                # MSCOCOの形式
                print("COCO Format:", pts[kinect2coco])


def main() -> None:
    # データはマトリョーシカ形式(.mkv)で保存してあるものとする．
    # k4arecorder sample.mkv で保存できる．
    parser = ArgumentParser(description="pyk4a player")
    parser.add_argument("--seek", type=float, help="Seek file to specified offset in seconds", default=0.0)
    parser.add_argument("FILE", type=str, help="Path to MKV file written by k4arecorder")

    args = parser.parse_args()
    filename: str = args.FILE
    offset: float = args.seek

    # ファイルのオープン
    playback = PyK4APlayback(filename)
    playback.open()

    # 何秒目から始めるか（マイクロ秒）
    if offset != 0.0:
        playback.seek(int(offset * 1000 * 1000))

    play(playback)

    playback.close()


if __name__ == "__main__":
    main()