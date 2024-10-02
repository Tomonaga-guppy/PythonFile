import cv2
import os

# マーカー画像を保存するディレクトリ
current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path_to_target = r"..\..\..\PythonDataFile\stroke\Aruco_markers"
marker_dir = os.path.abspath(os.path.join(current_dir, relative_path_to_target))

# marker_dir = os.path.dirname(__file__) + "/Aruco_markers"
print(f"marker_dir = {marker_dir}")

# ArUcoのライブラリを導入
aruco = cv2.aruco

# 6*6のマーカー，ID番号は50までの辞書を使う
dictionary = aruco.getPredefinedDictionary(aruco.DICT_7X7_50)

def main():

    # 10枚のマーカーを作るために10回繰り返す
    for i in range(10):
        ar_image = aruco.generateImageMarker(dictionary, i, 150)  # ID番号は i ，150x150ピクセルでマーカー画像を作る

        if not os.path.exists(marker_dir):  # ディレクトリが存在しない場合
            os.makedirs(marker_dir)

        fileName = marker_dir + "/ar" + str(i).zfill(2) + ".png"  # ファイル名を "ar0x.png" の様に作る
        # cv2.imwrite(fileName, ar_image)  # マーカー画像を保存する

        cv2.imshow("ArUco Marker", ar_image)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()  # メイン関数を実行
