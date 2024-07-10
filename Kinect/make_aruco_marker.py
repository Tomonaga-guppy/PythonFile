import cv2
import os
from fpdf import FPDF

# マーカー画像を保存するディレクトリ
marker_dir = os.path.dirname(__file__) + "/Aruco_markers"
print(f"marker_dir = {marker_dir}")

# ArUcoのライブラリを導入
aruco = cv2.aruco

# 6*6のマーカー，ID番号は50までの辞書を使う
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)

def main():
    pdf = FPDF()

    # 10枚のマーカーを作るために10回繰り返す
    for i in range(10):
        ar_image = aruco.generateImageMarker(dictionary, i, 150)  # ID番号は i ，150x150ピクセルでマーカー画像を作る

        if not os.path.exists(marker_dir):  # ディレクトリが存在しない場合
            os.makedirs(marker_dir)

        fileName = marker_dir + "/ar" + str(i).zfill(2) + ".png"  # ファイル名を "ar0x.png" の様に作る
        cv2.imwrite(fileName, ar_image)  # マーカー画像を保存する

        pdf.add_page()
        pdf.image(fileName, x=10, y=10, w=100)  # マーカー画像をPDFに配置
        print(f"saved {fileName} and added to PDF")

    # まとめたPDFファイルを保存
    pdfFileName = marker_dir + "/all_markers.pdf"
    pdf.output(pdfFileName)
    print(f"saved all markers in {pdfFileName}")

if __name__ == "__main__":
    main()  # メイン関数を実行
