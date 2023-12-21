import cv2
import os
import csv
import numpy as np

dir_path = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\2023_09_000\20230606_J2"
# テンプレートマッチングを実行する
template_path = r"C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/seal_template/seal.png"
image_path = dir_path + "/RGB_image/0001.png"
img = cv2.imread(image_path)
imgcopy = img.copy()

# OpenFace_result 0 frame, 5-72 x_pixel, 73-140 y_pixel
csv_file = dir_path + '/RGB_image.csv'
if os.path.isfile(csv_file) == False:
    csv_file = dir_path + '/OpenFace.csv'

OpenFace_result = []
with open(csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        OpenFace_result.append(row)


mask1_x=int(float(OpenFace_result[1][53]))    #48x
mask1_y=int(float(OpenFace_result[1][121]))    #48y
mask2_x=int(float(OpenFace_result[1][59]))    #54x
mask2_y=int(float(OpenFace_result[1][81]))    #8y

# 矩形のマスク画像の生成
width, height = 1280, 720
mask = np.zeros((height,width,3), dtype = np.uint8)
#矩形検出範囲の設定
mask = cv2.rectangle(mask, (mask1_x,mask1_y), (mask2_x,mask2_y), (255, 255, 255), -1)
#検出範囲をマスク
mask_img = cv2.bitwise_and(imgcopy, mask)







# 画像を読み込む
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
image = mask_img
# image = cv2.imread(mask_img, cv2.IMREAD_GRAYSCALE)

# AKAZE特徴量を抽出する
akaze = cv2.AKAZE_create()
kp1, des1 = akaze.detectAndCompute(template, None)
kp2, des2 = akaze.detectAndCompute(image, None)

# Brute-Force Matcherを使用して特徴量をマッチングする
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
print(matches)
matches = sorted(matches, key=lambda x: x.distance)

# マッチング結果を表示する
result = cv2.drawMatches(template, kp1, image, kp2[:], matches, None, flags=2)
# result = cv2.drawMatches(template, kp1, image, kp2, matches[:30], None, flags=2)
# result = cv2.drawMatches(template, kp1, image, kp2, matches[:100], None, flags=2)

# 結果を表示する
cv2.imshow("Template Matching Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

