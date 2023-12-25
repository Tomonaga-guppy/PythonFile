import cv2

dir_path = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\2023_09_000\20230606_J2"
# テンプレートマッチングを実行する
template_path = r"C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/seal_template/seal_1.png"
image_path = dir_path + "/RGB_image/0001.png"

img1 = cv2.imread(template_path)
img2 = cv2.imread(image_path)
# img1 = cv2.imread('./utsu1.png')
# img2 = cv2.imread('./utsu2.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
akaze = cv2.AKAZE_create()
kp1, des1 = akaze.detectAndCompute(gray1, None)
kp2, des2 = akaze.detectAndCompute(gray2, None)

akaze_img = cv2.drawKeypoints(img1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("img", akaze_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# print(matches)
# matches = sorted(matches, key = lambda x:x.distance)
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)


# # 結果を表示する
# cv2.imshow("Template Matching Result", img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()