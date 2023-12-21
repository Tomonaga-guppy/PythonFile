import cv2

# img1 = cv2.imread('./manta1.png')
# img2 = cv2.imread('./manta2.png')
img1 = cv2.imread('./utsu1.png')
img2 = cv2.imread('./utsu2.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
akaze = cv2.AKAZE_create()
kp1, des1 = akaze.detectAndCompute(gray1, None)
kp2, des2 = akaze.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
print(matches)
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)


# 結果を表示する
cv2.imshow("Template Matching Result", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()