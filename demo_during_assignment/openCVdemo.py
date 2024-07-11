import cv2
#メディアンフィルタで平滑化（空間フィルタリング）
img = cv2.imread("C:\\imagestest\\koron.jpg",cv2.IMREAD_GRAYSCALE)
dst=cv2.medianBlur(img,101)
cv2.namedWindow("koron",cv2.WINDOW_NORMAL)
cv2.imshow("koron",dst)

#白黒をカラーにして表現（トーンカーブ）
img2 = cv2.applyColorMap(img,cv2.COLORMAP_JET)
cv2.namedWindow("colorkoron",cv2.WINDOW_NORMAL)
cv2.imshow("colorkoron",img2)

#sobelフィルタ(エッジ検出)
img3 = cv2.Sobel(img,-10,0,1)
cv2.namedWindow("sobelkoron",cv2.WINDOW_NORMAL)
cv2.imshow("sobelkoron",img3)

cv2.waitKey(0)