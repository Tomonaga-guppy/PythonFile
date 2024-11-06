# 参考：https://openi.jp/blogs/main/software-opencv-aruco-charuco

import cv2
from PIL import Image
from fpdf import FPDF

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

squares_x = 5
squares_y = 6
square_length = 0.04
marker_length = 0.02
dpi = 600

charucoBoard = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)

# A4サイズの寸法
mm_to_inch = 25.4
a4_x = 210
a4_y = 297
a4_x_inch = a4_x / mm_to_inch
a4_y_inch = a4_y / mm_to_inch

image_x = round(a4_x_inch * dpi)
image_y = round(a4_y_inch * dpi)

image = charucoBoard.generateImage((image_x, image_y))

image = Image.fromarray(image)
image.save("charuco_board.png", dpi=(dpi, dpi))

pdf = FPDF(unit="mm", format="A4")
pdf.add_page()
pdf.image("charuco_board.png", x=0, y=0, w=a4_x, h=a4_y)
pdf.output("charuco_board.pdf")