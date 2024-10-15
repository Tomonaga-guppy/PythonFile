import cv2
import subprocess

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

squares_x = 5
squares_y = 7
square_length = 0.04
marker_length = 0.02
charucoBoard = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)

image_x = round(5 * 40 / 25.4 * 350)
image_y = round(7 * 40 / 25.4 * 350)
image = charucoBoard.generateImage((image_x, image_y))
cv2.imwrite("charuco_board.png", image)
# dpiを350dpiに変更
subprocess.run(['convert', '-density', '350', '-units', 'pixelsperinch', 'charuco_board.png', 'charuco_board.png'])