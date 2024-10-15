import cv2
import subprocess

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

squares_x = 3
squares_y = 5
square_length = 0.04
marker_length = 0.02
charucoBoard = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)

# A4サイズに合わせたピクセル数 (350dpi)
image_x = round(210 / 25.4 * 350)  # 幅 (210mm)
image_y = round(297 / 25.4 * 350)  # 高さ (297mm)
image = charucoBoard.generateImage((image_x, image_y))
cv2.imwrite("charuco_board2.png", image)
# dpiを350dpiに変更
subprocess.run(['convert', '-density', '350', '-units', 'pixelsperinch', 'charuco_board.png', 'charuco_board.png'])