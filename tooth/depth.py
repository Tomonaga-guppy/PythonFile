import cv2
import os
import numpy as np


root_dir = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\2023_11_17\20231117_b1"

# pic_close = os.path.join(root_dir,"RGB_image/0001.png")
# pic_open = os.path.join(root_dir,"RGB_image/0519.png")

# def onMouse(event, x, y, flags, params):
#     count = 0
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(f"pixel = {x}, {y}")
#         print(f"count = {count}")
#         count += 1
#     if count == 2:
#         cv2.destroyAllWindows()


# img_close = cv2.imread(pic_close)
# img_open = cv2.imread(pic_open)

# # cv2.imshow('close', img_close)
# # cv2.setMouseCallback('close', onMouse)
# # cv2.waitKey(0)

# cv2.imshow('open', img_open)
# cv2.setMouseCallback('open', onMouse)
# cv2.waitKey(0)

img_depth_path = os.path.join(root_dir,"Depth_image/0001.png")
# img_depth_path = os.path.join(root_dir,"0519.png")
img_depth = cv2.imread(img_depth_path, cv2.IMREAD_ANYDEPTH)
#depth_scale = mm/depth_data
depth_scale = 1.0000000474974513

pixel_list = np.array([[618, 291], [669, 333], [685, 363], [686, 400]])

for i in range(len(pixel_list)):
    seal_x_pixel = pixel_list[i][0]
    seal_y_pixel = pixel_list[i][1]
    # シールの奥行 depthから計算
    depth68 = img_depth[seal_y_pixel,seal_x_pixel]
    seal_z = depth68 * depth_scale
    print(f"pixel = {seal_x_pixel}, {seal_y_pixel},{seal_z}")