import os
import glob
import math
import subprocess
import csv
import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_09_000"

#depth_scale = mm/depth_data
depth_scale = 1.0000000474974513

def Coordinate(root_dir, depth_scale):
    pattern = os.path.join(root_dir, '*J2*/RGB_image')
    RGB_dirs = glob.glob(pattern, recursive=True)

    for i,RGB_dir in enumerate(RGB_dirs):
        dir_path = os.path.dirname(RGB_dir) + '/'

        # OpenFace_result 0 frame, 5-72 x_pixel, 73-140 y_pixel
        csv_file = dir_path + 'RGB_image.csv'
        if os.path.isfile(csv_file) == False:
            csv_file = dir_path + 'OpenFace.csv'

        OpenFace_result = []
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                OpenFace_result.append(row)

        #OpenFace実行しに作成されるファイル名を"RGB_image" → "OpenFace"に変更
        before_files = glob.glob(dir_path + "*RGB_image*.*")
        for before_file in before_files:
            after_file = before_file.replace("RGB_image", "OpenFace")
            if os.path.exists(after_file):
                os.remove(after_file)
            os.rename(before_file,after_file)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Xaxis')
        ax.set_ylabel('Zaxis')
        ax.set_zlabel('Yaxis')
        ax.set_title('3Dplot')
        ax.invert_zaxis()
        ax.view_init(elev=0, azim=0)

        frame_count = 1
        img_path = dir_path + "RGB_image/{:04d}.png".format(frame_count)
        img_depth_path = dir_path + "Depth_image/{:04d}.png".format(frame_count)

        img = cv2.imread(img_path)
        img_depth = cv2.imread(img_depth_path, cv2.IMREAD_ANYDEPTH)

        xpixel_scale = (float(OpenFace_result[frame_count][186])-float(OpenFace_result[frame_count][177]))/(float(OpenFace_result[frame_count][50])-float(OpenFace_result[frame_count][41]))
        ypixel_scale = (float(OpenFace_result[frame_count][242])-float(OpenFace_result[frame_count][236]))/(float(OpenFace_result[frame_count][106])-float(OpenFace_result[frame_count][100]))

        landmark_List=[]
        for i in range(68):
            # print(frame_count, i, i+141, i+209, i+277)
            x,y = float(OpenFace_result[frame_count][i+141]),float(OpenFace_result[frame_count][i+209])
            depthi = img_depth[int(float(OpenFace_result[frame_count][i+73])),int(float(OpenFace_result[frame_count][i+5]))] #整数型
            z = depthi * depth_scale

            xp,yp = float(OpenFace_result[frame_count][i+5]),float(OpenFace_result[frame_count][i+73])
            landmark_List.append([i,x,y,z,xp,yp])
        landmark_list_np = np.array(landmark_List)

        coordinate_list = []
        color_list = []

        for xpix in range(int(min(landmark_list_np[:,4])), int(max(landmark_list_np[:,4])),2):
            for ypix in range (int(min(landmark_list_np[:,5])), int(max(landmark_list_np[:,5])),2):
        # for xpix in range(0,1279,5):
        #     for ypix in range (0,719,5):
                # X = Xo + (x - xo)*scale
                x = float(0 + (xpix - 640)*xpixel_scale)
                y = float(0 + (ypix - 360)*ypixel_scale)
                z = img_depth[ypix,xpix] * depth_scale

                color = (img[ypix,xpix][::-1]/255.0).tolist()  #BGRからRGBにして正規化
                color_list.append(color)
                coordinate_list.append([frame_count,x,y,z])

        coordinate_list_np = np.array(coordinate_list)
        ax.scatter(coordinate_list_np[:,1], coordinate_list_np[:,3], coordinate_list_np[:,2], s=2, color=color_list)
        ax.scatter(landmark_list_np[:,1], landmark_list_np[:,3], landmark_list_np[:,2], s=10, color = "r")

        plt.show()


Coordinate(root_dir, depth_scale)

