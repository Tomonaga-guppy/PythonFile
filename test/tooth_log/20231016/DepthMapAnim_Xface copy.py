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
import mpl_toolkits.mplot3d.art3d as art3d

root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_09_000"

#depth_scale = mm/depth_data
depth_scale = 1.0000000474974513

def Coordinate(root_dir, depth_scale):
    pattern = os.path.join(root_dir, '*A2*/RGB_image')
    RGB_dirs = glob.glob(pattern, recursive=True)

    for i,RGB_dir in enumerate(RGB_dirs):
        dir_path = os.path.dirname(RGB_dir) + '/'
        print(f"dirpath = {dir_path}")
        npy_path = dir_path + 'XYZ_localdata.npy'
        # ファイルから配列を読み込む dataは[axis][frame][id]の順で並んでいる
        data = np.load(npy_path,allow_pickle=True)
        data[2,:,:] += 1
        print(data.shape)
        accel_path = os.path.join(dir_path,"accel_data.npy")
        accel = np.load(accel_path, allow_pickle=True)  #[frame][x,y,z]
        csv_file = dir_path + 'RGB_image.csv'
        if os.path.isfile(csv_file) == False:
            csv_file = dir_path + 'OpenFace.csv'

        OpenFace_result = []
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                OpenFace_result.append(row)
        # print(OpenFace_result[0])

        #OpenFace実行しに作成されるファイル名を"RGB_image" → "OpenFace"に変更
        before_files = glob.glob(dir_path + "*RGB_image*.*")
        for before_file in before_files:
            after_file = before_file.replace("RGB_image", "OpenFace")
            if os.path.exists(after_file):
                os.remove(after_file)
            os.rename(before_file,after_file)

        img_len = len(glob.glob(dir_path + "RGB_image/*.png"))
        depth_len = len(glob.glob(dir_path + "Depth_image/*.png"))
        end_frame = min(img_len,depth_len)

        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Z-axis')
        ax.set_zlabel('Y-axis')
        ax.set_title('3Dplot')
        ax.set_zlim(-100,100)
        ax.set_xlim(-100,100)
        ax.set_ylim(-100,100)
        ax.view_init(elev=0, azim=0)

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_xlabel('X-axis')
        ax2.set_ylabel('Z-axis')
        ax2.set_zlabel('Y-axis')
        ax2.set_title('3Dplot')
        a = 100
        ax2.set_zlim(-a,a)
        ax2.set_xlim(-a,a)
        ax2.set_ylim(-a,a)
        ax2.view_init(elev=0, azim=90)

        line1= art3d.Line3D([-a,a],[0,0],[0,0], color='black')
        line2= art3d.Line3D([0,0],[-a,a],[0,0], color='black')
        line3= art3d.Line3D([0,0],[0,0],[-a,a], color='black')
        ax.add_line(line1)
        ax.add_line(line2)
        ax.add_line(line3)
        line2_1= art3d.Line3D([-a,a],[0,0],[0,0], color='black')
        line2_2= art3d.Line3D([0,0],[-a,a],[0,0], color='black')
        line2_3= art3d.Line3D([0,0],[0,0],[-a,a], color='black')
        ax2.add_line(line2_1)
        ax2.add_line(line2_2)
        ax2.add_line(line2_3)

        global theta_nose_sum,theta_nose
        global scatter, scatter_point,scatter2, scatter_point2
        # scatter = ax.scatter([], [], [], s=2, color=[])
        # scatter_point = ax.scatter([], [], [], s=10, color="r")
        # scatter2 = ax2.scatter([], [], [], s=2, color=[])
        # scatter_point2 = ax2.scatter([], [], [], s=10, color="r")
        scatter = []
        scatter2 = []

        caliblation_time = 5
        fps = 30
        theta_nose = 0
        theta_camera = 0
        theta_nose_sum = 0

        def update(frame_count):
            X =[]
            Y =[]
            Z =[]
            global theta_nose_sum,theta_nose
            global scatter, scatter_point,scatter2, scatter_point2, scatter_seal, scatter_seal2

            img_path = dir_path + "RGB_image/{:04d}.png".format(frame_count)
            img_depth_path = dir_path + "Depth_image/{:04d}.png".format(frame_count)
            img = cv2.imread(img_path)
            img_depth = cv2.imread(img_depth_path, cv2.IMREAD_ANYDEPTH)

            xpixel_scale = (float(OpenFace_result[frame_count][186])-float(OpenFace_result[frame_count][177]))/(float(OpenFace_result[frame_count][50])-float(OpenFace_result[frame_count][41]))
            ypixel_scale = (float(OpenFace_result[frame_count][242])-float(OpenFace_result[frame_count][236]))/(float(OpenFace_result[frame_count][106])-float(OpenFace_result[frame_count][100]))
            landmark_List=[]
            coordinate_list = []
            color_list = []

            for i in range(68):
                # print(frame_count, i, i+141, i+209, i+277)
                x,y = float(OpenFace_result[frame_count][i+141]),float(OpenFace_result[frame_count][i+209])
                depthi = img_depth[int(float(OpenFace_result[frame_count][i+73])),int(float(OpenFace_result[frame_count][i+5]))] #整数型
                z = depthi * depth_scale

                xp,yp = float(OpenFace_result[frame_count][i+5]),float(OpenFace_result[frame_count][i+73])
                landmark_List.append([i,x,y,z,xp,yp])
            landmark_list_np = np.array(landmark_List)
            # print(f"landshape = {np.shape(landmark_list_np)}")

            for xpix in range(int(min(landmark_list_np[:,4])), int(max(landmark_list_np[:,4])),2):
                for ypix in range (int(min(landmark_list_np[:,5])), int(max(landmark_list_np[:,5])),2):
            # for xpix in range(0,1279,5):
            #     for ypix in range (0,719,5):
                    # X = Xo + (x - xo)*scale pixelとmmの変換 ちょい不安
                    x = float(0 + (xpix - 640)*xpixel_scale)
                    y = float(0 + (ypix - 360)*ypixel_scale)
                    z = img_depth[ypix,xpix] * depth_scale

                    color = (img[ypix,xpix][::-1]/255.0).tolist()  #BGRからRGBにして正規化
                    color_list.append(color)
                    coordinate_list.append([frame_count,x,y,z])

            coordinate_list_np = np.array(coordinate_list)
            print(frame_count)
            # 前のフレームの点をクリア
            try :
                scatter.remove()
                scatter2.remove()
                scatter_point.remove()
                scatter_point2.remove()
                scatter_seal.remove()
                scatter_seal2.remove()
            except Exception as e:
                # print(e)
                pass

            theta_camera_x =np.arctan(accel[frame_count][2]/abs(accel[frame_count][1]))

            if frame_count <= caliblation_time*fps:
                #e_XL (36 → 45のベクトル)
                bector_x = np.array([landmark_list_np[45][1]-landmark_list_np[36][1], landmark_list_np[45][2]-landmark_list_np[36][2], landmark_list_np[45][3]-landmark_list_np[36][3]])
                bector30_36 = np.array([landmark_list_np[36][1]-landmark_list_np[30][1],landmark_list_np[36][2]-landmark_list_np[30][2],landmark_list_np[36][3]-landmark_list_np[30][3]])
                c = - (np.dot(bector_x,bector30_36))/(np.linalg.norm(bector_x)**2)

                #e_yLnose
                #bector_Xと30からbectorXに下した垂線の交点をPとした、30とPを結ぶベクトルがbector_y_nose
                bector_y_nose = bector30_36 + c*bector_x
                bector_y_nose2= np.array([bector_y_nose[0],bector_y_nose[1], 0])
                #print(bector_y_nose, bector_y_nose2)
                theta_nose_sum += float(np.arccos(np.dot(bector_y_nose,bector_y_nose2)/(np.linalg.norm(bector_y_nose)*np.linalg.norm(bector_y_nose2))))
                theta_nose = (theta_nose_sum/(caliblation_time*fps)) - theta_camera
            else:
                # print(f"OpenFace_result[frame_count][186] = {OpenFace_result[frame_count][186]}")
                #e_XL (36 → 45のベクトル)
                bector_x = np.array([landmark_list_np[45][1]-landmark_list_np[36][1], landmark_list_np[45][2]-landmark_list_np[36][2], landmark_list_np[45][3]-landmark_list_np[36][3]])
                length_bector_x = ((bector_x[0])**2 + (bector_x[1])**2 + (bector_x[2])**2)**(1/2)
                base_bector_x = [bector_x[0]/length_bector_x, bector_x[1]/length_bector_x, bector_x[2]/length_bector_x]

                bector30_36 = np.array([landmark_list_np[36][1]-landmark_list_np[30][1],landmark_list_np[36][2]-landmark_list_np[30][2],landmark_list_np[36][3]-landmark_list_np[30][3]])
                c = -(np.dot(bector_x,bector30_36))/(np.linalg.norm(bector_x)**2)

                #e_yLnose
                #bector_Xと30からbectorXに下した垂線の交点をPとした、30とPを結ぶベクトルがbector_y_nose
                bector_y_nose = bector30_36 + c*bector_x
                base_bector_y_nose = bector_y_nose/np.linalg.norm(bector_y_nose)
                bector_Pposition = [bector_y_nose[0]+landmark_list_np[30][1],bector_y_nose[1]+landmark_list_np[30][2],bector_y_nose[2]+landmark_list_np[30][3]]

                #e_zLnose
                base_bector_z_nose = np.cross(base_bector_x,base_bector_y_nose)

                theta_x = theta_nose - theta_camera_x
                theta_y = 0
                theta_z = 0

                R_Cam_Nose = np.array([base_bector_x,base_bector_y_nose,base_bector_z_nose]).T
                R_Nose_Local = np.array([[1,0,0],[0,np.cos(theta_x), np.sin(theta_x)],[0, -np.sin(theta_x),np.cos(theta_x)]]).T  #x軸回転
                R_Local_Local2 = np.array([[np.cos(theta_y),0, -np.sin(theta_y)],[0, 1, 0],[np.sin(theta_y), 0, np.cos(theta_y),]]).T  #y軸回転
                R_Local2_Local3 = np.array([[np.cos(theta_z),np.sin(theta_z),0],[-np.sin(theta_z), np.cos(theta_z), 0],[0, 0, 1]]).T  #z軸回転

                # t = [landmark_list_np[33][1], landmark_list_np[33][2], landmark_list_np[33][3]]
                t = [bector_Pposition[0],bector_Pposition[1],bector_Pposition[2]]

                #Aは同次変換行列    (A_Cam_Noseはtheta_nose補正前までの回転と並進、原点は33番    A_Nose_LocalはXnose軸について鼻の角度分回転のみ)
                A_Cam_Nose = np.array([[R_Cam_Nose[0][0],R_Cam_Nose[0][1],R_Cam_Nose[0][2],t[0]],
                                    [R_Cam_Nose[1][0],R_Cam_Nose[1][1],R_Cam_Nose[1][2],t[1]],
                                    [R_Cam_Nose[2][0],R_Cam_Nose[2][1],R_Cam_Nose[2][2],t[2]],
                                    [0,0,0,1]])

                #x軸回転
                A_Nose_Local = np.array([[R_Nose_Local[0][0],R_Nose_Local[0][1],R_Nose_Local[0][2],0],
                                    [R_Nose_Local[1][0],R_Nose_Local[1][1],R_Nose_Local[1][2],0],
                                    [R_Nose_Local[2][0],R_Nose_Local[2][1],R_Nose_Local[2][2],0],
                                    [0,0,0,1]])

                #y軸回転
                A_Local_Local2 = np.array([[R_Local_Local2[0][0],R_Local_Local2[0][1],R_Local_Local2[0][2],0],
                                    [R_Local_Local2[1][0],R_Local_Local2[1][1],R_Local_Local2[1][2],0],
                                    [R_Local_Local2[2][0],R_Local_Local2[2][1],R_Local_Local2[2][2],0],
                                    [0,0,0,1]])

                #z軸回転
                A_Local2_Local3 = np.array([[R_Local2_Local3[0][0],R_Local2_Local3[0][1],R_Local2_Local3[0][2],0],
                                    [R_Local2_Local3[1][0],R_Local2_Local3[1][1],R_Local2_Local3[1][2],0],
                                    [R_Local2_Local3[2][0],R_Local2_Local3[2][1],R_Local2_Local3[2][2],0],
                                    [0,0,0,1]])

                demo = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

                # A_Cam_Local = A_Cam_Nose @ A_Nose_Local @ A_Local_Local2 @ A_Local2_Local3  #行列掛け算
                A_Cam_Local = A_Cam_Nose @ A_Nose_Local @ demo @ demo  #行列掛け算
                # A_Cam_Local = A_Cam_Nose @ A_Nose_Local @ A_Local_Local2 @ demo  #行列掛け算
                # A_Cam_Local = A_Cam_Nose @ A_Nose_Local @ demo @ A_Local2_Local3  #行列掛け算

                # X_seal = np.array([landmark_list_np[68][1], landmark_list_np[68][2], landmark_list_np[68][3],1])
                # XL_seal = np.dot(np.linalg.inv(A_G_L) ,X_seal)
                # XL_x_seal.append(XL_seal[0])
                # XL_y_seal.append(XL_seal[1])
                # XL_z_seal.append(XL_seal[2])

                for pixel in range(np.shape(coordinate_list_np)[0]):
                    Xid = np.array([coordinate_list_np[pixel][1], coordinate_list_np[pixel][2], coordinate_list_np[pixel][3],1])
                    XX = np.dot(np.linalg.inv(A_Cam_Local),Xid)
                    X.append(XX[0])
                    Y.append(XX[1])
                    Z.append(XX[2])

                # print('theta_nose [rad] = ', theta_nose)
                # print(f"Xshape = {np.shape(X)}")
                # print(f"coordinate[1] = {coordinate_list_np[1]}")

                scatter = ax.scatter(X, Z, Y, s=2, color=color_list)
                scatter2 = ax2.scatter(X, Z, Y, s=2, color=color_list)
                scatter_point = ax.scatter(data[0,frame_count - caliblation_time*fps -1,0:68], data[2,frame_count - caliblation_time*fps -1,0:68], data[1,frame_count - caliblation_time*fps -1,0:68], s=4, color = "b")
                scatter_point2 = ax2.scatter(data[0,frame_count - caliblation_time*fps -1,0:68], data[2,frame_count - caliblation_time*fps -1,0:68], data[1,frame_count - caliblation_time*fps -1,0:68], s=4, color = "b")
                scatter_seal = ax.scatter(data[0,frame_count - caliblation_time*fps -1,68], data[2,frame_count - caliblation_time*fps -1,68], data[1,frame_count - caliblation_time*fps -1,68], s=10, color = "m")
                scatter_seal2 = ax2.scatter(data[0,frame_count - caliblation_time*fps -1,68], data[2,frame_count - caliblation_time*fps -1,68], data[1,frame_count - caliblation_time*fps -1,68], s=10, color = "m")


        save_path = dir_path + '3dAnim.mp4'
        # アニメーションの作成
        anim = FuncAnimation(fig, update, frames=range(1,end_frame+1), interval=1000/fps, blit=False)
        # アニメーションを保存する
        anim.save(save_path, fps=30)

        # if os.path.isfile(save_path) == False:
        #     # アニメーションの作成
        #     anim = FuncAnimation(fig, update, frames=range(1,end_frame+1), interval=1000/fps, blit=False)
        #     # アニメーションを保存する
        #     anim.save(save_path, fps=30)

        # #初めの待機フレーム分を削る
        # cap = cv2.VideoCapture(save_path)
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # writer = cv2.VideoWriter_fourcc(*'mp4v')  # 出力動画のコーデックを指定
        # output_fps = cap.get(cv2.CAP_PROP_FPS)    # 入力動画のFPSを取得
        # output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 入力動画の幅を取得
        # output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 入力動画の高さを取得
        # out = cv2.VideoWriter(save_path, writer, output_fps, (output_width, output_height))
        # frame_count = 1

        # while frame_count <= total_frames:
        #     print(frame_count)
        #     if frame_count > 150:
        #         ret, frame = cap.read()
        #         if not ret:
        #             break
        #         out.write(frame)
        #         frame_count += 1
        #     else:
        #         frame_count += 1
        #         continue

        # cap.release()
        # out.release()
        # cv2.destroyAllWindows()


Coordinate(root_dir, depth_scale)

