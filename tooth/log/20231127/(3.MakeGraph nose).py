import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from scipy.signal import savgol_filter
from matplotlib.cm import ScalarMappable
import sys

root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_11_17"
# root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_09_000"

# if len(sys.argv) > 1:
#     root_dir = sys.argv[1]
# else:
#     print("ディレクトリパスが指定されていません。")
#     sys.exit()

caliblation_time0 = 5
caliblation_time = 5

#カメラ座標系 RealSensed435i https://www.intelrealsense.com/how-to-getting-imu-data-from-d435i-and-t265/  https://watako-lab.com/2019/02/15/3axis_acc/
# norm = 9.245
# a_x = 0.226
# a_y = -8.247
# a_z = 4.168
global theta_co_x, theta_y, theta_z
theta_y = np.deg2rad(0)
theta_z = np.deg2rad(0)
theta_co_x = np.deg2rad(-35)

# print(f"theta_camx = {theta_camera}")
# print(f"theta_camx[deg] = {np.rad2deg(theta_camera)}")

def MakeGraph(root_dir, fps):
    pattern = os.path.join(root_dir, '*a1*/result.npy')
    npy_files = glob.glob(pattern, recursive=True)
    num_npy_files = len(npy_files)

    for i,npy_file in enumerate(npy_files):
        print(f"{i + 1}/{num_npy_files} {npy_file}")
        dir_path = os.path.dirname(npy_file) + '/'
        aa=np.load(npy_file, allow_pickle=True)  #作製したnumpy配列は[frame][number][number, x, y, z]
        accel_path = os.path.join(dir_path,"accel_data.npy")
        accel = np.load(accel_path, allow_pickle=True)  #[frame][x,y,z]
        theta_camera = np.arccos(np.mean(abs(accel[:,1]))/(np.sqrt(np.mean(accel[:,1])**2+np.mean(accel[:,2])**2)))
        # print(f'aa.shape = {aa.shape}')
        XL_x_seal = []
        XL_y_seal = []
        XL_z_seal = []
        frame_count = []
        count = 0

        theta_nose_sum = 0
        X = []
        Y = []
        Z = []

        for frame_number in range(aa.shape[0]):
            #aa[0]はframe数
            theta_camera =np.arccos(abs(accel[frame_number][1])/(np.sqrt(accel[frame_number][1]**2+accel[frame_number][2]**2)))
            #鼻先(30)、左右目(36,45)or左右鼻翼(31,35)の位置ベクトル
            bector_30 = np.array([aa[frame_number][30][1], aa[frame_number][30][2], aa[frame_number][30][3]])
            bector_36 = np.array([aa[frame_number][36][1], aa[frame_number][36][2], aa[frame_number][36][3]])
            bector_45 = np.array([aa[frame_number][45][1], aa[frame_number][45][2], aa[frame_number][45][3]])
            bector_31 = np.array([aa[frame_number][31][1], aa[frame_number][31][2], aa[frame_number][31][3]])
            bector_35 = np.array([aa[frame_number][35][1], aa[frame_number][35][2], aa[frame_number][35][3]])

            # #e_x (36 → 45のベクトル)
            # bector_x = bector_45 - bector_36
            # base_bector_x = bector_x / np.linalg.norm(bector_x)
            # bector_30_36 = bector_36 - bector_30
            # c = - (np.dot(bector_x,bector_30_36))/(np.linalg.norm(bector_x)**2)

            # #e_y
            # #bector_Xと30からbectorXに下した垂線の交点をPとした、30とPを結ぶベクトルがbector_y_nose
            # bector_y_nose = bector_30_36 + c*bector_x
            # bector_Pposition = [bector_y_nose[0]+aa[frame_number][30][1],bector_y_nose[1]+aa[frame_number][30][2],bector_y_nose[2]+aa[frame_number][30][3]]
            # bector_y_nose2= np.array([bector_y_nose[0],bector_y_nose[1], 0])

            # e_x (31 → 35のベクトル)
            bector_x = bector_35 - bector_31
            base_bector_x = bector_x / np.linalg.norm(bector_x)
            bector_31_30 = bector_30 - bector_31
            c = - (np.dot(bector_x,bector_31_30))/(np.linalg.norm(bector_x)**2)

            # e_y
            # bector_Xと30からbectorXに下した垂線の交点をPとした、30とPを結ぶベクトルがbector_y_nose
            bector_y_nose = bector_31_30 + c*bector_x
            bector_Pposition = bector_30 - bector_y_nose
            bector_y_nose2= np.array([bector_y_nose[0],bector_y_nose[1], 0])

            if frame_number < caliblation_time0*fps:
                theta_nose_sum += float(np.arccos(np.dot(bector_y_nose,bector_y_nose2)/(np.linalg.norm(bector_y_nose)*np.linalg.norm(bector_y_nose2))))
                theta_nose = (theta_nose_sum/(caliblation_time0*fps))

            elif frame_number >= caliblation_time*fps:
                # for frame_number in range(caliblation_time*fps,aa.shape[0]):
                #e_y
                base_bector_y = bector_y_nose/np.linalg.norm(bector_y_nose)
                #e_z
                base_bector_z = np.cross(base_bector_x,base_bector_y)

                # 点を反時計回りにtheta回転 = 軸を時計回りにtheta回転  ==   点は反時計回りが正、軸は時計回りが正
                # https://qiita.com/suzuki-navi/items/60ef241b2dca499df794
                theta_x =  -( - theta_nose - theta_camera) + theta_co_x
                # theta_x =  -(theta_nose - theta_camera) + theta_co_x
                # print(f"thetax = {np.rad2deg(theta_x)}")

                #ベクトル変換https://eman-physics.net/math/linear08.html  グローバル座標とローカル座標https://programming-surgeon.com/script/coordinate-system/
                R_Cam_Nose = np.array([base_bector_x,base_bector_y,base_bector_z]).T
                R_Nose_Local = np.array([[1,0,0],[0,np.cos(theta_x), np.sin(theta_x)],[0, -np.sin(theta_x),np.cos(theta_x)]]).T  #x軸回転
                R_Local_Local2 = np.array([[np.cos(theta_y),0, -np.sin(theta_y)],[0, 1, 0],[np.sin(theta_y), 0, np.cos(theta_y),]]).T  #y軸回転
                R_Local2_Local3 = np.array([[np.cos(theta_z),np.sin(theta_z),0],[-np.sin(theta_z), np.cos(theta_z), 0],[0, 0, 1]]).T  #z軸回転
                # t = [aa[frame_number][33][1], aa[frame_number][33][2], aa[frame_number][33][3]]
                t = [bector_Pposition[0],bector_Pposition[1],bector_Pposition[2]]
                # t = [0, 0, 0]

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
                A_y = np.array([[R_Local_Local2[0][0],R_Local_Local2[0][1],R_Local_Local2[0][2],0],
                                    [R_Local_Local2[1][0],R_Local_Local2[1][1],R_Local_Local2[1][2],0],
                                    [R_Local_Local2[2][0],R_Local_Local2[2][1],R_Local_Local2[2][2],0],
                                    [0,0,0,1]])

                #z軸回転
                A_z = np.array([[R_Local2_Local3[0][0],R_Local2_Local3[0][1],R_Local2_Local3[0][2],0],
                                    [R_Local2_Local3[1][0],R_Local2_Local3[1][1],R_Local2_Local3[1][2],0],
                                    [R_Local2_Local3[2][0],R_Local2_Local3[2][1],R_Local2_Local3[2][2],0],
                                    [0,0,0,1]])

                demo = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

                # A_Cam_Local = A_Cam_Nose @ A_Nose_Local @ A_y @ A_z  #行列掛け算
                A_Cam_Local = A_Cam_Nose @ A_Nose_Local @ demo @ demo  #行列掛け算
                # A_Cam_Local = A_Cam_Nose @ A_Nose_Local @ A_y @ demo  #行列掛け算
                # A_Cam_Local = A_Cam_Nose @ A_Nose_Local @ demo @ A_z  #行列掛け算

                X_seal = np.array([aa[frame_number][68][1], aa[frame_number][68][2], aa[frame_number][68][3],1])

                # XL_seal = np.dot(np.linalg.inv(A_Cam_Local) ,X_seal)
                # if frame_number == 150:
                #     XL_seal_0 = np.dot(np.linalg.inv(A_Cam_Local) ,X_seal)
                #     t2 = [XL_seal_0[0], XL_seal_0[1], XL_seal_0[2]]
                # #原点移動
                # A_Local3_Local4 = np.array([[1, 0, 0, t2[0]],
                #                             [0, 1, 0, t2[1]],
                #                             [0, 0, 1, t2[2]],
                #                             [0, 0, 0, 1]])
                # A_Cam_Local = A_Cam_Local @ A_Local3_Local4  #行列掛け算


                XL_seal = np.dot(np.linalg.inv(A_Cam_Nose) ,X_seal)
                XL_seal = A_Nose_Local @ XL_seal
                # XL_seal = np.dot(np.linalg.inv(A_Cam_Local) ,X_seal)
                # XL_seal = np.dot(A_Nose_Local @ np.linalg.inv(A_Cam_Nose) ,X_seal)
                # XL_seal = np.dot(A_z @ A_y @ A_Nose_Local @ np.linalg.inv(A_Cam_Nose) ,X_seal)
                XL_x_seal.append(XL_seal[0])
                XL_y_seal.append(XL_seal[1])
                XL_z_seal.append(XL_seal[2])
                frame_count.append(count)

                count = count + 1

                for id in range(aa.shape[1]):
                    Xid = np.array([aa[frame_number][id][1], aa[frame_number][id][2], aa[frame_number][id][3],1])
                    # XX = np.dot(np.linalg.inv(A_Cam_Local),Xid)
                    # XX = np.dot(np.linalg.inv(A_Cam_Nose),Xid)
                    XX = np.linalg.inv(A_Cam_Nose) @ Xid
                    XX = A_Nose_Local @ XX
                    # XX = np.dot(A_z @ A_y @ A_Nose_Local @ np.linalg.inv(A_Cam_Nose),Xid)
                    X.append(XX[0])
                    Y.append(XX[1])
                    Z.append(XX[2])

        print('theta_nose [deg] = ', np.rad2deg(theta_nose))
        print('theta_cam [deg] = ', np.rad2deg(theta_camera))
        print('theta_co [deg] = ', np.rad2deg(theta_co_x))
        print('theta_x [deg] = ', np.rad2deg(theta_x))

        #ローカル座標のxyzをnpyファイルで保存
        X = np.array(X).reshape(((aa.shape[0]-caliblation_time*fps),aa.shape[1]))
        Y = np.array(Y).reshape(((aa.shape[0]-caliblation_time*fps),aa.shape[1]))
        Z = np.array(Z).reshape(((aa.shape[0]-caliblation_time*fps),aa.shape[1]))
        XYZdata = np.stack([X,Y,Z])
        path = dir_path + "XYZ_localdata.npy"
        np.save(path,XYZdata)
        print('XYZ_localdata is saved')

        # 散布図,線を描画
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1,2,2)  #1行2列つくって右に配置

        ax1.set_xlabel('X [mm]',fontsize=15)
        ax1.set_ylabel('Y [mm]',fontsize=15)

        ax1.minorticks_on()
        ax1.grid(which = "major", axis = "x", color = "gray", alpha = 0.5, linestyle = "-", linewidth = 1)
        ax1.grid(which = "major", axis = "y", color = "gray", alpha = 0.5, linestyle = "-", linewidth = 1)
        ax1.grid(which = "minor", axis = "x", color = "gray", alpha = 0.1, linestyle = "-", linewidth = 1)
        ax1.grid(which = "minor", axis = "y", color = "gray", alpha = 0.1, linestyle = "-", linewidth = 1)

        ax1.set_aspect('equal', adjustable='box')

        #https://datachemeng.com/wp-content/uploads/preprocessspectratimeseriesdata.pdf
        window_length = 11
        polyorder = 2
        XL_x_seal_SG = savgol_filter(XL_x_seal, window_length=window_length, polyorder=polyorder)
        XL_y_seal_SG = savgol_filter(XL_y_seal, window_length=window_length, polyorder=polyorder)
        XL_z_seal_SG = savgol_filter(XL_z_seal, window_length=window_length, polyorder=polyorder)

        # Create a color map based on time values
        cmap = plt.get_cmap('jet')
        normalize = plt.Normalize(min(frame_count), max(frame_count))
        colors = cmap(normalize(frame_count))

        # # ax1.scatter(XL_x_seal, XL_y_seal, c=colors, s=20)
        # ax1.scatter(XL_x_seal_SG, XL_y_seal_SG, c=colors, s=20)

        # ax1.scatter(XL_x_seal, XL_y_seal, c="k", s=20, alpha=0.4)
        ax1.scatter(XL_x_seal_SG, XL_y_seal_SG, c=colors, s=20)

        # Create a colorbar
        cbar = fig.colorbar(ScalarMappable(norm=normalize, cmap=cmap), ax=ax1)
        cbar.set_label('frame', fontsize=10)

        # 散布図を描画
        ax2 = fig.add_subplot(1,2,1)  #1行2列つくって左に配置

        ax2.set_xlabel('Z [mm]',fontsize=15)
        ax2.set_ylabel('Y [mm]',fontsize=15)

        ax2.minorticks_on()
        ax2.grid(which = "major", axis = "x", color = "gray", alpha = 0.5, linestyle = "-", linewidth = 1)
        ax2.grid(which = "major", axis = "y", color = "gray", alpha = 0.5, linestyle = "-", linewidth = 1)
        ax2.grid(which = "minor", axis = "x", color = "gray", alpha = 0.1, linestyle = "-", linewidth = 1)
        ax2.grid(which = "minor", axis = "y", color = "gray", alpha = 0.1, linestyle = "-", linewidth = 1)

        ax2.invert_xaxis()
        ax2.set_aspect('equal', adjustable='box')
        # ax2.scatter(XL_z_seal, XL_y_seal, c=colors, s=20)
        # # ax2.scatter(XL_z_seal_SG, XL_y_seal_SG, c=colors, s=20)

        # ax2.scatter(XL_z_seal, XL_y_seal, c="k", s=20, alpha=0.4)
        ax2.scatter(XL_z_seal_SG, XL_y_seal_SG, c=colors, s=20)

        plt.tight_layout()  # グラフのレイアウトを調整
        plt.savefig(dir_path + f"frontal&sagittal_nose_theta[{int(np.rad2deg(theta_co_x))},{int(np.rad2deg(theta_y))},{int(np.rad2deg(theta_z))}].png", bbox_inches='tight')
        print(f"fig is saved in frontal&sagittal_nose_theta[{int(np.rad2deg(theta_co_x))},{int(np.rad2deg(theta_y))},{int(np.rad2deg(theta_z))}].png")
        # plt.show()

        print(f"xlim = {max(XL_x_seal_SG)-min(XL_x_seal_SG):.1f}")
        print(f"xlim a= {min(XL_x_seal_SG)-XL_x_seal_SG[0]:.1f}")
        print(f"xlim b= {max(XL_x_seal_SG)-XL_x_seal_SG[0]:.1f}")

        print(f"ylim = {max(XL_y_seal_SG)-min(XL_y_seal_SG):.1f}")

        print(f"zlim = {XL_z_seal_SG[0]-min(XL_z_seal_SG):.1f}")
        # print(f"zlim a= {XL_z_seal_SG[0]-min(XL_z_seal_SG):.1f}")
        # print(f"zlim b= {max(XL_z_seal_SG)-XL_z_seal_SG[0]:.1f}")

MakeGraph(root_dir, 30)