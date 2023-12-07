import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from scipy.signal import savgol_filter
from matplotlib.cm import ScalarMappable
import sys

caliblation_time = 5

#カメラ座標系 RealSensed435i https://www.intelrealsense.com/how-to-getting-imu-data-from-d435i-and-t265/
norm = 9.245
a_x = 0.226
a_y = -8.247
a_z = 4.168
theta_camera = np.arcsin(a_z / norm) # 0.4393960220266065
theta_camera = 0
print(np.rad2deg(theta_camera))

root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_09_000"

# if len(sys.argv) > 1:
#     root_dir = sys.argv[1]
# else:
#     print("ディレクトリパスが指定されていません。")
#     sys.exit()

#カメラを水平に補正するための平行移動量計算 回転じゃないとダメ
# t_y = aa[f][33][3]*np.sin(theta_camera)*np.cos(theta_camera)
# t_z = - aa[f][33][3]*np.sin(theta_camera)*np.sin(theta_camera)


def MakeGraph(root_dir, fps, EndTimeOfTerm1, EndTimeOfTerm2):
    pattern = os.path.join(root_dir, '*J2*/result.npy')
    npy_files = glob.glob(pattern, recursive=True)
    print(npy_files)

    for i,npy_file in enumerate(npy_files):
        dir_path = os.path.dirname(npy_file) + '/'
        aa=np.load(npy_file, allow_pickle=True)
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
            f = frame_number

            if f < caliblation_time*fps:
                #e_XL (36 → 45のベクトル)
                bector_x = np.array([aa[f][45][1]-aa[f][36][1], aa[f][45][2]-aa[f][36][2], aa[f][45][4]-aa[f][36][4]])
                length_bector_x = ((bector_x[0])**2 + (bector_x[1])**2 + (bector_x[2])**2)**(1/2)
                base_bector_x = [bector_x[0]/length_bector_x, bector_x[1]/length_bector_x, bector_x[2]/length_bector_x]
                bector30_36 = np.array([aa[f][36][1]-aa[f][30][1],aa[f][36][2]-aa[f][30][2],aa[f][36][4]-aa[f][30][4]])
                c = - (np.dot(bector_x,bector30_36))/(np.linalg.norm(bector_x)**2)

                #e_yLnose
                #bector_Xと30からbectorXに下した垂線の交点をPとした、30とPを結ぶベクトルがbector_y_nose
                bector_y_nose = bector30_36 + c*bector_x
                bector_Pposition = [bector_y_nose[0]+aa[f][30][1],bector_y_nose[1]+aa[f][30][2],bector_y_nose[2]+aa[f][30][4]]
                bector_y_nose2= np.array([bector_y_nose[0],bector_y_nose[1], 0])
                #print(bector_y_nose, bector_y_nose2)
                theta_nose_sum += float(np.arccos(np.dot(bector_y_nose,bector_y_nose2)/(np.linalg.norm(bector_y_nose)*np.linalg.norm(bector_y_nose2))))
                theta_nose = (theta_nose_sum/(caliblation_time*fps)) - theta_camera
            else:
                for f in range(caliblation_time*fps,aa.shape[0]):
                    #e_XL (36 → 45のベクトル)
                    bector_x = np.array([aa[f][45][1]-aa[f][36][1], aa[f][45][2]-aa[f][36][2], aa[f][45][4]-aa[f][36][4]])
                    length_bector_x = ((bector_x[0])**2 + (bector_x[1])**2 + (bector_x[2])**2)**(1/2)
                    base_bector_x = [bector_x[0]/length_bector_x, bector_x[1]/length_bector_x, bector_x[2]/length_bector_x]

                    bector30_36 = np.array([aa[f][36][1]-aa[f][30][1],aa[f][36][2]-aa[f][30][2],aa[f][36][4]-aa[f][30][4]])
                    c = -(np.dot(bector_x,bector30_36))/(np.linalg.norm(bector_x)**2)

                    #e_yLnose
                    #bector_Xと30からbectorXに下した垂線の交点をPとした、30とPを結ぶベクトルがbector_y_nose
                    bector_y_nose = bector30_36 + c*bector_x
                    base_bector_y_nose = bector_y_nose/np.linalg.norm(bector_y_nose)
                    bector_Pposition = [bector_y_nose[0]+aa[f][30][1],bector_y_nose[1]+aa[f][30][2],bector_y_nose[2]+aa[f][30][4]]

                    #e_zLnose
                    base_bector_z_nose = np.cross(base_bector_x,base_bector_y_nose)

                    R_G_Nose = np.array([base_bector_x,base_bector_y_nose,base_bector_z_nose]).T
                    R_Nose_L = np.array([[1,0,0],[0,np.cos(theta_nose), np.sin(theta_nose)],[0, -np.sin(theta_nose),np.cos(theta_nose)]]).T
                    # t = [aa[f][33][1], aa[f][33][2], aa[f][33][4]]
                    t = [bector_Pposition[0],bector_Pposition[1],bector_Pposition[2]]

                    #Aは同次変換行列    (A_G_Noseはtheta_nose補正前までの回転と並進、原点は33番    A_Nose_LはXnose軸について鼻の角度分回転のみ)
                    A_G_Nose = np.array([[R_G_Nose[0][0],R_G_Nose[0][1],R_G_Nose[0][2],t[0]],
                                        [R_G_Nose[1][0],R_G_Nose[1][1],R_G_Nose[1][2],t[1]],
                                        [R_G_Nose[2][0],R_G_Nose[2][1],R_G_Nose[2][2],t[2]],
                                        [0,0,0,1]])

                    A_Nose_L = np.array([[R_Nose_L[0][0],R_Nose_L[0][1],R_Nose_L[0][2],0],
                                        [R_Nose_L[1][0],R_Nose_L[1][1],R_Nose_L[1][2],0],
                                        [R_Nose_L[2][0],R_Nose_L[2][1],R_Nose_L[2][2],0],
                                        [0,0,0,1]])

                    # A_Nose_L = np.array([[1,0,0,0],
                    #                     [0,1,0,0],
                    #                     [0,0,1,0],
                    #                     [0,0,0,1]])

                    A_G_L = np.dot(A_G_Nose,A_Nose_L)
                    X_seal = np.array([aa[f][68][1], aa[f][68][2], aa[f][68][4],1])
                    # XL_seal = np.cross((R1@R2).T ,XG_point)[0]
                    XL_seal = np.dot(np.linalg.inv(A_G_L) ,X_seal)
                    XL_x_seal.append(XL_seal[0])
                    XL_y_seal.append(XL_seal[1])
                    XL_z_seal.append(XL_seal[2])
                    frame_count.append(count)

                    count = count + 1

                    for id in range(aa.shape[1]):
                        Xid = np.array([aa[f][id][1], aa[f][id][2], aa[f][id][4],1])
                        XX = np.dot(np.linalg.inv(A_G_L),Xid)
                        X.append(XX[0])
                        Y.append(XX[1])
                        Z.append(XX[2])

                print('theta_nose [rad] = ', theta_nose)

                #ローカル座標のxyzをnpyファイルで保存
                X = np.array(X).reshape(((aa.shape[0]-caliblation_time*fps),aa.shape[1]))
                Y = np.array(Y).reshape(((aa.shape[0]-caliblation_time*fps),aa.shape[1]))
                Z = np.array(Z).reshape(((aa.shape[0]-caliblation_time*fps),aa.shape[1]))
                XYZdata = np.stack([X,Y,Z])
                path = dir_path + "XYZ_localdata_rotate.npy"
                np.save(path,XYZdata)
                print('XYZ_localdata is saved')


                # x1,x2,y1,y2=[],[],[],[]

                # for k in range(aa.shape[0]):
                #     if k < EndTimeOfTerm1*fps:
                #         x1.append(XL_x_seal[k] )
                #         y1.append(XL_y_seal[k] )
                #     elif EndTimeOfTerm1*fps <= k < EndTimeOfTerm2*fps :
                #         x2.append(XL_x_seal[k] )
                #         y2.append(XL_y_seal[k] )

                # x1,x2,y1,y2 = np.array(x1),np.array(x2),np.array(y1),np.array(y2)
                # x1,x2 = np.convolve(x1,np.ones(5)/5, mode='valid'),np.convolve(x2,np.ones(5)/5, mode='valid')
                # y1,y2 = np.convolve(y1,np.ones(5)/5, mode='valid'),np.convolve(y2,np.ones(5)/5, mode='valid')

                # 散布図,線を描画
                fig, ax = plt.subplots(dpi=300)

                ax.set_xlabel('X [mm]',fontsize=15)
                ax.set_ylabel('Y [mm]',fontsize=15)

                ax.minorticks_on()
                ax.grid(which = "both", color = "gray", alpha = 0.3, linestyle = "-", linewidth = 1)
                ax.grid(which="minor", color="gray", alpha = 0.3, linestyle=":")

                """
                ax.set_xlim(-6,4)
                ax.set_ylim(-66,-46)

                # 主目盛
                labels = ax.set_xticks(np.arange(-6,4+1,2))
                for t in labels :
                    t.set_path_effects([patheffects.Stroke(linewidth=3,foreground='white'),patheffects.Normal()])

                labels = ax.set_yticks(np.arange(-66,-46+1,2))
                for t in labels :
                    t.set_path_effects([patheffects.Stroke(linewidth=3,foreground='white'),patheffects.Normal()])

                # 補助目盛(minor)
                ax.set_xticks(np.linspace(-6,4,11),minor=True)
                ax.set_yticks(np.linspace(-66,-46,21),minor=True)
                ax.tick_params(which='minor',direction='inout',length=3)
                """

                ax.set_aspect('equal', adjustable='box')

                #plt.plot(x1, y1, label="right")
                #plt.plot(x2, y2, label="left")
                # plt.plot(x1, y1, label="front")
                # plt.plot(x2, y2, label="behind")

                XL_x_seal_SG = savgol_filter(XL_x_seal,9,3)
                XL_y_seal_SG = savgol_filter(XL_y_seal,9,3)
                XL_z_seal_SG = savgol_filter(XL_z_seal,9,3)

                # Create a color map based on time values
                cmap = plt.get_cmap('jet')
                normalize = plt.Normalize(min(frame_count), max(frame_count))
                colors = cmap(normalize(frame_count))

                sc = ax.scatter(XL_x_seal_SG, XL_y_seal_SG, c=colors, cmap='viridis', s=20)
                # ax.plot(XL_x_seal_SG, XL_y_seal_SG, color='k', linestyle="dashdot", linewidth=1)

                # Create a colorbar
                cbar = fig.colorbar(ScalarMappable(norm=normalize, cmap=cmap), ax=ax)
                cbar.set_label('frame', fontsize=10)

                #plt.plot(XL_x_seal,XL_y_seal,'.')
                #plt.plot(XL_x_seal_SG,XL_y_seal_SG,'.')

                plt.legend(bbox_to_anchor=(0.9, 1.1), loc='upper left', borderaxespad=0, fontsize=10)
                plt.savefig(dir_path + "frontal_rotate.png")
                print("Graph is saved in " + dir_path + "frontal.png")



                # x1,x2,y1,y2=[],[],[],[]

                # for i in range(aa.shape[0]):
                #     if i < EndTimeOfTerm1*fps:
                #             x1.append(XL_z_seal[id] )
                #             y1.append(XL_y_seal[id] )
                #     elif EndTimeOfTerm1*fps <= i < EndTimeOfTerm2*fps :
                #             x2.append(XL_z_seal[id] )
                #             y2.append(XL_y_seal[id] )

                # x1,x2,y1,y2 = np.array(x1),np.array(x2),np.array(y1),np.array(y2)
                # x1,x2 = np.convolve(x1,np.ones(5)/5, mode='valid'),np.convolve(x2,np.ones(5)/5, mode='valid')
                # y1,y2 = np.convolve(y1,np.ones(5)/5, mode='valid'),np.convolve(y2,np.ones(5)/5, mode='valid')

                # 散布図を描画
                fig, ax = plt.subplots(dpi=300)

                ax.set_xlabel('Z [mm]',fontsize=15)
                ax.set_ylabel('Y [mm]',fontsize=15)

                ax.minorticks_on()
                ax.grid(which = "both", color = "gray", alpha = 0.3, linestyle = "-", linewidth = 1)
                ax.grid(which="minor", color="gray", alpha = 0.3, linestyle=":")

                """
                # 主目盛
                labels = ax.set_xticks(np.arange(-28,-14+1,2))
                for t in labels :
                    t.set_path_effects([patheffects.Stroke(linewidth=3,foreground='white'),patheffects.Normal()])

                labels = ax.set_yticks(np.arange(-64,-48+1,2))
                for t in labels :
                    t.set_path_effects([patheffects.Stroke(linewidth=3,foreground='white'),patheffects.Normal()])

                # 補助目盛(minor)
                ax.set_xticks(np.linspace(-28,-14,15),minor=True)
                ax.set_yticks(np.linspace(-64,-48,17),minor=True)
                ax.tick_params(which='minor',direction='inout',length=3)
                """

                ax.invert_xaxis()
                ax.set_aspect('equal', adjustable='box')

                #plt.plot(x1, y1, label="right")
                #plt.plot(x2, y2, label="left")
                # plt.plot(x1, y1, label="front")
                # plt.plot(x2, y2, label="behind")

                sc = ax.scatter(XL_z_seal_SG, XL_y_seal_SG, c=colors, cmap='viridis', s=20)
                # ax.plot(XL_z_seal_SG, XL_y_seal_SG, color='k', linestyle="dashdot", linewidth=1)

                # Create a colorbar
                cbar = fig.colorbar(ScalarMappable(norm=normalize, cmap=cmap), ax=ax)
                cbar.set_label('frame', fontsize=10)

                #plt.plot(XL_z_seal_SG,XL_y_seal_SG,'.')

                plt.legend(loc='upper right')
                plt.savefig(dir_path + "sagittal_rotate.png", bbox_inches='tight')
                print("Graph is saved in " + dir_path + "sagittal.png")

                print(f"xlim = {max(XL_x_seal_SG)-min(XL_x_seal_SG):.2f}")
                print(f"xlim a= {min(XL_x_seal_SG)-XL_x_seal_SG[0]:.2f}")
                print(f"xlim b= {max(XL_x_seal_SG)-XL_x_seal_SG[0]:.2f}")

                print(f"ylim = {max(XL_y_seal_SG)-min(XL_y_seal_SG):.2f}")

                print(f"zlim = {max(XL_z_seal_SG)-min(XL_z_seal_SG):.2f}")
                print(f"zlim a= {XL_z_seal_SG[0]-min(XL_z_seal_SG):.2f}")
                print(f"zlim b= {max(XL_z_seal_SG)-XL_z_seal_SG[0]:.2f}")

                # # print(XL_x_seal)
                # print(max(XL_x_seal)-min(XL_x_seal))
                # print(max(XL_y_seal)-min(XL_y_seal))
                # print(max(XL_z_seal)-min(XL_z_seal))
                # # print(str(np.argmax(np.array(XL_x_seal))-np.argmin(np.array(XL_x_seal))))
                # # print(str(np.argmax(np.array(XL_y_seal))-np.argmin(np.array(XL_y_seal))))
                # # print(str(np.argmax(np.array(XL_z_seal))-np.argmin(np.array(XL_z_seal))))
                break




MakeGraph(root_dir, 30, 9,12)