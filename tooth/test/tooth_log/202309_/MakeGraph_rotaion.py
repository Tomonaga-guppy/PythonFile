import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

root_dir = 'C:/Users/Tomson/BRLAB/tooth/Temporomandibular_movement/movie/2023_06_14/20230614_183324'
caliblation_time = 5

def MakeGraph(root_dir, fps, EndTimeOfTerm1, EndTimeOfTerm2):
    pattern = os.path.join(root_dir, 'result.npy')
    npy_files = glob.glob(pattern, recursive=True)
    print(npy_files)

    for i,npy_file in enumerate(npy_files):
        dir_path = os.path.dirname(npy_file) + '/'
        aa=np.load(npy_file, allow_pickle=True)
        XL_x_seal = []
        XL_y_seal = []
        XL_z_seal = []

        theta_sum = 0
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
                base_bector_y_nose = bector_y_nose/np.linalg.norm(bector_y_nose)
                bector_Pposition = [bector_y_nose[0]+aa[f][30][1],bector_y_nose[1]+aa[f][30][2],bector_y_nose[2]+aa[f][30][4]]

                bector_y_nose2= np.array([bector_y_nose[0],bector_y_nose[1], bector_Pposition[2]])
                theta_sum += float(np.arccos(np.dot(bector_y_nose,bector_y_nose2)/(np.linalg.norm(bector_y_nose)*np.linalg.norm(bector_y_nose2))))
                theta = (theta_sum/(caliblation_time*fps))

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
                    R_Nose_L = np.array([[1,0,0],[0,np.cos(theta), np.sin(theta)],[0, -np.sin(theta),np.cos(theta)]]).T
                    # t = [aa[f][33][1], aa[f][33][2], aa[f][33][4]]
                    t = [bector_Pposition[0],bector_Pposition[1],bector_Pposition[2]]

                    #Aは同次変換行列    (A_G_Noseはtheta補正前までの回転と並進、原点は33番    A_Nose_LはXnose軸について鼻の角度分回転のみ)
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

                    for id in range(aa.shape[1]):
                        Xid = np.array([aa[f][id][1], aa[f][id][2], aa[f][id][4],1])
                        XX = np.dot(np.linalg.inv(A_G_L),Xid)
                        X.append(XX[0])
                        Y.append(XX[1])
                        Z.append(XX[2])


                print('theta [rad] = ', theta)

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
                fig, ax = plt.subplots()

                ax.set_xlabel('X [mm]',fontsize=15)
                ax.set_ylabel('Y [mm]',fontsize=15)

                ax.grid(which = "major", axis = "x", color = "gray", alpha = 0.1, linestyle = "-", linewidth = 1)
                ax.grid(which = "major", axis = "y", color = "gray", alpha = 0.1, linestyle = "-", linewidth = 1)

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
                plt.plot(XL_x_seal,XL_y_seal,'.')

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
                fig, ax = plt.subplots()

                ax.set_xlabel('Z [mm]',fontsize=15)
                ax.set_ylabel('Y [mm]',fontsize=15)

                ax.grid(which = "major", axis = "x", color = "gray", alpha = 0.1, linestyle = "-", linewidth = 1)
                ax.grid(which = "major", axis = "y", color = "gray", alpha = 0.1, linestyle = "-", linewidth = 1)

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
                plt.plot(XL_z_seal,XL_y_seal,'.')

                plt.legend(loc='upper right')
                plt.savefig(dir_path + "sagittal_rotate.png", bbox_inches='tight')
                print("Graph is saved in " + dir_path + "sagittal.png")

MakeGraph(root_dir, 30, 9,12)