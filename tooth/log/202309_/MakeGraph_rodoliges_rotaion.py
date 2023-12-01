import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

root_dir = 'C:/Users/Tomson/BRLAB/tooth/Temporomandibular_movement/movie/2023_06_05'
caliblation_time = 5

def MakeGraph(root_dir, fps, EndTimeOfTerm1, EndTimeOfTerm2):
    pattern = os.path.join(root_dir, 'result.npy')
    npy_files = glob.glob(pattern, recursive=True)
    print(npy_files)

    for i,npy_file in enumerate(npy_files):
        dir_path = os.path.dirname(npy_file) + '/'
        aa=np.load(npy_file, allow_pickle=True)
        # print(aa.shape[0])
        XL_x = []
        XL_y = []
        XL_z = []

        theta_sum = 0
        X = []
        Y = []
        Z = []

        for frame_number in range(aa.shape[0]):
            f = frame_number
            # print(aa.shape[f])
            # print("frame number = ",frame_number)
            # print(bool(frame_number < caliblation_time*fps))

            #e_XL (36 → 45のベクトル)
            bector_x = np.array([aa[f][45][1]-aa[f][36][1], aa[f][45][2]-aa[f][36][2], aa[f][45][4]-aa[f][36][4]])
            length_bector_x = ((bector_x[0])**2 + (bector_x[1])**2 + (bector_x[2])**2)**(1/2)
            base_bector_x = [bector_x[0]/length_bector_x, bector_x[1]/length_bector_x, bector_x[2]/length_bector_x]

            bector30_36 = np.array([aa[f][36][1]-aa[f][30][1],aa[f][36][2]-aa[f][30][2],aa[f][36][4]-aa[f][30][4]])
            c = -(np.dot(bector_x,bector30_36))/(np.linalg.norm(bector_x)**2)
            # print('bectorX=',bector_x,'bector30_36=',bector30_36)
            # print('np.dot(bector_x,bector30_36)',np.dot(bector_x,bector30_36),'bectorxnorm**2=',np.linalg.norm(bector_x)**2)
            # print('c=',c)
            # #bector_Xと30からbectorXに下した垂線の交点をPとした、30とPを結ぶベクトルがbector_y_nose
            bector_y_nose = bector30_36 + c*bector_x
            # print('bectorynose=',bector_y_nose)
            bector_Pposition = (bector_y_nose[0]+aa[f][30][1],bector_y_nose[1]+aa[f][30][2],bector_y_nose[2]+aa[f][30][4])


            if f < caliblation_time*fps:
                bector_y_nose2= np.array([bector_y_nose[0],bector_y_nose[1], bector_Pposition[2]])
                theta_sum += float(np.arccos(np.dot(bector_y_nose,bector_y_nose2)/(np.linalg.norm(bector_y_nose)*np.linalg.norm(bector_y_nose2))))
                theta = -(theta_sum/(caliblation_time*fps))

            else:
                for f in range(caliblation_time*fps,aa.shape[0]):
                    # print('theta=',theta)
                    #aa.shape[0]=474

                    #e_YL (30 → Pのベクトルをe_XLについてtheta回転) ロドリゲスの回転公式
                    n = np.array([base_bector_x[0], base_bector_x[1], base_bector_x[2]])
                    R_xL =    np.array([[n[0]*n[0]*(1-np.cos(theta)) + np.cos(theta), n[0]*n[1]*(1-np.cos(theta))-n[2]*np.sin(theta), n[0]*n[2]*(1-np.cos(theta)) + n[2]+np.sin(theta)],
                                        [n[1]*n[2]*(1-np.cos(theta)) + n[2]*np.sin(theta), n[1]*n[1]*(1-np.cos(theta))+np.cos(theta), n[1]*n[2]*(1-np.cos(theta)) - n[0]+np.sin(theta)],
                                        [n[0]*n[2]*(1-np.cos(theta)) + n[1]*np.sin(theta), n[1]*n[2]*(1-np.cos(theta)) + n[0]+np.sin(theta), n[2]*n[2]*(1-np.cos(theta))+np.cos(theta)]])
                    bector_y = np.dot(R_xL,bector_y_nose)
                    # print('bector_y=',bector_y)
                    length_bector_y = ((bector_y[0])**2 + (bector_y[1])**2 + (bector_y[2])**2)**(1/2)
                    base_bector_y = [bector_y[0]/length_bector_y, bector_y[1]/length_bector_y, bector_y[2]/length_bector_y]

                    #e_ZL (X,Yの基本ベクトルの外積)
                    vector_product_x = base_bector_x[1]*base_bector_y[2] - base_bector_x[2]*base_bector_y[1]
                    vector_product_y = base_bector_x[2]*base_bector_y[0] - base_bector_x[0]*base_bector_y[2]
                    vector_product_z = base_bector_x[0]*base_bector_y[1] - base_bector_x[1]*base_bector_y[0]
                    base_bector_z = [vector_product_x, vector_product_y, vector_product_z]

                    # print('R_xL=',R_xL)
                    # print('bector_y_nose',bector_y_nose,'bector_y_nose.shape',bector_y_nose.shape)
                    # print('R_xL=', R_xL,'R_xL.shape = ',R_xL.shape)
                    # print('bector_y = ',bector_y,'bector_y.shape = ', bector_y.shape)
                    # print('base_y',f,bector_y)
                    # print('base_ynose',f,bector_y_nose)
                    # print(base_bector_x[0]*base_bector_y[0]+base_bector_x[1]*base_bector_y[1]+base_bector_x[2]*base_bector_y[2])

                    #座標変換(68) XL = M^t*XG - M^t*t
                    #M~t = (ex,ey,ez)^t, t = 33 - 原点
                    t = [aa[f][33][1], aa[f][33][2], aa[f][33][4]]
                    XG = [aa[f][68][1], aa[f][68][2], aa[f][68][4]]
                    #XGt = XG -t
                    XGt = [XG[0]-t[0], XG[1]-t[1], XG[2]-t[2]]
                    XL_x.append(base_bector_x[0]*XGt[0] + base_bector_x[1]*XGt[1] + base_bector_x[2]*XGt[2])
                    XL_y.append(base_bector_y[0]*XGt[0] + base_bector_y[1]*XGt[1] + base_bector_y[2]*XGt[2])
                    XL_z.append(base_bector_z[0]*XGt[0] + base_bector_z[1]*XGt[1] + base_bector_z[2]*XGt[2])

                    for id in range(aa.shape[1]):
                        Xid = [aa[f][id][1], aa[f][id][2], aa[f][id][4]]
                        Xidt = [Xid[0]-t[0], Xid[1]-t[1], Xid[2]-t[2]]
                        # print(type((base_bector_x[0]*Xidt[0] + base_bector_x[1]*Xidt[1] + base_bector_x[2]*XGidt[2]).tolist))
                        X.append((base_bector_x[0]*Xidt[0] + base_bector_x[1]*Xidt[1] + base_bector_x[2]*Xidt[2]))
                        Y.append((base_bector_y[0]*Xidt[0] + base_bector_y[1]*Xidt[1] + base_bector_y[2]*Xidt[2]))
                        Z.append((base_bector_z[0]*Xidt[0] + base_bector_z[1]*Xidt[1] + base_bector_z[2]*Xidt[2]))


                #ローカル座標のxyzをnpyファイルで保存
                # print('xtype=',type(X))
                X = np.array(X).reshape(((aa.shape[0]-caliblation_time*fps),aa.shape[1]))
                Y = np.array(Y).reshape(((aa.shape[0]-caliblation_time*fps),aa.shape[1]))
                # print('Y=',Y)
                Z = np.array(Z).reshape(((aa.shape[0]-caliblation_time*fps),aa.shape[1]))
                XYZdata = np.stack([X,Y,Z])
                # print(np.shape(XYZdata))
                path = dir_path + "XYZ_localdata_rodoliges_rotate.npy"
                np.save(path,XYZdata)
                print('XYZ_localdata is saved')

                x1,x2,y1,y2=[],[],[],[]

                for k in range(aa.shape[0]):
                    if k < EndTimeOfTerm1*fps:
                        x1.append(XL_x[k] )
                        y1.append(XL_y[k] )
                    elif EndTimeOfTerm1*fps <= k < EndTimeOfTerm2*fps :
                        x2.append(XL_x[k] )
                        y2.append(XL_y[k] )

                x1,x2,y1,y2 = np.array(x1),np.array(x2),np.array(y1),np.array(y2)
                x1,x2 = np.convolve(x1,np.ones(5)/5, mode='valid'),np.convolve(x2,np.ones(5)/5, mode='valid')
                y1,y2 = np.convolve(y1,np.ones(5)/5, mode='valid'),np.convolve(y2,np.ones(5)/5, mode='valid')

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
                plt.plot(x1, y1, label="front")
                plt.plot(x2, y2, label="behind")

                plt.legend(bbox_to_anchor=(0.9, 1.1), loc='upper left', borderaxespad=0, fontsize=10)
                plt.savefig(dir_path + "frontal__rodoliges_rotate.png")
                print("Graph is saved in " + dir_path + "frontal.png")

                x1,x2,y1,y2=[],[],[],[]

                for i in range(aa.shape[0]):
                    if i < EndTimeOfTerm1*fps:
                            x1.append(XL_z[id] )
                            y1.append(XL_y[id] )
                    elif EndTimeOfTerm1*fps <= i < EndTimeOfTerm2*fps :
                            x2.append(XL_z[id] )
                            y2.append(XL_y[id] )

                x1,x2,y1,y2 = np.array(x1),np.array(x2),np.array(y1),np.array(y2)
                x1,x2 = np.convolve(x1,np.ones(5)/5, mode='valid'),np.convolve(x2,np.ones(5)/5, mode='valid')
                y1,y2 = np.convolve(y1,np.ones(5)/5, mode='valid'),np.convolve(y2,np.ones(5)/5, mode='valid')

                # 散布図を描画
                fig, ax = plt.subplots()

                ax.set_xlabel('Z [mm]',fontsize=15)
                ax.set_ylabel('Y [mm]',fontsize=15)

                ax.grid(which = "major", axis = "x", color = "gray", alpha = 0.1, linestyle = "-", linewidth = 1)
                ax.grid(which = "major", axis = "y", color = "gray", alpha = 0.1, linestyle = "-", linewidth = 1)

                """
                ax.set_xlim(-28,-14)
                ax.set_ylim(-64,-48)

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
                plt.plot(x1, y1, label="front")
                plt.plot(x2, y2, label="behind")

                plt.legend(loc='upper right')
                plt.savefig(dir_path + "sagittal_rodoliges_rotate.png", bbox_inches='tight')
                print("Graph is saved in " + dir_path + "sagittal.png")

MakeGraph(root_dir, 30, 9,12)