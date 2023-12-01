import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

root_dir = 'C:/Users/Tomson/BRLAB/tooth/Temporomandibular_movement/movie/2023_06_05'

def MakeGraph(root_dir, fps, EndTimeOfTerm1, EndTimeOfTerm2):

    pattern = os.path.join(root_dir, 'result.npy')
    npy_files = glob.glob(pattern, recursive=True)
    print(npy_files)

    for i,npy_file in enumerate(npy_files):
        dir_path = os.path.dirname(npy_file) + '/'
        aa=np.load(npy_file, allow_pickle=True)

        print('aa.shape[0]=',aa.shape[0])

        XL_x = []
        XL_y = []
        XL_z = []

        for i in range(aa.shape[0]):

            #e_XL (36 → 45のベクトル)
            bector_x = [aa[i][45][1]-aa[i][36][1], aa[i][45][2]-aa[i][36][2], aa[i][45][4]-aa[i][36][4]]
            length_bector_x = ((bector_x[0])**2 + (bector_x[1])**2 + (bector_x[2])**2)**(1/2)
            base_bector_x = [bector_x[0]/length_bector_x, bector_x[1]/length_bector_x, bector_x[2]/length_bector_x]

            #e_YL (33 → 28のベクトル)
            bector_y = [aa[i][28][1]-aa[i][33][1], aa[i][28][2]-aa[i][33][2], 0]
            length_bector_y = ((bector_y[0])**2 + (bector_y[1])**2 + (bector_y[2])**2)**(1/2)
            base_bector_y = [bector_y[0]/length_bector_y, bector_y[1]/length_bector_y, bector_y[2]/length_bector_y]

            """
            #e_ZL (27→XLベクトル上の点)
            bector36_27 = [aa[i][27][1]-aa[i][36][1], aa[i][27][2]-aa[i][36][2], aa[i][27][4]-aa[i][36][4]]
            n = base_bector_x[0]*bector36_27[0] + base_bector_x[1]*bector36_27[1] + base_bector_x[2]*bector36_27[2]
            bector_z = [bector36_27[0] - n*base_bector_x[0], bector36_27[1] - n*base_bector_x[1], bector36_27[2] - n*base_bector_x[2]]
            length_bector_z = ((bector_z[0])**2 + (bector_z[1])**2 + (bector_z[2])**2)**(1/2)
            base_bector_z = [bector_z[0]/length_bector_z, bector_z[1]/length_bector_z, bector_z[2]/length_bector_z]

            #e_YL （Z,Xの基本ベクトルの外積）
            vector_product_x = base_bector_z[1]*base_bector_x[2] - base_bector_z[2]*base_bector_x[1]
            vector_product_y = base_bector_z[2]*base_bector_x[0] - base_bector_z[0]*base_bector_x[2]
            vector_product_z = base_bector_z[0]*base_bector_x[1] - base_bector_z[1]*base_bector_x[0]
            base_bector_y = [vector_product_x, vector_product_y, vector_product_z]
            """

            #e_ZL (X,Yの基本ベクトル)
            vector_product_x = base_bector_x[1]*base_bector_y[2] - base_bector_x[2]*base_bector_y[1]
            vector_product_y = base_bector_x[2]*base_bector_y[0] - base_bector_x[0]*base_bector_y[2]
            vector_product_z = base_bector_x[0]*base_bector_y[1] - base_bector_x[1]*base_bector_y[0]
            base_bector_z = [vector_product_x, vector_product_y, vector_product_z]

            #座標変換(68) XL = M^t*XG - M^t*t
            #M~t = (ex,ey,ez)^t, t = 33 - 原点
            t = [aa[i][33][1], aa[i][33][2], aa[i][33][4]]
            XG = [aa[i][68][1], aa[i][68][2], aa[i][68][4]]
            #XGt = XG -t
            XGt = [XG[0]-t[0], XG[1]-t[1], XG[2]-t[2]]
            XL_x.append(base_bector_x[0]*XGt[0] + base_bector_x[1]*XGt[1] + base_bector_x[2]*XGt[2])
            XL_y.append(base_bector_y[0]*XGt[0] + base_bector_y[1]*XGt[1] + base_bector_y[2]*XGt[2])
            XL_z.append(base_bector_z[0]*XGt[0] + base_bector_z[1]*XGt[1] + base_bector_z[2]*XGt[2])

            #######################################
            for j in range(aa.shape[1]):
                XGj = [aa[i][j][1], aa[i][j][2], aa[i][j][4]]
                XGjt = [XGj[0]-t[0], XGj[1]-t[1], XGj[2]-t[2]]
                X.append(base_bector_x[0]*XGjt[0] + base_bector_x[1]*XGjt[1] + base_bector_x[2]*XGjt[2])
                Y.append(base_bector_y[0]*XGjt[0] + base_bector_y[1]*XGjt[1] + base_bector_y[2]*XGjt[2])
                Z.append(base_bector_z[0]*XGjt[0] + base_bector_z[1]*XGjt[1] + base_bector_z[2]*XGjt[2])


        #ローカル座標のxyzをnpyファイルで保存
        X = np.array(X).reshape((519,69))
        Y = np.array(Y).reshape((519,69))
        Z = np.array(Z).reshape((519,69))
        print('X=',np.shape(X))
        XYZdata = np.stack([X,Y,Z])
        print(np.shape(XYZdata))
        path = dir_path + "XYZ_localdata.npy"
        np.save(path,XYZdata)
##################################################
        x1,x2,y1,y2=[],[],[],[]

        for i in range(aa.shape[0]):
            if i < EndTimeOfTerm1*fps:
                if (XL_x[i] != None or XL_y[i] !=None) and XL_y[i] <0:
                    x1.append(XL_x[i] )
                    y1.append(XL_y[i] )

            elif EndTimeOfTerm1*fps <= i < EndTimeOfTerm2*fps :
                if (XL_x[i] != None or XL_y[i] !=None) and XL_y[i] < 0:
                    x2.append(XL_x[i] )
                    y2.append(XL_y[i] )

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
        plt.savefig(dir_path + "frontal.png")
        print("Graph is saved in " + dir_path + "frontal.png")

        x1,x2,y1,y2=[],[],[],[]

        for i in range(aa.shape[0]):
            if i < EndTimeOfTerm1*fps:
                if (XL_z[i] != None or XL_y[i] !=None) and XL_y[i] < 0:
                    x1.append(XL_z[i] )
                    y1.append(XL_y[i] )
            elif EndTimeOfTerm1*fps <= i < EndTimeOfTerm2*fps :
                if (XL_z[i] != None or XL_y[i] !=None) and XL_y[i] < 0:
                    x2.append(XL_z[i] )
                    y2.append(XL_y[i] )

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
        plt.savefig(dir_path + "sagittal.png", bbox_inches='tight')
        print("Graph is saved in " + dir_path + "sagittal.png")

MakeGraph(root_dir, 30, 9,12)