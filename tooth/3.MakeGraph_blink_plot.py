import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from scipy.signal import savgol_filter
from matplotlib.cm import ScalarMappable
import sys
import csv
import pandas as pd
import trimesh
import matplotlib.ticker as mticker

root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_12_20"

# corrective_condition = [[2,-5],[5,0],[5,-10],[4,-15],[4,0],[5,-4.578]]  #a,b,c,d,e,fの順 [calibration_time, theta_co_x]
corrective_condition = [[2,-5],[5,0],[5,-10],[4,-15],[4,0],[5,0]]  #a,b,c,d,e,fの順 [calibration_time, theta_co_x]
fps = 30

grid_draw = True  #Trueでグリッドを描画
frame_draw = False #Trueで各点にframe番号を表示
seal_3dplot = False  #Trueでシールの3D軌道作成
ear = True  #Trueでblink_list_npy（まばたき補正）を使用
ply_create = False #Trueでplyファイル作成
blink_comparison = True  #Trueでまばたき補正前後の比較

global theta_co_x, theta_co_y, theta_co_z
theta_co_x = np.deg2rad(0)
theta_co_y = np.deg2rad(0)
theta_co_z = np.deg2rad(0)


def main():
    global npy_file, dir_path
    pattern = os.path.join(root_dir, '*a/landmark_cam.npy')
    npy_files = glob.glob(pattern, recursive=True)
    num_npy_files = len(npy_files)

    for i,npy_file in enumerate(npy_files):
        print(f"{i + 1}/{num_npy_files} {npy_file}")
        dir_path = os.path.dirname(npy_file) + '/'

        if blink_comparison:
            df_sg = CaliculateCoordinate(root_dir, dir_path, False)  #まばたき補正なし
            df_sg_blink = CaliculateCoordinate(root_dir, dir_path, True)  #まばたき補正あり
            MakeGraph(df_sg_blink,df_sg)
        if not blink_comparison:
            df_sg_blink = CaliculateCoordinate(root_dir, dir_path, True)  #まばたき補正あり
            MakeGraph(df_sg_blink)


def CaliculateCoordinate(root_dir, dir_path, blink_correction):  #座標変換
    id = os.path.basename(os.path.dirname(dir_path))
    if id == "20231117_a": calibration_time, theta_co_x = corrective_condition[0][0], np.deg2rad(corrective_condition[0][1])
    if id == "20231117_b": calibration_time, theta_co_x = corrective_condition[1][0], np.deg2rad(corrective_condition[1][1])
    if id == "20231218_c": calibration_time, theta_co_x = corrective_condition[2][0], np.deg2rad(corrective_condition[2][1])
    if id == "20231218_d": calibration_time, theta_co_x = corrective_condition[3][0], np.deg2rad(corrective_condition[3][1])
    if id == "20231218_e": calibration_time, theta_co_x = corrective_condition[4][0], np.deg2rad(corrective_condition[4][1])
    if id == "20231218_f": calibration_time, theta_co_x = corrective_condition[5][0], np.deg2rad(corrective_condition[5][1])

    if blink_correction:
        print(f"calibration_time = {calibration_time}, theta_co_x = {np.rad2deg(theta_co_x)}")
    aa=np.load(npy_file, allow_pickle=True)  #作製したnumpy配列は[frame][number][number, x, y, z]
    accel_path = os.path.join(dir_path,"accel_data.npy")
    accel = np.load(accel_path, allow_pickle=True)  #[frame][x,y,z]
    theta_camera = np.arccos(np.mean(abs(accel[:,1]))/(np.sqrt(np.mean(accel[:,1])**2+np.mean(accel[:,2])**2)))

    blink_frame_npy = np.load(dir_path + "blink_list.npy")
    if blink_correction:
        print(f"blink_frame_npy = {blink_frame_npy}")

    XL_x_seal = []
    XL_y_seal = []
    XL_z_seal = []
    theta_nose_sum = 0
    X = []
    Y = []
    Z = []

    frame_number = 1

    df_vector_30 = pd.DataFrame(aa[:,30,1:])
    df_vector_36 = pd.DataFrame(aa[:,36,1:])
    df_vector_45 = pd.DataFrame(aa[:,45,1:])

    if blink_correction:
        #まばたきしているフレームの目の位置はまばたき前後で線形補間をおこなう
        df_vector_36.loc[blink_frame_npy,:] = np.nan
        df_vector_36 = df_vector_36.interpolate(method='linear', limit_direction='both')
        df_vector_45.loc[blink_frame_npy,:] = np.nan
        df_vector_45 = df_vector_45.interpolate(method='linear', limit_direction='both')

    while True:
        #鼻先(30)、左右目(36,45)の位置ベクトル
        vector_30 = np.array([df_vector_30.iloc[frame_number-1,0], df_vector_30.iloc[frame_number-1,1], df_vector_30.iloc[frame_number-1,2]])
        vector_36 = np.array([df_vector_36.iloc[frame_number-1,0], df_vector_36.iloc[frame_number-1,1], df_vector_36.iloc[frame_number-1,2]])
        vector_45 = np.array([df_vector_45.iloc[frame_number-1,0], df_vector_45.iloc[frame_number-1,1], df_vector_45.iloc[frame_number-1,2]])

        #e_x (36 → 45のベクトル)
        vector_x = vector_45 - vector_36
        base_vector_x = vector_x / np.linalg.norm(vector_x)
        vector_30_36 = vector_36 - vector_30
        c = - (np.dot(vector_x,vector_30_36))/(np.linalg.norm(vector_x)**2)

        #e_y
        #vector_Xと30からvectorXに下した垂線の交点をPとした、30とPを結ぶベクトルがvector_y_nose
        vector_y_nose = vector_30_36 + c*vector_x
        vector_Pposition = vector_y_nose + vector_30
        vector_y_nose2= np.array([vector_y_nose[0],vector_y_nose[1], 0])

        if frame_number < calibration_time*fps:
            theta_nose_sum += float(np.arccos(np.dot(vector_y_nose,vector_y_nose2)/(np.linalg.norm(vector_y_nose)*np.linalg.norm(vector_y_nose2))))
            theta_nose = (theta_nose_sum/frame_number)
            # print(f"theta_nose = {np.rad2deg(theta_nose)}")

        elif frame_number >= calibration_time*fps:
            #e_y
            base_vector_y = vector_y_nose/np.linalg.norm(vector_y_nose)
            #e_z
            base_vector_z = np.cross(base_vector_x,base_vector_y)

            # 点を反時計回りにtheta回転 = 軸を時計回りにtheta回転  ==   点は反時計回りが正、軸は時計回りが正
            # https://qiita.com/suzuki-navi/items/60ef241b2dca499df794
            theta_x =  -theta_nose  +theta_camera + theta_co_x
            # if frame_number == 457 and id == "20231218_f":  #f最大開口時に鼻先検出位置のずれにより約5°ずれていることを確認したため補正
            #     theta_x = theta_x + np.deg2rad(-5)

            #ベクトル変換https://eman-physics.net/math/linear08.html  グローバル座標とローカル座標https://programming-surgeon.com/script/coordinate-system/
            R_Cam_Nose = np.array([base_vector_x,base_vector_y,base_vector_z]).T
            R_Nose_Local = np.array([[1,0,0],[0,np.cos(theta_x), np.sin(theta_x)],[0, -np.sin(theta_x),np.cos(theta_x)]]).T  #x軸回転
            R_Local_Local2 = np.array([[np.cos(theta_co_y),0, -np.sin(theta_co_y)],[0, 1, 0],[np.sin(theta_co_y), 0, np.cos(theta_co_y),]]).T  #y軸回転
            R_Local2_Local3 = np.array([[np.cos(theta_co_z),np.sin(theta_co_z),0],[-np.sin(theta_co_z), np.cos(theta_co_z), 0],[0, 0, 1]]).T  #z軸回転
            t = [vector_Pposition[0],vector_Pposition[1],vector_Pposition[2]]

            #Aは同次変換行列    (A_Cam_Noseはtheta_nose補正前までの回転と並進、原点は33番    A_xはXnose軸について鼻の角度分回転のみ)
            A_Cam_Nose = np.array([[R_Cam_Nose[0][0],R_Cam_Nose[0][1],R_Cam_Nose[0][2],t[0]],
                                [R_Cam_Nose[1][0],R_Cam_Nose[1][1],R_Cam_Nose[1][2],t[1]],
                                [R_Cam_Nose[2][0],R_Cam_Nose[2][1],R_Cam_Nose[2][2],t[2]],
                                [0,0,0,1]])

            #x軸回転
            A_x = np.array([[R_Nose_Local[0][0],R_Nose_Local[0][1],R_Nose_Local[0][2],0],
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

            A_rotate = A_x @ demo @ demo  #行列掛け算

            X_seal = np.array([aa[frame_number-1][68][1], aa[frame_number-1][68][2], aa[frame_number-1][68][3],1])

            # XL_seal = np.dot(np.linalg.inv(A_rotate) ,X_seal)
            # if frame_number == 150:
            #     XL_seal_0 = np.dot(np.linalg.inv(A_rotate) ,X_seal)
            #     t2 = [XL_seal_0[0], XL_seal_0[1], XL_seal_0[2]]
            # #原点移動
            # A_Local3_Local4 = np.array([[1, 0, 0, t2[0]],
            #                             [0, 1, 0, t2[1]],
            #                             [0, 0, 1, t2[2]],
            #                             [0, 0, 0, 1]])
            # A_rotate = A_rotate @ A_Local3_Local4  #行列掛け算

            XL_seal = np.linalg.inv(A_Cam_Nose) @ X_seal
            XL_seal = A_x @ XL_seal

            # print(f"XL_seal_0 = {XL_seal_0}")
            # XL_seal = A_y @ XL_seal
            XL_x_seal.append(XL_seal[0])
            XL_y_seal.append(XL_seal[1])
            XL_z_seal.append(XL_seal[2])

            for landmark_num in range(aa.shape[1]):
                X_land = np.array([aa[frame_number-1][landmark_num][1], aa[frame_number-1][landmark_num][2], aa[frame_number-1][landmark_num][3],1])
                XX = np.linalg.inv(A_Cam_Nose) @ X_land
                XX = A_rotate @ XX
                XX = XX - XL_seal  #顎運動開始時のシールの位置を原点とする
                X.append(XX[0])
                Y.append(XX[1])
                Z.append(XX[2])

            if ply_create == True:
                # if frame_number == 60 or frame_number == 308:  #a
                # if frame_number == 150 or frame_number == 514:  #b
                # if frame_number == 150 or frame_number == 287:  #c
                # if frame_number == 120 or frame_number == 367:  #d
                # if frame_number == 120 or frame_number == 410:  #e
                if frame_number == 150 or frame_number == 456:  #f
                    if os.path.isfile(dir_path + f"plycam/random_cloud{frame_number}.ply"):
                        random_ply_path = dir_path + f"plycam/random_cloud{frame_number}.ply"
                        print(random_ply_path)
                        # PLYファイルを読み込む
                        mesh = trimesh.load_mesh(random_ply_path)
                        color_of_vertices = np.array(mesh.visual.vertex_colors[:,:3])
                        x = np.array(mesh.vertices)[:,0]
                        y = np.array(mesh.vertices)[:,1]
                        z = np.array(mesh.vertices)[:,2]
                        XL = []
                        for vertices_num in range(len(mesh.vertices)):
                            xg = np.array([x[vertices_num], y[vertices_num], z[vertices_num], 1])
                            xl = np.linalg.inv(A_Cam_Nose) @ xg
                            xl = A_rotate @ xl
                            XL.append([xl[0],xl[1],xl[2]])

                        XL = np.array(XL)
                        ply_path = dir_path + "ply"
                        if not os.path.exists(ply_path):
                            os.mkdir(ply_path)

                        # PLYファイルに書き込む
                        header = f"ply\nformat ascii 1.0\nelement vertex {len(mesh.vertices)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                        with open(dir_path + f"ply/random_cloud{frame_number}.ply", "w") as ply_file:
                            ply_file.write(header)
                            for vertex in range(len(mesh.vertices)):
                                vertex = list(map(str, XL[vertex,:])) + list(map(str, map(int, color_of_vertices[vertex,:])))
                                ply_file.write(" ".join(vertex) + "\n")

                    if os.path.isfile(dir_path + f"plycam/face_cloud{frame_number}.ply"):
                        random_ply_path = dir_path + f"plycam/face_cloud{frame_number}.ply"
                        print(random_ply_path)
                        # PLYファイルを読み込む
                        mesh = trimesh.load_mesh(random_ply_path)
                        x = np.array(mesh.vertices)[:,0]
                        y = np.array(mesh.vertices)[:,1]
                        z = np.array(mesh.vertices)[:,2]
                        XL = []
                        for vertices_num in range(len(mesh.vertices)):
                            xg = np.array([x[vertices_num], y[vertices_num], z[vertices_num], 1])
                            xl = np.linalg.inv(A_Cam_Nose) @ xg
                            xl = A_rotate @ xl
                            XL.append([xl[0],xl[1],xl[2]])

                        XL = np.array(XL)
                        ply_path = dir_path + "ply"
                        if not os.path.exists(ply_path):
                            os.mkdir(ply_path)

                        # PLYファイルに書き込む
                        header = f"ply\nformat ascii 1.0\nelement vertex {len(mesh.vertices)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
                        with open(dir_path + f"ply/face_cloud{frame_number}.ply", "w") as ply_file:
                            ply_file.write(header)
                            for vertex in range(len(mesh.vertices)):
                                vertex = list(map(str, XL[vertex,:]))
                                ply_file.write(" ".join(vertex) + "\n")

        frame_number += 1
        if frame_number > aa.shape[0]:  #frame_numberがaa.shape[0]を超えたらwhileを抜ける
            break

    # print('theta_nose [deg] = ', np.rad2deg(theta_nose))
    # print('theta_cam [deg] = ', np.rad2deg(theta_camera))
    # print('theta_co_x [deg] = ', np.rad2deg(theta_co_x))
    # print('theta_x [deg] = ', np.rad2deg(theta_x))

    #ローカル座標のxyzをnpyファイルで保存
    X = np.array(X).reshape(((aa.shape[0]-(calibration_time*fps-1)),aa.shape[1]))
    Y = np.array(Y).reshape(((aa.shape[0]-(calibration_time*fps-1)),aa.shape[1]))
    Z = np.array(Z).reshape(((aa.shape[0]-(calibration_time*fps-1)),aa.shape[1]))
    XYZdata = np.stack([X,Y,Z])
    path = dir_path + "landmark_local.npy"
    np.save(path,XYZdata)
    # print('XYZ_localdata is saved')

    data = {'x': XL_x_seal,
            'y': XL_y_seal,
            'z': XL_z_seal}

    df = pd.DataFrame(data)

    # #df['y']をフレーム数を横軸としてグラフを描画
    # plt.figure()
    # ax_a = plt.subplot(1,2,1)
    # ax_a.plot(df['z'])

    df_sg = pd.DataFrame(index=df.index)
    # 各列データを平滑化して、結果をdf_sgに格納
    #SG法   https://mimikousi.com/smoothing_savgol/
    window_length = 7 #奇数に設定． 窓枠を増やすとより平滑化される
    polyorder = 2  #window_lengthよりも小さく． 次数が大きい方がノイズを強調する

    for col in df.columns:
        df_sg[col] = savgol_filter(df[col], window_length=window_length, polyorder=polyorder)

    XL_x_seal_SG = df_sg['x']
    XL_y_seal_SG = df_sg['y']
    XL_z_seal_SG = df_sg['z']

    data_num = df_sg.shape[0]
    mo =  np.sqrt((XL_y_seal_SG[1:] - XL_y_seal_SG[0])**2 + (XL_z_seal_SG[1:] - XL_z_seal_SG[0])**2)  #最大開口量
    rs_mo = max(mo)
    mo_index = mo.idxmax()
    # print(f"mo = {mo}")

    rs_xr = min(XL_x_seal_SG)-XL_x_seal_SG[0]
    rs_xl = max(XL_x_seal_SG)-XL_x_seal_SG[0]
    rs_y = XL_y_seal_SG[mo_index]-XL_y_seal_SG[0]
    rs_z = XL_z_seal_SG[mo_index]-XL_z_seal_SG[0]

    if blink_correction:
        print(f"max_mo_frame = {mo_index}({mo_index+fps*calibration_time})")
        # print(f"df[max_mo_frame] = {df.iloc[mo_index,:]}")
        # print(f"df_sg[max_mo_frame] = {df_sg.iloc[mo_index,:]}")

        print(f"before_opening = y:{ XL_y_seal_SG[0]},z:{ XL_z_seal_SG[0]}")
        print(f"max_opening = y:{ XL_y_seal_SG[mo_index]},z:{ XL_z_seal_SG[mo_index]}")

        print(f"RS xr,xl,y,z = {rs_xr:.1f}, {rs_xl:.1f}, {rs_y:.1f}, {rs_z:.1f}, rs_mo = {rs_mo:.1f}")

        id = os.path.basename(os.path.dirname(dir_path))
        mkg_xr, mkg_xl, mkg_y, mkg_z, mkg_mo = 0, 0, 0, 0, 0
        try:
            mkg_result_path = os.path.join(root_dir,"mkg_result.csv")
            # CSVファイルを読み込みモードで開く
            with open(mkg_result_path, 'r', newline='') as file:
                reader = csv.reader(file)
                mkg_result = [row for row in reader]
                for i in range(0,len(mkg_result)):
                    if mkg_result[i][0] == str(id):
                        mkg_xr = float(mkg_result[i][1])
                        mkg_xl = float(mkg_result[i][2])
                        mkg_y = float(mkg_result[i][3])
                        mkg_z = float(mkg_result[i][4])
                        mkg_mo = float(mkg_result[i][5])
                        print(f"mkg a,b,c,d = {mkg_xr}, {mkg_xl}, {mkg_y}, {mkg_z}, mkg_mo = {mkg_mo:.1f}")
                        print(f"error a,b,c,d = {rs_xr-mkg_xr:.1f}, {rs_xl-mkg_xl:.1f}, {rs_y-mkg_y:.1f}, {rs_z-mkg_z:.1f}, error_mo = {rs_mo-mkg_mo:.1f}")
        except:
            pass

    return df_sg


def MakeGraph(df_sg_blink, *df_sg):  #グラフ作成
    if len(df_sg) != 0:
        df_sg = df_sg[0]
        print(df_sg)
        x,y,z = df_sg['x'], df_sg['y'], df_sg['z']  #まばたき補正前
    x_b,y_b,z_b = df_sg_blink['x'], df_sg_blink['y'], df_sg_blink['z']  #まばたき補正後
    data_num = df_sg_blink.shape[0]
    # 散布図,線を描画
    fig = plt.figure(figsize=(14, 5))

    id = os.path.basename(os.path.dirname(dir_path))
    if id == "20231218_f":
        fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(1,2,2)  #1行2列つくって右に配置

    ax1.set_xlabel('X-axis [mm]',fontsize=20)
    ax1.set_ylabel('Y-axis [mm]',fontsize=20)

    #軸範囲を設定 a([-10,10][-30,5][5,-20])
    axis_range = [[int(min(x_b)-6),int(max(x_b)+6)],[int(min(y_b)-10),int(max(y_b)+5)],[int(min(z_b)-5),int(max(z_b)+5)]]
    ax1.set_xlim(axis_range[0][:])
    ax1.set_ylim(axis_range[1][:])

    ax1.minorticks_on()
    if grid_draw:
        print(f"axis_range = {axis_range}")
        ax1.set_xticks(np.arange(axis_range[0][0]-axis_range[0][0]%5+5, axis_range[0][1], 5))
        ax1.set_xticks(np.arange(axis_range[0][0], axis_range[0][1], 1), minor=True)
        ax1.set_yticks(np.arange(axis_range[1][0]-axis_range[1][0]%5+5, axis_range[1][1], 5))
        ax1.set_yticks(np.arange(axis_range[1][0], axis_range[1][1], 1), minor=True)
        ax1.grid(which = "major", axis = "x", color = "gray", alpha = 1, linestyle = "-", linewidth = 1)
        ax1.grid(which = "major", axis = "y", color = "gray", alpha = 1, linestyle = "-", linewidth = 1)
        ax1.grid(which = "minor", axis = "x", color = "gray", alpha = 0.3, linestyle = "-", linewidth = 1)
        ax1.grid(which = "minor", axis = "y", color = "gray", alpha = 0.3, linestyle = "-", linewidth = 1)
        ax1.set_xticklabels(ax1.get_xticks(), fontsize=15)
        ax1.set_yticklabels(ax1.get_yticks(), fontsize=15)

    if frame_draw:
        #各点にframe番号を表示
        for i in range(data_num):
            ax1.annotate(i, (x_b[i], y_b[i]), fontsize=6, path_effects=[patheffects.withStroke(linewidth=1, foreground="w")])

    #0の位置に線
    ax1.plot([0,0], [axis_range[1][0],axis_range[1][1]], color='black', linewidth = 2)
    ax1.plot([axis_range[0][0],axis_range[0][1]], [0,0], color='black', linewidth = 2)
    ax1.set_aspect('equal', adjustable='box')

    #カラーマップ設定
    cmap = plt.get_cmap('jet')
    normalize = plt.Normalize(1, data_num)
    colors = cmap(normalize(range(1,data_num)))

    # カラーバーを表示
    # cbar = fig.colorbar(ScalarMappable(norm=normalize, cmap=cmap), ax=ax1,ticks=mticker.MultipleLocator(base=30))
    # cbar.ax.tick_params(labelsize=12)
    # cbar.set_label('frame [-]', fontsize=15)

    ax1.plot(x_b[1:], y_b[1:], alpha = 0.3)

    ax1.scatter(x_b[1:], y_b[1:], c="b", s=15, alpha = 0.7)
    if len(df_sg) != 0:
        ax1.plot(x[1:], y[1:], c="r", alpha=0.3)
        ax1.scatter(x[1:], y[1:], c="r", s=15, alpha = 0.3)
    # ax1.scatter(XL_x_seal_SG[1:], XL_y_seal_SG[1:], c=colors, s=15, alpha = 0.7)
    # ax1.scatter(XL_x_seal_SG[0], XL_y_seal_SG[0], c=[(255/255,165/255,0)], s=200, marker="*")
    # ax1.scatter(XL_x_seal_SG[mo_index], XL_y_seal_SG[mo_index], c="b", s=200, marker="*")


    # 散布図を描画
    ax2 = fig.add_subplot(1,2,1)  #1行2列つくって左に配置

    ax2.set_xlabel('Z-axis [mm]',fontsize=20)
    ax2.set_ylabel('Y-axis [mm]',fontsize=20)

    #軸範囲を設定
    ax2.set_xlim(axis_range[2][:])
    ax2.set_ylim(axis_range[1][:])

    ax2.minorticks_on()
    if grid_draw:
        ax2.set_xticks(np.arange(axis_range[2][0]-axis_range[2][0]%5+5, axis_range[2][1], 5))
        ax2.set_xticks(np.arange(axis_range[2][0], axis_range[2][1], 1), minor=True)
        ax2.set_yticks(np.arange(axis_range[1][0]-axis_range[1][0]%5+5, axis_range[1][1], 5))
        ax2.set_yticks(np.arange(axis_range[1][0], axis_range[1][1], 1), minor=True)
        ax2.grid(which = "major", axis = "x", color = "gray", alpha = 1, linestyle = "-", linewidth = 1)
        ax2.grid(which = "major", axis = "y", color = "gray", alpha = 1, linestyle = "-", linewidth = 1)
        ax2.grid(which = "minor", axis = "x", color = "gray", alpha = 0.3, linestyle = "-", linewidth = 1)
        ax2.grid(which = "minor", axis = "y", color = "gray", alpha = 0.3, linestyle = "-", linewidth = 1)
        ax2.set_xticklabels(ax2.get_xticks(), fontsize=15)
        ax2.set_yticklabels(ax2.get_yticks(), fontsize=15)

    if frame_draw:
        #各点にframe番号を表示
        for i in range(data_num):
            ax2.annotate(i, (z_b[i], y_b[i]), fontsize=6, path_effects=[patheffects.withStroke(linewidth=1, foreground="w")])

    ax2.plot([0,0], [axis_range[1][0],axis_range[1][1]], color='black', linewidth = 2)
    ax2.plot([axis_range[2][0],axis_range[2][1]], [0,0], color='black', linewidth = 2)
    ax2.invert_xaxis()
    ax2.set_aspect('equal', adjustable='box')

    # if ear: ax2.scatter(XL_z_seal_SG[blink_frame_npy], XL_y_seal_SG[blink_frame_npy], c="r", s=200, alpha = 1.0)

    ax2.plot(z_b[1:], y_b[1:], color="b",alpha=0.3)
    ax2.scatter(z_b[1:], y_b[1:], c="b", s=15, alpha=0.7)
    if len(df_sg) != 0:
        ax2.plot(z[1:], y[1:], color="r", alpha=0.3)
        ax2.scatter(z[1:], y[1:], c="r", s=15, alpha=0.3)
    # ax2.scatter(XL_z_seal_SG[1:], XL_y_seal_SG[1:], c=colors, s=15, alpha = 0.7)
    # ax2.scatter(XL_z_seal_SG[0], XL_y_seal_SG[0], c=[(255/255,165/255,0)], s=200, marker="*")
    # ax2.scatter(XL_z_seal_SG[mo_index], XL_y_seal_SG[mo_index], c="b", s=200, marker="*")

    plt.tight_layout()  # グラフのレイアウトを調整

    # plt.show()
    if frame_draw and ear:
        png_name = f"frontal&sagittal_frame_ear_theta[{int(np.rad2deg(theta_co_x))}].png"
    if frame_draw and not(ear):
        png_name = f"frontal&sagittal_frame_theta[{int(np.rad2deg(theta_co_x))}].png"
    if not(frame_draw) and ear:
        png_name = f"frontal&sagittal_ear_theta[{int(np.rad2deg(theta_co_x))}].png"
    if not(frame_draw) and not(ear):
        png_name = f"frontal&sagittal_theta[{int(np.rad2deg(theta_co_x))}].png"

    plt.savefig(dir_path+png_name, bbox_inches='tight', transparent=True)
    print(f"fig is saved in frontal&sagittal_theta[{int(np.rad2deg(theta_co_x))}].png")
    plt.show()


    if seal_3dplot:
        #3次元プロット
        fig = plt.gcf()
        plt.close(fig)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X [mm]',fontsize=15)
        ax.set_ylabel('Z [mm]',fontsize=15)
        ax.set_zlabel('Y [mm]',fontsize=15)
        ax.scatter(x_b[1:], z_b[1:], y_b[1:], c=colors, s=15, alpha = 0.7)
        ax.scatter(x_b[0], z_b[0], y_b[0], c=[(255/255,165/255,0)], s=200, marker="*")
        # ax.scatter(x_b[mo_index], z_b[mo_index], y_b[mo_index], c=[(0/255,0/255,255/255)], s=200, marker="*")
        ax.plot(x_b[1:], z_b[1:], y_b[1:], alpha = 0.3)
        #アスペクト比を揃える
        max_range = np.array([x_b.max()-x_b.min(), z_b.max()-z_b.min(), y_b.max()-y_b.min()]).max() / 2.0
        mid_x = (x_b.max()+x_b.min()) * 0.5
        mid_z = (z_b.max()+z_b.min()) * 0.5
        mid_y = (y_b.max()+y_b.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_z - max_range, mid_z + max_range)
        ax.set_zlim(mid_y - max_range, mid_y + max_range)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

