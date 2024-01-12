import trimesh
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_12_20"
dir_paths = glob.glob(os.path.join(root_dir,"*f/OpenFace.avi"))

landmark_plot = True  #Trueでランドマークをプロット
caliblation_time_list = [2,5,5,4,4,5]  #a-fまでのキャリブレーション時間

for i,dir_path in enumerate(dir_paths):
    dir_path = os.path.dirname(dir_path)
    # if os.path.isfile(dir_path + r"\ply\random_cloud150.ply") == False or os.path.isfile(dir_path + r"\ply\random_cloud512.ply") == False :
    #     print(f'{dir_path}にはplyファイルがありません.')
    #     continue
    if landmark_plot == True:
        id = os.path.basename(dir_path)
        print(f"id = {id}")
        if id == "20231117_a": caliblation_time = caliblation_time_list[0]
        if id == "20231117_b": caliblation_time = caliblation_time_list[1]
        if id == "20231218_c": caliblation_time = caliblation_time_list[2]
        if id == "20231218_d": caliblation_time = caliblation_time_list[3]
        if id == "20231218_e": caliblation_time = caliblation_time_list[4]
        if id == "20231218_f": caliblation_time = caliblation_time_list[5]
        landmark_path = dir_path + r"\landmark_local.npy"
        landmark = np.load(landmark_path,allow_pickle=True)   #dataは[axis][frame][id]の順で並んでいる

    frame1 = 150
    frame2 = 457
    ply_path = dir_path + r"\ply\random_cloud"+ str(frame1) +".ply"
    ply_path2 = dir_path + r"\ply\random_cloud" + str(frame2) + ".ply"
    # ply_path = dir_path + r"\ply\random_cloud150.ply"
    # ply_path2 = dir_path + r"\ply\random_cloud457.ply"
    print(f"{i+1}/{len(dir_paths)}: {dir_path}")
    mesh = trimesh.load_mesh(ply_path)  # PLYファイルを読み込む
    mesh2 = trimesh.load_mesh(ply_path2)  # PLYファイルを読み込む
    colors = np.array(mesh.visual.vertex_colors)/255  #meshから色の情報を取得
    colors2 = np.array(mesh2.visual.vertex_colors)/255  #meshから色の情報を取得
    # 読み込んだ点群を色付きでmatplotlibで可視化
    fig = plt.figure()
    xlim = [-150,50]
    ylim = [-150,50]
    ax = fig.add_subplot(1,2,1)
    ax.scatter(mesh.vertices[:,2], mesh.vertices[:,1], c=colors, s=1)
    if landmark_plot:
        ax.scatter(landmark[2,frame1-caliblation_time*30,30], landmark[1,frame1-caliblation_time*30,30], c='r', s=9)
    ax.set_xlabel('Z-axis', fontsize=15)
    ax.set_ylabel('Y-axis', fontsize=15)
    #グリッドを表示
    ax.grid()
    #x,y軸に少し薄めの黒で線を引く
    ax.axhline(0, color='black', alpha=0.4)
    ax.axvline(0, color='black', alpha=0.4)
    #描画範囲設定
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    #アスペクト比を1:1に
    ax.set_aspect('equal', adjustable='box')


    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(mesh2.vertices[:,2], mesh2.vertices[:,1], c=colors2, s=1)
    if landmark_plot:
        ax2.scatter(landmark[2,frame2-caliblation_time*30,30], landmark[1,frame2-caliblation_time*30,30], c='r', s=9)
    ax2.set_xlabel('Z-axis', fontsize=15)
    ax2.set_ylabel('Y-axis', fontsize=15)
    #グリッドを表示
    ax2.grid()
    #x,y軸に少し薄めの黒で線を引く
    ax2.axhline(0, color='black', alpha=0.4)
    ax2.axvline(0, color='black', alpha=0.4)
    #描画範囲設定
    ax2.set_xlim(xlim[0],xlim[1])
    ax2.set_ylim(ylim[0],ylim[1])
    #アスペクト比を1:1に
    ax2.set_aspect('equal', adjustable='box')

    ply_face_path = dir_path + r"\ply\face_cloud" + str(frame1) + ".ply"
    ply_face_path2 = dir_path + r"\ply\face_cloud" + str(frame2) + ".ply"
    mesh_face = trimesh.load_mesh(ply_face_path)  # PLYファイルを読み込む
    mesh_face2 = trimesh.load_mesh(ply_face_path2)  # PLYファイルを読み込む
    print(f"before_open_seal_x,y,z= {mesh_face[68,:]}")
    print(f"open_seal_x,y,z= {mesh_face2[68,:]}")
    seal_disp_z = mesh_face2[68,2] - mesh_face[68,2]
    seal_disp_y = mesh_face2[68,1] - mesh_face[68,1]
    print(f"seal_disp_z = {seal_disp_z}, seal_disp_y = {seal_disp_y}")

    plt.savefig(dir_path + "/sagittal.png")
    # plt.tight_layout()
    plt.show()

    # #2枚を重ねて表示
    # plt.close(fig)
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.scatter(mesh.vertices[:,2], mesh.vertices[:,1], c=colors, s=1)
    # ax.scatter(mesh2.vertices[:,2], mesh2.vertices[:,1], c=colors2, s=1)
    # ax.set_xlabel('Z-axis', fontsize=15)
    # ax.set_ylabel('Y-axis', fontsize=15)
    # #グリッドを表示
    # ax.grid()
    # #x,y軸に少し薄めの黒で線を引く
    # ax.axhline(0, color='black', alpha=0.4)
    # ax.axvline(0, color='black', alpha=0.4)
    # #描画範囲設定
    # ax.set_xlim(xlim[0],xlim[1])
    # ax.set_ylim(ylim[0],ylim[1])
    # #アスペクト比を1:1に
    # ax.set_aspect('equal', adjustable='box')
    # plt.show()