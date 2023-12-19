import trimesh
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

root_dir = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\scale"
dir_paths = glob.glob(os.path.join(root_dir,"*d/OpenFace.avi"))

for i,dir_path in enumerate(dir_paths):
    dir_path = os.path.dirname(dir_path)
    # if os.path.isfile(dir_path + r"\ply\random_cloud150.ply") == False or os.path.isfile(dir_path + r"\ply\random_cloud512.ply") == False :
    #     print(f'{dir_path}にはplyファイルがありません.')
    #     continue
    ply_path = dir_path + r"\ply\random_cloud150.ply"
    # ply_path2 = dir_path + r"\ply\random_cloud514.ply"
    # ply_path = dir_path + r"\ply\random_cloud60.ply"
    # ply_path2 = dir_path + r"\ply\random_cloud291.ply"
    print(f"{i+1}/{len(dir_paths)}: {dir_path}")
    mesh = trimesh.load_mesh(ply_path)  # PLYファイルを読み込む
    # mesh2 = trimesh.load_mesh(ply_path2)  # PLYファイルを読み込む
    colors = np.array(mesh.visual.vertex_colors)/255  #meshから色の情報を取得
    # colors2 = np.array(mesh2.visual.vertex_colors)/255  #meshから色の情報を取得
    # 読み込んだ点群を色付きでmatplotlibで可視化
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.scatter(mesh.vertices[:,2], mesh.vertices[:,1], c=colors, s=1)
    ax.set_xlabel('Z-axis', fontsize=15)
    ax.set_ylabel('Y-axis', fontsize=15)
    #グリッドを表示
    ax.grid()
    #x,y軸に少し薄めの黒で線を引く
    ax.axhline(0, color='black', alpha=0.4)
    ax.axvline(0, color='black', alpha=0.4)
    #描画範囲設定
    ax.set_xlim(-100,100)
    ax.set_ylim(-150,50)
    #アスペクト比を1:1に
    ax.set_aspect('equal', adjustable='box')


    # ax2 = fig.add_subplot(1,2,2)
    # ax2.scatter(mesh2.vertices[:,2], mesh2.vertices[:,1], c=colors2, s=1)
    # ax2.set_xlabel('Z-axis', fontsize=15)
    # ax2.set_ylabel('Y-axis', fontsize=15)
    # #グリッドを表示
    # ax2.grid()
    # #x,y軸に少し薄めの黒で線を引く
    # ax2.axhline(0, color='black', alpha=0.4)
    # ax2.axvline(0, color='black', alpha=0.4)
    # #描画範囲設定
    # ax2.set_xlim(-100,100)
    # ax2.set_ylim(-150,50)
    # #アスペクト比を1:1に
    # ax2.set_aspect('equal', adjustable='box')

    # ply_face_path = dir_path + r"\ply\face_cloud150.ply"
    # ply_face_path2 = dir_path + r"\ply\face_cloud514.ply"
    # # ply_face_path = dir_path + r"\ply\face_cloud60.ply"
    # # ply_face_path2 = dir_path + r"\ply\face_cloud291.ply"
    # mesh_face = trimesh.load_mesh(ply_face_path)  # PLYファイルを読み込む
    # mesh_face2 = trimesh.load_mesh(ply_face_path2)  # PLYファイルを読み込む
    # print(f"before_open_seal_x,y,z= {mesh_face[68,:]}")
    # print(f"open_seal_x,y,z= {mesh_face2[68,:]}")
    # seal_disp_z = mesh_face2[68,2] - mesh_face[68,2]
    # seal_disp_y = mesh_face2[68,1] - mesh_face[68,1]
    # print(f"seal_disp_z = {seal_disp_z}, seal_disp_y = {seal_disp_y}")

    # plt.savefig(dir_path + "/scale.png")
    # plt.tight_layout()
    plt.show()