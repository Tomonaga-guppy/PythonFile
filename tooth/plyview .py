import trimesh
import matplotlib.pyplot as plt

ply_path1 = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\2023_11_17\20231117_b1\ply\face_cloud30.ply"
# ply_path2 = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\2023_11_17\20231117_b1\plycam\face_cloud30.ply"
# PLYファイルを読み込む
mesh1 = trimesh.load_mesh(ply_path1)
# mesh2 = trimesh.load_mesh(ply_path2)
# min_y_row = mesh.vertices[mesh.vertices[:, 1].argmin()]

# 読み込んだ点群を色付きでmatplotlibで可視化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mesh1.vertices[:,0], mesh1.vertices[:,1], mesh1.vertices[:,2], c='r', s=1, label='mesh')  
ax.scatter(0,0,0, c='b', s=10)  
# ax.scatter(mesh2.vertices[:,0], mesh2.vertices[:,1], mesh2.vertices[:,2], c='b', s=1, label='cam_mesh') 
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.legend()
plt.show()



# pixel_ply_path = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\2023_11_17\20231117_b1\plycam\pixel_cloud60.ply"
# pixel_mesh = trimesh.load_mesh(ply_path1)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pixel_mesh.vertices[:,0], pixel_mesh.vertices[:,1], pixel_mesh.vertices[:,2], c='r', s=0.1)  # プロット点を小さくして
# # ax.scatter(mesh2.vertices[:,0], mesh2.vertices[:,1], mesh2.vertices[:,2], c='b', s=0.1, label='mesh2cam')  # プロット点を小さくして
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# plt.legend()
# plt.show()