import trimesh
import matplotlib.pyplot as plt

# ply_path1 = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\2023_11_17\20231117_b1\ply\face_cloud30.ply"
# # ply_path2 = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\2023_11_17\20231117_b1\plycam\face_cloud30.ply"
# # PLYファイルを読み込む
# mesh1 = trimesh.load_mesh(ply_path1)
# # mesh2 = trimesh.load_mesh(ply_path2)
# # min_y_row = mesh.vertices[mesh.vertices[:, 1].argmin()]

# # 読み込んだ点群を色付きでmatplotlibで可視化
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(mesh1.vertices[:,0], mesh1.vertices[:,1], mesh1.vertices[:,2], c='r', s=1, label='mesh')
# ax.scatter(0,0,0, c='b', s=10)
# # ax.scatter(mesh2.vertices[:,0], mesh2.vertices[:,1], mesh2.vertices[:,2], c='b', s=1, label='cam_mesh')
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# plt.legend()
# plt.show()



ply_pathp = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\2023_09_000\20230606_J2\ply\random_cloud30_p.ply"
ply_path0 = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\2023_09_000\20230606_J2\ply\random_cloud30_0.ply"
ply_path68_0 = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\2023_09_000\20230606_J2\ply\face_cloud30_0.ply"
ply_pathm = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\2023_09_000\20230606_J2\ply\random_cloud30_m.ply"
meshp = trimesh.load_mesh(ply_pathp)
mesh0 = trimesh.load_mesh(ply_path0)
mesh68_0 = trimesh.load_mesh(ply_path68_0)
meshm = trimesh.load_mesh(ply_pathm)
colorsp = meshp.visual.vertex_colors
colors0 = mesh0.visual.vertex_colors
colorsm = meshm.visual.vertex_colors
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(meshp.vertices[:,2], meshp.vertices[:,1], c=colorsp/255, s=0.1, alpha=0.1)  
ax.scatter(mesh0.vertices[:,2], mesh0.vertices[:,1], c=colors0/255, s=0.1)  
ax.scatter(meshm.vertices[:,2], meshm.vertices[:,1], c=colorsm/255, s=0.1, alpha=0.1)  
ax.scatter
lim = (-150,150)
ax.set_aspect('equal')
ax.set_xlim(lim)
ax.set_ylim(-150,150)
ax.set_xlabel('Z-axis')
ax.set_ylabel('Y-axis')
ax.invert_xaxis()
plt.legend()
plt.show()