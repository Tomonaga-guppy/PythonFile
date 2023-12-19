
import numpy as np
import cv2
import matplotlib.pyplot as plt


# def save_frames(video_path, output_dir):
#     # 動画を読み込む
#     video = cv2.VideoCapture(video_path)

#     # フレーム数を取得する
#     frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#     # フレームごとに保存する
#     for i in range(frame_count):
#             # フレームを読み込む
#             ret, frame = video.read()

#             # フレームが正常に読み込まれた場合は保存する
#             if ret:
#                 # if i == 0 or i == 410:  # 1フレーム目と410フレーム目を保存
#                 # 画像を保存する範囲を指定
#                 frame = frame[700:1180,600:1080]  # 1920*1080 →480*480に切り取り
#                 frame_path = f"{output_dir}/frame_{i}.jpg"
#                 cv2.imwrite(frame_path, frame)

#     # 動画を解放する
#     video.release()
# # 動画のパスと保存先ディレクトリを指定する
# video_path = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\scale\20231117_b1\b1.mp4"
# output_dir = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\scale\20231117_b1\yoko"
# # フレームを保存する
# save_frames(video_path, output_dir)

#スケール
point1 = np.array([363.8, 329.9])
point2 = np.array([403.6, 292.9])
pixel_norm = np.linalg.norm(point1-point2)
scale = 33.0/pixel_norm
print(f"scale = {scale}")

#顎運動前
mkg1_bo = np.array([436.2, 261.6])  #mkg基準点1
mkg2_bo = np.array([332.2, 264.8])  #mkg基準点2
marker_bo = np.array([351.5, 304.1])  #MKGのマーカー位置
seal_bo = np.array([351.5, 333.4])  #シール位置
point_bo_list = [mkg1_bo, mkg2_bo, marker_bo, seal_bo]

vec_mkg_bo = mkg1_bo - mkg2_bo
vec_z_bo = vec_mkg_bo.copy()
base_z_bo = vec_z_bo/np.linalg.norm(vec_z_bo)
theta_z = np.deg2rad(-90)
rot_z = np.array([[np.cos(theta_z), np.sin(theta_z)], [ -np.sin(theta_z), np.cos(theta_z)]]).T  # y軸作成用回転行列
vec_y_bo = rot_z @ vec_z_bo
base_y_bo = vec_y_bo / np.linalg.norm(vec_y_bo)

Rot_bo = np.array([[base_z_bo[0], base_z_bo[1]], [base_y_bo[0], base_y_bo[1]]]).T

marker_bo_2 = Rot_bo.T @ marker_bo - Rot_bo.T @ mkg2_bo
seal_bo_2 = Rot_bo.T @ seal_bo - Rot_bo.T @ mkg2_bo


#最大開口時
mkg1_ao = np.array([456.8, 237.2])  #mkg基準点1
mkg2_ao = np.array([353.9, 252.6])  #mkg基準点2
marker_ao = np.array([324.7, 342.0])  #MKGのマーカー位置
seal_ao = np.array([313.7, 369.5])  #シール位置
point_ao_list = [mkg1_ao, mkg2_ao, marker_ao, seal_ao]

vec_mkg_ao = mkg1_ao - mkg2_ao
vec_z_ao = vec_mkg_ao.copy()
base_z_ao = vec_z_ao/np.linalg.norm(vec_z_ao)
vec_y_ao = rot_z @ vec_z_ao
base_vec_y_ao = vec_y_ao / np.linalg.norm(vec_y_ao)

Rot_ao = np.array([[base_z_ao[0], base_z_ao[1]], [base_vec_y_ao[0], base_vec_y_ao[1]]]).T

marker_ao_2 = Rot_ao.T @ marker_ao - Rot_ao.T @ mkg2_ao
seal_ao_2 = Rot_ao.T @ seal_ao - Rot_ao.T @ mkg2_ao


marker_bo_2 = marker_bo_2 * scale
seal_bo_2 = seal_bo_2 * scale
marker_ao_2 = marker_ao_2 * scale
seal_ao_2 = seal_ao_2 * scale

marker_disp = marker_ao_2-marker_bo_2
seal_disp = seal_ao_2-seal_bo_2

vec_mkg_norm = np.linalg.norm(vec_z_bo)*scale

print(f"marker_disp = {marker_disp}")
print(f"seal_disp = {seal_disp}")
print(f"diff = {(seal_disp) - (marker_disp)}")

# グラフの設定
plt.figure()
#point_bo_listとpoint\ao_listの点を画像に表示
# 保存した画像をグラフに描画する
image1_path = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\scale\20231117_b\yoko\frame_0.jpg"
image2_path = r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\scale\20231117_b\yoko\frame_400.jpg"

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.scatter(marker_bo[0], marker_bo[1], c='b', s=20)
plt.scatter(seal_bo[0], seal_bo[1], c='r', s=20)
plt.scatter(mkg1_bo[0], mkg1_bo[1], c='w', s=3)
plt.scatter(mkg2_bo[0], mkg2_bo[1], c='w', s=3)
plt.quiver(mkg2_bo[0], mkg2_bo[1], vec_mkg_bo[0], vec_mkg_bo[1], angles='xy', scale_units='xy', scale=1, color='w')
plt.quiver(mkg2_bo[0], mkg2_bo[1], -vec_y_bo[0], -vec_y_bo[1], angles='xy', scale_units='xy', scale=1, color='w')
# plt.scatter(point1[0], point1[1], c='w', s=5)  #スケール用の点
# plt.scatter(point2[0], point2[1], c='w', s=5)
plt.title("before opening")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.scatter(marker_ao[0], marker_ao[1], c='b', s=20)
plt.scatter(seal_ao[0], seal_ao[1], c='r', s=20)
plt.scatter(mkg1_ao[0], mkg1_ao[1], c='w', s=3)
plt.scatter(mkg2_ao[0], mkg2_ao[1], c='w', s=3)
plt.quiver(mkg2_ao[0], mkg2_ao[1], vec_mkg_ao[0], vec_mkg_ao[1], angles='xy', scale_units='xy', scale=1, color='w')
plt.quiver(mkg2_ao[0], mkg2_ao[1], -vec_y_ao[0], -vec_y_ao[1], angles='xy', scale_units='xy', scale=1, color='w')





plt.title("max opening")

plt.savefig(r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\scale\20231117_b\point.png")
plt.show()
fig = plt.gcf()
plt.close(fig)


# ベクトルを描画
plt.quiver(0, 0, vec_mkg_norm, 0, angles='xy', scale_units='xy', scale=1, color='black')
plt.quiver(0, 0, 0, -vec_mkg_norm, angles='xy', scale_units='xy', scale=1, color='black')
plt.scatter(marker_bo_2[0], marker_bo_2[1], c='b', s = 50, alpha=0.5)
plt.scatter(seal_bo_2[0], seal_bo_2[1], c='r',  s=50, alpha=0.5)
plt.scatter(marker_ao_2[0], marker_ao_2[1], c='b', label='marker', s=50)
plt.scatter(seal_ao_2[0], seal_ao_2[1], c='r', label='seal', s=50)
# グラフの軸ラベルと凡例を設定
plt.xlabel('X')
plt.ylabel('Y')
#y軸を反転
plt.gca().invert_yaxis()
plt.legend()
# #関連する点がすべて移るようにグラフの表示範囲を設定
plt.xlim([-50,70])
plt.ylim([-100,20])
#x軸,y軸に少し薄めの黒で線を引く
plt.axhline(0, color='black', alpha=0.4)
plt.axvline(0, color='black', alpha=0.4)
#グリッドを表示
plt.grid()
#アスペクト比を1:1に
plt.gca().set_aspect('equal', adjustable='box')
# グラフを表示
plt.savefig(r"C:\Users\zutom\BRLAB\tooth\Temporomandibular_movement\movie\scale\20231117_b\yokoscale.png")
# plt.show()