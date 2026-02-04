"""
ViTPoseに対応するようにMocapデータを解析して比較ようデータを作成する。
歩行イベントはmocapデータから算出し，ViTPoseの歩行周期に対応する範囲で歩行指標を算出．
"""

import pandas as pd
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

import json
import m_opti as opti


# 同期用
def check_imu_sync_frame_diff(imu_path):
    """
    imuからgoproとmocapの撮影開始フレームを取得する
    """
    sync_imu_df = pd.read_csv(imu_path, sep=",", header=None)
    # ext data 行のみ抽出
    ext_df = sync_imu_df[sync_imu_df[0] == "ext data"].reset_index(drop=True)

    # 念のため int 化
    ext_df[2] = ext_df[2].astype(int)
    ext_df[3] = ext_df[3].astype(int)

    # 変化点検出
    # 2列目 LED: 1 → 0
    col2_fall = ext_df.index[
        (ext_df[2].shift(1) == 1) & (ext_df[2] == 0)
    ]
    # 3列目 Mocap: 0 → 1
    col3_rise = ext_df.index[
        (ext_df[3].shift(1) == 0) & (ext_df[3] == 1)
    ]

    # フレームずれ量の計算
    if len(col2_fall) > 0 and len(col3_rise) > 0:
        # 最初のイベント同士で比較
        sync_frame_diff_100hz = (col3_rise[0] - col2_fall[0])
        sync_frame_diff_60hz = int(round(sync_frame_diff_100hz * (60/100))) # 60Hzに変換
    else:
        print("変化点が検出できませんでした")
        sync_frame_diff_60hz = None
    return sync_frame_diff_60hz



# =========================
# Plot settings (change here if needed)
# =========================
# 角度プロットのY範囲（関節・自由度ごと）
# ※4_1 の計算結果（CSV/npz）には影響しません。プロットの見た目のみ。
YLIM_BY_ANGLE = {
    # Hip
    "Hip_FlEx":  (-40,  50),
    "Hip_AdAb":  (-30,  30),
    "Hip_InEx":  (-30,  30),

    # Knee
    "Knee_FlEx": (-10,  75),
    "Knee_AdAb": (-30,  30),
    "Knee_InEx": (-30,  30),

    # Ankle (Plantar/Dorsi)
    "Ankle_PlDo": (-40,  40),
    "Ankle_AdAb": (-30,  30),
    "Ankle_InEx": (-30,  30),
}

def ylims_for_angle_key(k: str):
    """
    例: 'R_Hip_FlEx' -> (-40, 50)
    """
    if not isinstance(k, str) or "_" not in k:
        return None
    # R_ / L_ を落とす
    base = k.split("_", 1)[1]
    return YLIM_BY_ANGLE.get(base, None)



def process_one(csv_path_dir: Path, frame_diff, gopro_gait_cycle):
    """
    subX/theraY-0/mocap を1つ処理
    """
    # goproでの歩行周期(60hz)を取得
    gopro_gait_cycles_r = gopro_gait_cycle["gait_cycle_r"]
    gopro_gait_cycles_l = gopro_gait_cycle["gait_cycle_l"]
    print(f"gopro_gait_cycles_r: {gopro_gait_cycles_r}")
    print(f"gopro_gait_cycles_l: {gopro_gait_cycles_l}")
    
    # mocapで対応する解析開始フレームおよび歩行周期(60hz)を算出
    def _cycles_gopro_to_mocap(cycles_gopro, offset):
        """
        mocapで対応する歩行周期(60hz)を算出
        cycles: [[ic, ic_opp, to, ic_end], ...]
        """
        out = []
        for c in cycles_gopro:
            if c is None or len(c) != 4:
                continue
            out.append([int((int(c[0]) + offset)),
                        int((int(c[1]) + offset)),
                        int((int(c[2]) + offset)),
                        int((int(c[3]) + offset))])
        return out
    
    gait_cycles_r_check = _cycles_gopro_to_mocap(gopro_gait_cycles_r, frame_diff)
    gait_cycles_l_check = _cycles_gopro_to_mocap(gopro_gait_cycles_l, frame_diff)
    
    print(f"mocap gait_cycles_r_check: {gait_cycles_r_check}")
    print(f"mocap gait_cycles_l_check: {gait_cycles_l_check}")
    
    all_frames_r = [frame for cycle in gait_cycles_r_check for frame in cycle if frame is not None]
    all_frames_l = [frame for cycle in gait_cycles_l_check for frame in cycle if frame is not None]
    start_frame = min(all_frames_r + all_frames_l) - 30  # 前後に余裕を持たせる
    end_frame = max(all_frames_r + all_frames_l) + 30
    
    csv_path = next(csv_path_dir.glob("[0-9]*-[0-9]*-[0-9]*.csv"), None)
    if csv_path is None:
        print(f"No CSV file found in {csv_path_dir}")
        return

    print(f"Processing: {csv_path}")

    try:
        keypoints_mocap, full_range, abs_frames = opti.read_3d_optitrack(csv_path, start_frame, end_frame)
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return
    print(f"abs_frames: {abs_frames[0] - abs_frames[-1]}")
    
    # 絶対フレーム番号から相対フレーム番号への変換対応
    abs_to_rel = {int(a): i for i, a in enumerate(abs_frames)}

    if keypoints_mocap.size == 0:
        print(f"Skipping {csv_path}: No valid data")
        return
    
    print(f"csv_path = {csv_path}")
    print(f"keypoints_mocap shape: {keypoints_mocap.shape}")

    # バターワースフィルタのサンプリング周波数を動的に設定
    rasi = np.array([keypoints_mocap[:, 10, x] for x in range(3)]).T
    lasi = np.array([keypoints_mocap[:, 2, x] for x in range(3)]).T
    rpsi = np.array([keypoints_mocap[:, 14, x] for x in range(3)]).T
    lpsi = np.array([keypoints_mocap[:, 6, x] for x in range(3)]).T
    rank = np.array([keypoints_mocap[:, 8, x] for x in range(3)]).T
    lank = np.array([keypoints_mocap[:, 0, x] for x in range(3)]).T
    rank2 = np.array([keypoints_mocap[:, 9, x] for x in range(3)]).T
    lank2 = np.array([keypoints_mocap[:, 1, x] for x in range(3)]).T
    rknee = np.array([keypoints_mocap[:, 12, x] for x in range(3)]).T
    lknee = np.array([keypoints_mocap[:, 4, x] for x in range(3)]).T
    rknee2 = np.array([keypoints_mocap[:, 13, x] for x in range(3)]).T
    lknee2 = np.array([keypoints_mocap[:, 5, x] for x in range(3)]).T
    rtoe = np.array([keypoints_mocap[:, 15, x] for x in range(3)]).T
    ltoe = np.array([keypoints_mocap[:, 7, x] for x in range(3)]).T
    rhee = np.array([keypoints_mocap[:, 11, x] for x in range(3)]).T
    lhee = np.array([keypoints_mocap[:, 3, x] for x in range(3)]).T

    ######################################################
    # 完全におまけ 骨盤座標の確認
    rasi_test = rasi.copy()
    lasi_test = lasi.copy()
    rpsi_test = rpsi.copy()
    lpsi_test = lpsi.copy()
    
    plot_target = {"RASI": rasi_test, "LASI": lasi_test, "RPSI": rpsi_test, "LPSI": lpsi_test}

    fig = plt.figure(figsize=(15, 12))
    plt.suptitle(f"Pelvis Markers Timeseries: {csv_path.stem}", fontsize=16)
    full_range_abs = [f + start_frame for f in full_range]  #元の絶対フレーム番号に変換
    # print(f"full_range_abs(絶対フレーム番号): {full_range_abs}")

    # 各マーカーのXYZ座標を時系列プロット
    for idx, (name, data) in enumerate(plot_target.items(), 1):
        # X座標
        ax1 = fig.add_subplot(4, 3, idx * 3 - 2)
        ax1.plot(full_range_abs, data[:, 0], label=f"{name} X", color='r')
        ax1.set_ylabel('X [m]')
        ax1.set_title(f"{name} X coordinate")
        ax1.grid(True)
        ax1.legend()
        
        # Y座標
        ax2 = fig.add_subplot(4, 3, idx * 3 - 1)
        ax2.plot(full_range_abs, data[:, 1], label=f"{name} Y", color='g')
        ax2.set_ylabel('Y [m]')
        ax2.set_title(f"{name} Y coordinate")
        ax2.grid(True)
        ax2.legend()
        
        # Z座標
        ax3 = fig.add_subplot(4, 3, idx * 3)
        ax3.plot(full_range_abs, data[:, 2], label=f"{name} Z", color='b')
        ax3.set_ylabel('Z [m]')
        ax3.set_title(f"{name} Z coordinate")
        ax3.grid(True)
        ax3.legend()

    plt.tight_layout()
    plt.savefig(csv_path_dir / f"pelvis_markers_timeseries_{csv_path.stem}.png")
    plt.close()
    ######################################################
    
    angle_list = []
    sac2hee_r_list = []
    sac2hee_l_list = []
    sac2toe_r_list = []
    sac2toe_l_list = []
    heel_z_list = []
    toe_z_list = []


    full_range_abs = [f + start_frame for f in full_range]  #元の絶対フレーム番号に変換
    
    # full_range = range(1, len(rasi))  #差分取るために0からではなく1フレーム目からにする
    print(f"full_range(開始点は1フレーム後から): {full_range}")

    hip_list = []
    angle_list = []

    for frame_num in full_range:
        #メモ
        d_asi = np.linalg.norm(rasi[frame_num,:] - lasi[frame_num,:])
        d_leg = (np.linalg.norm(rank[frame_num,:] - rasi[frame_num,:]) + np.linalg.norm(lank[frame_num, :] - lasi[frame_num,:])) / 2
        r = 0.0127 #[m] Opti確認：https://www.optitrack.jp/products/accessories/marker.html
        h = 1.7 #[m]
        k = h/1.7
        beta = 0.1 * np.pi #[rad]
        theta = 0.496 #[rad]
        c = 0.115 * d_leg - 0.0153  #SKYCOMだと0.00153だけどDavisモデルは0.0153  https://wiki.has-motion.com/doku.php?id=visual3d:documentation:modeling:segments:hip_joint_landmarks
        x_dis = 0.1288 * d_leg - 0.04856

        """
        変更後
        """
        # skycom + davis
        x_rthigh = -(x_dis +r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
        x_lthigh = -(x_dis +r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
        y_rthigh = +(c * np.sin(theta) - d_asi/2)
        y_lthigh = -(c * np.sin(theta)- d_asi/2)
        z_rthigh = -(x_dis + r) * np.sin(beta) - c * np.cos(theta) * np.cos(beta)
        z_lthigh = -(x_dis + r) * np.sin(beta) - c * np.cos(theta) * np.cos(beta)
        rthigh_pelvis = np.array([x_rthigh, y_rthigh, z_rthigh]).T
        lthigh_pelvis = np.array([x_lthigh, y_lthigh, z_lthigh]).T

        # 骨盤原点1 ASISの中点
        hip_0 = (rasi[frame_num,:] + lasi[frame_num,:]) / 2
        # 仙骨 PSISの中点
        sacrum = (rpsi[frame_num,:] + lpsi[frame_num,:]) / 2

        #骨盤節座標系1（原点はhip_0）
        e_x0_pelvis_0 = (hip_0 - sacrum)/np.linalg.norm(hip_0 - sacrum)
        e_y_pelvis_0 = (lasi[frame_num,:] - rasi[frame_num,:])/np.linalg.norm(lasi[frame_num,:] - rasi[frame_num,:])
        e_z_pelvis_0 = np.cross(e_x0_pelvis_0, e_y_pelvis_0)/np.linalg.norm(np.cross(e_x0_pelvis_0, e_y_pelvis_0))
        e_x_pelvis_0 = np.cross(e_y_pelvis_0, e_z_pelvis_0)
        
        # Davidモデルを参考に骨盤座標系1を骨盤の座標系として定義
        e_x_pelvis = e_x_pelvis_0
        e_y_pelvis = e_y_pelvis_0
        e_z_pelvis = e_z_pelvis_0
        rot_pelvis = np.array([e_x_pelvis, e_y_pelvis, e_z_pelvis]).T


                    
        transformation_matrix = np.array([[e_x_pelvis_0[0], e_y_pelvis_0[0], e_z_pelvis_0[0], hip_0[0]],
                                            [e_x_pelvis_0[1], e_y_pelvis_0[1], e_z_pelvis_0[1], hip_0[1]],
                                            [e_x_pelvis_0[2], e_y_pelvis_0[2], e_z_pelvis_0[2], hip_0[2]],
                                            [0,       0,       0,       1]])

        #グローバル座標に変換して再度計算
        rthigh = np.dot(transformation_matrix, np.append(rthigh_pelvis, 1))[:3]
        lthigh = np.dot(transformation_matrix, np.append(lthigh_pelvis, 1))[:3]
        hip = (rthigh + lthigh) / 2

        hip_list.append(hip)
        hip_array = np.array(hip_list)

        #必要な原点の設定
        rshank = (rknee[frame_num, :] + rknee2[frame_num, :]) / 2
        lshank = (lknee[frame_num, :] + lknee2[frame_num, :]) / 2
        rfoot = (rank[frame_num,:] + rank2[frame_num,:]) / 2
        lfoot = (lank[frame_num, :] + lank2[frame_num,:]) / 2

        #右大腿節座標系（原点はrthigh）
        e_y0_rthigh = rknee2[frame_num, :] - rknee[frame_num, :]
        e_z_rthigh = (rshank - rthigh)/np.linalg.norm(rshank - rthigh)
        e_x_rthigh = np.cross(e_y0_rthigh, e_z_rthigh)/np.linalg.norm(np.cross(e_y0_rthigh, e_z_rthigh))
        e_y_rthigh = np.cross(e_z_rthigh, e_x_rthigh)
        rot_rthigh = np.array([e_x_rthigh, e_y_rthigh, e_z_rthigh]).T

        #左大腿節座標系（原点はlthigh）
        e_y0_lthigh = lknee[frame_num, :] - lknee2[frame_num, :]
        e_z_lthigh = (lshank - lthigh)/np.linalg.norm(lshank - lthigh)
        e_x_lthigh = np.cross(e_y0_lthigh, e_z_lthigh)/np.linalg.norm(np.cross(e_y0_lthigh, e_z_lthigh))
        e_y_lthigh = np.cross(e_z_lthigh, e_x_lthigh)
        rot_lthigh = np.array([e_x_lthigh, e_y_lthigh, e_z_lthigh]).T

        #右下腿節座標系（原点はrshank）
        e_y0_rshank = rknee2[frame_num, :] - rknee[frame_num, :]
        e_z_rshank = (rfoot - rshank)/np.linalg.norm(rfoot - rshank)
        e_x_rshank = np.cross(e_y0_rshank, e_z_rshank)/np.linalg.norm(np.cross(e_y0_rshank, e_z_rshank))
        e_y_rshank = np.cross(e_z_rshank, e_x_rshank)
        rot_rshank = np.array([e_x_rshank, e_y_rshank, e_z_rshank]).T

        #左下腿節座標系（原点はlshank）
        e_y0_lshank = lknee[frame_num, :] - lknee2[frame_num, :]
        e_z_lshank = (lfoot - lshank)/np.linalg.norm(lfoot - lshank)
        e_x_lshank = np.cross(e_y0_lshank, e_z_lshank)/np.linalg.norm(np.cross(e_y0_lshank, e_z_lshank))
        e_y_lshank = np.cross(e_z_lshank, e_x_lshank)
        rot_lshank = np.array([e_x_lshank, e_y_lshank, e_z_lshank]).T

        #右足節座標系 AIST参照（原点はrfoot）
        e_x_rfoot = (rtoe[frame_num,:] - rhee[frame_num,:]) / np.linalg.norm(rtoe[frame_num,:] - rhee[frame_num,:])
        e_y0_rfoot = rank2[frame_num,:] - rank[frame_num,:]
        e_z_rfoot = np.cross(e_x_rfoot, e_y0_rfoot)/np.linalg.norm(np.cross(e_x_rfoot, e_y0_rfoot))
        e_y_rfoot = np.cross(e_z_rfoot, e_x_rfoot)
        rot_rfoot = np.array([e_x_rfoot, e_y_rfoot, e_z_rfoot]).T
        

        #左足節座標系 AIST参照（原点はlfoot）
        e_x_lfoot = (ltoe[frame_num,:] - lhee[frame_num, :]) / np.linalg.norm(ltoe[frame_num,:] - lhee[frame_num, :])
        e_y0_lfoot = lank[frame_num,:] - lank2[frame_num,:]
        e_z_lfoot = np.cross(e_x_lfoot, e_y0_lfoot)/np.linalg.norm(np.cross(e_x_lfoot, e_y0_lfoot))
        e_y_lfoot = np.cross(e_z_lfoot, e_x_lfoot)
        rot_lfoot = np.array([e_x_lfoot, e_y_lfoot, e_z_lfoot]).T

        # 相対回転行列の計算
        r_hip_realative_rotation = np.dot(np.linalg.inv(rot_rthigh), rot_pelvis)  #骨盤節に合わせるための大腿節の回転行列
        l_hip_realative_rotation = np.dot(np.linalg.inv(rot_lthigh), rot_pelvis)
        r_knee_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rthigh)  #大腿節に合わせるための下腿節の回転行列
        l_knee_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lthigh)
        r_ankle_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rfoot)  #足節に合わせるための下腿節の回転行列
        l_ankle_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lfoot)

        r_hip_angle_rot = R.from_matrix(r_hip_realative_rotation)
        l_hip_angle_rot = R.from_matrix(l_hip_realative_rotation)
        r_knee_angle_rot = R.from_matrix(r_knee_realative_rotation)
        l_knee_angle_rot = R.from_matrix(l_knee_realative_rotation)
        r_ankle_angle_rot = R.from_matrix(r_ankle_realative_rotation)
        l_ankle_angle_rot = R.from_matrix(l_ankle_realative_rotation)

        # 回転行列から回転角を計算 XYZ大文字だと内因性，xyz小文字だと外因性
        # 屈曲-伸展
        r_hip_angle_flex = r_hip_angle_rot.as_euler('YZX', degrees=True)[0]
        l_hip_angle_flex = l_hip_angle_rot.as_euler('YZX', degrees=True)[0]
        r_knee_angle_flex = r_knee_angle_rot.as_euler('YZX', degrees=True)[0]
        l_knee_angle_flex = l_knee_angle_rot.as_euler('YZX', degrees=True)[0]
        r_ankle_angle_pldo = r_ankle_angle_rot.as_euler('YZX', degrees=True)[0]
        l_ankle_angle_pldo = l_ankle_angle_rot.as_euler('YZX', degrees=True)[0]
        
        # 内旋外旋
        r_hip_angle_inex = r_hip_angle_rot.as_euler('YZX', degrees=True)[1]
        l_hip_angle_inex = l_hip_angle_rot.as_euler('YZX', degrees=True)[1]
        r_knee_angle_inex = r_knee_angle_rot.as_euler('YZX', degrees=True)[1]
        l_knee_angle_inex = l_knee_angle_rot.as_euler('YZX', degrees=True)[1]
        r_ankle_angle_inex = r_ankle_angle_rot.as_euler('YZX', degrees=True)[1]
        l_ankle_angle_inex = l_ankle_angle_rot.as_euler('YZX', degrees=True)[1]

        # 内転外転
        r_hip_angle_adab = r_hip_angle_rot.as_euler('YZX', degrees=True)[2]
        l_hip_angle_adab = l_hip_angle_rot.as_euler('YZX', degrees=True)[2]
        r_knee_angle_adab = r_knee_angle_rot.as_euler('YZX', degrees=True)[2]
        l_knee_angle_adab = l_knee_angle_rot.as_euler('YZX', degrees=True)[2]
        r_ankle_angle_adab = r_ankle_angle_rot.as_euler('YZX', degrees=True)[2]
        l_ankle_angle_adab = l_ankle_angle_rot.as_euler('YZX', degrees=True)[2]

        angle_list.append([r_hip_angle_flex, l_hip_angle_flex, r_knee_angle_flex, l_knee_angle_flex, r_ankle_angle_pldo, l_ankle_angle_pldo,
                                r_hip_angle_inex, l_hip_angle_inex, r_knee_angle_inex, l_knee_angle_inex, r_ankle_angle_inex, l_ankle_angle_inex,
                                r_hip_angle_adab, l_hip_angle_adab, r_knee_angle_adab, l_knee_angle_adab, r_ankle_angle_adab, l_ankle_angle_adab])

        plot_flag = False
        if plot_flag:
            # print(frame_num)  #相対フレーム数
            if frame_num == 98:
                fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1, 2)
                ax.set_zlim(-2, 1)
                #frame数を表示
                ax.text2D(0.5, 0.01, f"frame = {frame_num}", transform=ax.transAxes)
                #方向を設定
                ax.view_init(elev=0, azim=0)

                ax.scatter(rasi[frame_num,:][0], rasi[frame_num,:][1], rasi[frame_num,:][2], color='black', s=5)
                ax.scatter(lasi[frame_num,:][0], lasi[frame_num,:][1], lasi[frame_num,:][2], color='black', s=5)
                ax.scatter(rpsi[frame_num,:][0], rpsi[frame_num,:][1], rpsi[frame_num,:][2], color='black', s=5)
                ax.scatter(lpsi[frame_num,:][0], lpsi[frame_num,:][1], lpsi[frame_num,:][2], color='black', s=5)
                ax.scatter(rank[frame_num,:][0], rank[frame_num,:][1], rank[frame_num,:][2], color='black', s=5)
                ax.scatter(lank[frame_num,:][0], lank[frame_num,:][1], lank[frame_num,:][2], color='black', s=5)
                ax.scatter(rank2[frame_num,:][0], rank2[frame_num,:][1], rank2[frame_num,:][2], color='black', s=5)
                ax.scatter(lank2[frame_num,:][0], lank2[frame_num,:][1], lank2[frame_num,:][2], color='black', s=5)
                ax.scatter(rknee[frame_num,:][0], rknee[frame_num,:][1], rknee[frame_num,:][2], color='black', s=5)
                ax.scatter(lknee[frame_num,:][0], lknee[frame_num,:][1], lknee[frame_num,:][2], color='black', s=5)
                ax.scatter(rknee2[frame_num,:][0], rknee2[frame_num,:][1], rknee2[frame_num,:][2], color='black', s=5)
                ax.scatter(lknee2[frame_num,:][0], lknee2[frame_num,:][1], lknee2[frame_num,:][2], color='black', s=5)
                ax.scatter(rtoe[frame_num,:][0], rtoe[frame_num,:][1], rtoe[frame_num,:][2], color='black', s=5)
                ax.scatter(ltoe[frame_num,:][0], ltoe[frame_num,:][1], ltoe[frame_num,:][2], color='black', s=5)
                ax.scatter(rhee[frame_num,:][0], rhee[frame_num,:][1], rhee[frame_num,:][2], color='black', s=5)
                ax.scatter(lhee[frame_num, :][0], lhee[frame_num, :][1], lhee[frame_num, :][2], color='black', s=5)
                
                ax.scatter(rfoot[0], rfoot[1], rfoot[2], label='rfoot')
                ax.scatter(lfoot[0], lfoot[1], lfoot[2], label='lfoot')
                ax.scatter(rshank[0], rshank[1], rshank[2], label='rshank')
                ax.scatter(lshank[0], lshank[1], lshank[2], label='lshank')
                # ax.scatter(lumbar[0], lumbar[1], lumbar[2], label='lumbar')
                ax.scatter(hip[0], hip[1], hip[2], label='hip')
                ax.scatter(rthigh[0], rthigh[1], rthigh[2], label='rthigh')
                ax.scatter(lthigh[0], lthigh[1], lthigh[2], label='lthigh')

                e_x_pelvis = e_x_pelvis * 0.1
                e_y_pelvis = e_y_pelvis * 0.1
                e_z_pelvis = e_z_pelvis * 0.1
                e_x_rthigh = e_x_rthigh * 0.1
                e_y_rthigh = e_y_rthigh * 0.1
                e_z_rthigh = e_z_rthigh * 0.1
                e_x_lthigh = e_x_lthigh * 0.1
                e_y_lthigh = e_y_lthigh * 0.1
                e_z_lthigh = e_z_lthigh * 0.1
                e_x_rshank = e_x_rshank * 0.1
                e_y_rshank = e_y_rshank * 0.1
                e_z_rshank = e_z_rshank * 0.1
                e_x_lshank = e_x_lshank * 0.1
                e_y_lshank = e_y_lshank * 0.1
                e_z_lshank = e_z_lshank * 0.1
                e_x_rfoot = e_x_rfoot * 0.1
                e_y_rfoot = e_y_rfoot * 0.1
                e_z_rfoot = e_z_rfoot * 0.1
                e_x_lfoot = e_x_lfoot * 0.1
                e_y_lfoot = e_y_lfoot * 0.1
                e_z_lfoot = e_z_lfoot * 0.1

                ax.plot([hip[0], hip[0] + e_x_pelvis[0]], [hip[1], hip[1] + e_x_pelvis[1]], [hip[2], hip[2] + e_x_pelvis[2]], color='red')
                ax.plot([hip[0], hip[0] + e_y_pelvis[0]], [hip[1], hip[1] + e_y_pelvis[1]], [hip[2], hip[2] + e_y_pelvis[2]], color='green')
                ax.plot([hip[0], hip[0] + e_z_pelvis[0]], [hip[1], hip[1] + e_z_pelvis[1]], [hip[2], hip[2] + e_z_pelvis[2]], color='blue')

                ax.plot([rthigh[0], rthigh[0] + e_x_rthigh[0]], [rthigh[1], rthigh[1] + e_x_rthigh[1]], [rthigh[2], rthigh[2] + e_x_rthigh[2]], color='red')
                ax.plot([rthigh[0], rthigh[0] + e_y_rthigh[0]], [rthigh[1], rthigh[1] + e_y_rthigh[1]], [rthigh[2], rthigh[2] + e_y_rthigh[2]], color='green')
                ax.plot([rthigh[0], rthigh[0] + e_z_rthigh[0]], [rthigh[1], rthigh[1] + e_z_rthigh[1]], [rthigh[2], rthigh[2] + e_z_rthigh[2]], color='blue')

                ax.plot([lthigh[0], lthigh[0] + e_x_lthigh[0]], [lthigh[1], lthigh[1] + e_x_lthigh[1]], [lthigh[2], lthigh[2] + e_x_lthigh[2]], color='red')
                ax.plot([lthigh[0], lthigh[0] + e_y_lthigh[0]], [lthigh[1], lthigh[1] + e_y_lthigh[1]], [lthigh[2], lthigh[2] + e_y_lthigh[2]], color='green')
                ax.plot([lthigh[0], lthigh[0] + e_z_lthigh[0]], [lthigh[1], lthigh[1] + e_z_lthigh[1]], [lthigh[2], lthigh[2] + e_z_lthigh[2]], color='blue')

                ax.plot([rshank[0], rshank[0] + e_x_rshank[0]], [rshank[1], rshank[1] + e_x_rshank[1]], [rshank[2], rshank[2] + e_x_rshank[2]], color='red')
                ax.plot([rshank[0], rshank[0] + e_y_rshank[0]], [rshank[1], rshank[1] + e_y_rshank[1]], [rshank[2], rshank[2] + e_y_rshank[2]], color='green')
                ax.plot([rshank[0], rshank[0] + e_z_rshank[0]], [rshank[1], rshank[1] + e_z_rshank[1]], [rshank[2], rshank[2] + e_z_rshank[2]], color='blue')

                ax.plot([lshank[0], lshank[0] + e_x_lshank[0]], [lshank[1], lshank[1] + e_x_lshank[1]], [lshank[2], lshank[2] + e_x_lshank[2]], color='red')
                ax.plot([lshank[0], lshank[0] + e_y_lshank[0]], [lshank[1], lshank[1] + e_y_lshank[1]], [lshank[2], lshank[2] + e_y_lshank[2]], color='green')
                ax.plot([lshank[0], lshank[0] + e_z_lshank[0]], [lshank[1], lshank[1] + e_z_lshank[1]], [lshank[2], lshank[2] + e_z_lshank[2]], color='blue')

                ax.plot([rfoot[0], rfoot[0] + e_x_rfoot[0]], [rfoot[1], rfoot[1] + e_x_rfoot[1]], [rfoot[2], rfoot[2] + e_x_rfoot[2]], color='red')
                ax.plot([rfoot[0], rfoot[0] + e_y_rfoot[0]], [rfoot[1], rfoot[1] + e_y_rfoot[1]], [rfoot[2], rfoot[2] + e_y_rfoot[2]], color='green')
                ax.plot([rfoot[0], rfoot[0] + e_z_rfoot[0]], [rfoot[1], rfoot[1] + e_z_rfoot[1]], [rfoot[2], rfoot[2] + e_z_rfoot[2]], color='blue')

                ax.plot([lfoot[0], lfoot[0] + e_x_lfoot[0]], [lfoot[1], lfoot[1] + e_x_lfoot[1]], [lfoot[2], lfoot[2] + e_x_lfoot[2]], color='red')
                ax.plot([lfoot[0], lfoot[0] + e_y_lfoot[0]], [lfoot[1], lfoot[1] + e_y_lfoot[1]], [lfoot[2], lfoot[2] + e_y_lfoot[2]], color='green')
                ax.plot([lfoot[0], lfoot[0] + e_z_lfoot[0]], [lfoot[1], lfoot[1] + e_z_lfoot[1]], [lfoot[2], lfoot[2] + e_z_lfoot[2]], color='blue')
                
                e_x_pelvis_0 = e_x_pelvis_0 * 0.1
                e_y_pelvis_0 = e_y_pelvis_0 * 0.1
                e_z_pelvis_0 = e_z_pelvis_0 * 0.1
                ax.scatter(hip_0[0], hip_0[1], hip_0[2], label='hip_0', color='black')
                ax.plot([hip_0[0], hip_0[0] + e_x_pelvis_0[0]], [hip_0[1], hip_0[1] + e_x_pelvis_0[1]], [hip_0[2], hip_0[2] + e_x_pelvis_0[2]], color='red')
                ax.plot([hip_0[0], hip_0[0] + e_y_pelvis_0[0]], [hip_0[1], hip_0[1] + e_y_pelvis_0[1]], [hip_0[2], hip_0[2] + e_y_pelvis_0[2]], color='green')
                ax.plot([hip_0[0], hip_0[0] + e_z_pelvis_0[0]], [hip_0[1], hip_0[1] + e_z_pelvis_0[1]], [hip_0[2], hip_0[2] + e_z_pelvis_0[2]], color='blue')
                
                plt.legend()
                plt.show()
        
        #仙骨とかかとのベクトル計算
        sacuram = (rpsi[frame_num, :] + lpsi[frame_num, :]) / 2
        sac2hee_r = rhee[frame_num, :] - sacuram
        sac2hee_r_list.append(sac2hee_r)
        sac2hee_l = lhee[frame_num, :] - sacuram
        sac2hee_l_list.append(sac2hee_l)
        
        #仙骨とつま先のベクトル計算
        sac2toe_r = rtoe[frame_num, :] - sacuram
        sac2toe_r_list.append(sac2toe_r)
        sac2toe_l = ltoe[frame_num, :] - sacuram
        sac2toe_l_list.append(sac2toe_l)
        
        # 踵のZ座標記録
        heel_z_list.append((rhee[frame_num, 2], lhee[frame_num, 2]))
        toe_z_list.append((rtoe[frame_num, 2], ltoe[frame_num, 2]))

    angle_array = np.array(angle_list)
    angle_df = pd.DataFrame(angle_array, columns=["R_Hip_FlEx", "L_Hip_FlEx", "R_Knee_FlEx", "L_Knee_FlEx", "R_Ankle_PlDo", "L_Ankle_PlDo",
                                                "R_Hip_InEx", "L_Hip_InEx", "R_Knee_InEx", "L_Knee_InEx", "R_Ankle_InEx", "L_Ankle_InEx",
                                                "R_Hip_AdAb", "L_Hip_AdAb", "R_Knee_AdAb", "L_Knee_AdAb", "R_Ankle_AdAb", "L_Ankle_AdAb"], index=np.arange(len(angle_array)))

    # 角度データの連続性保つ
    angle_df['R_Hip_FlEx'] = np.where(angle_df['R_Hip_FlEx'] < -90, angle_df['R_Hip_FlEx'] + 360, angle_df['R_Hip_FlEx'])
    angle_df['L_Hip_FlEx'] = np.where(angle_df['L_Hip_FlEx'] < -90, angle_df['L_Hip_FlEx'] + 360, angle_df['L_Hip_FlEx'])
    angle_df['R_Knee_FlEx'] = np.where(angle_df['R_Knee_FlEx'] < -90, angle_df['R_Knee_FlEx'] + 360, angle_df['R_Knee_FlEx'])
    angle_df['L_Knee_FlEx'] = np.where(angle_df['L_Knee_FlEx'] < -90, angle_df['L_Knee_FlEx'] + 360, angle_df['L_Knee_FlEx'])
    angle_df['R_Ankle_PlDo'] = np.where(angle_df['R_Ankle_PlDo'] < -90, angle_df['R_Ankle_PlDo'] + 360, angle_df['R_Ankle_PlDo'])
    angle_df['L_Ankle_PlDo'] = np.where(angle_df['L_Ankle_PlDo'] < -90, angle_df['L_Ankle_PlDo'] + 360, angle_df['L_Ankle_PlDo'])
                
    # Hip, Knee, Ankle角度のオフセット補正
    if 'R_Hip_FlEx' in angle_df.columns:
        for frame in angle_df.index:
            angle_df.loc[frame, 'R_Hip_FlEx'] = angle_df.at[frame, 'R_Hip_FlEx'] - 180
    if 'L_Hip_FlEx' in angle_df.columns:
        for frame in angle_df.index:
            angle_df.loc[frame, 'L_Hip_FlEx'] = angle_df.at[frame, 'L_Hip_FlEx'] - 180
    if 'R_Knee_FlEx' in angle_df.columns:
        for frame in angle_df.index:
            angle_df.loc[frame, 'R_Knee_FlEx'] = - angle_df.at[frame, 'R_Knee_FlEx']
    if 'L_Knee_FlEx' in angle_df.columns:
        for frame in angle_df.index:
            angle_df.loc[frame, 'L_Knee_FlEx'] = - angle_df.at[frame, 'L_Knee_FlEx']
    if 'R_Ankle_PlDo' in angle_df.columns:
        for frame in angle_df.index:
            angle_df.loc[frame, 'R_Ankle_PlDo'] = 180 - angle_df.at[frame, 'R_Ankle_PlDo']
    if 'L_Ankle_PlDo' in angle_df.columns:
        for frame in angle_df.index:
            angle_df.loc[frame, 'L_Ankle_PlDo'] = 180 - angle_df.at[frame, 'L_Ankle_PlDo']
    
    if 'R_Hip_InEx' in angle_df.columns:
        for frame in angle_df.index:
            angle_df.loc[frame, 'R_Hip_InEx'] = angle_df.at[frame, 'R_Hip_InEx'] #外旋ex+, 内旋in-
    if 'L_Hip_InEx' in angle_df.columns:
        for frame in angle_df.index:
            angle_df.loc[frame, 'L_Hip_InEx'] = - angle_df.at[frame, 'L_Hip_InEx'] # 外旋ex+, 内旋in-
    if 'R_Hip_AdAb' in angle_df.columns:
        for frame in angle_df.index:
            angle_df.loc[frame, 'R_Hip_AdAb'] = angle_df.at[frame, 'R_Hip_AdAb']  # 外転ab+, 内転ad-
    if 'L_Hip_AdAb' in angle_df.columns:
        for frame in angle_df.index:
            angle_df.loc[frame, 'L_Hip_AdAb'] = - angle_df.at[frame, 'L_Hip_AdAb'] # 外転ab+, 内転ad-
    
    # DataFrameのインデックスを絶対フレーム番号に設定
    # 60Hzデータの場合、start_frameからの60Hz絶対フレーム番号
    absolute_frame_indices = np.array(full_range) + start_frame

    # print(f"absolute_frame_indices = {absolute_frame_indices}")
    absolute_frame_indices = absolute_frame_indices[1:]
    # print(f"absolute_frame_indices = {absolute_frame_indices}")

    sac2hee_r_array = np.array(sac2hee_r_list)
    rsac2hee_z = sac2hee_r_array[:, 2]
    sac2hee_l_array = np.array(sac2hee_l_list)
    lsac2hee_z = sac2hee_l_array[:, 2]
    
    sac2toe_r_array = np.array(sac2toe_r_list)
    rsac2toe_z = sac2toe_r_array[:, 2]
    sac2toe_l_array = np.array(sac2toe_l_list)
    lsac2toe_z = sac2toe_l_array[:, 2]

    # =========================
    # 関節角度プロット (60Hz) - 3段（Hip/Knee/Ankle）
    # =========================
    try:
        frames_for_plot = abs_frames
        if len(frames_for_plot) == len(angle_df) + 1:
            frames_for_plot = frames_for_plot[1:]
        elif len(frames_for_plot) != len(angle_df):
            frames_for_plot = np.arange(len(angle_df))
    except Exception:
        frames_for_plot = np.arange(len(angle_df))
        
    # 描画の際にはgoproのフレームと合わせるためにオフセットを引く
    frames_for_plot_disp = frames_for_plot - frame_diff

    # 関節角度データを60Hz絶対フレーム番号(ViTPoseに合わせた)で保存
    angle_df_abs = angle_df.copy()
    angle_df_abs.index = frames_for_plot_disp
    # print(f"frame_for_plot_disp = {frames_for_plot_disp}")
    # print(f"angle_df_abs.index = {angle_df_abs.index}")
    angle_df_abs.to_csv(csv_path.parent / f"angle_60Hz_{csv_path.name}")
    
    def plot_three_timeseries(title, keys_r, keys_l, angles_dict, fname,
                            frames=None, ic_r=None, ic_l=None):
        """
        keys_r: ["R_Hip_FlEx","R_Knee_FlEx","R_Ankle_PlDo"] など
        keys_l: ["L_Hip_FlEx","L_Knee_FlEx","L_Ankle_PlDo"] など
        frames: x軸（絶対フレーム or 相対フレーム）を指定できる
        ic_r / ic_l: 初期接地フレーム（frames と同じ基準の値）
        3段：股→膝→足、各段にR/Lを重ね描き
        """
        assert len(keys_r) == 3 and len(keys_l) == 3

        # フレーム数（R/Lで長さが違う可能性に備えて最小に合わせる）
        n_r = len(angles_dict[keys_r[0]]) if keys_r[0] in angles_dict else 0
        n_l = len(angles_dict[keys_l[0]]) if keys_l[0] in angles_dict else 0
        n = min(n_r, n_l) if (n_r > 0 and n_l > 0) else max(n_r, n_l)

        if frames is None:
            x = np.arange(n)
        else:
            x = np.asarray(frames)[:n]
            # 念のため：長さが合わなければ相対にフォールバック
            if len(x) != n:
                x = np.arange(n)

        # IC（範囲外除外・重複排除）
        x_min, x_max = float(np.min(x)), float(np.max(x))
        ic_r = [] if ic_r is None else sorted({int(v) for v in ic_r if x_min <= int(v) <= x_max})
        ic_l = [] if ic_l is None else sorted({int(v) for v in ic_l if x_min <= int(v) <= x_max})
        
        # 表示用にフレームオフセットを引く
        ic_r = [ic_r_ori - frame_diff for ic_r_ori in ic_r]
        ic_l = [ic_l_ori - frame_diff for ic_l_ori in ic_l]

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

        for ax, kr, kl in zip(axes, keys_r, keys_l):
            has_r = kr in angles_dict
            has_l = kl in angles_dict

            if not has_r and not has_l:
                ax.set_title(f"{kr} / {kl} (missing)")
                ax.grid(True)
                continue

            if has_r:
                y_r = np.asarray(angles_dict[kr], dtype=float)[:n]
                # ax.plot(x, y_r, label=kr, color="tab:orange")
                ax.plot(frames_for_plot_disp[:n], y_r, label=kr, color="tab:orange")

            if has_l:
                y_l = np.asarray(angles_dict[kl], dtype=float)[:n]
                # ax.plot(x, y_l, label=kl, color="tab:blue")
                ax.plot(frames_for_plot_disp[:n], y_l, label=kl, color="tab:blue")
            # IC縦線（破線）
            for i, f in enumerate(ic_r):
                ax.axvline(f, linestyle="--", linewidth=1.0,
                        color="tab:orange", alpha=0.6,
                        label="IC_R" if i == 0 else None)
            for i, f in enumerate(ic_l):
                ax.axvline(f, linestyle="--", linewidth=1.0,
                        color="tab:blue", alpha=0.6,
                        label="IC_L" if i == 0 else None)

            ax.set_ylabel("Angle [deg]")

            ref_key = kr if has_r else kl
            ylims = ylims_for_angle_key(ref_key)
            if ylims is not None:
                ax.set_ylim(*ylims)

            ax.grid(True)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Frame")
        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(csv_path_dir / fname)
        plt.close()

    angles_dict = {c: angle_df[c].to_numpy() for c in angle_df.columns}
    
    event_frame_dict = opti.calc_gait_events(rsac2hee_z, lsac2hee_z, rsac2toe_z, lsac2toe_z, start_frame, csv_path_dir)
    
    gait_cycles_r_rel, gait_cycles_l_rel = opti.calc_gait_cycles(event_frame_dict)
    
    print(f"右足の歩行周期 IC→対IC→TO→次IC (60Hz): {gait_cycles_r_rel}")    
    print(f"左足の歩行周期  IC→対IC→TO→次IC (60Hz): {gait_cycles_l_rel}")
    
    gait_cycles_r_abs = []
    for ic_rel, ic_l_rel, to_rel, ic_next_rel in gait_cycles_r_rel:
        ic_abs = ic_rel + start_frame
        ic_l_abs = ic_l_rel + start_frame
        to_abs = to_rel + start_frame
        ic_next_abs = ic_next_rel + start_frame
        gait_cycles_r_abs.append([ic_abs, ic_l_abs, to_abs, ic_next_abs])
    gait_cycles_l_abs = []
    for ic_rel, ic_r_rel, to_rel, ic_next_rel in gait_cycles_l_rel:
        ic_abs = ic_rel + start_frame
        ic_r_abs = ic_r_rel + start_frame
        to_abs = to_rel + start_frame
        ic_next_abs = ic_next_rel + start_frame
        gait_cycles_l_abs.append([ic_abs, ic_r_abs, to_abs, ic_next_abs])
    print(f"右足の歩行周期 (60Hz絶対フレーム): {gait_cycles_r_abs}")
    print(f"左足の歩行周期 (60Hz絶対フレーム): {gait_cycles_l_abs}")
    
    gait_cycles_r_abs = opti.select_valid_gait_cycles(gait_cycles_r_abs, gait_cycles_r_check, tolerance=5)
    gait_cycles_l_abs = opti.select_valid_gait_cycles(gait_cycles_l_abs, gait_cycles_l_check, tolerance=5)

    print(f"フィルタリング後の右足歩行周期 (60Hz絶対フレーム): {gait_cycles_r_abs}")
    print(f"フィルタリング後の左足歩行周期 (60Hz絶対フレーム): {gait_cycles_l_abs}")
    
    # 最終的に使用した絶対フレーム範囲
    all_cycles = gait_cycles_r_abs + gait_cycles_l_abs
    final_start_frame_60hz = min([cycle[0] for cycle in all_cycles])
    final_start_frame_60hz_abs = final_start_frame_60hz + start_frame
    final_end_frame_60hz = max([cycle[-1] for cycle in all_cycles])
    final_end_frame_60hz_abs = final_end_frame_60hz + start_frame
    print(f"最終的に使用したフレーム範囲 (60Hz絶対フレーム): {final_start_frame_60hz_abs} 〜 {final_end_frame_60hz_abs}")
    
    # rel のIC（index）
    ic_r_rel = [c[0] for c in gait_cycles_r_rel]
    ic_l_rel = [c[0] for c in gait_cycles_l_rel]

    # abs のIC（フレーム番号）
    ic_r_abs = [int(abs_frames[i]) for i in ic_r_rel if 0 <= i < len(abs_frames)]
    ic_l_abs = [int(abs_frames[i]) for i in ic_l_rel if 0 <= i < len(abs_frames)]

    # 相対フレーム番号での歩行サイクルリスト
    start_frame_rel = min([c[0] for c in gait_cycles_r_rel] + [c[0] for c in gait_cycles_l_rel])
    end_frame_rel   = max([c[-1] for c in gait_cycles_r_rel] + [c[-1] for c in gait_cycles_l_rel])

    # 絶対フレーム番号を相対フレーム番号に変換
    gait_cycles_r_rel = [[frame - start_frame for frame in cycle] for cycle in gait_cycles_r_abs]
    gait_cycles_l_rel = [[frame - start_frame for frame in cycle] for cycle in gait_cycles_l_abs]

    gait_cycles_r_rel_df = pd.DataFrame(gait_cycles_r_rel, columns=["IC", "IC_opp", "TO", "IC_next"])
    gait_cycles_l_rel_df = pd.DataFrame(gait_cycles_l_rel, columns=["IC", "IC_opp", "TO", "IC_next"])
    gait_cycles_r_rel_df.to_csv(csv_path.parent / "gait_cycles_r_rel.csv", index=False)
    gait_cycles_l_rel_df.to_csv(csv_path.parent / "gait_cycles_l_rel.csv", index=False)
    
    # 歩行周期をViTPoseの絶対フレーム番号に変換
    print(f"gait_cycles_r_rel = {gait_cycles_r_rel}")
    print(f"gait_cycles_l_rel = {gait_cycles_l_rel}")
    print(f"gait_cycles_r_abs = {gait_cycles_r_abs}")
    print(f"gait_cycles_l_abs = {gait_cycles_l_abs}")
    
    
    used_key = csv_path.stem
    plot_three_timeseries(
        f"Timeseries Flex/Ext (R & L) - mocap ({used_key})",
        ["R_Hip_FlEx", "R_Knee_FlEx", "R_Ankle_PlDo"],
        ["L_Hip_FlEx", "L_Knee_FlEx", "L_Ankle_PlDo"],
        angles_dict,
        f"timeseries_FlEx_{used_key}.png",
        frames=frames_for_plot,     
        ic_r=ic_r_abs, ic_l=ic_l_abs
    )

    plot_three_timeseries(
        f"Timeseries Ad/Ab (R & L) - mocap ({used_key})",
        ["R_Hip_AdAb", "R_Knee_AdAb", "R_Ankle_AdAb"],
        ["L_Hip_AdAb", "L_Knee_AdAb", "L_Ankle_AdAb"],
        angles_dict,
        f"timeseries_AdAb_{used_key}.png",
        frames=frames_for_plot,
        ic_r=ic_r_abs, ic_l=ic_l_abs
    )

    plot_three_timeseries(
        f"Timeseries In/Ex (R & L) - mocap ({used_key})",
        ["R_Hip_InEx", "R_Knee_InEx", "R_Ankle_InEx"],
        ["L_Hip_InEx", "L_Knee_InEx", "L_Ankle_InEx"],
        angles_dict,
        f"timeseries_InEx_{used_key}.png",
        frames=frames_for_plot,
        ic_r=ic_r_abs, ic_l=ic_l_abs
    )

    
    # 開始フレームが右足のどのサイクルに含まれるか確認（すべてのサイクルをチェック）
    start_is_right = False
    for cycle in gait_cycles_r_rel:
        if cycle[0] == start_frame_rel:  # サイクルの開始フレーム（IC）
            start_is_right = True
            break
    if start_is_right:
        start_heel_pos = rhee[start_frame_rel][2]
        print(f"{start_frame}フレーム  右足初期接地から開始，右足踵位置: {start_heel_pos:.3f} m")
    else:
        start_heel_pos = lhee[start_frame_rel][2]
        print(f"{start_frame}フレーム  左足初期接地から開始，左足踵位置: {start_heel_pos:.3f} m")

    # 終了フレームが右足のどのサイクルに含まれるか確認（すべてのサイクルをチェック）
    end_is_right = False
    for cycle in gait_cycles_r_rel:
        if cycle[-1] == end_frame_rel:  # サイクルの終了フレーム（次のIC）
            end_is_right = True
            break

    if end_is_right:
        end_heel_pos = rhee[end_frame_rel][2]
        print(f"{end_frame}フレーム  右足初期接地で終了，右足踵位置: {end_heel_pos:.3f} m")
    else:
        end_heel_pos = lhee[end_frame_rel][2]
        print(f"{end_frame}フレーム  左足初期接地で終了，左足踵位置: {end_heel_pos:.3f} m")
        
    ###########################################
    # 歩行パラメータの計算（Mocap）
    ###########################################
    def calculate_gait_parameters(gait_cycles, hip_array, rhee, lhee, side, sampling_freq=60):
        """
        歩行パラメータを計算
        
        算出する指標:
        - gait_speed: 歩行速度 [m/s]
        - swing_stance_ratio: 時間的対称性（遊脚期比）ここではまだ算出不可
        - stride_time: ストライド時間 [s]
        - stride_width: 歩隔 [m]
    
        Parameters:
        -----------
        gait_cycles : list
            [[ic_start, to, ic_end], ...] の形式
        hip_array : np.ndarray
            骨盤中心の3D座標 (n_frames, 3)
        rhee : np.ndarray
            右踵の3D座標 (n_frames, 3)
        lhee : np.ndarray
            左踵の3D座標 (n_frames, 3)
        sampling_freq : int
            サンプリング周波数 [Hz]
            
        Returns:
        --------
        gait_params : list of dict
            各歩行周期のパラメータ
        """
        gait_params = []
        
        for cycle_idx, (ic_start, ic_opp, to, ic_end) in enumerate(gait_cycles):
            # ストライド時間 [s]
            stride_time = (ic_end - ic_start) / sampling_freq
            
            # 遊脚期時間 [s]
            swing_duration = (ic_end - to) / sampling_freq
            
            # 歩行速度 [m/s]
            # 骨盤の移動距離を時間で割る
            hip_displacement = np.linalg.norm(hip_array[ic_end] - hip_array[ic_start])
            gait_speed = hip_displacement / stride_time
            
            # 歩隔[m](とステップ長[m])
            if side == 'R':
                a = np.linalg.norm(lhee[ic_opp] - rhee[ic_start])
                b = np.linalg.norm(rhee[ic_end] - lhee[ic_opp])
                c = np.linalg.norm(rhee[ic_end] - rhee[ic_start])
                step_length = (b**2 + c**2 -a**2) / (2*c)
                stride_width = np.sqrt(b**2 - step_length**2)
            else:
                a = np.linalg.norm(rhee[ic_opp] - lhee[ic_start])
                b = np.linalg.norm(lhee[ic_end] - rhee[ic_opp])
                c = np.linalg.norm(lhee[ic_end] - lhee[ic_start])
                step_length = (b**2 + c**2 -a**2) / (2*c)
                stride_width = np.sqrt(b**2 - step_length**2)
            
            params = {
                'cycle_index': cycle_idx,
                'gait_speed': gait_speed,
                'stride_time': stride_time,
                'swing_duration': swing_duration,
                'stride_width': stride_width,
            }
            gait_params.append(params)
        
        return gait_params
    
    # 右足と左足の歩行パラメータを計算
    gait_params_r = calculate_gait_parameters(gait_cycles_r_rel, hip_array, rhee, lhee, side='R', sampling_freq=60)
    gait_params_l = calculate_gait_parameters(gait_cycles_l_rel, hip_array, rhee, lhee, side='L', sampling_freq=60)
    
    # 時間的対称性指標（遊脚期比のSI）を算出　右麻痺前提 歩行周期ごとでの算出が難しいため全体平均で算出
    swing_duration_para = np.array([p['swing_duration'] for p in gait_params_r]).mean()
    swing_duration_nonpara = np.array([p['swing_duration'] for p in gait_params_l]).mean()
    symmetry_index_sw = (swing_duration_para - swing_duration_nonpara) / (0.5 * (swing_duration_para + swing_duration_nonpara)) * 100

        
    # symmetry_index_swも保存　一度の施行で一回算出する値なので各サイクルに同じ値を入れておく
    for i_cycle, _ in enumerate(gait_cycles_r_rel):
        gait_params_r[i_cycle]['SI_sw'] = symmetry_index_sw
        
        # 最大関節角度をまとめる 右麻痺前提
    for i_cycle, cycle_frames in enumerate(gait_cycles_r_rel):
        ic_start = int(cycle_frames[0])
        ic_end = int(cycle_frames[3])    
        # cycle_frames: [ic, ic_opp, to, ic_end]
        print(f"gait cycle {i_cycle}: frames {cycle_frames}")
        hip_flex = angles_dict["R_Hip_FlEx"][ic_start:ic_end+1]
        knee_flex = angles_dict["R_Knee_FlEx"][ic_start:ic_end+1]
        ankle_pl = angles_dict["R_Ankle_PlDo"][ic_start:ic_end+1]
        hip_abad = angles_dict["R_Hip_AdAb"][ic_start:ic_end+1]
        hip_max_ext = - np.min(hip_flex) # 股関節最大伸展　伸展は負の値になるので正にするために符号反転
        knee_max_flex = np.max(knee_flex)  # 膝関節最大屈曲
        ankle_max_do = np.max(ankle_pl) # 足関節最大背屈
        hip_max_ab = np.max(hip_abad)  # 股関節最大外転
        gait_params_r[i_cycle]['hip_max_ext'] = hip_max_ext 
        gait_params_r[i_cycle]['knee_max_flex'] = knee_max_flex
        gait_params_r[i_cycle]['ankle_max_do'] = ankle_max_do
        gait_params_r[i_cycle]['hip_max_ab'] = hip_max_ab
        
    # 歩行パラメータをCSVに保存
    gait_params_r_df = pd.DataFrame(gait_params_r)
    gait_params_l_df = pd.DataFrame(gait_params_l)
    
    gait_params_r_df.to_csv(csv_path_dir / f"gait_parameters_R_{csv_path.stem}.csv", index=False)
    gait_params_l_df.to_csv(csv_path_dir / f"gait_parameters_L_{csv_path.stem}.csv", index=False)
    
    symmetry_df = pd.DataFrame({
        'SI_swing_duration': [symmetry_index_sw]
    })
    symmetry_df.to_csv(csv_path_dir / f"SI_{csv_path.stem}.csv", index=False)
    
    print(f"\n歩行パラメータ（Mocap）:")
    print(f"右足: 平均歩行速度 = {np.mean([p['gait_speed'] for p in gait_params_r]):.2f} m/s")
    print(f"左足: 平均歩行速度 = {np.mean([p['gait_speed'] for p in gait_params_l]):.2f} m/s")
    print(f"右足: 平均歩隔 = {np.mean([p['stride_width'] for p in gait_params_r]):.2f} m")
    print(f"左足: 平均歩隔 = {np.mean([p['stride_width'] for p in gait_params_l]):.2f} m")

def main():
    root_dir = Path(r"G:\gait_pattern\2025_shuron_BR9G")

    # sub1~sub10, thera1-0~thera10-0 を総当たり
    for sub_i in range(1, 11):
        # # 被験者の対象しぼるならここで指定 不要ならコメントアウト
        # if sub_i not  in [3]:
        #     print(f"[SKIP] not check {sub_i} now")
        #     continue
        sub_dir = root_dir / f"sub{sub_i}"
        if not sub_dir.exists():
            print(f"[SKIP] missing: {sub_dir}")
            continue

        thera_dir = sub_dir / f"thera{sub_i}-0"
        csv_path_dir = thera_dir / "mocap"
        if not csv_path_dir.exists():
            print(f"[SKIP] missing: {csv_path_dir}")
            continue
        
        sync_imu_dir = thera_dir / "IMU"
        imu_path = next(sync_imu_dir.glob("*SYNC*.csv"), None)
        if imu_path is None:
            print(f"[WARN] no sync IMU found: {imu_path}")
            continue
        # gopro発光とmocap撮影開始のフレーム差(60hz)を取得
        gopro_mocap_frame_diff = check_imu_sync_frame_diff(imu_path)
        
        gopro_trimming_json = thera_dir / "gopro" / "trimming_info.json"
        with open(str(gopro_trimming_json), "r", encoding="utf-8") as f:
            gopro_trimming_data = json.load(f)
        # goproの発光からトリミング開始までのフレーム差(60hz)を取得
        gopro_cut_frame_diff = gopro_trimming_data["trimming_settings"]["start_frame_relative"]
        
        # goproとmocapのフレーム差(60hz)を計算
        frame_diff = gopro_cut_frame_diff - gopro_mocap_frame_diff
        frame_diff_df = pd.DataFrame({'frame_diff': [frame_diff],})
        frame_diff_df.to_csv(csv_path_dir / f"frame_diff.csv", index=False)
        
        print(f"[INFO] gopro_mocap_frame_diff = {gopro_mocap_frame_diff}, gopro_cut_frame_diff = {gopro_cut_frame_diff}, frame_diff = {frame_diff}")
            
        gopro_gait_cycle_dir = thera_dir / "ViTPose_results"
        gopro_gait_cycle_path = next(gopro_gait_cycle_dir.glob("gait_cycles_*.npz"), None)
        if gopro_gait_cycle_path is None:
            print(f"[WARN] no gopro gait cycle {gopro_gait_cycle_path}")
            continue
        gopro_gait_cycle = np.load(gopro_gait_cycle_path, allow_pickle=True)

        print("\n")
        print("=" * 80)
        print(f"[RUN] {csv_path_dir}")
        print("=" * 80)

        try:
            process_one(csv_path_dir, frame_diff, gopro_gait_cycle)
        except Exception as e:
            print(f"[ERROR] {csv_path_dir}: {e}")
            continue
    
if __name__ == "__main__":
    main()