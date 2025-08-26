import pandas as pd
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample
import json
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

def read_3d_optitrack(csv_path, down_hz):
    df = pd.read_csv(csv_path, skiprows=[0, 1, 2, 4], header=[0, 2])  #Motive

    if down_hz:
        # 100Hz → 60Hz のダウンサンプリング
        # サンプル数を60/100 = 0.6倍にリサンプリング
        original_length = len(df)
        target_length = int(original_length * 60 / 100)
        
        # 各列に対してリサンプリングを適用
        df_resampled = pd.DataFrame()
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:  # 数値列のみリサンプリング
                resampled_data = resample(df[col].dropna(), target_length)
                df_resampled[col] = resampled_data
            else:
                # 非数値列は単純にインデックスを調整
                step = len(df) / target_length
                indices = [int(i * step) for i in range(target_length)]
                df_resampled[col] = df[col].iloc[indices].reset_index(drop=True)
        
        df_down = df_resampled.reset_index(drop=True)
    else:
        df_down = df

    # 全てのマーカーセット
    marker_set = ["RASI", "LASI","RPSI","LPSI","RKNE","LKNE", "RTHI", "LTHI", "RANK","LANK", "RTIB", "LTIB","RTOE","LTOE","RHEE","LHEE",
                "RSHO", "LSHO","C7", "T10", "CLAV", "STRN", "RBAK", "RKNE2", "LKNE2", "RANK2", "LANK2"]

    marker_set_df = df_down[[col for col in df_down.columns if any(marker in col[0] for marker in marker_set)]].copy()

    print(f"Marker set dataframe shape: {marker_set_df.shape}")

    # マーカーデータが存在しない場合やファイル名に"marker_set"が含まれる場合はスキップ
    if marker_set_df.shape[1] == 0 or "marker_set" in os.path.basename(csv_path):
        print(f"No marker data found or already processed file: {csv_path}. Skipping...")
        return None, None, None

    success_frame_list = []

    for frame in range(0, len(marker_set_df)):
        if not marker_set_df.iloc[frame].isna().any():
            success_frame_list.append(frame)

    if not success_frame_list:
        print(f"No valid frames found in {csv_path}. Skipping...")
        return None, None, None

    full_range = range(min(success_frame_list), max(success_frame_list) + 1)
    success_df = marker_set_df.reindex(full_range)
    interpolate_success_df = success_df.interpolate(method='spline', order=3)

    for i, index in enumerate(full_range):
        marker_set_df.loc[index, :] = interpolate_success_df.iloc[i, :]
    
    # 実際に存在するマーカーのリストを取得
    available_markers = []
    for marker in marker_set:
        if any(marker in col[0] for col in marker_set_df.columns):
            available_markers.append(marker)
    
    print(f"Available markers: {available_markers}")
    
    marker_set_df.to_csv(os.path.join(os.path.dirname(csv_path), f"marker_set_{os.path.basename(csv_path)}"))

    keypoints = marker_set_df.values
    keypoints_mocap = keypoints.reshape(-1, len(available_markers), 3)  #xyzで組になるように変形

    return keypoints_mocap, full_range, available_markers

def butter_lowpass_fillter(data, order, cutoff_freq, frame_list, sampling_freq=60):
    nyquist_freq = sampling_freq / 2
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data[frame_list])
    data_fillter = np.copy(data)
    data_fillter[frame_list] = y
    return data_fillter

def get_marker_index(marker_name, available_markers):
    """マーカー名からインデックスを取得"""
    try:
        return available_markers.index(marker_name)
    except ValueError:
        return None

def safe_filter_marker(keypoints_mocap, marker_idx, axis, full_range, sampling_freq):
    """安全にマーカーデータをフィルタリング"""
    if marker_idx is not None:
        return butter_lowpass_fillter(keypoints_mocap[:, marker_idx, axis], order=4, cutoff_freq=6, frame_list=full_range, sampling_freq=sampling_freq)
    else:
        return np.zeros(len(keypoints_mocap))

def create_animation(all_markers_data, available_markers, full_range, output_path):
    """3Dアニメーションを作成してmp4で保存"""
    print(f"Creating animation for {len(full_range)} frames...")
    
    # 高品質な設定でfigureを作成
    plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'  # ffmpegのパスを指定（必要に応じて）
    fig = plt.figure(figsize=(16, 12), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # 軸の設定
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    
    # データの範囲を計算
    all_x = []
    all_y = []
    all_z = []
    for marker_data in all_markers_data.values():
        all_x.extend(marker_data[:, 0])
        all_y.extend(marker_data[:, 1])
        all_z.extend(marker_data[:, 2])
    
    margin = 0.3
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_zlim(min(all_z) - margin, max(all_z) + margin)
    
    # マーカーの色設定（より見やすい色を使用）
    colors = plt.cm.Set3(np.linspace(0, 1, len(available_markers)))
    
    # 進捗バーの設定
    progress_bar = tqdm(total=len(full_range), desc="Rendering frames", unit="frame")
    
    def animate(frame_idx):
        ax.clear()
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        ax.set_zlim(min(all_z) - margin, max(all_z) + margin)
        
        # 背景色を設定
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # グリッドを薄く表示
        ax.grid(True, alpha=0.3)
        
        frame_num = full_range[frame_idx] if frame_idx < len(full_range) else full_range[-1]
        
        # 各マーカーを描画（サイズと透明度を調整）
        for i, (marker_name, marker_data) in enumerate(all_markers_data.items()):
            if frame_num < len(marker_data):
                ax.scatter(marker_data[frame_num, 0], marker_data[frame_num, 1], marker_data[frame_num, 2], 
                          c=[colors[i]], s=80, alpha=0.8, label=marker_name, edgecolors='black', linewidth=0.5)
        
        # 骨格の線を描画（太さと色を調整）
        connections = [
            ('RASI', 'LASI'), ('RASI', 'RPSI'), ('LASI', 'LPSI'), ('RPSI', 'LPSI'),  # 骨盤
            ('RASI', 'RKNE'), ('LASI', 'LKNE'),  # 股関節-膝
            ('RKNE', 'RANK'), ('LKNE', 'LANK'),  # 膝-足首
            ('RANK', 'RTOE'), ('LANK', 'LTOE'),  # 足首-つま先
            ('RANK', 'RHEE'), ('LANK', 'LHEE'),  # 足首-踵
            ('RSHO', 'LSHO'), ('RSHO', 'C7'), ('LSHO', 'C7'),  # 肩
            # 追加の接続
            ('RKNE', 'RKNE2'), ('LKNE', 'LKNE2'),  # 膝マーカー
            ('RANK', 'RANK2'), ('LANK', 'LANK2'),  # 足首マーカー
        ]
        
        for marker1, marker2 in connections:
            if marker1 in all_markers_data and marker2 in all_markers_data:
                if frame_num < len(all_markers_data[marker1]) and frame_num < len(all_markers_data[marker2]):
                    p1 = all_markers_data[marker1][frame_num]
                    p2 = all_markers_data[marker2][frame_num]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', alpha=0.7, linewidth=2)
        
        # タイトルとフレーム情報
        ax.set_title(f'3D Motion Capture Data - Frame: {frame_num}', fontsize=14, fontweight='bold')
        
        # 凡例を調整（小さくして右上に配置）
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=0.7)
        
        # カメラアングルを設定（よりよい視点に調整）
        ax.view_init(elev=20, azim=45)
        
        # 進捗バーを更新
        progress_bar.update(1)
    
    # アニメーション作成
    print("Starting animation rendering...")
    anim = FuncAnimation(fig, animate, frames=len(full_range), interval=50, repeat=True, blit=False)
    
    # mp4で保存（高品質設定）
    print("Saving animation as MP4...")
    try:
        # FFMpegWriterを使用してmp4保存
        from matplotlib.animation import FFMpegWriter
        
        writer = FFMpegWriter(fps=30, metadata=dict(artist='Motion Capture Analysis'), bitrate=5000)
        
        # 保存の進捗を表示
        save_progress = tqdm(desc="Saving MP4", unit="frame")
        
        def progress_callback(current_frame, total_frames):
            save_progress.update(1)
        
        anim.save(output_path, writer=writer, progress_callback=progress_callback)
        save_progress.close()
        
    except ImportError:
        # FFMpegが利用できない場合はpillowを使用してmp4として保存を試行
        print("FFMpeg not available, trying alternative method...")
        try:
            anim.save(output_path, writer='ffmpeg', fps=30, bitrate=5000)
        except:
            # 最後の手段としてgifで保存
            gif_path = output_path.replace('.mp4', '.gif')
            print(f"MP4 save failed, saving as GIF instead: {gif_path}")
            anim.save(gif_path, writer='pillow', fps=20)
    
    progress_bar.close()
    plt.close()
    print(f"Animation saved: {output_path}")

def main():
    down_hz = True  # True: 100Hz→60Hz変換, False: 100Hzのまま
    csv_path_dir = r"G:\gait_pattern\20250811_br\sub0\thera0-14\mocap"
    csv_paths = glob.glob(os.path.join(csv_path_dir, "*.csv"))
    
    # 既に処理されたファイルを除外
    csv_paths = [path for path in csv_paths if "marker_set" not in os.path.basename(path) and "angle" not in os.path.basename(path)]
    
    print(f"Found {len(csv_paths)} files to process")

    for i, csv_path in enumerate(tqdm(csv_paths, desc="Processing files", unit="file")):
        print(f"\nProcessing file {i+1}/{len(csv_paths)}: {os.path.basename(csv_path)}")
        
        result = read_3d_optitrack(csv_path, down_hz)
        if result[0] is None:
            continue
            
        keypoints_mocap, full_range, available_markers = result
        print(f"Available markers: {available_markers}")

        # サンプリング周波数を設定
        sampling_freq = 60 if down_hz else 100

        # 全マーカーのデータを保存する辞書
        all_markers_data = {}

        # 各マーカーのインデックスを取得
        marker_indices = {}
        for marker in available_markers:
            marker_indices[marker] = get_marker_index(marker, available_markers)

        # 全マーカーのフィルタリングされたデータを取得（進捗表示付き）
        print("Filtering marker data...")
        for marker in tqdm(available_markers, desc="Filtering markers", unit="marker"):
            if marker_indices[marker] is not None:
                marker_data = np.array([
                    safe_filter_marker(keypoints_mocap, marker_indices[marker], x, full_range, sampling_freq) 
                    for x in range(3)
                ]).T
                all_markers_data[marker] = marker_data

        # 基本的なマーカーが存在するかチェック
        required_markers = ["RASI", "LASI", "RPSI", "LPSI", "RKNE", "LKNE", "RANK", "LANK", "RTOE", "LTOE", "RHEE", "LHEE"]
        missing_markers = [marker for marker in required_markers if marker not in available_markers]
        
        if missing_markers:
            print(f"Warning: Missing required markers: {missing_markers}")
            print("Skipping angle calculation for this file.")
            
            # マーカー座標のみ保存
            print("Saving coordinates data...")
            coordinates_df = pd.DataFrame()
            for marker in tqdm(available_markers, desc="Saving coordinates", unit="marker"):
                if marker in all_markers_data:
                    coordinates_df[f"{marker}_X"] = all_markers_data[marker][:, 0]
                    coordinates_df[f"{marker}_Y"] = all_markers_data[marker][:, 1]
                    coordinates_df[f"{marker}_Z"] = all_markers_data[marker][:, 2]
            
            coordinates_df.index = coordinates_df.index + full_range.start
            filename_suffix = "60Hz" if down_hz else "100Hz"
            coordinates_df.to_csv(os.path.join(os.path.dirname(csv_path), f"coordinates_{filename_suffix}_{os.path.basename(csv_path)}"))
            
            # アニメーション作成（mp4形式）
            animation_path = os.path.join(os.path.dirname(csv_path), f"animation_{filename_suffix}_{os.path.splitext(os.path.basename(csv_path))[0]}.mp4")
            create_animation(all_markers_data, available_markers, full_range, animation_path)
            
            continue

        # 必要なマーカーデータを取得（フィルタリング済み）
        rasi = all_markers_data.get("RASI", np.zeros((len(keypoints_mocap), 3)))
        lasi = all_markers_data.get("LASI", np.zeros((len(keypoints_mocap), 3)))
        rpsi = all_markers_data.get("RPSI", np.zeros((len(keypoints_mocap), 3)))
        lpsi = all_markers_data.get("LPSI", np.zeros((len(keypoints_mocap), 3)))
        rank = all_markers_data.get("RANK", np.zeros((len(keypoints_mocap), 3)))
        lank = all_markers_data.get("LANK", np.zeros((len(keypoints_mocap), 3)))
        rank2 = all_markers_data.get("RANK2", np.zeros((len(keypoints_mocap), 3)))
        lank2 = all_markers_data.get("LANK2", np.zeros((len(keypoints_mocap), 3)))
        rknee = all_markers_data.get("RKNE", np.zeros((len(keypoints_mocap), 3)))
        lknee = all_markers_data.get("LKNE", np.zeros((len(keypoints_mocap), 3)))
        rknee2 = all_markers_data.get("RKNE2", np.zeros((len(keypoints_mocap), 3)))
        lknee2 = all_markers_data.get("LKNE2", np.zeros((len(keypoints_mocap), 3)))
        rtoe = all_markers_data.get("RTOE", np.zeros((len(keypoints_mocap), 3)))
        ltoe = all_markers_data.get("LTOE", np.zeros((len(keypoints_mocap), 3)))
        rhee = all_markers_data.get("RHEE", np.zeros((len(keypoints_mocap), 3)))
        lhee = all_markers_data.get("LHEE", np.zeros((len(keypoints_mocap), 3)))

        angle_list = []
        dist_list = []
        bector_list = []

        print("Calculating joint angles...")
        for frame_num in tqdm(full_range, desc="Calculating angles", unit="frame"):
            #メモ
            d_asi = np.linalg.norm(rasi[frame_num,:] - lasi[frame_num,:])
            d_leg = (np.linalg.norm(rank[frame_num,:] - rasi[frame_num,:]) + np.linalg.norm(lank[frame_num, :] - lasi[frame_num,:]) / 2)
            r = 0.0127 #[m] Opti確認：https://www.optitrack.jp/products/accessories/marker.html
            h = 1.76 #[m]
            k = h/1.7
            beta = 0.1 * np.pi #[rad]
            theta = 0.496 #[rad]
            c = 0.115 * d_leg - 0.00153
            x_dis = 0.1288 * d_leg - 0.04856

            # skycom + davis
            x_rthigh = -(x_dis +r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
            x_lthigh = -(x_dis +r) * np.cos(beta) + c * np.cos(theta) * np.sin(beta)
            y_rthigh = +(c * np.sin(theta) - d_asi/2)
            y_lthigh = -(c * np.sin(theta)- d_asi/2)
            z_rthigh = -(x_dis + r) * np.sin(beta) + c * np.cos(theta) * np.cos(beta)
            z_lthigh = -(x_dis + r) * np.sin(beta) + c * np.cos(theta) * np.cos(beta)
            rthigh_pelvis = np.array([x_rthigh, y_rthigh, z_rthigh]).T
            lthigh_pelvis = np.array([x_lthigh, y_lthigh, z_lthigh]).T

            hip_0 = (rasi[frame_num,:] + lasi[frame_num,:]) / 2
            lumbar = (0.47 * (rasi[frame_num,:] + lasi[frame_num,:]) / 2 + 0.53 * (rpsi[frame_num,:] + lpsi[frame_num,:]) / 2) + 0.02 * k * np.array([0, 0, 1])

            #骨盤節座標系（原点はhip）
            e_y0_pelvis = lasi[frame_num,:] - rasi[frame_num,:]
            e_z_pelvis = (lumbar - hip_0)/np.linalg.norm(lumbar - hip_0)
            e_x_pelvis = np.cross(e_y0_pelvis, e_z_pelvis)/np.linalg.norm(np.cross(e_y0_pelvis, e_z_pelvis))
            e_y_pelvis = np.cross(e_z_pelvis, e_x_pelvis)
            rot_pelvis = np.array([e_x_pelvis, e_y_pelvis, e_z_pelvis]).T

            transformation_matrix = np.array([[e_x_pelvis[0], e_y_pelvis[0], e_z_pelvis[0], hip_0[0]],
                                                [e_x_pelvis[1], e_y_pelvis[1], e_z_pelvis[1], hip_0[1]],
                                                [e_x_pelvis[2], e_y_pelvis[2], e_z_pelvis[2], hip_0[2]],
                                                [0,       0,       0,       1]])

            #モーキャプの座標系に変換してもう一度計算
            rthigh = np.dot(transformation_matrix, np.append(rthigh_pelvis, 1))[:3]
            lthigh = np.dot(transformation_matrix, np.append(lthigh_pelvis, 1))[:3]
            hip = (rthigh + lthigh) / 2

            e_y0_pelvis = lthigh - rthigh
            e_z_pelvis = (lumbar - hip)/np.linalg.norm(lumbar - hip)
            e_x_pelvis = np.cross(e_y0_pelvis, e_z_pelvis)/np.linalg.norm(np.cross(e_y0_pelvis, e_z_pelvis))
            e_y_pelvis = np.cross(e_z_pelvis, e_x_pelvis)
            rot_pelvis = np.array([e_x_pelvis, e_y_pelvis, e_z_pelvis]).T

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
            e_z_rshank = (rshank - rfoot)/np.linalg.norm(rshank - rfoot)
            e_x_rshank = np.cross(e_y0_rshank, e_z_rshank)/np.linalg.norm(np.cross(e_y0_rshank, e_z_rshank))
            e_y_rshank = np.cross(e_z_rshank, e_x_rshank)
            rot_rshank = np.array([e_x_rshank, e_y_rshank, e_z_rshank]).T

            #左下腿節座標系（原点はlshank）
            e_y0_lshank = lknee[frame_num, :] - lknee2[frame_num, :]
            e_z_lshank = (lshank - lfoot)/np.linalg.norm(lshank - lfoot)
            e_x_lshank = np.cross(e_y0_lshank, e_z_lshank)/np.linalg.norm(np.cross(e_y0_lshank, e_z_lshank))
            e_y_lshank = np.cross(e_z_lshank, e_x_lshank)
            rot_lshank = np.array([e_x_lshank, e_y_lshank, e_z_lshank]).T

            #右足節座標系 AIST参照（原点はrfoot）
            e_z_rfoot = (rtoe[frame_num,:] - rhee[frame_num,:]) / np.linalg.norm(rtoe[frame_num,:] - rhee[frame_num,:])
            e_y0_rfoot = rank[frame_num,:] - rank2[frame_num,:]
            e_x_rfoot = np.cross(e_z_rfoot, e_y0_rfoot)/np.linalg.norm(np.cross(e_z_rfoot, e_y0_rfoot))
            e_y_rfoot = np.cross(e_z_rfoot, e_x_rfoot)
            rot_rfoot = np.array([e_x_rfoot, e_y_rfoot, e_z_rfoot]).T

            #左足節座標系 AIST参照（原点はlfoot）
            e_z_lfoot = (ltoe[frame_num,:] - lhee[frame_num, :]) / np.linalg.norm(ltoe[frame_num,:] - lhee[frame_num, :])
            e_y0_lfoot = lank2[frame_num,:] - lank[frame_num,:]
            e_x_lfoot = np.cross(e_z_lfoot, e_y0_lfoot)/np.linalg.norm(np.cross(e_z_lfoot, e_y0_lfoot))
            e_y_lfoot = np.cross(e_z_lfoot, e_x_lfoot)
            rot_lfoot = np.array([e_x_lfoot, e_y_lfoot, e_z_lfoot]).T

            # as_eulerが大文字(内因性の回転角度)となるよう設定
            r_hip_realative_rotation = np.dot(np.linalg.inv(rot_pelvis), rot_rthigh)
            r_hip_angle = R.from_matrix(r_hip_realative_rotation).as_euler('YZX', degrees=True)[0]
            l_hip_realative_rotation = np.dot(np.linalg.inv(rot_pelvis), rot_lthigh)
            l_hip_angle = R.from_matrix(l_hip_realative_rotation).as_euler('YZX', degrees=True)[0]
            r_knee_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rthigh)
            r_knee_angle =  R.from_matrix(r_knee_realative_rotation).as_euler('YZX', degrees=True)[0]
            l_knee_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lthigh)
            l_knee_angle = R.from_matrix(l_knee_realative_rotation).as_euler('YZX', degrees=True)[0]
            r_ankle_realative_rotation = np.dot(np.linalg.inv(rot_rshank), rot_rfoot)
            r_ankle_angle = R.from_matrix(r_ankle_realative_rotation).as_euler('YZX', degrees=True)[0]
            l_ankle_realative_rotation = np.dot(np.linalg.inv(rot_lshank), rot_lfoot)
            l_ankle_angle = R.from_matrix(l_ankle_realative_rotation).as_euler('YZX', degrees=True)[0]

            r_hip_angle = 360 + r_hip_angle if r_hip_angle < 0 else r_hip_angle
            l_hip_angle = 360 + l_hip_angle if l_hip_angle < 0 else l_hip_angle
            r_knee_angle = 360 + r_knee_angle if r_knee_angle < 0 else r_knee_angle
            l_knee_angle = 360 + l_knee_angle if l_knee_angle < 0 else l_knee_angle
            r_ankle_angle = 360 + r_ankle_angle if r_ankle_angle < 0 else r_ankle_angle
            l_ankle_angle = 360 + l_ankle_angle if l_ankle_angle < 0 else l_ankle_angle

            r_hip_angle = 180 - r_hip_angle
            l_hip_angle = 180 - l_hip_angle
            r_knee_angle = 180 - r_knee_angle
            l_knee_angle = 180 - l_knee_angle
            r_ankle_angle = 90 - r_ankle_angle
            l_ankle_angle = 90 - l_ankle_angle

            angles = [r_hip_angle, l_hip_angle, r_knee_angle, l_knee_angle, r_ankle_angle, l_ankle_angle]
            angle_list.append(angles)

            #骨盤とかかとの距離計算
            dist = np.linalg.norm(lhee[frame_num, :] - hip[:])
            bector = lhee[frame_num, :] - hip[:]
            dist_list.append(dist)
            bector_list.append(bector)

        # データ保存
        print("Saving data...")
        
        # 角度データの保存
        angle_array = np.array(angle_list)
        angle_df = pd.DataFrame({
            "r_hip_angle": angle_array[:, 0], 
            "r_knee_angle": angle_array[:, 2], 
            "r_ankle_angle": angle_array[:, 4], 
            "l_hip_angle": angle_array[:, 1], 
            "l_knee_angle": angle_array[:, 3], 
            "l_ankle_angle": angle_array[:, 5]
        })
        angle_df.index = angle_df.index + full_range.start
        
        # 全マーカー座標データの保存
        coordinates_df = pd.DataFrame()
        for marker in available_markers:
            if marker in all_markers_data:
                coordinates_df[f"{marker}_X"] = all_markers_data[marker][:, 0]
                coordinates_df[f"{marker}_Y"] = all_markers_data[marker][:, 1]
                coordinates_df[f"{marker}_Z"] = all_markers_data[marker][:, 2]
        
        coordinates_df.index = coordinates_df.index + full_range.start
        
        # ファイル名に適切なサンプリング周波数を記載
        filename_suffix = "60Hz" if down_hz else "100Hz"
        angle_df.to_csv(os.path.join(os.path.dirname(csv_path), f"angle_{filename_suffix}_{os.path.basename(csv_path)}"))
        coordinates_df.to_csv(os.path.join(os.path.dirname(csv_path), f"coordinates_{filename_suffix}_{os.path.basename(csv_path)}"))
        
        # アニメーション作成（mp4形式）
        animation_path = os.path.join(os.path.dirname(csv_path), f"animation_{filename_suffix}_{os.path.splitext(os.path.basename(csv_path))[0]}.mp4")
        create_animation(all_markers_data, available_markers, full_range, animation_path)

        if down_hz:
            print("Processing initial contact detection...")
            bector_array = np.array(bector_list)
            lhee_pel_z = bector_array[:, 2]
            ic_df = pd.DataFrame({"frame":full_range, "lhee_pel_z":lhee_pel_z})
            ic_df = ic_df.sort_values(by="lhee_pel_z", ascending=False)
            ic_list = ic_df.head(30)["frame"].values
            print(f"ic_list = {ic_list}")

            filtered_list = []
            skip_values = set()
            for value in ic_list:
                if value in skip_values:
                    continue
                filtered_list.append(value)
                skip_values.update(range(value - 10, value + 11))
            filtered_list = sorted(filtered_list)
            print(f"フィルタリング後のリスト:{filtered_list}")
            np.save(os.path.join(os.path.dirname(csv_path), f"ic_frame_{os.path.splitext(os.path.basename(csv_path))[0]}"), filtered_list)

    print("\nAll files processed successfully!")

if __name__ == "__main__":
    main()