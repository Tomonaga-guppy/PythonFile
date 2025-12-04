import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import m_opti as opti

def downsample_to_60hz(data, sampling_freq=100):
    """
    100Hzのデータを60Hzにダウンサンプリング
    
    Parameters:
    -----------
    data : np.ndarray
        shape (n_frames, n_joints, 3) または (n_frames, 3)
    sampling_freq : int
        元のサンプリング周波数 [Hz]
    
    Returns:
    --------
    downsampled_data : np.ndarray
        60Hzにダウンサンプリングされたデータ
    """
    ratio = sampling_freq / 60.0  # 100/60 = 1.666...
    n_frames_60hz = int(len(data) / ratio)
    
    if data.ndim == 3:
        # (n_frames, n_joints, 3)
        downsampled = np.zeros((n_frames_60hz, data.shape[1], data.shape[2]))
        for joint_idx in range(data.shape[1]):
            for coord_idx in range(data.shape[2]):
                frame_indices = np.arange(n_frames_60hz) * ratio
                downsampled[:, joint_idx, coord_idx] = np.interp(
                    frame_indices, 
                    np.arange(len(data)), 
                    data[:, joint_idx, coord_idx]
                )
    elif data.ndim == 2:
        # (n_frames, 3)
        downsampled = np.zeros((n_frames_60hz, data.shape[1]))
        for coord_idx in range(data.shape[1]):
            frame_indices = np.arange(n_frames_60hz) * ratio
            downsampled[:, coord_idx] = np.interp(
                frame_indices,
                np.arange(len(data)),
                data[:, coord_idx]
            )
    else:
        raise ValueError(f"Unsupported data dimension: {data.ndim}")
    
    return downsampled

def create_stick_figure_comparison_video(csv_path_dir, start_frame, end_frame, 
                                         geometry_json_path, frame_offset_cut=394):
    """
    モーションキャプチャとOpenPose3Dのスティックフィギュア比較動画を作成
    3つの視点（横、前、斜め）から保存
    
    Parameters:
    -----------
    csv_path_dir : Path
        CSVファイルのディレクトリパス
    start_frame : int
        開始フレーム（100Hz基準）
    end_frame : int
        終了フレーム（100Hz基準）
    geometry_json_path : Path
        geometry.jsonのパス
    frame_offset_cut : int
        LED発光フレームと動画トリミング開始フレームの差（60Hz）
    """
    
    # モーキャプデータの読み込み
    csv_paths = list(csv_path_dir.glob("*.csv"))
    csv_paths = [path for path in csv_paths if not path.name.startswith("marker_set_")]
    csv_paths = [path for path in csv_paths if not path.name.startswith("angle_")]
    csv_paths = [path for path in csv_paths if not path.name.startswith("before_")]
    csv_paths = [path for path in csv_paths if not path.name.startswith("after_")]
    csv_paths = [path for path in csv_paths if not path.name.startswith("normalized_")]
    csv_paths = [path for path in csv_paths if not path.name.startswith("gait_parameters_")]
    csv_paths = [path for path in csv_paths if not path.name.startswith("symmetry_indices_")]
    
    if len(csv_paths) == 0:
        print("CSVファイルが見つかりません")
        return
    
    csv_path = csv_paths[0]
    print(f"Processing: {csv_path}")
    
    # モーキャプデータの読み込み（100Hz）
    keypoints_mocap, full_range = opti.read_3d_optitrack(
        csv_path, start_frame, end_frame, geometry_path=geometry_json_path
    )
    
    # 関節点にバターワースフィルタを適用（100Hz）
    sampling_freq = 100
    
    # 主要な関節点を抽出してフィルタリング
    rasi = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 10, x], order=4, cutoff_freq=6, 
                                                frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    lasi = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 2, x], order=4, cutoff_freq=6, 
                                                frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    rpsi = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 14, x], order=4, cutoff_freq=6, 
                                                frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    lpsi = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 6, x], order=4, cutoff_freq=6, 
                                                frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    rank = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 8, x], order=4, cutoff_freq=6, 
                                                frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    lank = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 0, x], order=4, cutoff_freq=6, 
                                                frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    rknee = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 12, x], order=4, cutoff_freq=6, 
                                                 frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    lknee = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 4, x], order=4, cutoff_freq=6, 
                                                 frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    rtoe = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 15, x], order=4, cutoff_freq=6, 
                                                frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    ltoe = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 7, x], order=4, cutoff_freq=6, 
                                                frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    rhee = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 11, x], order=4, cutoff_freq=6, 
                                                frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    lhee = np.array([opti.butter_lowpass_filter(keypoints_mocap[:, 3, x], order=4, cutoff_freq=6, 
                                                frame_list=full_range, sampling_freq=sampling_freq) for x in range(3)]).T
    
    # 股関節中心を計算（簡易版）
    hip_center = (rasi + lasi + rpsi + lpsi) / 4
    rhip = (rasi + rpsi) / 2
    lhip = (lasi + lpsi) / 2
    
    # 60Hzにダウンサンプリング
    hip_center_60hz = downsample_to_60hz(hip_center, 100)
    rhip_60hz = downsample_to_60hz(rhip, 100)
    lhip_60hz = downsample_to_60hz(lhip, 100)
    rknee_60hz = downsample_to_60hz(rknee, 100)
    lknee_60hz = downsample_to_60hz(lknee, 100)
    rank_60hz = downsample_to_60hz(rank, 100)
    lank_60hz = downsample_to_60hz(lank, 100)
    rtoe_60hz = downsample_to_60hz(rtoe, 100)
    ltoe_60hz = downsample_to_60hz(ltoe, 100)
    rhee_60hz = downsample_to_60hz(rhee, 100)
    lhee_60hz = downsample_to_60hz(lhee, 100)
    
    # OpenPoseデータの読み込み
    openpose_npz_path = csv_path_dir.parent / "3d_kp_data_openpose_kalman.npz"
    if not openpose_npz_path.exists():
        print(f"OpenPose3Dデータが見つかりません: {openpose_npz_path}")
        return
    
    openpose_data = np.load(openpose_npz_path)
    op_frame = openpose_data['frame']
    op_filt_data = openpose_data['filt']  # shape: (num_frames, num_joints, 3)
    
    # タイミング合わせ
    base_point = 0
    hip_z_opti = hip_center[:, 2]
    base_passing_frame = np.argmax(hip_z_opti > base_point) + start_frame
    
    hip_z_op = op_filt_data[:, 8, 2]
    base_passing_idx_op = np.argmax(hip_z_op > base_point)
    base_passing_frame_op = op_frame[base_passing_idx_op]
    
    frame_offset_60Hz = base_passing_frame_op + frame_offset_cut - base_passing_frame * 0.6
    mc_frame_offset = (frame_offset_cut - frame_offset_60Hz) / 0.6
    
    print(f"フレームオフセット: {frame_offset_60Hz} (60Hz), {mc_frame_offset} (100Hz)")
    
    # OpenPoseの関節点を抽出
    neck_op = op_filt_data[:, 1, :]
    midhip_op = op_filt_data[:, 8, :]
    rhip_op = op_filt_data[:, 9, :]
    rknee_op = op_filt_data[:, 10, :]
    rankle_op = op_filt_data[:, 11, :]
    rhee_op = op_filt_data[:, 24, :]
    rtoe_op = (op_filt_data[:, 22, :] + op_filt_data[:, 23, :]) / 2
    lhip_op = op_filt_data[:, 12, :]
    lknee_op = op_filt_data[:, 13, :]
    lankle_op = op_filt_data[:, 14, :]
    lhee_op = op_filt_data[:, 21, :]
    ltoe_op = (op_filt_data[:, 19, :] + op_filt_data[:, 20, :]) / 2
    
    # フレーム範囲を計算
    start_frame_op = int((start_frame - mc_frame_offset) * 0.6)
    end_frame_op = int((end_frame - mc_frame_offset) * 0.6)
    
    # 有効範囲のチェック
    start_frame_op = max(0, start_frame_op)
    end_frame_op = min(len(op_filt_data), end_frame_op)
    n_frames = min(len(hip_center_60hz), end_frame_op - start_frame_op)
    
    print(f"動画フレーム数: {n_frames}")
    print(f"OpenPose範囲: {start_frame_op} - {end_frame_op}")
    
    # 3つの視点を定義: (elevation, azimuth, view_name)
    views = [
        (10, -90, 'side'),      # 横視点
        (10, 0, 'front'),       # 正面視点
        (20, -45, 'diagonal')   # 斜め視点
    ]
    
    output_dir = csv_path_dir.parent / "comparison_videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 各視点で動画を作成
    for elev, azim, view_name in views:
        print(f"\n視点 '{view_name}' の動画を作成中...")
        
        # アニメーション作成
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.cla()
            
            # 軸の設定
            ax.set_xlabel('X [m]', fontsize=12)
            ax.set_ylabel('Y [m]', fontsize=12)
            ax.set_zlabel('Z [m]', fontsize=12)
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1, 2)
            ax.set_zlim(-2, 1)
            ax.view_init(elev=elev, azim=azim)
            
            ax.set_title(f'Mocap (solid) vs OpenPose (dashed) - Frame: {frame} - View: {view_name}', 
                        fontsize=14, fontweight='bold')
            
            # Mocapのスティックフィギュア (実線)
            # 胴体
            ax.plot([hip_center_60hz[frame, 0], rhip_60hz[frame, 0]], 
                   [hip_center_60hz[frame, 1], rhip_60hz[frame, 1]], 
                   [hip_center_60hz[frame, 2], rhip_60hz[frame, 2]], 
                   'b-', linewidth=3, label='Mocap - Body', alpha=0.8)
            ax.plot([hip_center_60hz[frame, 0], lhip_60hz[frame, 0]], 
                   [hip_center_60hz[frame, 1], lhip_60hz[frame, 1]], 
                   [hip_center_60hz[frame, 2], lhip_60hz[frame, 2]], 
                   'b-', linewidth=3, alpha=0.8)
            
            # 右脚
            ax.plot([rhip_60hz[frame, 0], rknee_60hz[frame, 0]], 
                   [rhip_60hz[frame, 1], rknee_60hz[frame, 1]], 
                   [rhip_60hz[frame, 2], rknee_60hz[frame, 2]], 
                   'r-', linewidth=3, label='Mocap - Right', alpha=0.8)
            ax.plot([rknee_60hz[frame, 0], rank_60hz[frame, 0]], 
                   [rknee_60hz[frame, 1], rank_60hz[frame, 1]], 
                   [rknee_60hz[frame, 2], rank_60hz[frame, 2]], 
                   'r-', linewidth=3, alpha=0.8)
            ax.plot([rhee_60hz[frame, 0], rtoe_60hz[frame, 0]], 
                   [rhee_60hz[frame, 1], rtoe_60hz[frame, 1]], 
                   [rhee_60hz[frame, 2], rtoe_60hz[frame, 2]], 
                   'r-', linewidth=3, alpha=0.8)
            
            # 左脚
            ax.plot([lhip_60hz[frame, 0], lknee_60hz[frame, 0]], 
                   [lhip_60hz[frame, 1], lknee_60hz[frame, 1]], 
                   [lhip_60hz[frame, 2], lknee_60hz[frame, 2]], 
                   'g-', linewidth=3, label='Mocap - Left', alpha=0.8)
            ax.plot([lknee_60hz[frame, 0], lank_60hz[frame, 0]], 
                   [lknee_60hz[frame, 1], lank_60hz[frame, 1]], 
                   [lknee_60hz[frame, 2], lank_60hz[frame, 2]], 
                   'g-', linewidth=3, alpha=0.8)
            ax.plot([lhee_60hz[frame, 0], ltoe_60hz[frame, 0]], 
                   [lhee_60hz[frame, 1], ltoe_60hz[frame, 1]], 
                   [lhee_60hz[frame, 2], ltoe_60hz[frame, 2]], 
                   'g-', linewidth=3, alpha=0.8)
            
            # Mocap関節点
            ax.scatter(hip_center_60hz[frame, 0], hip_center_60hz[frame, 1], hip_center_60hz[frame, 2], 
                      c='blue', s=100, marker='o', edgecolors='black', linewidth=1.5, alpha=0.8)
            ax.scatter(rknee_60hz[frame, 0], rknee_60hz[frame, 1], rknee_60hz[frame, 2], 
                      c='red', s=100, marker='o', edgecolors='black', linewidth=1.5, alpha=0.8)
            ax.scatter(lknee_60hz[frame, 0], lknee_60hz[frame, 1], lknee_60hz[frame, 2], 
                      c='green', s=100, marker='o', edgecolors='black', linewidth=1.5, alpha=0.8)
            
            # OpenPoseのスティックフィギュア (破線、mm → m に変換)
            op_frame_idx = frame + start_frame_op
            if op_frame_idx < len(op_filt_data):
                scale = 0.001  # mm to m
                
                # 胴体
                ax.plot([midhip_op[op_frame_idx, 0]*scale, neck_op[op_frame_idx, 0]*scale], 
                       [midhip_op[op_frame_idx, 1]*scale, neck_op[op_frame_idx, 1]*scale], 
                       [midhip_op[op_frame_idx, 2]*scale, neck_op[op_frame_idx, 2]*scale], 
                       'b--', linewidth=2.5, label='OpenPose - Body', alpha=0.7)
                
                # 右脚
                ax.plot([midhip_op[op_frame_idx, 0]*scale, rhip_op[op_frame_idx, 0]*scale], 
                       [midhip_op[op_frame_idx, 1]*scale, rhip_op[op_frame_idx, 1]*scale], 
                       [midhip_op[op_frame_idx, 2]*scale, rhip_op[op_frame_idx, 2]*scale], 
                       'r--', linewidth=2.5, label='OpenPose - Right', alpha=0.7)
                ax.plot([rhip_op[op_frame_idx, 0]*scale, rknee_op[op_frame_idx, 0]*scale], 
                       [rhip_op[op_frame_idx, 1]*scale, rknee_op[op_frame_idx, 1]*scale], 
                       [rhip_op[op_frame_idx, 2]*scale, rknee_op[op_frame_idx, 2]*scale], 
                       'r--', linewidth=2.5, alpha=0.7)
                ax.plot([rknee_op[op_frame_idx, 0]*scale, rankle_op[op_frame_idx, 0]*scale], 
                       [rknee_op[op_frame_idx, 1]*scale, rankle_op[op_frame_idx, 1]*scale], 
                       [rknee_op[op_frame_idx, 2]*scale, rankle_op[op_frame_idx, 2]*scale], 
                       'r--', linewidth=2.5, alpha=0.7)
                ax.plot([rhee_op[op_frame_idx, 0]*scale, rtoe_op[op_frame_idx, 0]*scale], 
                       [rhee_op[op_frame_idx, 1]*scale, rtoe_op[op_frame_idx, 1]*scale], 
                       [rhee_op[op_frame_idx, 2]*scale, rtoe_op[op_frame_idx, 2]*scale], 
                       'r--', linewidth=2.5, alpha=0.7)
                
                # 左脚
                ax.plot([midhip_op[op_frame_idx, 0]*scale, lhip_op[op_frame_idx, 0]*scale], 
                       [midhip_op[op_frame_idx, 1]*scale, lhip_op[op_frame_idx, 1]*scale], 
                       [midhip_op[op_frame_idx, 2]*scale, lhip_op[op_frame_idx, 2]*scale], 
                       'g--', linewidth=2.5, label='OpenPose - Left', alpha=0.7)
                ax.plot([lhip_op[op_frame_idx, 0]*scale, lknee_op[op_frame_idx, 0]*scale], 
                       [lhip_op[op_frame_idx, 1]*scale, lknee_op[op_frame_idx, 1]*scale], 
                       [lhip_op[op_frame_idx, 2]*scale, lknee_op[op_frame_idx, 2]*scale], 
                       'g--', linewidth=2.5, alpha=0.7)
                ax.plot([lknee_op[op_frame_idx, 0]*scale, lankle_op[op_frame_idx, 0]*scale], 
                       [lknee_op[op_frame_idx, 1]*scale, lankle_op[op_frame_idx, 1]*scale], 
                       [lknee_op[op_frame_idx, 2]*scale, lankle_op[op_frame_idx, 2]*scale], 
                       'g--', linewidth=2.5, alpha=0.7)
                ax.plot([lhee_op[op_frame_idx, 0]*scale, ltoe_op[op_frame_idx, 0]*scale], 
                       [lhee_op[op_frame_idx, 1]*scale, ltoe_op[op_frame_idx, 1]*scale], 
                       [lhee_op[op_frame_idx, 2]*scale, ltoe_op[op_frame_idx, 2]*scale], 
                       'g--', linewidth=2.5, alpha=0.7)
                
                # OpenPose関節点
                ax.scatter(midhip_op[op_frame_idx, 0]*scale, midhip_op[op_frame_idx, 1]*scale, 
                          midhip_op[op_frame_idx, 2]*scale, c='cyan', s=80, marker='s', 
                          edgecolors='black', linewidth=1.5, alpha=0.7)
                ax.scatter(rknee_op[op_frame_idx, 0]*scale, rknee_op[op_frame_idx, 1]*scale, 
                          rknee_op[op_frame_idx, 2]*scale, c='orange', s=80, marker='s', 
                          edgecolors='black', linewidth=1.5, alpha=0.7)
                ax.scatter(lknee_op[op_frame_idx, 0]*scale, lknee_op[op_frame_idx, 1]*scale, 
                          lknee_op[op_frame_idx, 2]*scale, c='lime', s=80, marker='s', 
                          edgecolors='black', linewidth=1.5, alpha=0.7)
            
            # 凡例を最初のフレームのみ表示
            if frame == 0:
                ax.legend(loc='upper right', fontsize=10)
            
            return fig,
        
        # アニメーション作成
        anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/60, blit=False)
        
        # 動画保存
        output_path = output_dir / f"stick_comparison_{view_name}_{csv_path.stem}.mp4"
        
        writer = FFMpegWriter(fps=60, metadata=dict(artist='Me'), bitrate=5000)
        anim.save(output_path, writer=writer)
        
        print(f"動画を保存しました: {output_path}")
        plt.close()
    
    print(f"\n全ての動画を保存しました: {output_dir}")

def main():
    # 設定
    csv_path_dir = Path(r"G:\gait_pattern\20250811_br\sub1\thera0-3\mocap")
    geometry_json_path = Path(r"G:\gait_pattern\20250811_br\sub0\thera0-16\mocap\geometry.json")
    
    # フレーム範囲の設定
    if str(csv_path_dir) == r"G:\gait_pattern\20250811_br\sub1\thera0-3\mocap":
        start_frame = 943
        end_frame = 1400
        frame_offset_cut = 394 + 5
    else:
        start_frame = 0
        end_frame = 100
        frame_offset_cut = 394
    
    # スティックフィギュア比較動画を作成
    create_stick_figure_comparison_video(
        csv_path_dir, 
        start_frame, 
        end_frame, 
        geometry_json_path,
        frame_offset_cut
    )

if __name__ == "__main__":
    main()