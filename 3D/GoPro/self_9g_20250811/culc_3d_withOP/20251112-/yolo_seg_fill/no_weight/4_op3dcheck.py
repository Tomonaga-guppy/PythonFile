"""
OpenPose 3Dデータの関節位置とベクトルをインタラクティブに描画するスクリプト
スライダーでフレームを変更可能
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # インタラクティブバックエンドに変更
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path

# =============================================================================
# 設定
# =============================================================================
FRAME_RATE = 60
VECTOR_LENGTH = 300  # ベクトルの長さ [mm]

# =============================================================================
# OpenPoseスケルトン定義
# =============================================================================
def get_lower_body_connections():
    """下半身のスケルトン接続定義"""
    connections = [
        (8, 9),   # MidHip -> RHip
        (8, 12),  # MidHip -> LHip
        (9, 10),  # RHip -> RKnee
        (10, 11), # RKnee -> RAnkle
        (12, 13), # LHip -> LKnee
        (13, 14), # LKnee -> LAnkle
        (11, 24), # RAnkle -> RHeel
        (11, 22), # RAnkle -> RBigToe
        (14, 21), # LAnkle -> LHeel
        (14, 19), # LAnkle -> LBigToe
        (1, 8),   # Neck -> MidHip
    ]
    colors = [
        'green',   # MidHip -> RHip
        'green',   # MidHip -> LHip
        'red',     # RHip -> RKnee
        'red',     # RKnee -> RAnkle
        'blue',    # LHip -> LKnee
        'blue',    # LKnee -> LAnkle
        'red',     # RAnkle -> RHeel
        'red',     # RAnkle -> RBigToe
        'blue',    # LAnkle -> LHeel
        'blue',    # LAnkle -> LBigToe
        'purple',  # Neck -> MidHip
    ]
    return connections, colors


def normalize_vector(vec):
    """ベクトルを正規化"""
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


class InteractiveVectorViewer:
    """インタラクティブなベクトルビューア"""
    
    def __init__(self, data_3d, title="OpenPose 3D Vectors"):
        self.data_3d = data_3d
        self.title = title
        self.num_frames = len(data_3d)
        self.current_frame = 0
        self.is_playing = False
        self.timer = None
        
        # Figure作成
        self.fig = plt.figure(figsize=(16, 12))
        
        # 3Dプロット用のAxes（スライダー用スペースを確保）
        self.ax = self.fig.add_axes([0.05, 0.15, 0.9, 0.8], projection='3d')
        
        # 軸設定
        self.ax.set_xlim(-2000, 2000)
        self.ax.set_ylim(-2000, 2000)
        self.ax.set_zlim(0, 2000)
        self.ax.set_xlabel('Z-axis (mm) - Forward', fontsize=12, labelpad=10)
        self.ax.set_ylabel('X-axis (mm) - Sideways', fontsize=12, labelpad=10)
        self.ax.set_zlabel('Y-axis (mm) - Up', fontsize=12, labelpad=10)
        self.ax.view_init(elev=10, azim=45)
        self.ax.set_autoscale_on(False)
        
        # スケルトン線を初期化
        self.connections, self.skeleton_colors = get_lower_body_connections()
        self.skeleton_lines = [self.ax.plot([], [], [], color=c, lw=2.5)[0] 
                               for c in self.skeleton_colors]
        
        # 関節点
        self.joint_scatter = self.ax.scatter([], [], [], c='green', s=30, 
                                              depthshade=True, zorder=5)
        
        # ベクトル
        self.vector_colors = ['purple', 'cyan', 'orange', 'magenta', 
                              'darkred', 'darkblue', 'salmon', 'lightblue', 'coral', 'skyblue']
        self.vector_labels = ['pel_vec', 'n_axis', 'n_axis_adab', 'n_axis_inex',
                              'thigh_R', 'thigh_L', 'shank_R', 'shank_L', 'foot_R', 'foot_L']
        self.vector_linewidths = [2, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        
        self.vector_quivers = []
        for color, label, lw in zip(self.vector_colors, self.vector_labels, self.vector_linewidths):
            lc = Line3DCollection([], colors=[color], linewidths=lw, label=label)
            self.ax.add_collection3d(lc)
            self.vector_quivers.append(lc)
        
        # 凡例
        self.ax.legend(loc='upper right', fontsize=8)
        
        # タイトル
        self.title_text = self.ax.text2D(0.5, 0.95, title, transform=self.ax.transAxes, 
                                          fontsize=16, ha='center')
        self.frame_text = self.ax.text2D(0.02, 0.02, "", transform=self.ax.transAxes, 
                                          fontsize=12, va='bottom',
                                          bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))
        
        # スライダー
        self.ax_slider = self.fig.add_axes([0.15, 0.05, 0.55, 0.03])
        self.slider = Slider(
            ax=self.ax_slider,
            label='Frame',
            valmin=0,
            valmax=self.num_frames - 1,
            valinit=0,
            valstep=1
        )
        self.slider.on_changed(self.on_slider_change)
        
        # 再生/停止ボタン
        self.ax_play = self.fig.add_axes([0.75, 0.05, 0.08, 0.03])
        self.btn_play = Button(self.ax_play, '▶ Play')
        self.btn_play.on_clicked(self.toggle_play)
        
        # 前へ/次へボタン
        self.ax_prev = self.fig.add_axes([0.84, 0.05, 0.05, 0.03])
        self.btn_prev = Button(self.ax_prev, '◀')
        self.btn_prev.on_clicked(self.prev_frame)
        
        self.ax_next = self.fig.add_axes([0.90, 0.05, 0.05, 0.03])
        self.btn_next = Button(self.ax_next, '▶')
        self.btn_next.on_clicked(self.next_frame)
        
        # 初期フレームを描画
        self.update_frame(0)
    
    def transform(self, pt):
        """座標変換 (X, Y, Z) -> (Z, X, Y)"""
        return np.array([pt[2], pt[0], pt[1]])
    
    def update_frame(self, frame_idx):
        """フレームを更新"""
        self.current_frame = int(frame_idx)
        keypoints = self.data_3d[self.current_frame]
        
        # 関節点の抽出
        neck = keypoints[1]
        midhip = keypoints[8]
        rhip = keypoints[9]
        rknee = keypoints[10]
        rankle = keypoints[11]
        lhip = keypoints[12]
        lknee = keypoints[13]
        lankle = keypoints[14]
        rhee = keypoints[24]
        lhee = keypoints[21]
        rtoe = (keypoints[22] + keypoints[23]) / 2
        ltoe = (keypoints[19] + keypoints[20]) / 2
        
        # ベクトル計算
        pel_vec = neck - midhip
        thigh_r = rknee - rhip
        thigh_l = lknee - lhip
        shank_r = rankle - rknee
        shank_l = lankle - lknee
        foot_r = rtoe - rhee
        foot_l = ltoe - lhee
        
        # 回転軸
        n_axis = rhip - lhip
        n_axis_adab = np.cross(pel_vec, n_axis)
        n_axis_inex = pel_vec
        
        # スケルトン描画
        for i, (start_idx, end_idx) in enumerate(self.connections):
            start = self.transform(keypoints[start_idx])
            end = self.transform(keypoints[end_idx])
            if not (np.isnan(start).any() or np.isnan(end).any()):
                self.skeleton_lines[i].set_data_3d([start[0], end[0]], 
                                                   [start[1], end[1]], 
                                                   [start[2], end[2]])
            else:
                self.skeleton_lines[i].set_data_3d([], [], [])
        
        # 関節点描画
        valid_joints = []
        joint_indices = [1, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]
        for idx in joint_indices:
            pt = self.transform(keypoints[idx])
            if not np.isnan(pt).any():
                valid_joints.append(pt)
        
        if valid_joints:
            valid_joints = np.array(valid_joints)
            self.joint_scatter._offsets3d = (valid_joints[:, 0], 
                                             valid_joints[:, 1], 
                                             valid_joints[:, 2])
        
        # ベクトル描画
        origin_midhip = self.transform(midhip)
        origin_rhip = self.transform(rhip)
        origin_lhip = self.transform(lhip)
        origin_rknee = self.transform(rknee)
        origin_lknee = self.transform(lknee)
        origin_rankle = self.transform(rankle)
        origin_lankle = self.transform(lankle)
        
        vectors_data = [
            (origin_midhip, pel_vec),
            (origin_midhip, n_axis),
            (origin_midhip, n_axis_adab),
            (origin_midhip, n_axis_inex),
            (origin_rhip, thigh_r),
            (origin_lhip, thigh_l),
            (origin_rknee, shank_r),
            (origin_lknee, shank_l),
            (origin_rankle, foot_r),
            (origin_lankle, foot_l),
        ]
        
        for i, (origin, vec) in enumerate(vectors_data):
            if not np.isnan(origin).any() and not np.isnan(vec).any():
                vec_norm = normalize_vector(vec)
                vec_transformed = self.transform(vec_norm * VECTOR_LENGTH)
                self.vector_quivers[i].set_segments([[[origin[0], origin[1], origin[2]],
                                                      [origin[0] + vec_transformed[0],
                                                       origin[1] + vec_transformed[1],
                                                       origin[2] + vec_transformed[2]]]])
            else:
                self.vector_quivers[i].set_segments([])
        
        # フレーム情報更新
        self.frame_text.set_text(f"Frame: {self.current_frame + 1}/{self.num_frames}")
        
        # 再描画
        self.fig.canvas.draw_idle()
    
    def on_slider_change(self, val):
        """スライダー変更時のコールバック"""
        self.update_frame(int(val))
    
    def toggle_play(self, event):
        """再生/停止切り替え"""
        if self.is_playing:
            self.is_playing = False
            self.btn_play.label.set_text('▶ Play')
            if self.timer is not None:
                self.timer.stop()
        else:
            self.is_playing = True
            self.btn_play.label.set_text('⏸ Stop')
            self.timer = self.fig.canvas.new_timer(interval=int(1000/FRAME_RATE))
            self.timer.add_callback(self.play_next_frame)
            self.timer.start()
    
    def play_next_frame(self):
        """次のフレームを再生"""
        if self.is_playing:
            next_frame = (self.current_frame + 1) % self.num_frames
            self.slider.set_val(next_frame)
    
    def prev_frame(self, event):
        """前のフレームへ"""
        prev_frame = max(0, self.current_frame - 1)
        self.slider.set_val(prev_frame)
    
    def next_frame(self, event):
        """次のフレームへ"""
        next_frame = min(self.num_frames - 1, self.current_frame + 1)
        self.slider.set_val(next_frame)
    
    def show(self):
        """表示"""
        plt.show()


def main():
    """メイン処理"""
    # データパス設定
    data_dir = Path(r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0")
    openpose_npz_path = data_dir / "3d_kp_data_openpose_yoloseg.npz"
    
    if not openpose_npz_path.exists():
        print(f"データが見つかりません: {openpose_npz_path}")
        return
    
    # データ読み込み
    print(f"データ読み込み: {openpose_npz_path}")
    data = np.load(openpose_npz_path)
    op_filt_data = data['butter_filt']
    valid_range = data.get('valid_frame_range', [0, len(op_filt_data) - 1])
    
    print(f"データ形状: {op_filt_data.shape}")
    print(f"有効フレーム範囲: {valid_range}")
    
    # 有効範囲のデータを抽出
    start_frame, end_frame = int(valid_range[0]), int(valid_range[1])
    op_filt_data_valid = op_filt_data[start_frame:end_frame + 1]
    
    # ビューア作成・表示
    viewer = InteractiveVectorViewer(
        op_filt_data_valid,
        title=f"OpenPose 3D Vectors (Frames {start_frame}-{end_frame})"
    )
    viewer.show()
    
    print("\n処理完了")


if __name__ == "__main__":
    main()