import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm

# --- 骨格情報の定義 ---
BODY_25_MAPPING = {
    0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
    10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
    15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
    20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel",
}

SKELETON_CONNECTIONS = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
    (14, 19), (14, 20), (14, 21), (11, 22), (11, 23), (11, 24)
]

def main():
    # --- 1. パス設定 ---
    video_dir = Path(r"G:\gait_pattern\20250807_br\ngait")
    input_csv_path = video_dir / "keypoints_3d_world_origin.csv"
    output_video_path = video_dir / "skeleton_3d_animation.mp4"

    print(f"\n{'='*60}")
    print("3D骨格アニメーションの作成を開始します。")
    print(f"{'='*60}")
    print(f"入力CSV: {input_csv_path}")
    print(f"出力動画: {output_video_path}")

    # --- 2. データの読み込みと前処理 ---
    if not input_csv_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_csv_path}")
        return

    df = pd.read_csv(input_csv_path)
    df.dropna(subset=['x', 'y', 'z'], inplace=True)

    df = df[
        (df['x'] > -10000) & (df['x'] < 10000) &
        (df['y'] > -10000) & (df['y'] < 10000) &
        (df['z'] > -10000) & (df['z'] < 10000)
    ]

    confidence_threshold = 0.6
    original_rows = len(df)
    df = df[(df['confidence_L'] >= confidence_threshold) & (df['confidence_R'] >= confidence_threshold)]
    print(f"\n信頼度フィルタリング (閾値: {confidence_threshold}):")
    print(f"  {original_rows}行 -> {len(df)}行 にフィルタリングしました。")

    if df.empty:
        print("エラー: フィルタリング後に有効な3D座標データが残っていません。")
        return

    frames = sorted(df['frame'].unique())

    # --- 3. 3Dプロットの準備 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    z_min, z_max = df['z'].min(), df['z'].max()

    x_center, y_center, z_center = df['x'].mean(), df['y'].mean(), df['z'].mean()
    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0

    # --- 4. アニメーションの更新関数を定義 ---
    def update(frame_num):
        ax.cla()

        frame_data = df[df['frame'] == frame_num]
        if frame_data.empty:
            return

        coords = frame_data.set_index('keypoint_id')

        # YとZを入れ替えてプロット
        ax.scatter(coords['x'], coords['z'], coords['y'], c='red', marker='o')

        for p1_id, p2_id in SKELETON_CONNECTIONS:
            if p1_id in coords.index and p2_id in coords.index:
                p1 = coords.loc[p1_id]
                p2 = coords.loc[p2_id]
                ax.plot([p1['x'], p2['x']], [p1['z'], p2['z']], [p1['y'], p2['y']], 'b-')

        # 毎回、軸の範囲とラベルを再設定
        ax.set_xlim(x_center - max_range, x_center + max_range)
        ax.set_ylim(z_center - max_range, z_center + max_range)
        ax.set_zlim(y_center - max_range, y_center + max_range)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z (mm)")
        ax.set_zlabel("Y (mm)")
        ax.set_title(f"3D Skeleton Animation (Frame: {frame_num})")

        # ★★★ 変更点: 軸の向きを揃える ★★★
        ax.invert_xaxis()
        ax.invert_zaxis()

    # --- 5. アニメーションの生成と保存 ---
    print(f"\nアニメーションを生成しています... (フレーム数: {len(frames)})")
    ani = FuncAnimation(fig, update, frames=frames, interval=1000/60)

    progress_callback = lambda i, n: pbar.update()
    with tqdm(total=len(frames), desc="動画ファイル保存中") as pbar:
        ani.save(output_video_path, writer='ffmpeg', fps=60, progress_callback=progress_callback)

    plt.close(fig)
    print(f"\nアニメーションの保存が完了しました。")
    print(f"-> {output_video_path}")

if __name__ == '__main__':
    main()
