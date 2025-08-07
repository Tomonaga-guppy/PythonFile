import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm

# --- 骨格情報の定義 ---
# 5_triangulate.py と同じマッピングを使用
BODY_25_MAPPING = {
    0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
    10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
    15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
    20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel",
}

# BODY_25モデルの骨格を定義する。各タプルは接続する2つのキーポイントのIDを示す。
SKELETON_CONNECTIONS = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
    (14, 19), (14, 20), (14, 21), (11, 22), (11, 23), (11, 24)
]

def main():
    # --- 1. パス設定 ---
    video_dir = Path(r"G:\gait_pattern\20250807_br\Tpose")
    input_csv_path = video_dir / "keypoints_3d_49d5_udOP.csv"
    output_video_path = video_dir / "skeleton_3d_animation_49d5_udOP.mp4"

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
    # NaN値を含む行は描画に問題を起こすため、座標計算前に除外
    df.dropna(subset=['x', 'y', 'z'], inplace=True)

    # -10000から10000の範囲外にある異常なデータを除外
    df = df[
        (df['x'] > -10000) & (df['x'] < 10000) &
        (df['y'] > -10000) & (df['y'] < 10000) &
        (df['z'] > -10000) & (df['z'] < 10000)
    ]

    print(f"フィルタリング後のdf:\n{df}")

    if df.empty:
        print("エラー: 有効な3D座標データがCSVにありません。")
        return

    frames = df['frame'].unique()

    # --- 3. 3Dプロットの準備 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 軸の範囲をデータ全体から決定し、固定する
    # これによりアニメーション中に視点がぶれないようにする
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    z_min, z_max = df['z'].min(), df['z'].max()

    print(f"データの範囲: X({x_min}, {x_max}), Y({y_min}, {y_max}), Z({z_min}, {z_max})")

    # 見やすいように少しマージンを追加
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range)

    ax.set_xlim((x_min + x_max - max_range) / 2, (x_min + x_max + max_range) / 2)
    ax.set_ylim((y_min + y_max - max_range) / 2, (y_min + y_max + max_range) / 2)
    ax.set_zlim((z_min + z_max - max_range) / 2, (z_min + z_max + max_range) / 2)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D Skeleton Animation")
    # Y軸とZ軸の向きを一般的な3D座標系に合わせる
    ax.invert_zaxis()


    # --- 4. アニメーションの更新関数を定義 ---
    def update(frame_num):
        ax.cla() # 前のフレームの描画をクリア

        # 現在のフレームのデータを抽出
        frame_data = df[df['frame'] == frame_num]
        if frame_data.empty:
            return

        # 座標データを取得
        coords = frame_data[['keypoint_id', 'x', 'y', 'z']].set_index('keypoint_id')

        # キーポイントを点でプロット
        ax.scatter(coords['x'], coords['y'], coords['z'], c='red', marker='o')

        # 骨格を線でプロット
        for p1_id, p2_id in SKELETON_CONNECTIONS:
            # 接続する両方のキーポイントが存在する場合のみ線を描画
            if p1_id in coords.index and p2_id in coords.index:
                p1 = coords.loc[p1_id]
                p2 = coords.loc[p2_id]
                ax.plot([p1['x'], p2['x']], [p1['y'], p2['y']], [p1['z'], p2['z']], 'b-')

        # 毎回、軸の範囲とラベルを再設定
        ax.set_xlim((x_min + x_max - max_range) / 2, (x_min + x_max + max_range) / 2)
        ax.set_ylim((y_min + y_max - max_range) / 2, (y_min + y_max + max_range) / 2)
        ax.set_zlim((z_min + z_max - max_range) / 2, (z_min + z_max + max_range) / 2)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z (mm)")
        ax.set_zlabel("Y (mm)")
        ax.set_title(f"3D Skeleton Animation (Frame: {frame_num})")
        # ax.invert_zaxis()
        # ax.view_init(elev=0, azim=0)

    # --- 5. アニメーションの生成と保存 ---
    print("\nアニメーションを生成しています... (フレーム数: {})".format(len(frames)))
    # frames=frames を指定することで、実際にデータが存在するフレームのみを対象にする
    ani = FuncAnimation(fig, update, frames=frames, interval=1000/60) # intervalはミリ秒 (60fps相当)

    # tqdmを使ってプログレスバーを表示
    progress_callback = lambda i, n: pbar.update()
    with tqdm(total=len(frames), desc="動画ファイル保存中") as pbar:
        ani.save(output_video_path, writer='ffmpeg', fps=60, progress_callback=progress_callback)

    plt.close(fig) # メモリ解放
    print(f"\nアニメーションの保存が完了しました。")
    print(f"-> {output_video_path}")

if __name__ == '__main__':
    main()