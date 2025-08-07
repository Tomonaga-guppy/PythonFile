import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

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
    # --- 1. パスと設定 ---
    video_dir = Path(r"G:\gait_pattern\20250807_br\Tpose")
    input_csv_path = video_dir / "keypoints_3d_distorted_35.csv"

    # プロットするフレーム番号を指定
    target_frame = 100

    # 出力する画像ファイル名
    output_image_path = video_dir / f"skeleton_3d_frame_{target_frame}_dist_35mm.png"

    print(f"\n{'='*60}")
    print(f"フレーム {target_frame} の3D骨格プロットを作成します。")
    print(f"{'='*60}")
    print(f"入力CSV: {input_csv_path}")
    print(f"出力画像: {output_image_path}")

    # --- 2. データの読み込み ---
    if not input_csv_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_csv_path}")
        return

    df = pd.read_csv(input_csv_path)
    df.dropna(subset=['x', 'y', 'z'], inplace=True)

    if df.empty:
        print("エラー: 有効な3D座標データがCSVにありません。")
        return

    # -10000から10000の範囲外にある異常なデータを除外
    df = df[
        (df['x'] > -10000) & (df['x'] < 10000) &
        (df['y'] > -10000) & (df['y'] < 10000) &
        (df['z'] > -10000) & (df['z'] < 10000)
    ]

    # --- 3. 指定フレームのデータを抽出 ---
    frame_data = df[df['frame'] == target_frame]
    if frame_data.empty:
        print(f"エラー: フレーム {target_frame} のデータが見つかりません。")
        return

    # 座標データを取得
    coords = frame_data[['keypoint_id', 'x', 'y', 'z']].set_index('keypoint_id')

    # --- 4. 3Dプロットの準備と描画 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # キーポイントを点でプロット
    ax.scatter(coords['x'], coords['y'], coords['z'], c='red', marker='o', label='Keypoints')

    # 骨格を線でプロット
    for i, (p1_id, p2_id) in enumerate(SKELETON_CONNECTIONS):
        if p1_id in coords.index and p2_id in coords.index:
            p1 = coords.loc[p1_id]
            p2 = coords.loc[p2_id]
            # 最初の線にだけラベルを追加して凡例に表示
            label = 'Skeleton' if i == 0 else ''
            ax.plot([p1['x'], p2['x']], [p1['y'], p2['y']], [p1['z'], p2['z']], 'b-', label=label)

    # --- 5. プロットの体裁を整える ---
    # 軸の範囲をデータに基づいて設定
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    z_min, z_max = df['z'].min(), df['z'].max()

    # x_min, x_max = -5000, 5000
    # y_min, y_max = -5000, 5000
    # z_min, z_max = -5000, 5000

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range) * 1.1 # 10%のマージン

    ax.set_xlim((x_min + x_max - max_range) / 2, (x_min + x_max + max_range) / 2)
    ax.set_ylim((y_min + y_max - max_range) / 2, (y_min + y_max + max_range) / 2)
    ax.set_zlim((z_min + z_max - max_range) / 2, (z_min + z_max + max_range) / 2)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"3D Skeleton (Frame: {target_frame})")
    ax.legend() # 凡例を表示

    # Y軸とZ軸の向きを一般的な3D座標系に合わせる
    ax.invert_zaxis()
    # # 見やすい角度に視点を調整
    # ax.view_init(elev=20, azim=-75)

    # --- 6. 保存と表示 ---
    plt.savefig(output_image_path, dpi=150)
    print(f"\nプロットを画像として保存しました。")
    print(f"-> {output_image_path}")

    plt.show() # プロットを画面に表示

if __name__ == '__main__':
    main()