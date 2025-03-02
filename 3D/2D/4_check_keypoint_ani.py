from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import sys
from tqdm import tqdm

root_dir = Path(r"G:\gait_pattern\20241114_ota_test\gopro\sagi")


def animate_keypoints(data_dict, condition):
    """
    二人のキーポイント座標をアニメーションで表示する関数。

    Args:
        data_dict: キーポイントデータを含む辞書。
                   'sub*_df_filter_dict' の形式で、
                   各値は pandas DataFrame。
    """

    keys = list(data_dict.keys())
    people_num = len(keys)
    frames = data_dict[list(data_dict.keys())[0]].index

    # FigureとAxesオブジェクトを作成
    fig, ax = plt.subplots(figsize=(10, 8))  # サイズを調整
    ax.set_aspect('equal')  # アスペクト比を保持

    # 描画範囲の計算
    x_min, x_max = 0, 3840
    y_max, y_min = 0, 2160  #わかりやすくするためy軸を反転
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    #信頼度(p)が低い場合は描画しない
    threshold = 0.2

    # 関節間の接続関係 (OpenPoseの標準的な18関節モデルに基づく)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),         # 右腕
        (1, 5), (5, 6), (6, 7),                 # 左腕
        (1, 8), (8, 9), (9, 10), (10, 11),      # 右足
        (1, 12), (12, 13), (13, 14), (14, 15),   # 左足
        (0, 16), (0, 17),                      # 顔
        (1, 16),
        (1,17)
    ]

    # 各人物の Scatter オブジェクトと接続線を格納するリスト
    scatters = []
    lines = []
    colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown'] #10人まで色分け

    for ipeople in range(people_num):
        color = colors[ipeople % len(colors)]  # 色を循環
        scat = ax.scatter([], [], s=50, c=color, label=f'Person {ipeople+1}')
        scatters.append(scat)
        person_lines = [ax.plot([], [], c=color, lw=2, alpha=0.7)[0] for _ in connections]
        lines.append(person_lines)
    # lines1 = [ax.plot([], [], c='blue', lw=2, alpha=0.7)[0] for _ in connections]  # 接続線, alphaは透明度
    # lines2 = [ax.plot([], [], c='red', lw=2, alpha=0.7)[0] for _ in connections]

    title_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=12) #タイトル

    def update(frame_index):
        frame = frames[frame_index]
        artists = []  # 更新が必要なアーティストを格納

        for i, (key, df) in enumerate(data_dict.items()):
            x = df.loc[frame].filter(like='_x').values
            y = df.loc[frame].filter(like='_y').values
            p = df.loc[frame].filter(like='_p').values

            x_filtered = np.where(p > threshold, x, np.nan)
            y_filtered = np.where(p > threshold, y, np.nan)
            scatters[i].set_offsets(np.c_[x_filtered, y_filtered])
            artists.append(scatters[i])

            for j, (start, end) in enumerate(connections):
                if p[start] > threshold and p[end] > threshold:
                    lines[i][j].set_data([x[start], x[end]], [y[start], y[end]])
                else:
                    lines[i][j].set_data([], [])
                artists.append(lines[i][j])

        title_text.set_text(f'{condition} Frame: {frame}')
        artists.append(title_text) #タイトル
        return artists

    # アニメーションを作成
    ani = animation.FuncAnimation(
        fig, update, frames=range(len(frames)), interval=30, blit=False  # interval: フレーム間の時間(ミリ秒)
    )

    ax.legend()  # 凡例を表示
    plt.tight_layout()  # レイアウトの調整
    ani_save_path = root_dir / f"{condition}_ani.mp4"
    ani.save(ani_save_path, writer="ffmpeg", fps=60)  # アニメーションを保存
    # plt.show()
    plt.close(fig)  # プロットを閉じる


def main():
    pickles = root_dir.glob("*asgait*.pickle")
    print(f"pickles: {pickles}")

    keypoints_df_dict = {}

    for pickle_path in pickles:
        with open(pickle_path, "rb") as f:
            keypoints_df = pickle.load(f)
        keypoints_df_dict[pickle_path.stem] = keypoints_df

    # print(f"keypoints_df_dict: {keypoints_df_dict}")


    keys = list(keypoints_df_dict.keys())
    # print(f"keys: {keys}")
    # print(f"keys[0]: {keys[0]}")
    # #keys: ['sub0_abngait_openpose_df_dict', 'sub0_abngait_openpose_df_spline_dict', 'sub0_abngait_openpose_df_filter_dict']
    # #keys[0]: sub0_abngait_openpose_df_dict


    for i, condition in enumerate(keys):
        print(f"{i+1}/{len(keys)} {condition}を処理中...")
        animate_keypoints(keypoints_df_dict[condition], condition)
        print(f"    {condition}のアニメーション作成が完了しました。")

    print(f"すべてのアニメーションの作成が完了しました。")
    sys.exit()

if __name__ == "__main__":
    main()

