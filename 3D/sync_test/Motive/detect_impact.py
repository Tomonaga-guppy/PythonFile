import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample
import json
from pathlib import Path  # pathlibをインポート

def read_3d_optitrack(csv_path):
    """
    OptiTrackの3Dデータを読み込む
    """
    # csv_pathがPathオブジェクトでも文字列でも動作するようにstr()で変換
    df = pd.read_csv(str(csv_path), skiprows=[0, 1, 2, 4], header=[0, 2])
    marker_set = ["Label"]
    marker_set_df = df[[col for col in df.columns if any(marker in col[0] for marker in marker_set)]].copy()
    return marker_set_df

def main():
    csv_path_dir = Path(r"G:\gait_pattern\20250915_synctest\Motive")
    
    try:
        csv_path = list(csv_path_dir.glob("*106.csv"))[0]
    except IndexError:
        print(f"エラー: '{csv_path_dir}' 内に一致するファイルが見つかりません。")
        return

    df_original = read_3d_optitrack(csv_path)
    
    y_series = df_original.loc[:, ('MarkerSet 01:Label', 'Y')]
    df_y = y_series.to_frame(name='Y')

    df_y['Y_diff'] = df_y['Y'].diff().fillna(0)
    df_y['Y_diff_diff'] = df_y['Y_diff'].diff().fillna(0)
    
    # print(df_y.head(10))
    
    # 衝突のフレームを検出
    impact_frame = df_y['Y_diff_diff'].idxmax()

    # y_median = df_y['Y'].median()
    # pos_threshold = 0.01
    # vel_threshold_impact = -0.0005
    # vel_threshold_stop = 0.0004
    
    # cond1 = (df_y['Y'] - y_median).abs() <= pos_threshold
    # cond2 = df_y['Y_diff'].shift(1) < vel_threshold_impact
    # cond3 = df_y['Y_diff'].abs() <= vel_threshold_stop

    # impact_frames_indices = df_y[cond1 & cond2 & cond3].index.tolist()


    df_z = df_original.loc[:, ('MarkerSet 01:Label', 'Z')]

    if impact_frame:
        print(f"検出された衝突フレーム: {impact_frame}")

        # (グラフ描画部分は変更なし)
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        fig.suptitle('Y Position', fontsize=16)

        ax1.plot(df_y.index, df_y['Y'], marker='.', linestyle='-', label='Y Position', color="tab:blue")
        ax1.axvline(x=impact_frame, color='r', linestyle='--', label='Impact Frame')
        ax1.set_title('Y Position over Frames')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Y Position (m)')
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(df_y.index, df_y['Y_diff'], marker='.', linestyle='-', label='Y diff', color="tab:orange")
        ax2.axvline(x=impact_frame, color='r', linestyle='--', label='Impact Frame')
        ax2.set_title('Y diff over Frames')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Y diff (m/frame)')
        ax2.grid(True)
        ax2.legend()

        ax3.plot(df_y.index, df_y['Y_diff_diff'], marker='.', linestyle='-', label='Y diff diff', color="tab:green")
        ax3.axvline(x=impact_frame, color='r', linestyle='--', label='Impact Frame')
        ax3.set_title('Y diff diff over Frames')
        ax3.set_xlabel('Frame Number')
        ax3.set_ylabel('Y diff diff (m/frame^2)')
        ax3.grid(True)
        ax3.legend()
        
        # ax3.plot(df_z.index, df_z, marker='.', linestyle='-', label='Z Position', color="tab:green")
        # ax3.axvline(x=impact_frame, color='r', linestyle='--', label='Impact Frame')
        # ax3.set_title('Z Position over Frames')
        # ax3.set_xlabel('Frame Number')
        # ax3.set_ylabel('Z Position (m)')
        # ax3.grid(True)
        # ax3.legend()
        
        plt.tight_layout()
        plt.savefig(csv_path.with_name(f"{csv_path.stem}_impact_detection.png"))
        plt.show()
        plt.close(fig)

        file_id = csv_path.stem.split('_')[1]
        json_path = csv_path_dir.parent / f"{file_id}_motive_impact_info.json"
        
        json_data = {
            "impact_frame_number": int(impact_frame)
        }

        with open(json_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
        print(f"衝突フレーム(モーキャプ基準)を '{json_path}' に保存しました。")
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle('Y Position', fontsize=16)
        ax1.plot(df_y.index, df_y['Y'], marker='.', linestyle='-', label='Y Position', color = "tab:blue")
        ax1.set_title('Y Position over Frames')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Y Position (m)')
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(df_y.index, df_y['Y_diff'], marker='.', linestyle='-', label='Y diff', color="tab:orange")
        ax2.set_title('Y diff over Frames')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Y diff (m/frame)')
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        plt.show() # グラフを表示する場合

        


if __name__ == "__main__":
    main()