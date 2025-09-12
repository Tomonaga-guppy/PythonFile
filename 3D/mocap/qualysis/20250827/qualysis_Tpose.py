"""
QualysisのTポーズデータから、指定された主要マーカーの
平均的なグローバル座標を抽出し、JSONファイルとして保存します。
"""

import pandas as pd
import numpy as np
import json
# ユーザーが作成したモジュールと標準ライブラリをインポート
import module_mocap as moc  # ユーザー定義モジュール
from pathlib import Path
import matplotlib.pyplot as plt

def get_marker_data(df: pd.DataFrame, marker_name: str) -> np.ndarray | None:
    """
    データフレームから指定されたマーカーの3次元座標データを抽出します。
    列が存在しない場合は、Noneを返します。（警告は呼び出し元でまとめて表示）
    """
    required_columns = [f'{marker_name} X', f'{marker_name} Y', f'{marker_name} Z']
    if all(col in df.columns for col in required_columns):
        return df[required_columns].to_numpy()
    else:
        return None

def calculate_and_save_reference_positions(markers: dict, output_path: Path):
    """
    マーカーのグローバル座標の時系列データから、各マーカーの平均位置を計算し、
    JSONファイルに保存します。
    """
    
    averaged_global_positions = {}
    for name, data in markers.items():
        if data is not None:
            # Tポーズのような静的なデータでは、外れ値の影響が少ないため平均値を使用
            averaged_global_positions[name] = np.mean(data, axis=0).tolist()

    if not averaged_global_positions:
        print("警告: 保存対象のマーカーデータが一つも存在しませんでした。JSONファイルは作成されません。")
        return

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(averaged_global_positions, f, indent=4)
    print(f"各マーカーの平均グローバル座標を '{output_path}' に保存しました。")

def plot_marker_data(markers: dict, output_path: Path):
    """
    各マーカーの3次元座標を時系列でプロットし、PNGファイルとして保存・表示します。
    """
    num_markers = len([data for data in markers.values() if data is not None])
    if num_markers == 0:
        print("警告: プロットするマーカーデータがありません。")
        return
        
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    colors = plt.cm.get_cmap('hsv', num_markers)
    marker_names = [name for name, data in markers.items() if data is not None]

    axes[0].set_title('Marker Trajectories - X-axis')
    axes[1].set_title('Marker Trajectories - Y-axis')
    axes[2].set_title('Marker Trajectories - Z-axis')

    for i, (name, data) in enumerate(markers.items()):
        if data is not None:
            frames = np.arange(len(data))
            axes[0].plot(frames, data[:, 0], label=f'{name} X', color=colors(i))
            axes[1].plot(frames, data[:, 1], label=f'{name} Y', color=colors(i))
            axes[2].plot(frames, data[:, 2], label=f'{name} Z', color=colors(i))

    for ax in axes:
        ax.set_xlabel('Frame')
        ax.set_ylabel('Position (mm)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"時系列プロットを '{output_path}' に保存しました。")
    plt.show()



def main():
    tsv_dir = Path(r"G:\gait_pattern\20250827_fukuyama\qualisys\psub_label\qtm")
    # Tポーズファイルを対象とする
    tsv_files_list = list(tsv_dir.glob("*0001*.tsv"))
    num_files = len(tsv_files_list)
    
    if num_files == 0:
        print(f"ディレクトリ '{tsv_dir}' に対象のTポーズファイルが見つかりません。")
        return

    for itsv, tsv_file in enumerate(tsv_files_list):
        print(f"Processing {itsv+1}/{num_files}: {tsv_file.name}")
        
        full_df = moc.read_tsv(tsv_file)
        if full_df.empty:
            print(f"警告: ファイル '{tsv_file.name}' が空か、読み込みに失敗しました。スキップします。")
            continue

        target_df = full_df.copy()
        nan_df = target_df.replace(0, np.nan)

        butter_df = moc.butterworth_filter(nan_df, cutoff=6, order=4, fs=120)

        # 処理対象のマーカーリストを固定
        marker_names = ['RASI', 'LASI', 'RPSI', 'LPSI', 'RILC', 'LILC']
        print(f"処理対象マーカー: {', '.join(marker_names)}")

        markers_data = {}
        found_markers = []
        not_found_markers = []

        for name in marker_names:
            data = get_marker_data(butter_df, name)
            markers_data[name] = data
            if data is not None:
                found_markers.append(name)
            else:
                not_found_markers.append(name)
        
        if found_markers:
            print(f"検出されたマーカー: {', '.join(found_markers)}")
        if not_found_markers:
            print(f"警告: 次のマーカーが見つかりませんでした: {', '.join(not_found_markers)}")

        # 出力ファイル名を生成 (例: sub1-0001_ref_pos.json)
        output_filename = tsv_file.with_suffix('.json').name.replace(tsv_file.stem, f"{tsv_file.stem}_ref_pos")
        output_path = tsv_file.parent / output_filename

        calculate_and_save_reference_positions(markers_data, output_path)
        
        # プロット処理を追加
        output_plot_path = tsv_file.with_suffix('.png').name.replace(tsv_file.stem, f"{tsv_file.stem}_plots")
        plot_path = tsv_file.parent / output_plot_path
        plot_marker_data(markers_data, plot_path)

if __name__ == "__main__":
    main()

