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

def calculate_and_save_average_global_positions(markers: dict, output_path: Path):
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
        
        calculate_and_save_average_global_positions(markers_data, output_path)
        
        print("-" * 50)

if __name__ == "__main__":
    main()

