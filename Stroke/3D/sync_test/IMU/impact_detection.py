import pandas as pd
import json
import numpy as np
from pathlib import Path

def analyze_sync_data(csv_file):
    """
    agsとext dataが混在したCSVファイルから計測開始位置を抽出
    """
    print(f"'{csv_file}' を読み込んでいます...")
    df = pd.read_csv(csv_file, encoding='cp932')
    print("読み込み完了。")

    port0_change_df = df[(df['Port 0'] == 0) & (df['Port 0'].shift(1) == 1)]
    if port0_change_df.empty:
        print("-> Port0が1から0に変化するイベントは見つかりませんでした。")
        return None, None, None, None
    port0_ext_event_timestamp = port0_change_df['Timestamp_Ext'].iloc[0]
    port0_ext_event_index = port0_change_df.index[0]

    port1_change_df = df[(df['Port1'] == 1) & (df['Port1'].shift(1) == 0)]
    if port1_change_df.empty:
        print("-> Port1が0から1に変化するイベントは見つかりませんでした。")
        return None, None, None, None
    port1_ext_event_timestamp = port1_change_df['Timestamp_Ext'].iloc[0]
    port1_ext_event_index = port1_change_df.index[0]
    
    print(f"Port0の変化イベント (1→0) のタイムスタンプ (Timestamp_Ext): {int(port0_ext_event_timestamp)} (Index: {port0_ext_event_index})")
    print(f"Port1の変化イベント (0→1) のタイムスタンプ (Timestamp_Ext): {int(port1_ext_event_timestamp)} (Index: {port1_ext_event_index})")
    diff_port0to1_frame = port1_ext_event_index - port0_ext_event_index
    diff_port0to1_time = diff_port0to1_frame * 10  # 100Hzなので10ms刻み
    print(f"Port0イベントからPort1イベントまでのフレーム数: {diff_port0to1_frame} フレーム,  {diff_port0to1_time} ms")

    time_difference = (df['Timestamp_Acc'] - port0_ext_event_timestamp).abs()
    closest_index = time_difference.idxmin()
    ags_closest_timestamp = df.loc[closest_index, 'Timestamp_Acc']
    print(f"最も近い 'ags' のタイムスタンプ (Timestamp_Acc): {int(ags_closest_timestamp)}")

    return df, port0_ext_event_timestamp, port1_ext_event_timestamp, ags_closest_timestamp, diff_port0to1_frame

def extract_data_from_timestamp(df, start_timestamp, acc_column_name):
    """
    指定されたタイムスタンプを開始点として、特定の列のデータを抽出する関数
    """
    print(f"\nタイムスタンプ '{start_timestamp}' を開始点としてデータを抽出中...")
    start_indices = df[df['Timestamp_Acc'] == start_timestamp].index
    if start_indices.empty:
        print(f"-> エラー: タイムスタンプ '{start_timestamp}' がデータ内に見つかりません。")
        return None
    start_index = start_indices[0]
    columns_to_extract = ['Timestamp_Acc', acc_column_name]
    extracted_df = df.loc[start_index:, columns_to_extract]
    reset_df = extracted_df.reset_index(drop=True)
    return reset_df

def find_sharp_increase(df, acc_column_name, threshold=10000):
    """
    加速度の急激な増加点を検出する関数
    """
    acc_diff = df[acc_column_name].diff()
    sharp_increase_points = df[acc_diff > threshold]
    if sharp_increase_points.empty:
        print("-> 急激な増加は見つかりませんでした。")
        return None, None
    first_increase_point = sharp_increase_points.iloc[0]
    frame_number = first_increase_point.name
    timestamp = first_increase_point['Timestamp_Acc']
    return frame_number, timestamp

# --- メイン処理 ---
if __name__ == "__main__":
    csv_dir = Path(r'G:\gait_pattern\20250915_synctest\IMU\sub1\thera0-3\IMU')
    csv_path = list(csv_dir.glob("*sync*.csv"))[0]
    imu_df, p0_ts, p1_ts, ags_ts, diff_port0to1_frame = analyze_sync_data(csv_path)
    print(f"p0_ts: {p0_ts}, p1_ts: {p1_ts}, ags_ts: {ags_ts}, diff_port0to1_frame: {diff_port0to1_frame}")

    acc_z_series = extract_data_from_timestamp(imu_df, ags_ts, 'Acc_Y 0.1[mG]')
    
    if acc_z_series is not None:
        acc_threshold = 10000
        result_increase = find_sharp_increase(acc_z_series, 'Acc_Y 0.1[mG]', threshold=acc_threshold)

        if result_increase is not None and result_increase[0] is not None:
            impact_frame_number, impact_timestamp = result_increase
            elapsed_time_ms = (impact_timestamp - ags_ts)
            print(f"衝突検出フレーム(100Hz): {impact_frame_number}, 経過時間: {elapsed_time_ms}ms, 衝突時タイムスタンプ: {impact_timestamp}")
        else:
            impact_frame_number, impact_timestamp, elapsed_time_ms = None, None, None

        impact_frame_number_py = int(impact_frame_number) if impact_frame_number is not None else None
        impact_timestamp_py = float(impact_timestamp) if impact_timestamp is not None else None
        elapsed_time_ms_py = float(elapsed_time_ms) if elapsed_time_ms is not None else None
        diff_port0to1_frame = int(diff_port0to1_frame) if diff_port0to1_frame is not None else None
        
        json_data = {
            "impact_frame_number": impact_frame_number_py,
            "impact_timestamp": impact_timestamp_py,
            "elapsed_time_ms": elapsed_time_ms_py,
            "diff_port0to1_frame": diff_port0to1_frame,
            "diff_port0to1_time_ms": diff_port0to1_frame * 10
        }

        id = csv_path.parent.parent.stem.split('-')[1]
        if len(id.split('_')) > 1:
            id = id.split('_')[0]
        output_json_path = csv_path.parent.parent.parent.parent.with_name(f"{id}_imu_impact_info.json")

        with open(output_json_path, 'w', encoding='utf-8') as f: # Pathオブジェクトをそのまま渡せる
            json.dump(json_data, f, indent=4)
        print(f"結果を '{output_json_path}' に保存しました。")