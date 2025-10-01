"""
骨盤の補間が妥当かどうかを確認するために、RASIマーカーのデータを特定のフレーム範囲で0に置き換えるスクリプト
"""


import pandas as pd
import numpy as np

def process_tsv_file(input_filename, output_filename, zero_ranges):
    """
    TSVファイルを処理し、指定されたフレーム範囲のRASIデータを0に置き換えます。

    Args:
        input_filename (str): 入力TSVファイル名
        output_filename (str): 出力TSVファイル名
        zero_ranges (list of tuples): 0に置き換えるフレーム範囲のリスト。
                                     例: [(10, 50), (100, 150)]
    """
    try:
        # --- 1. ヘッダー情報の読み込み ---
        # ファイルの先頭からヘッダー部分（データが始まる前の行）を読み込みます。
        # このファイルでは12行目までがヘッダーです。
        with open(input_filename, 'r') as f:
            header_lines = [next(f) for _ in range(10)]

        # --- 2. データ部分の読み込み ---
        # pandasを使用して、12行目をヘッダーとしてデータフレームに読み込みます。
        # タブ区切りなので sep='\t' を指定します。
        df = pd.read_csv(input_filename, sep='\t', header=10)

        # --- 3. RASI座標の列名を確認 ---
        # RASIマーカーのX, Y, Z座標に対応する列名を特定します。
        rasi_columns = ['RASI X', 'RASI Y', 'RASI Z']
        
        # 指定された列がデータフレームに存在するかチェックします。
        missing_cols = [col for col in rasi_columns if col not in df.columns]
        if missing_cols:
            print(f"エラー: 次の列が見つかりませんでした: {', '.join(missing_cols)}")
            return

        # --- 4. 指定されたフレーム範囲のデータを0に置換 ---
        # フレーム範囲のリストをループ処理します。
        for start_frame, end_frame in zero_ranges:
            # 指定されたフレーム範囲内の行を特定するための条件を作成します。
            # (df['Frame'] >= start_frame) & (df['Frame'] <= end_frame) で
            # start_frameからend_frameまでの範囲を指定します。
            condition = (df['Frame'] >= start_frame) & (df['Frame'] <= end_frame)

            # locを使って、条件に一致する行のRASI列の値をnp.nan（非数）に設定します。
            df.loc[condition, rasi_columns] = int(0)
            print(f"フレーム {start_frame} から {end_frame} のRASIデータを0に置換しました。")

        # --- 5. 結果を新しいTSVファイルに書き出し ---
        # 新しいファイルを開き、まず保持していたヘッダー情報を書き込みます。
        with open(output_filename, 'w') as f:
            f.writelines(header_lines)

        # 処理済みのデータフレームを、ヘッダーなし、インデックスなし、タブ区切りで
        # ファイルに追記します。na_rep='nan'で欠損値を'nan'という文字列で表現します。
        df.to_csv(output_filename, sep='\t', index=False, header=True, mode='a', na_rep='nan')
        
        print(f"\n処理が完了しました。結果は '{output_filename}' に保存されました。")

    except FileNotFoundError:
        print(f"エラー: ファイル '{input_filename}' が見つかりません。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

# --- プログラムの実行 ---
if __name__ == "__main__":
    # 入力ファイル名
    input_file = r"G:\gait_pattern\20250827_fukuyama\qualisys\psub_label\qtm\test_20241016\sub4_com_nfpa0001.tsv"

    # 出力ファイル名
    output_file = r"G:\gait_pattern\20250827_fukuyama\qualisys\psub_label\qtm\test_20241016\sub4_com_nfpa0001_deleted.tsv"
    
    

    # nanに置換したいフレーム範囲をリストで指定
    frame_ranges_to_nan = [(17, 36), (46, 55), (67, 94), (180, 197), (204, 213), (240, 269), (286, 305), (308, 317), (377, 386)]
    # frame_ranges_to_nan = [
    #     (10, 20),
    #     (100, 200),
    #     (340, 355),
    #     (390, 500),
    #     (600, 800),
    #     (850, 900),
    #     (1000, 1100)
    # ]

    # 関数を実行
    process_tsv_file(input_file, output_file, frame_ranges_to_nan)