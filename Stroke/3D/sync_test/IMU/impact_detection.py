import pandas as pd
import numpy as np

def analyze_sync_data(csv_file):
    """
    agsとext dataが混在したCSVファイルを分析する関数
    """
    try:
        # --- 1. ファイルの読み込み ---
        # ヘッダーの空白文字に対応するため、区切り文字を空白(正規表現 \s+)に設定
        # また、前回の文字コードエラー対策として encoding='cp932' を指定
        print(f"'{csv_file}' を読み込んでいます...")
        df = pd.read_csv(csv_file, encoding='cp932', sep='\s+')
        print("読み込み完了。")
        
        print("【診断】実際に読み込まれた列名リスト:", df.columns)

        # --- 2. ext dataでPort0が0から1に変化した時刻を探す ---
        print("\nStep 1: 'ext data'のPort0の変化点を検索中...")
        
        # .shift() を使って1つ前の行と比較し、0から1への変化点を見つける
        change_df = df[(df['Port 0'] == 1) & (df['Port 0'].shift(1) == 0)]

        if change_df.empty:
            print("-> Port0が0から1に変化するイベントは見つかりませんでした。")
            return

        # 変化した最初のイベントのタイムスタンプを取得
        ext_event_timestamp = change_df['Timestamp_Ext'].iloc[0]
        print(f"-> 発見！ Port0の変化点のタイムスタンプ (Timestamp_Ext): {int(ext_event_timestamp)}")

        # --- 3. 最も近いagsのタイムスタンプを探す ---
        print("\nStep 2: 上記時刻に最も近い 'ags' のタイムスタンプを検索中...")

        # Timestamp_Acc と ext_event_timestamp の差（絶対値）を計算
        time_difference = (df['Timestamp_Acc'] - ext_event_timestamp).abs()
        
        # 差が最小となる行のインデックス（位置）を見つける
        closest_index = time_difference.idxmin()
        
        # そのインデックスを使って、最も近いTimestamp_Accの値を取得
        ags_closest_timestamp = df.loc[closest_index, 'Timestamp_Acc']
        
        print(f"-> 発見！ 最も近い 'ags' のタイムスタンプ (Timestamp_Acc): {int(ags_closest_timestamp)}")
        
        print("\n---")
        print("✅ 処理が完了しました。")
        print("---")


    except FileNotFoundError:
        print(f"エラー: ファイル '{csv_file}' が見つかりません。")
        print("スクリプトと同じフォルダにCSVファイルを置いてください。")
    except KeyError as e:
        print(f"エラー: 必要な列が見つかりませんでした: {e}")
        print("CSVファイルのヘッダー名が 'Timestamp_Ext', 'Timestamp_Acc', 'Port0' となっているか確認してください。")
    except Exception as e:
        print(f"処理中に予期せぬエラーが発生しました: {e}")


# --- メイン処理 ---
if __name__ == "__main__":
    # ここに分析したいCSVファイルの名前を入力してください
    csv_file_name = r'G:\gait_pattern\20250915_synctest\IMU\sub0\thera0-4\IMU\sync_2025-9-15-8-44-1.csv'
    
    analyze_sync_data(csv_file_name)