import cv2
import numpy as np
import matplotlib.pyplot as plt # グラフ描画ライブラリをインポート
from pathlib import Path
import csv
import json
import glob
import pandas as pd

# ==============================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼【ここを修正】▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ==============================================================================
# 引数に start_frame_rel を追加
def create_annotated_video(video_path, df, output_path, start_frame_rel, impact_frame=None):
    """
    マーカーの検出位置をフレーム上に描画した動画を生成する。

    Args:
        video_path (Path): 元となる動画ファイルのパス。
        df (pd.DataFrame): フレーム番号をインデックスとし、'CenterX', 'CenterY'列を持つデータフレーム。
        output_path (Path): 生成する動画の保存先パス。
        start_frame_rel (int): 処理の開始フレーム番号（オフセット）。
        impact_frame (int, optional): 特にハイライトするフレーム番号. Defaults to None.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"エラー: 動画ファイル '{video_path}' を開けません。")
        return

    # 動画のプロパティを取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 動画書き出しの設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # コーデック
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"マーカー位置を描画した動画の生成を開始します... -> {output_path}")

    frame_num = 0 # トリミング後の動画のフレームカウンター (0, 1, 2, ...)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 座標検索用のフレーム番号を計算（オフセットを追加）
        lookup_frame_num = frame_num + start_frame_rel

        # 座標データが存在するか確認
        if lookup_frame_num in df.index:
            # 座標を取得
            center_x = int(df.loc[lookup_frame_num, 'X'])
            center_y = int(df.loc[lookup_frame_num, 'Y'])

            # マーカー位置に円を描画
            cv2.circle(frame, (center_x, center_y), 20, (0, 255, 0), -1) # 緑色の塗りつぶした円
            
            # 座標テキストを描画
            text = f"({center_x}, {center_y})"
            cv2.putText(frame, text, (center_x + 15, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # フレーム番号を描画（動画内の番号と、全体での相対番号を両方表示）
        frame_text = f"Frame: {frame_num} (Rel: {lookup_frame_num})"
        cv2.putText(frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 着地フレームの場合、特別なテキストを描画
        if lookup_frame_num == impact_frame:
            impact_text = "IMPACT FRAME"
            text_size = cv2.getTextSize(impact_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            cv2.putText(frame, impact_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # フレームを動画に書き込む
        out.write(frame)
        
        # 進捗表示
        if frame_num % 100 == 0:
            print(f"  処理中... {frame_num} / {total_frames} フレーム")

        frame_num += 1

    print("動画の生成が完了しました。")
    # リソースを解放
    cap.release()
    out.release()
# ==============================================================================
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲【修正ここまで】▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# ==============================================================================


def find_bottom_frame_by_aruco(video_path, start_frame_rel):
    """
    ArUcoマーカーを用いて、動画内でマーカーが最も下に到達したフレームを特定し、
    y座標の推移をグラフで可視化する。

    Args:
        video_path (str): 解析対象の動画ファイルへのパス。

    Returns:
        int: 最下点のフレーム番号。見つからない場合は-1。
        pd.DataFrame: マーカー座標のデータフレーム。
    """
    
    y_csv_path = video_path.parent / f"{video_path.stem}_coordinates.csv"
    
    # グラフ用のデータを保存するリストを初期化
    frame_numbers_list = []
    y_coords_list = []
    x_coords_list = []
    
    # --- ① データ読み込みセクション ---
    if y_csv_path.exists():
        print(f"既存のCSVファイルが見つかりました: {y_csv_path}")
        try:
            with open(y_csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                try:
                    # ヘッダー名からXとYの列番号を自動で特定
                    x_col_index = header.index('CenterX')
                    y_col_index = header.index('CenterY')
                except ValueError:
                    print("エラー: CSVのヘッダーに 'CenterX' または 'CenterY' が見つかりません。")
                    return -1, pd.DataFrame() # 古いフォーマットの場合は処理を中断

                for row in reader:
                    frame_numbers_list.append(int(row[0]))
                    x_coords_list.append(float(row[x_col_index])) # X座標を読み込む
                    y_coords_list.append(float(row[y_col_index])) # Y座標を読み込む
            print(f"CSVファイル '{y_csv_path}' からデータを読み込みました。")
        except IOError as e:
            print(f"エラー: CSVファイル '{y_csv_path}' の読み込みに失敗しました。 {e}")
            return -1, pd.DataFrame()
    else:
        print("CSVファイルが見つからないため、動画からマーカーを検出します。")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"エラー: 動画ファイル '{video_path}' を開けません。")
            return -1, pd.DataFrame()

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        current_frame_number = start_frame_rel

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            corners, ids, rejected = detector.detectMarkers(frame)

            if ids is not None:
                marker_corners = corners[0][0]
                center_y = np.mean(marker_corners[:, 1])
                center_x = np.mean(marker_corners[:, 0])
                frame_numbers_list.append(current_frame_number)
                x_coords_list.append(center_x)
                y_coords_list.append(center_y)
                print(f"検出成功Frame {current_frame_number}: Marker center Y = {center_y}")
            
            current_frame_number += 1

        cap.release()
        cv2.destroyAllWindows()

        # 検出した座標データを新しいCSVファイルに保存
        try:
            with open(y_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Frame', 'CenterX', 'CenterY'])
                for i in range(len(frame_numbers_list)):
                    writer.writerow([frame_numbers_list[i], x_coords_list[i], y_coords_list[i]])
            print(f"座標データを '{y_csv_path}' として保存しました。")
        except IOError as e:
            print(f"エラー: CSVファイル '{y_csv_path}' の書き込みに失敗しました。 {e}")

    # --- ② データ分析セクション ---
    if not frame_numbers_list:
        print("分析するデータがありません。")
        return -1, pd.DataFrame()
        
    # 地面に到達したフレームを算出
    df = pd.DataFrame(
        {'X': x_coords_list, 'Y': y_coords_list}, 
        index=frame_numbers_list
    )
    df.index.name = 'Frame'

    print(f"作成されたDataFrame:\n{df.head()}")

    df['Y_diff'] = df['Y'].diff()
    df['Y_diff_diff'] = df['Y_diff'].diff()

    # 条件1: X座標が中央値から100ピクセル以内
    x_median = df['X'].median()
    print(f"X座標の中央値: {x_median}")
    pos_x_threshold = 100
    print(df['X'] - x_median)
    cond1 = (df['X'] - x_median).abs() <= pos_x_threshold
    
    # cond1がFalseのフレーム（外れ値）を特定
    is_outlier = ~cond1

    # 外れ値とその前後1フレームをまとめて除外対象とする
    # window=3: 自分と前後1フレームの3つを見る
    # center=True: 自分を中心に見る
    # 3フレームの窓の中の合計値を計算し、その合計が0より大きい（＝少なくとも1つのフレームが外れ値である）場合にTrueを返す
    exclude_frames_mask = is_outlier.rolling(window=5, center=True, min_periods=1).sum() > 0

    # 最終的に候補とするフレームの条件（除外対象の逆）
    final_cond = ~exclude_frames_mask

    candidate_frames = df.loc[final_cond]
    print(f"X座標の条件を満たす候補フレーム数: {len(candidate_frames)}")

    # 候補の中からY軸加速度が最大のものを探す
    if not candidate_frames.empty and not candidate_frames['Y_diff_diff'].isnull().all():
        impact_frame = candidate_frames['Y_diff_diff'].idxmax()
        print(f"最終的な衝突フレーム: {impact_frame}")
    else:
        impact_frame = None
        print("衝突フレームが見つかりませんでした。")

    # --- ③ グラフ描画セクション ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Marker Center Y-coordinate and Velocity over Frames', fontsize=16)

    ax1.plot(df.index, df['Y'], marker='.', linestyle='-', label='Marker Y-coordinate', color = "tab:blue")
    if impact_frame is not None:
        ax1.axvline(x=impact_frame, color='r', linestyle='--', label='Impact Frame')
    ax1.set_title('Marker Center Y-coordinate over Frames')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Y-coordinate')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(df.index, df['Y_diff'], marker='.', linestyle='-', label='Marker Y-velocity', color="tab:orange")
    if impact_frame is not None:
        ax2.axvline(x=impact_frame, color='r', linestyle='--', label='Impact Frame')
    ax2.set_title('Marker Center Y-velocity over Frames')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Y-velocity')
    ax2.grid(True)
    ax2.legend()
    
    ax3.plot(df.index, df['Y_diff_diff'], marker='.', linestyle='-', label='Marker Y-acceleration', color="tab:green")
    if impact_frame is not None:
        ax3.axvline(x=impact_frame, color='r', linestyle='--', label='Impact Frame')
    ax3.set_title('Marker Center Y-acceleration over Frames')
    ax3.set_xlabel('Frame Number')
    ax3.set_ylabel('Y-acceleration')
    ax3.grid(True)
    ax3.legend()

    graph_filename = str(video_path.parent / f"{video_path.stem}_marker_y_coordinate_graph.png")
    plt.savefig(graph_filename)
    print(f"グラフを '{graph_filename}' として保存しました。")
    plt.show() # GUI表示をコメントアウトすると自動実行時に止まらなくなる
    plt.close()
    
    return impact_frame, df

# LED発光を0フレーム目として切り抜いた動画を処理
video_file_dir = Path(r"G:\gait_pattern\20250915_synctest\GoPro")
video_file_path = Path(glob.glob(str(video_file_dir / "*5*trimed*.mp4"))[0])

gopro_trim_info_path = video_file_path.parent / f"{video_file_path.stem.split('_')[0]}_gopro_trimming_info.json"
with open(gopro_trim_info_path, 'r') as f:
    gopro_trim_info = json.load(f)
    print(f"読み込んだトリミング情報: {gopro_trim_info}")

fps = gopro_trim_info['original_video_info']['fps']
led_flash_frame = gopro_trim_info['trimming_settings']['reference_frame']
start_frame_rel = gopro_trim_info['trimming_settings']['start_frame_relative']
end_frame_rel = gopro_trim_info['trimming_settings']['end_frame_relative']
print(f"動画のFPS: {fps}, LED発光フレーム: {led_flash_frame}, 開始フレーム(相対値): {start_frame_rel}, 終了フレーム(相対値): {end_frame_rel}")

impact_frame, df_coords = find_bottom_frame_by_aruco(video_file_path, start_frame_rel)

impact_time = impact_frame / fps if impact_frame is not None and impact_frame != -1 else -1
print(f"impact_frame: {impact_frame}, impact_time: {impact_time:.2f}秒")

# データフレームが空でない場合のみ、動画を生成
if not df_coords.empty:
    output_video_path = video_file_path.parent / f"{video_file_path.stem}_annotated.mp4"
    # 引数に start_frame_rel を追加
    create_annotated_video(video_file_path, df_coords, output_video_path, start_frame_rel, impact_frame)
else:
    print("座標データがないため、アノテーション付き動画は生成されませんでした。")

gopro_impact_info = {
    "impact_frame_ledbase": int(impact_frame) if impact_frame is not None and impact_frame != -1 else None,
    "impact_time_ledbase": impact_time
}
# JSONファイル名を修正（with_nameは親ディレクトリを維持しつつファイル名部分だけを変更する）
gopro_impact_info_path = video_file_path.with_name(f"{video_file_path.stem.split('_')[0]}_gopro_impact_info.json")
with open(gopro_impact_info_path, 'w', encoding='utf-8') as f:
    json.dump(gopro_impact_info, f, indent=4)
print(f"{gopro_impact_info_path}を保存しました。")