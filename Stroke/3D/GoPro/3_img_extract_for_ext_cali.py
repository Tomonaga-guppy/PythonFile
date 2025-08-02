import cv2
import os

def extract_frames_from_videos(num_frames_to_extract=40):
    """
    複数の動画から等間隔で指定枚数のフレームを抽出し、保存する。
    最初の動画を基準にして抽出タイミングを決定する。
    """
    # --- 設定項目 ---
    # (プレフィックス, 動画ファイルのパス) のタプルのリスト
    video_sources = [
        ('fl', r'G:\gait_pattern\20250717_br\ext_cali\fl\8x6\cali_trim.mp4'),
        ('fr', r'G:\gait_pattern\20250717_br\ext_cali\fr\8x6\cali_trim.mp4')
    ]
    # 保存する画像ファイルの形式
    image_format = '.png'
    # --- 設定はここまで ---

    print("フレーム抽出処理を開始します...")

    # 1. 基準となる動画から抽出するフレーム番号を計算する
    base_video_path = video_sources[0][1]
    if not os.path.exists(base_video_path):
        print(f"エラー: 基準動画が見つかりません: {base_video_path}")
        return

    cap_base = cv2.VideoCapture(base_video_path)
    if not cap_base.isOpened():
        print(f"エラー: 基準動画を開けません: {base_video_path}")
        return

    total_frames = int(cap_base.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_base.release()
    print(f"基準動画の総フレーム数: {total_frames}")

    # 抽出するフレーム番号のリストを生成
    # (例: 4000フレームの動画から40枚なら、0, 100, 200...番目のフレーム)
    frame_indices = [int(i * (total_frames / num_frames_to_extract)) for i in range(num_frames_to_extract)]
    print(f"抽出するフレーム番号を {num_frames_to_extract} 個計算しました。")


    # 2. 各動画からフレームを抽出して保存する
    for prefix, video_path in video_sources:
        print(f"\n'{video_path}' の処理を開始...")

        # 動画ファイルが存在するかチェック
        if not os.path.exists(video_path):
            print(f"  警告: 動画ファイルが見つかりません。スキップします。")
            continue

        # 保存先フォルダを作成
        output_dir = os.path.join(os.path.dirname(video_path), 'cali_frames')
        os.makedirs(output_dir, exist_ok=True)
        print(f"  画像を保存するフォルダ: {output_dir}")

        # 動画を読み込む
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  エラー: 動画を開けませんでした。")
            continue

        extracted_count = 0
        for i, frame_num in enumerate(frame_indices):
            # 指定したフレーム番号に移動
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            # フレームを読み込む
            ret, frame = cap.read()

            if ret:
                if prefix == 'fl':
                    # fl用のプレフィックスを付ける
                    serial_num = '0'
                elif prefix == 'fr':
                    # fr用のプレフィックスを付ける
                    serial_num = '1'
                # ファイル名を生成 (例: fl_01.jpg)
                output_filename = f"{i+1:04d}_{serial_num}.{image_format}"
                output_filepath = os.path.join(output_dir, output_filename)

                # 画像を保存
                cv2.imwrite(output_filepath, frame)
                extracted_count += 1
            else:
                print(f"  警告: フレーム番号 {frame_num} の読み込みに失敗しました。")

        print(f"  {extracted_count} 枚のフレームを抽出しました。")
        cap.release()

    print("\nすべての処理が完了しました。")

if __name__ == '__main__':
    extract_frames_from_videos(num_frames_to_extract=40)
