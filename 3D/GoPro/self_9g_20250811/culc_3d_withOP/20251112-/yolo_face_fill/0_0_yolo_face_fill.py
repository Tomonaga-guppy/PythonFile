from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
from pathlib import Path
from tqdm import tqdm

# 1. 学習済みモデルをロード
model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11x-face-detection", filename="model.pt")
model = YOLO(model_path)

condition_list = ["thera1-0"]
direction_list = ["fl_yolo", "fr_yolo"]

def apply_black_fill(image, x1, y1, x2, y2):
    # 指定範囲を黒で塗りつぶす関数
    color = [0, 0, 0]  # 黒色のBGR値
    width = x2 - x1
    height = y2 - y1
    # 中心を基準にして矩形のサイズを2倍に
    new_x1 = x1 - width // 2
    new_y1 = y1 - height // 2
    new_x2 = x2 + width // 2
    new_y2 = y2 + height // 2
    # 画像の範囲を超えないように調整
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image.shape[1], new_x2)
    new_y2 = min(image.shape[0], new_y2)
    cv2.rectangle(image, (new_x1, new_y1), (new_x2, new_y2), color, -1)
    return image


for direction in direction_list:
    for condition in condition_list:
        # 2. undistorted_oriフォルダから画像ファイルを読み込む
        input_dir = Path(fr"g:\gait_pattern\20250811_br\sub1\{condition}\{direction}\undistorted_ori")
        
        if not input_dir.exists():
            print(f"フォルダが存在しません: {input_dir}")
            continue
        
        # 画像ファイルを取得（重複を除外）
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
            image_files.extend(input_dir.glob(ext))
            
        # 重複を除外（Pathオブジェクトのセットを作成）
        image_files = sorted(set(image_files))
        
        if len(image_files) == 0:
            print(f"画像ファイルが見つかりません: {input_dir}")
            continue
        
        print(f"処理する画像ファイル数: {len(image_files)} in {input_dir}")
        
        # 出力フォルダを作成
        output_dir = input_dir.parent / "undistorted"
        output_dir.mkdir(exist_ok=True)
        
        # 動画設定用に最初の画像から情報を取得
        first_img = cv2.imread(str(image_files[0]))
        if first_img is None:
            print(f"最初の画像の読み込みに失敗しました: {image_files[0]}")
            continue
        
        height, width = first_img.shape[:2]
        fps = 60.0  # フレームレート
        
        # 動画出力の設定
        video_output_path = input_dir.parent / f"undistorted.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))
        
        # 各画像ファイルを処理
        for image_file in tqdm(image_files, desc=f"Processing {direction}/{condition}"):
            # 画像を読み込む
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"画像の読み込みに失敗しました: {image_file}")
                continue
            
            # 3. モデルで推論を実行
            results = model(img, verbose=False)  # verbose=False で出力を抑制
            
            # 4. 検出された顔の情報を収集
            all_boxes = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                    all_boxes.append((x1, y1, x2, y2))
            
            # 5. 方向に応じた顔の除外処理
            boxes_to_fill = []
            
            if direction == "fl_yolo":
                # fl_yoloの場合の処理
                # まず、x≤2280かつy≤1107の範囲内の顔を黒塗り対象に追加
                remaining_boxes = []
                for box in all_boxes:
                    x1, y1, x2, y2 = box
                    if x1 < 2280 < x2 and y1 < 1107 < y2:
                    # if x2 <= 2280 and y2 <= 1107:
                        # この範囲内の顔は黒塗り
                        boxes_to_fill.append(box)
                    else:
                        # この範囲外の顔は残す
                        remaining_boxes.append(box)
                
                # 残りの顔が2つ以上ある場合、x座標が最大のもの以外を黒塗り
                if len(remaining_boxes) >= 2:
                    # x座標が最大の顔を見つける（x2が最大のものを保持）
                    max_x_box = max(remaining_boxes, key=lambda box: box[2])
                    for box in remaining_boxes:
                        if box != max_x_box:
                            boxes_to_fill.append(box)
            
            elif direction == "fr_yolo":
                # fr_yoloの場合の処理
                # 顔が2つ以上検出されている場合、x座標が最小のもの以外を黒塗り
                if len(all_boxes) >= 2:
                    # x座標が最小の顔を見つける（x1が最小のものを保持）
                    min_x_box = min(all_boxes, key=lambda box: box[0])
                    for box in all_boxes:
                        if box != min_x_box:
                            boxes_to_fill.append(box)
            
            # 6. 黒塗り対象の顔を塗りつぶす
            for box in boxes_to_fill:
                x1, y1, x2, y2 = box
                img = apply_black_fill(img, x1, y1, x2, y2)
            
            # 7. 画像を保存
            output_file = output_dir / image_file.name
            cv2.imwrite(str(output_file), img)
            
            # 8. 動画に書き込み
            video_writer.write(img)
        
        # 動画ファイルを閉じる
        video_writer.release()
        print(f"動画を保存しました: {video_output_path}")

print("処理が完了しました。")