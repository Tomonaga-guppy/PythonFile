from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 1. 全身検出可能なモデルに変更 (標準のYOLOv11xを使用)
# 初回実行時に自動でダウンロードされます
model = YOLO("yolo11x.pt")

condition_list = ["thera1-0"]
direction_list = ["fr_yoloseg_crop"]
# direction_list = ["fl_yoloseg_crop", "fr_yoloseg_crop"]

def create_person_canvas(original_img, target_box):
    """
    黒背景の画像を作成し、target_boxの領域だけ元の画像からコピーする関数
    これにより、画像の解像度(4K)と座標系を維持したまま、対象人物以外を消去できる。
    """
    h, w = original_img.shape[:2]
    
    # 4Kサイズの真っ黒な画像を作成
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    x1, y1, x2, y2 = target_box
    
    # 座標が画像範囲を超えないようにクリップ
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # 元画像の該当エリアを、黒キャンバスの同じ位置にコピー
    canvas[y1:y2, x1:x2] = original_img[y1:y2, x1:x2]
    
    return canvas

for direction in direction_list:
    for condition in condition_list:
        # パスの設定（適宜変更してください）
        input_dir = Path(fr"g:\gait_pattern\20250811_br\sub1\{condition}\{direction}\undistorted")
        
        if not input_dir.exists():
            print(f"フォルダが存在しません: {input_dir}")
            continue
        
        image_files = sorted(list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png')))
        
        if len(image_files) == 0:
            print(f"画像ファイルが見つかりません: {input_dir}")
            continue
        
        print(f"処理する画像ファイル数: {len(image_files)} in {input_dir}")
        
        # 出力フォルダ
        output_dir = input_dir.parent / "undistorted_crop_4k"
        output_dir.mkdir(exist_ok=True)
        
        # 動画設定
        first_img = cv2.imread(str(image_files[0]))
        height, width = first_img.shape[:2]
        fps = 60.0
        video_output_path = input_dir.parent / f"undistorted_crop_4k.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))
        
        for image_file in tqdm(image_files, desc=f"Processing {direction}"):
            img = cv2.imread(str(image_file))
            if img is None: continue
            
            # 3. 推論 (classes=0 は 'person' クラスのみを検出する設定)
            results = model(img, classes=[0], verbose=False)
            
            # 検出された人物のボックスを収集
            person_boxes = []
            for result in results:
                for box in result.boxes:
                    # xyxy形式で座標取得
                    coords = [int(c) for c in box.xyxy[0]]
                    person_boxes.append(coords)
            
            target_box = None
            
            # 人物が検出されなかった場合は、そのまま黒画像(あるいは元画像)にするなどの処理
            if not person_boxes:
                # 誰もいない場合は黒画像を出力してスキップ
                canvas = np.zeros_like(img)
                cv2.imwrite(str(output_dir / image_file.name), canvas)
                video_writer.write(canvas)
                continue

            # 4. 対象人物の選択ロジック (以前のスクリプトの逆を行う)
            # 以前は「マスクする対象」を選んでいましたが、今回は「残す対象(Target)」を選びます
            
            if len(person_boxes) == 1:
                target_box = person_boxes[0]
            else:
                # 2人以上いる場合の選別ロジック
                if direction == "fl_yoloseg_crop":
                    # 以前のコード: 右の人(Max X)以外を黒塗り = 右の人を残す
                    # 今回のコード: 右の人(Max X)をターゲットにする
                    target_box = max(person_boxes, key=lambda box: box[2]) # x2が大きい＝右側
                    
                elif direction == "fr_yoloseg_crop":
                    # 以前のコード: 左の人(Min X)以外を黒塗り = 左の人を残す
                    # 今回のコード: 左の人(Min X)をターゲットにする
                    target_box = min(person_boxes, key=lambda box: box[0]) # x1が小さい＝左側

            # 5. キャンバス処理（ここが重要：座標維持のため）
            if target_box:
                # バウンディングボックスに少しマージン（余白）を持たせる
                margin = 50 # 50ピクセルくらい余白を追加
                x1, y1, x2, y2 = target_box
                target_box_padded = (
                    max(0, x1 - margin),
                    max(0, y1 - margin),
                    min(width, x2 + margin),
                    min(height, y2 + margin)
                )
                
                # 指定した人だけ残した画像を生成
                final_img = create_person_canvas(img, target_box_padded)
            else:
                # フォールバック（万が一ロジックからもれた場合）
                final_img = np.zeros_like(img)

            # 保存
            cv2.imwrite(str(output_dir / image_file.name), final_img)
            video_writer.write(final_img)
        
        video_writer.release()
        print(f"完了: {video_output_path}")

print("全処理完了")