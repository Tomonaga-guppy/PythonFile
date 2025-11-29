from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 1. セグメンテーション用のモデルをロード
model = YOLO('yolo11x-seg.pt')  # セグメンテーションモデル

condition_list = ["thera1-0"]
# direction_list = ["fl_yoloPAseg"]
direction_list = ["fl_yoloPAseg", "fr_yoloPAseg"]

def apply_mask_to_background(image, target_mask, color=(0, 0, 0), alpha=0.7, expand_ratio=0.1):
    """
    対象のperson以外（背景と他の人）をマスクする関数
    
    Parameters:
    -----------
    expand_ratio : float
        マスクを拡大する割合（0.1 = 10%拡大）
    """
    if target_mask is None or target_mask.size == 0:
        return image
    
    # マスクの境界を検出
    contours, _ = cv2.findContours(target_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # マスクを拡張
    if len(contours) > 0:
        # マスクの面積から膨張カーネルのサイズを計算
        mask_area = cv2.contourArea(contours[0])
        # 面積の平方根を基準にカーネルサイズを決定（10%拡大）
        kernel_size = int(np.sqrt(mask_area) * expand_ratio)
        kernel_size = max(kernel_size, 5)  # 最小値を5に設定
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # 奇数にする
        
        # 膨張処理を適用
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded_mask = cv2.dilate(target_mask, kernel, iterations=1)
    else:
        expanded_mask = target_mask
    
    # 拡張したマスクを反転（対象以外の領域を取得）
    inverse_mask = (expanded_mask == 0).astype(np.uint8)
    
    # オーバーレイ用の画像を作成
    overlay = image.copy()
    overlay[inverse_mask > 0] = color
    
    result = overlay.copy()
    
    # # 元画像とオーバーレイをブレンド
    # result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    # # 対象の境界線を描画
    # contours_expanded, _ = cv2.findContours(expanded_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(result, contours_expanded, -1, (0, 255, 0), 2)
    
    return result


for direction in direction_list:
    for condition in condition_list:
        # 2. undistorted_oriフォルダから画像ファイルを読み込む
        input_dir = Path(fr"g:\gait_pattern\BR9G_shuron\sub1\{condition}\{direction}\undistorted")
        
        if not input_dir.exists():
            print(f"フォルダが存在しません: {input_dir}")
            continue
        
        # 画像ファイルを取得（重複を除外）
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
            image_files.extend(input_dir.glob(ext))
            
        # 重複を除外
        image_files = sorted(set(image_files))
        
        if len(image_files) == 0:
            print(f"画像ファイルが見つかりません: {input_dir}")
            continue
        
        # ##########
        # ##########
        # # デバッグ用に枚数を制限
        # image_files = image_files[::5]  # 5枚に1枚だけ処理
        # ##########
        # ##########
        
        print(f"処理する画像ファイル数: {len(image_files)} in {input_dir}")
        
        # 出力フォルダを作成
        output_dir = input_dir.parent / "undistorted_PAseg"
        output_dir.mkdir(exist_ok=True)
        
        # 動画設定用に最初の画像から情報を取得
        first_img = cv2.imread(str(image_files[0]))
        if first_img is None:
            print(f"最初の画像の読み込みに失敗しました: {image_files[0]}")
            continue
        
        height, width = first_img.shape[:2]
        fps = 60.0
        
        # 動画出力の設定
        video_output_path = input_dir.parent / f"undistorted_PAseg_{direction}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))
        
        # モデルのクラス名を取得
        class_names = model.names
        # personクラスのIDを取得（通常は0）
        person_class_id = None
        for cls_id, cls_name in class_names.items():
            if cls_name.lower() == 'person':
                person_class_id = cls_id
                break
        
        if person_class_id is None:
            print("Error: 'person' クラスが見つかりません")
            continue
        
        print(f"Person class ID: {person_class_id}")
        
        # 各画像ファイルを処理
        for image_file in tqdm(image_files, desc=f"Processing {direction}/{condition}"):
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"画像の読み込みに失敗しました: {image_file}")
                continue
            
            # 3. セグメンテーション推論を実行
            results = model(img, verbose=False)
            
            # 4. personクラスのマスクとバウンディングボックスを収集
            person_masks = []
            person_boxes = []
            
            for result in results:
                if result.masks is not None:
                    # マスクデータを取得
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for mask, box, cls_id in zip(masks, boxes, classes):
                        # personクラスのみを対象
                        if cls_id == person_class_id:
                            # マスクを画像サイズにリサイズ
                            mask_resized = cv2.resize(mask, (width, height))
                            mask_binary = (mask_resized > 0.5).astype(np.uint8)
                            
                            x1, y1, x2, y2 = [int(coord) for coord in box]
                            person_masks.append(mask_binary)
                            person_boxes.append((x1, y1, x2, y2))
            
            # 5. 前方のpersonを決定して、それ以外をマスクする
            if len(person_boxes) >= 2:  # 2人以上のpersonが検出された場合
                if direction == "fl_yoloPAseg":
                    # x座標が大きいほう（左側）のpersonを前方として選択
                    target_idx = max(range(len(person_boxes)), key=lambda i: person_boxes[i][0])
                elif direction == "fr_yoloPAseg":
                    # x座標が小さいほう（右側）のpersonを前方として選択
                    target_idx = min(range(len(person_boxes)), key=lambda i: person_boxes[i][2])
                else:
                    target_idx = None
                
                if target_idx is not None:
                    # 前方のperson以外をマスクする
                    img = apply_mask_to_background(
                        img, 
                        person_masks[target_idx], 
                        color=(0, 0, 0),  # 黒色でマスク
                        alpha=0.7  # マスクの濃さ
                    )
            
            elif len(person_boxes) == 1:  # 1人のみ検出された場合
                # その1人を前方として、背景をマスク
                img = apply_mask_to_background(
                    img, 
                    person_masks[0], 
                    color=(0, 0, 0),
                    alpha=0.7
                )
            
            # 7. 画像を保存
            output_file = output_dir / image_file.name
            cv2.imwrite(str(output_file), img)
            
            # 8. 動画に書き込み
            video_writer.write(img)
        
        # 動画ファイルを閉じる
        video_writer.release()
        print(f"動画を保存しました: {video_output_path}")

print("処理が完了しました。")