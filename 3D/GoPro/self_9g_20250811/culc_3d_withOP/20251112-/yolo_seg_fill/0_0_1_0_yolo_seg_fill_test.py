from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 1. セグメンテーション用のモデルをロード
model = YOLO('yolo11x-seg.pt')  # セグメンテーションモデル

condition_list = ["thera1-0"]
direction_list = ["fl_yoloseg", "fr_yoloseg"]

# クラスごとの色を定義（BGR形式）
def get_color_palette(num_classes=80):
    """
    クラスごとに異なる色を生成
    """
    np.random.seed(42)  # 再現性のため固定
    palette = []
    for i in range(num_classes):
        # HSV色空間で色相を均等に分散させて、鮮やかな色を生成
        hue = int(180 * i / num_classes)
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        palette.append(tuple(map(int, color_bgr)))
    return palette

# 色パレットを生成
COLOR_PALETTE = get_color_palette(80)

def apply_segmentation_overlay(image, masks, classes, alpha=0.5, show_borders=True):
    """
    セグメンテーション結果をカラーオーバーレイとして表示
    
    Parameters:
    -----------
    image : numpy.ndarray
        元画像
    masks : list of numpy.ndarray
        各オブジェクトのセグメンテーションマスク
    classes : list of int
        各マスクに対応するクラスID
    alpha : float
        オーバーレイの透明度（0.0-1.0）
    show_borders : bool
        境界線を表示するかどうか
    """
    overlay = image.copy()
    
    for mask, cls_id in zip(masks, classes):
        if mask is None or mask.size == 0:
            continue
        
        # クラスIDに対応する色を取得
        color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]
        
        # マスク領域に色を塗る
        overlay[mask > 0] = color
        
        # 境界線を描画
        if show_borders:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 2)
    
    # 元画像とオーバーレイをブレンド
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    return result

def draw_labels(image, boxes, classes, class_names):
    """
    検出されたオブジェクトにラベルを描画
    
    Parameters:
    -----------
    image : numpy.ndarray
        画像
    boxes : list of tuple
        バウンディングボックス座標 [(x1, y1, x2, y2), ...]
    classes : list of int
        クラスID
    class_names : dict
        クラスIDと名前の対応
    """
    for box, cls_id in zip(boxes, classes):
        x1, y1, x2, y2 = box
        color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]
        
        # バウンディングボックスを描画
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # ラベルテキストを作成
        label = class_names.get(cls_id, f"Class {cls_id}")
        
        # ラベル背景を描画
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        
        # ラベルテキストを描画
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
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
            
        # 重複を除外
        image_files = sorted(set(image_files))
        
        if len(image_files) == 0:
            print(f"画像ファイルが見つかりません: {input_dir}")
            continue
        
        """
        テスト用に100枚刻みで処理
        """
        image_files = image_files[::100]
        
        print(f"処理する画像ファイル数: {len(image_files)} in {input_dir}")
        
        # 出力フォルダを作成
        output_dir = input_dir.parent / "undistorted_seg"
        output_dir.mkdir(exist_ok=True)
        
        # 動画設定用に最初の画像から情報を取得
        first_img = cv2.imread(str(image_files[0]))
        if first_img is None:
            print(f"最初の画像の読み込みに失敗しました: {image_files[0]}")
            continue
        
        height, width = first_img.shape[:2]
        fps = 60.0
        
        # 動画出力の設定
        video_output_path = input_dir.parent / f"undistorted_seg_{direction}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))
        
        # モデルのクラス名を取得
        class_names = model.names
        
        # 各画像ファイルを処理
        for image_file in tqdm(image_files, desc=f"Processing {direction}/{condition}"):
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"画像の読み込みに失敗しました: {image_file}")
                continue
            
            # 3. セグメンテーション推論を実行
            results = model(img, verbose=False)
            
            # 4. 検出されたオブジェクトのマスク、バウンディングボックス、クラスを収集
            all_masks = []
            all_boxes = []
            all_classes = []
            
            for result in results:
                if result.masks is not None:
                    # マスクデータを取得
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (mask, box, cls_id) in enumerate(zip(masks, boxes, classes)):
                        # マスクを画像サイズにリサイズ
                        mask_resized = cv2.resize(mask, (width, height))
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)
                        
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        all_masks.append(mask_binary)
                        all_boxes.append((x1, y1, x2, y2))
                        all_classes.append(cls_id)
            
            # 5. セグメンテーション結果をカラーオーバーレイで表示
            img = apply_segmentation_overlay(img, all_masks, all_classes, alpha=0.5, show_borders=True)
            
            # 6. ラベルを描画
            img = draw_labels(img, all_boxes, all_classes, class_names)
            
            # 7. 画像を保存
            output_file = output_dir / image_file.name
            cv2.imwrite(str(output_file), img)
            
            # 8. 動画に書き込み
            video_writer.write(img)
        
        # 動画ファイルを閉じる
        video_writer.release()
        print(f"動画を保存しました: {video_output_path}")

print("処理が完了しました。")