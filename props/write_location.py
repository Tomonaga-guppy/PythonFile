from pathlib import Path
import cv2


#連続した歩行画像におおよその位置を記入して動画にするプログラム
img_dir = Path(r"G:\gait_pattern\20241126_br9g\gopro\fl\abngait_1_op")
# img_dir = Path(r"G:\gait_pattern\20241126_br9g\gopro\fr\abngait_1_op")
img_files = img_dir.glob("*.jpg")

#動画を作成
#動画の保存先
video_path = img_dir.with_name("op_with_location.mp4")
#動画のフレームレート
fps = 60
#動画のサイズ
video_size = (1920, 1080)
#動画のコーデック
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, video_size)

#文字の設定
font_scale = 3 # 文字のスケール (大きくする)
thickness = 4 # 文字の太さ (スケールに合わせて太くする)
font = cv2.FONT_HERSHEY_SIMPLEX
text_color = (255, 255, 255) # 白色
position = (50, 100) # 開始位置 (Y座標を少し下げて見切れにくくする)
bg_color = (0, 0, 0)      # 黒色背景
text_position = (50, 100) # テキスト位置
padding = 10              # 背景パディング

#-4mスタート
# 左側の動画の場合
positon_frame = [760, 817, 854, 882, 933, 977, 1015, 1062, 1104, 1158, 1201, 1250, 1290, 1333]
# 右側の動画の場合 -3.5スタート？
# positon_frame = [757, 802, 845, 886, 937, 972, 1012, 1060, 1114, 1150, 1200, 1242, 1288, 1330]

for frame, img in enumerate(img_files):
    print(f"frame:{frame}")
    if frame < positon_frame[0]:
        continue
    if frame > 1500:
    # if frame > 817:
        break

    #画像を読み込む
    img = cv2.imread(str(img))
    #画像のサイズを変更する
    img = cv2.resize(img, video_size)
    #画像に位置を記入する
    if frame in range(positon_frame[0], positon_frame[1]):
        stand_pos = "-4m"
    elif frame in range(positon_frame[1], positon_frame[2]):
        stand_pos = "-3.5m"
    elif frame in range(positon_frame[2], positon_frame[3]):
        stand_pos = "-3m"
    elif frame in range(positon_frame[3], positon_frame[4]):
        stand_pos = "-2.5m"
    elif frame in range(positon_frame[4], positon_frame[5]):
        stand_pos = "-2m"
    elif frame in range(positon_frame[5], positon_frame[6]):
        stand_pos = "-1.5m"
    elif frame in range(positon_frame[6], positon_frame[7]):
        stand_pos = "-1m"
    elif frame in range(positon_frame[7], positon_frame[8]):
        stand_pos = "-0.5m"
    elif frame in range(positon_frame[8], positon_frame[9]):
        stand_pos = "0m"
    elif frame in range(positon_frame[9], positon_frame[10]):
        stand_pos = "0.5m"
    elif frame in range(positon_frame[10], positon_frame[11]):
        stand_pos = "1m"
    elif frame in range(positon_frame[11], positon_frame[12]):
        stand_pos = "1.5m"
    elif frame in range(positon_frame[12], positon_frame[13]):
        stand_pos = "2m"
    elif frame > positon_frame[13]:
        stand_pos = "2.5m"

    text = f"position: {stand_pos}"
    # --- 背景色付きテキスト描画 ---
    # 1. 文字サイズ取得
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 2. 背景矩形座標計算 (padding を使用)
    rect_x1, rect_y1 = text_position[0] - padding, text_position[1] - text_height - baseline - padding # 左上
    rect_x2, rect_y2 = text_position[0] + text_width + padding, text_position[1] + padding # 右下

    # 3. 背景矩形描画 (塗りつぶし)
    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, cv2.FILLED)

    # 4. テキスト描画 (背景の上に重ねる)
    cv2.putText(img, text, text_position, font, font_scale, text_color, thickness, cv2.LINE_AA)
    # --- 描画処理ここまで ---

    #動画に書き込む
    video_writer.write(img)


video_writer.release()
cv2.destroyAllWindows()
print("動画作成完了")