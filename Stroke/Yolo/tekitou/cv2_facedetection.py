# OpenCVを使用して動画内の顔を検出し、顔部分を黒で塗りつぶすスクリプト（全然うまくいかない）
import cv2

# 顔検出用のカスケード分類器を読み込む
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 動画ファイルへのパス (ファイル名を必ず含めてください)
video_path = r"g:\gait_pattern\20250228_ota\data\20250221\sub0\thera0-3\sagi\Undistort.mp4"

# 動画ファイルから映像を取得
cap = cv2.VideoCapture(video_path)

# 動画が正常に開かれたか確認
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # 1フレームずつ読み込む
    ret, frame = cap.read()
    if not ret:
        print("Reached the end of the video or an error occurred.")
        break

    # グレースケール変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 顔検出
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    # 検出した顔を塗りつぶす
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), thickness=-1)  # 黒で塗りつぶし

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
        break

#リソースを解放
cap.release()
cv2.destroyAllWindows()