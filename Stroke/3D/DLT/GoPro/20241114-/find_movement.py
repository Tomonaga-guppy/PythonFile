import cv2
from pathlib import Path

filepath = Path(r"G:\gait_pattern\20241112 (1)\gopro\front_l\gait.MP4")
cap = cv2.VideoCapture(str(filepath))

avg = None

while True:
    # 1フレームずつ取得する。
    ret, frame = cap.read()
    if not ret:
        break

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 比較用のフレームを取得する
    if avg is None:
        avg = gray.copy().astype("float")
        continue

    # 現在のフレームと移動平均との差を計算
    cv2.accumulateWeighted(gray, avg, 0.6)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # デルタ画像を閾値処理を行う
    thresh = cv2.threshold(frameDelta, 60, 255, cv2.THRESH_BINARY)[1]
    # thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
    # 画像の閾値に輪郭線を入れる
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # 結果を出力
    frame_resize = cv2.resize(frame, (960, 540))
    cv2.imshow("Frame", frame_resize)
    key = cv2.waitKey(1)
    if key == 27: # ESCキーを押すと終了
        break

cap.release()
cv2.destroyAllWindows()
