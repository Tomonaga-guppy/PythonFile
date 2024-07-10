import cv2
import cv2.aruco as aruco

# Arucoマーカーの辞書を定義
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

detector = aruco.ArucoDetector(aruco_dict, parameters)

# カメラのキャプチャを開始
cap = cv2.VideoCapture(0)

while True:
    # カメラからフレームを読み込む
    ret, frame = cap.read()

    if not ret:
        break

    # Arucoマーカーの検出
    corners, ids, rejectedImgPoints = detector.detectMarkers(frame)

    # 検出されたマーカーを描画
    if len(corners) > 0:
        aruco.drawDetectedMarkers(frame, corners, ids)

    # 結果を表示
    cv2.imshow('Aruco Marker Detection', frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラのリリースとウィンドウの破棄
cap.release()
cv2.destroyAllWindows()
