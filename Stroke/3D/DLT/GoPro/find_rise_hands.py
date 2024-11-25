import cv2
from pathlib import Path

movie_path = Path(r"G:\gait_pattern\20241112 (1)\gopro\front_l\gait.MP4")
cap = cv2.VideoCapture(str(movie_path))

prev_frame = None
threshold_height = 50  #適当

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    if prev_frame is not None:
        diff = cv2.absdiff(edges, prev_frame)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > threshold_height:  # 腕の大きさに応じた閾値を設定
                # 振り上げと判断
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        frame_resize = cv2.resize(frame, (960, 540))
        edges_resize = cv2.resize(edges, (960, 540))
        diff_resize = cv2.resize(diff, (960, 540))
        thresh_resize = cv2.resize(thresh, (960, 540))

        cv2.imshow("frame", frame_resize)
        cv2.imshow("edges", edges_resize)
        cv2.imshow("diff", diff_resize)
        cv2.imshow("thresh", thresh_resize)

    prev_frame = edges

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
