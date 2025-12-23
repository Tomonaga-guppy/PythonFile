import json
from pathlib import Path
import cv2

# ====== ここだけ書き換え ======
IMG_PATH  = Path(r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\gopro\fr\vit_dir\vis\000000.jpg")  # 元画像 or vis画像
JSON_PATH = Path(r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\gopro\fr\vit_dir\keypoints\frame_00000_keypoints.json")
OUT_PATH  = Path(r"G:\gait_pattern\BR9G_shuron\sub1\thera1-0\gopro\fr\vit_dir\debug_order_frame00000.jpg")
PERSON_IDX = 0   # peopleが複数いるときに何番目を描くか
CONF_TH = 0.2
# ==============================

def main():
    img = cv2.imread(str(IMG_PATH))
    if img is None:
        raise RuntimeError(f"image not found: {IMG_PATH}")

    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    people = data.get("people", [])
    if not people:
        raise RuntimeError("no people in json")

    person = people[min(PERSON_IDX, len(people)-1)]
    k = person.get("pose_keypoints_2d", [])
    if len(k) < 25*3:
        raise RuntimeError(f"pose_keypoints_2d length is too short: {len(k)}")

    # index i の (x,y,score) を取り出して、番号を描く
    for i in range(25):
        x = float(k[3*i + 0])
        y = float(k[3*i + 1])
        s = float(k[3*i + 2])
        if s < CONF_TH:
            continue
        xi, yi = int(round(x)), int(round(y))
        cv2.circle(img, (xi, yi), 4, (0, 255, 0), -1)
        cv2.putText(img, str(i), (xi+5, yi-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(OUT_PATH), img)
    print("saved:", OUT_PATH)

if __name__ == "__main__":
    main()
