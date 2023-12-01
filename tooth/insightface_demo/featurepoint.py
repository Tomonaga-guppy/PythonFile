import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

#参考元 https://github.com/deepinsight/insightface/blob/master/alignment/coordinate_reg/image_infer.py

root_dir = r"C:\Users\zutom\.vscode\PythonFile\tooth\insightface_demo"

if __name__ == '__main__':
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_3d_68'])
    # app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    # img = ins_get_image('t1')
    img = cv2.imread(os.path.join(root_dir,"test.jpg"))

    faces = app.get(img)
    #assert len(faces)==6
    tim = img.copy()
    color = (200, 160, 75)
    for face in faces:
        lmk = face.landmark_3d_68
        # lmk = face.landmark_2d_106
        lmk = np.round(lmk).astype(int)
        lmk = (np.array(lmk))[:, :2]

        for i in range(lmk.shape[0]):
            p = tuple(lmk[i])
            cv2.circle(tim, p, 1, color, 1, cv2.LINE_AA)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.2
            font_color = (0, 0, 0)  # テキストの色 (BGR)
            thickness = 1  # テキストの太さ
            cv2.putText(tim, f"{i}", p, font, font_scale, font_color, thickness)

    cv2.imwrite(os.path.join(root_dir,"68point.jpg"), tim)
    # cv2.imwrite(os.path.join(root_dir,"106point.jpg"), tim)