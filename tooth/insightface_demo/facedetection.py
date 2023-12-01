import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

import os
root_dir = r"C:\Users\zutom\.vscode\PythonFile\tooth\insightface_demo"

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
img = cv2.imread(os.path.join(root_dir,"test.jpg"))
# img = ins_get_image('t1')
faces = app.get(img)


#https://github.com/deepinsight/insightface/blob/master/python-package/insightface/app/face_analysis.py
# 元の関数をバックアップ
original_draw_on = FaceAnalysis.draw_on

# オーバーライドする新しい関数 元の関数はnp.int（旧)が使われているため
def draw_on_override(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(int)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

            #for key, value in face.items():
            #    if key.startswith('landmark_3d'):
            #        print(key, value.shape)
            #        print(value[0:10,:])
            #        lmk = np.round(value).astype(int)
            #        for l in range(lmk.shape[0]):
            #            color = (255, 0, 0)
            #            cv2.circle(dimg, (lmk[l][0], lmk[l][1]), 1, color,
            #                       2)
        return dimg
# 新しい関数をオーバーライド
FaceAnalysis.draw_on = draw_on_override


rimg = app.draw_on(img, faces)  #ここ
cv2.imwrite(os.path.join(root_dir,"facedetection.jpg"), rimg)