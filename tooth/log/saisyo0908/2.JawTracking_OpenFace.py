import os
import glob
import math
import subprocess
import csv
import cv2
import numpy as np
import sys

root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_09_000"

# if len(sys.argv) > 1:
#     root_dir = sys.argv[1]
# else:
#     print("ディレクトリパスが指定されていません。")
#     sys.exit()

command = 'C:/OpenFace_2.2.0_win_x64/FeatureExtraction.exe -2Dfp -3Dfp -tracked '

seal_template1 = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/seal_template/seal.png"
seal_template2 = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/seal_template/seal_black.png"

# seal_template1 = "C:/Users/brlab/Desktop/tooth/Temporomandibular_movement/seal_template/seal.png"
# seal_template2 ="C:/Users/brlab/Desktop/tooth/Temporomandibular_movement/seal_template/seal_black.png"


#depth_scale = mm/depth_data
depth_scale = 1.0000000474974513

def OpenFace(root_dir):
    pattern = os.path.join(root_dir, '*J2*/*_original.mp4')
    mp4_files = glob.glob(pattern, recursive=True)
    print('mp4files=',mp4_files)
    for i,video_file in enumerate(mp4_files):
        dir_path = os.path.dirname(video_file) + '/'
        video_out_path = dir_path + os.path.basename(video_file).split('.')[0] + '_sealDetection.mp4'

        # OpenFaceで顔の推定
        inputpath = '-f ' + video_file + ' '
        outpath = '-out_dir ' + dir_path
        print ('OpenFace =',command + inputpath + outpath)
        subprocess.run (command + inputpath + outpath )
        # OpenFaceの結果表示
        print('dir_path = ',dir_path)
        # OpenFace_result 0 frame, 5-72 x_pixel, 73-140 y_pixel
        csv_file = dir_path + os.path.basename(video_file).split('.')[0] + '.csv'
        OpenFace_result = []
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                OpenFace_result.append(row)

        # 動画を読み込む
        cap = cv2.VideoCapture(video_file)
        cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter(video_out_path,fourcc, fps, (cap_width, cap_height))

        frame_count=1
        seal=[[0,0]]
        List=[]

        while True:
            count = 0
            success, img = cap.read()
            # img_depth (y,x) = (720, 1280)
            img_depth = cv2.imread(dir_path + "{:04d}.png".format(frame_count), cv2.IMREAD_ANYDEPTH)
            if img is None or img_depth is None:
                break

            # OpenFace座標格納(mm)
            landmark_List=[]
            for i in range(68):
                x,y,z = float(OpenFace_result[frame_count][i+141]),float(OpenFace_result[frame_count][i+209]),float(OpenFace_result[frame_count][i+277])

                depthi = img_depth[int(float(OpenFace_result[frame_count][i+73])),int(float(OpenFace_result[frame_count][i+5]))]
                """
                depth33 = img_depth[int(float(OpenFace_result[frame_count][106])),int(float(OpenFace_result[frame_count][38]))]
                depthz = float(OpenFace_result[frame_count][334]) + (float(depthi) - float(depth33))*depth_scale
                """
                landmark_List.append([i,x,y,z,depthi*depth_scale])

            # シールを見つける処理 参照点 5,8,11,57 → 48,10→  48,10のx,8のy
            mask1_x=int(float(OpenFace_result[frame_count][53]))    #48x
            mask1_y=int(float(OpenFace_result[frame_count][121]))    #48y
            mask2_x=int(float(OpenFace_result[frame_count][59]))    #54x
            mask2_y=int(float(OpenFace_result[frame_count][81]))    #8y

            value = []
            values, tl = SealDetection(cap_height, cap_width, img, mask1_x, mask2_x, mask1_y, mask2_y)
            value = np.array(values)

            cv2.circle(img, (tl[0],tl[1]), 1, (255, 120, 255), -1)
            seal_x = tl[0]
            seal_y = tl[1]
            # detect = False
            # for v in value:
            #     if  float(OpenFace_result[frame_count][53]) <= v[0] <= float(OpenFace_result[frame_count][59]) and \
            #         float(OpenFace_result[frame_count][130]) <= v[1] <= float(OpenFace_result[frame_count][81]):


            #         if frame_count !=1 and detect == True:
            #             seal_x,seal_y = CloseLength(tl[0],tl[1],seal_x,seal_y,v[0],v[1])

            #         seal_x = v[0]
            #         seal_y = v[1]
            #         detect = True
            # #dif = diff(seal[frame_count -1][0],seal[frame_count -1][1],seal_x,seal_y)
            # if detect == False:
            #     seal_x = seal[frame_count -1][0]
            #     seal_y = seal[frame_count -2][1]
            seal.append([seal_x,seal_y])

            # シールの単位換算 (pixel → mm) 参照点 x:36-45 y:27-33
            xpixel_scale = (float(OpenFace_result[frame_count][186])-float(OpenFace_result[frame_count][177]))/(float(OpenFace_result[frame_count][50])-float(OpenFace_result[frame_count][41]))
            ypixel_scale = (float(OpenFace_result[frame_count][242])-float(OpenFace_result[frame_count][236]))/(float(OpenFace_result[frame_count][106])-float(OpenFace_result[frame_count][100]))

            # X68 = X33 + (x68 - x33)*scale
            x = float(OpenFace_result[frame_count][174]) + (seal_x - float(OpenFace_result[frame_count][38]))*xpixel_scale
            y = float(OpenFace_result[frame_count][242]) + (seal_y - float(OpenFace_result[frame_count][106]))*ypixel_scale

            # シールの奥行 depthから計算 参照点 z:33
            # Z68 = Z33 + (depth68(x,y)-depth33(x,y))*depth_scale
            depth68 = img_depth[seal_y,seal_x]
            depth33 = img_depth[int(float(OpenFace_result[frame_count][106])),int(float(OpenFace_result[frame_count][38]))]
            z = float(OpenFace_result[frame_count][334]) + (float(depth68) - float(depth33))*depth_scale

            landmark_List.append([68,x,y,z,depth68*depth_scale])
            cv2.circle(img, (int(seal_x),int(seal_y)), 5, (255, 0, 255), -1)
            List.append(landmark_List)

            print(frame_count)
            frame_count +=1
            writer.write(img)

        cap.release()
        writer.release
        print("Movie is saved in " + video_file)

        NumpyList = np.array(List)
        print(np.shape(NumpyList))
        path = dir_path + "result.npy"
        np.save(path,NumpyList)
        #作製したnumpy配列は[フレーム数-1[landmark number[number, x, y, z]]]


# def SealDetection(cap_height,cap_width,img):
def SealDetection(cap_height,cap_width,img,mask1_x,mask2_x,mask1_y,mask2_y):
    np.value = []
    tl = (0, 0)

    count = 0

    # 矩形のマスク画像の生成
    mask = np.zeros((cap_height,cap_width,3), dtype = np.uint8)
    #矩形検出範囲の設定
    # mask = cv2.rectangle(mask, (mask1_x-30,mask1_y-30), (mask2_x+30,mask2_y+30), (255, 255, 255), -1)
    mask = cv2.rectangle(mask, (mask1_x,mask1_y), (mask2_x,mask2_y), (255, 255, 255), -1)
    # cv2.rectangle(mask, (500, 440), (650, 500), (255, 255, 255), -1)
    #検出範囲をマスク
    mask_img = cv2.bitwise_and(img, mask)
    #ガウガシアンフィルタで平滑化
    blur_img = cv2.GaussianBlur(mask_img, (5,5), 0)

    template = cv2.imread(seal_template1)
    image_gray = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    for i in range(90,110,10):
        t2 = template_gray.copy()
        t3 = cv2.resize(t2, None, fx=i/100, fy=i/100, interpolation=cv2.INTER_CUBIC)
        w, h = t3.shape
        result = cv2.matchTemplate(image_gray, t3, cv2.TM_CCOEFF_NORMED)
        # 最も類似度が高い位置を取得する。
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

        # 描画する。
        # tl = maxLoc[0], maxLoc[1]
        # br = maxLoc[0] + w, maxLoc[1] + h
        center_x = int(maxLoc[0]+w/2)
        center_y = int(maxLoc[1]+h/2)
        radius = int((w+h)/4)

    # #グレースケール画像の生成
    # mask2_img = cv2.cvtColor(mask_img,cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('./mask2_img.jpg',mask2_img)
    # #円のハフ変換
    # #dpは大きいほど基準が緩く、小さいほど基準が厳しい
    # circles = cv2.HoughCircles(mask2_img,cv2.HOUGH_GRADIENT,dp = 2.0,minDist=50,param1=100)
    # if circles != None:
    #     print(1)
    #     circles = np.uint16(np.around(circles))
    #     for circle in circles[0, :]:
    #         # 円周を描画する
    #         cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 165, 255), 3)
    #         # 中心点を描画する
    #         cv2.circle(img, (circle[0], circle[1]), 2, (0, 0, 255), -1)

    # 矩形のマスク画像の生成
    mask2 = np.zeros((cap_height,cap_width,3), dtype = np.uint8)
    #矩形検出範囲の設定
    # mask2 = cv2.rectangle(mask2, tl ,br, (255, 255, 255), -1)
    mask2 = cv2.circle(mask2, (center_x,center_y), radius=radius-1, color=(255, 255, 255), thickness=-1)
    #検出範囲をマスク
    mask2_img = cv2.bitwise_and(img, mask2)
    # cv2.imwrite(root_dir + '20230606_mkg2/mask.jpg',mask2_img)
    #ガウガシアンフィルタで平滑化
    blur2_img = cv2.GaussianBlur(mask2_img, (5,5), 0)
    # エッジ検出 Canny法で二値化
    edges2 = cv2.Canny(blur2_img, 10, 150)
    # 確率的ハフ変換で直線の検出 threshold 10~20 各要素は検出した線分の始点と終点 (x1, y2, x2, y2) のタプル。1つも線分が見つからない場合は None を返す。
    lines = cv2.HoughLinesP(edges2, 1, np.pi / 180, threshold=15, minLineLength=10, maxLineGap=20)
    # cv2.imwrite('./edges.jpg',edges2)
    # 直線を描画する。
    if lines is not None:
        for x1, y1, x2, y2 in lines.squeeze(axis=1):
            cv2.line(img, (x1,y1), (x2,y2), (0, 0, 255), 1)
            # cv2.circle(img, (x1,y1), 5, (255, 0, 0), -1)
            # cv2.circle(img, (x2,y2), 5, (0, 255, 0), -1)
    # cv2.imshow('edges2',edges2)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    """
    # 直交する直線の交点を計算する
    if len(lines) >= 2:
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]

                # y=kx+b
                if abs(x2-x1) < 5: # y軸に平行
                    k1 = float('inf')
                    b1 = x1
                else:
                    k1 = (y2 - y1) / (x2 - x1)
                    b1 = y1 - k1 * x1
                if abs(x4-x3) < 5: # y軸に平行
                    k2 = float('inf')
                    b2 = x3
                else:
                    k2 = (y4 - y3) / (x4 - x3)
                    b2 = y3 - k2 * x3
                if abs(k1 - k2) < 1e-4 or (k1 == float('inf') and k2 == float('inf')):
                        continue

                if not math.isnan(abs(k1 - k2)): #float('inf')-float('inf')
                    if k1 ==float('inf'):
                        x = b1
                        y = k2 * x + b2
                    elif k2 == float('inf'):
                        x = b2
                        y = k1 * x + b1
                    else:
                        x = (b2 - b1) / (k1 - k2)
                        y = k1 * x + b1
                #交点が線分上にあるかどうか
                if ((min(x1,x2)-10 <= x <= max(x1,x2)+10) and\
                    (min(y1,y2)-10 <= y <= max(y1,y2)+10) and\
                    (min(x3,x4)-10 <= x <= max(x3,x4)+10) and\
                    (min(y3,y4)-10 <= y <= max(y3,y4)+10)):
                    #直線が直交するかどうか
                    if (k1 == float('inf') and abs(k2) < 0.35) or (k2 == float('inf') and abs(k1) < 0.35):
                        # cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
                        np.value.append([int(x),int(y)])
                    elif k1 != float('inf') and k2 != float('inf'):
                        theta = np.arctan2(k2-k1,1+k1*k2)*180/np.pi
                        #thetaもとは80~100
                        if 85<abs(theta)<95:
                            # cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)
                            np.value.append([int(x),int(y)])
    """

    template = cv2.imread(seal_template2)
    image_gray = cv2.cvtColor(blur2_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    for i in range(90,110,10):
        t2 = template_gray.copy()
        t3 = cv2.resize(t2, None, fx=i/100, fy=i/100, interpolation=cv2.INTER_CUBIC)
        w, h = t3.shape
        result = cv2.matchTemplate(image_gray, t3, cv2.TM_CCOEFF_NORMED)
        # 最も類似度が高い位置を取得する。
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        tl = maxLoc[0], maxLoc[1]

    # cv2.imshow('img',img)
    # cv2.imshow('edges',edges)
    # cv2.waitKey(0)
    return np.value,tl

def CloseLength(x0,y0,x1,y1,x2,y2):
    diff1 = diff(x0,y0,x1,y1)
    diff2 = diff(x0,y0,x2,y2)
    if diff1<=diff2:
        return x1,y1
    else:
        return x2,y2

def diff(x1,y1,x2,y2):
    diff = math.sqrt((x2-x1)**2+(y2-y1)**2)
    return diff





OpenFace(root_dir)
