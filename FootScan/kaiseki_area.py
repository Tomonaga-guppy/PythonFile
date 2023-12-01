# 各ポイントの色を、red と green をそれぞれ1~10,11~20,...のように10ずつの範囲で該当する色
# （例)Red:11~20,Green:41~50）に振り分けて、どのＲＧ値の色がいくつあるのかを出力する

from mailbox import NoSuchMailboxError
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import openpyxl as px
import csv
import pandas as pd
from openpyxl import load_workbook
# 棒グラフ作成(グラフ本体、データ参照情報の定義)に必要となるクラス
from openpyxl.chart import BarChart, Reference, Series
# パターン(模様)塗り潰しに必要なクラス
from openpyxl.drawing.fill import PatternFillProperties, ColorChoice
# 個別データ(項目名・カテゴリ名)情報の定義に必要なクラス
from openpyxl.chart.marker import DataPoint
# データラベル情報の定義に必要なクラス
from openpyxl.chart.label import DataLabel, DataLabelList
from openpyxl.drawing.line import LineProperties
from openpyxl.chart.shapes import GraphicalProperties



def maskAT(name,im, pix):
    im = cv2.flip(im, 1) # 反転
    x = 0
    while x <= 2:
        x += 1
        
        if x == 1:
            img = im[95:3445,55:1235]
        elif x == 2: 
            img = im[95:3445,1300:2480]
        elif x == 3:
            break
        

        wrs = "/Users/BRLAB/others/img3/sole" + str(x) + ".jpg"
        cv2.imwrite(wrs, img)
        imgcope = cv2.imread(wrs)

        img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        img_Lab_L, img_Lab_a, img_Lab_b = cv2.split(img_Lab)

        #L*a*b* 色空間は、ヒトの感覚に合わせて考案された色空間である。
        # L* は明度で、0 ≤ L* ≤ 100 の値を取りうる。a* および b* は色味を表している。
        # a* は緑色と赤色を表し、a* が正の大きな値になるほど赤みが強くなり、負の大きな値になるほど緑みが強くなり、また 0 であれば無彩色となる。
        # また、b* は青色と黄色を表し、a* が正の大きな値になるほど黄色みが強くなり、負の大きな値になるほど青みが強くなり、また 0 であれば無彩色となる。
        # a* および b* の最小値と最大値は、明度 L の値によって異なる
        # L*a*b* 色空間において、L* は 0 ≤ L* ≤ 100、a* および b* はマイナスからプラスまでの値を取りうる。
        # OpenCV においては、0 ≤ L ≤ 100、-127 ≤ a ≤ 127 および -127 ≤ b ≤ 127 で定義されている。
        # ただし、OpenCV の cv2.cvtColor メソッドを用いて変換された場合は、0 ≤ L ≤ 255、0 ≤ a ≤ 255 および 0 ≤ b ≤ 255 となる。
        # OpenCV で L*a*b* 色空間を取り扱うときは、ここを注意する必要がある。

        img_L_r = cv2.adaptiveThreshold(img_Lab_L, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, pix, 0)
        
        #adaptiveThreshold メソッドを用いれば、あるピクセルを 2 値化したい場合、
        # そのピクセルを中心とする n×n ピクセルのデータを用いて、閾値計算を行い、2 値化を行う。
        # adaptiveThreshold は次のように、6 つの引数を使って指定する。
        #cv2.adaptiveThreshold(img, maxValue, adaptiveMethod, thresholdType, blockSize, C)
        #img は、入力画像で、maxValue は閾値を満たすピクセルに与える値となる。
        # 続いて、adaptiveMethod は適応閾値処理の種類を指定して平均値 ADAPTIVE_THRESH_MEAN_C あるいは
        # 標準化された平均値 ADAPTIVE_THRESH_GAUSSIAN_C のどちらかを指定する。
        # また、thresholdType には、上で示した 5 つの閾値処理の種類を指定する。
        # blockSize は、近傍ピクセルのサイズを奇数で指定する。
        # また、C には定数を指定する。閾値処理がすべて終えたときに得られた閾値から最終的に定数 C の値を引いて、最終結果として出力される。

        res = "/Users/BRLAB/others/img3/result_3area_L_" + str(name) + str(x) + ".jpg"
        cv2.imwrite(res, img_L_r)

    #繋ぐ___________________________________________________________
    x = 1
    wr1_L = "/Users/BRLAB/others/img3/result_3area_L_" + str(name) + str(x) + ".jpg"
    i01_L = cv2.imread(wr1_L)
    wrs_1 = "/Users/BRLAB/others/img3/sole" + str(x) + ".jpg"
    i01 = cv2.imread(wrs_1)
    x = 2
    wr2_L = "/Users/BRLAB/others/img3/result_3area_L_" + str(name) + str(x) + ".jpg"
    i02_L = cv2.imread(wr2_L)
    wrs_2 = "/Users/BRLAB/others/img3/sole" + str(x) + ".jpg"
    i02 = cv2.imread(wrs_2)

    im_h_L = cv2.hconcat([i01_L, i02_L])
    res_L = "/Users/BRLAB/others/img3/result_3area_L" + str(name) + str(pix) + ".jpg"
    cv2.imwrite(res_L, im_h_L)
    im_h = cv2.hconcat([i01, i02])
    res = "/Users/BRLAB/others/img3/sole" + str(name) +".jpg"
    cv2.imwrite(res, im_h)
    os.remove(wrs_1)
    os.remove(wrs_2)

    x = 0
    imgdate = "/Users/BRLAB/others/img3/sole" + str(name) +".jpg"
    img_date = cv2.imread(imgdate)

    while x <= 2:
        x += 1
        if x == 1:
            img = im_h_L[0:3350,0:1180]
            img_d = img_date[0:3350,0:1180]
        elif x == 2: 
            img = im_h_L[0:3350,1180:2360]
            img_d = img_date[0:3350,1180:2360]
        elif x == 3:
            break
        (takasa, haba, nanka) = img.shape
        chushin_y = int(takasa/2)
        chushin_x = int(haba/2)
        wr1_L = "/Users/BRLAB/others/img3/result_3area_L_" + str(name) + str(x) + ".jpg"

        mask_b = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # imgcope_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)# HSV 色相(Hue)彩度(Saturation)明度(Value・Brightness)
        #imgcope_HSV = cv2.GaussianBlur(imgcope_HSV, (5, 5), 0)
        #imgco_H, imgco_S, imgco_V = cv2.split(imgcope_HSV)
        #mask_b = cv2.dilate(mask_b,np.ones((10,10),np.uint8),iterations = 1)
        wr_b = "/Users/BRLAB/others/img3/z_mask_b" + str(x) + ".jpg"
        cv2.imwrite(wr_b, mask_b)
        # 凸包（hull） ゲット！！！
        # 作成できる外接矩形を全て試し、足裏形状を囲うもののみ選択
        contours, hierarchy = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        mask_b = cv2.imread(wr_b)
        for i in range(len(contours)):
            # 最も大きい輪郭に外接矩形を作成し、画像に重ね書き
            rect = cv2.minAreaRect(contours[i])
            ((xx,yy),(w, h),th) = cv2.minAreaRect(contours[i])

            if w*h > 1000000 : # >1000000 は大きい輪郭のみを残すため(330*330 ぐらい)
                #     points = cv2.boxPoints(box)、
                # 引数 ... box : ((左上X座標, 左上Y座標)，(幅, 高さ)，回転角)
                # 戻り値 ... points : 4隅の頂点

                box = cv2.boxPoints(rect) # 外接矩形は描かないことにしたよ。
                box = np.int0(box)
                hull = cv2.convexHull(contours[i]) #  凸包
                mask_b = cv2.drawContours(mask_b,[hull],0,(0,0,255),3) # 凸包
                #img_bb = cv2.drawContours(img_d,[hull],0,(0,0,255),3) # 凸包
        #wr_c = "/Users/BRLAB/others/img3/z_mask_b" + str(x) + str(name) + ".jpg"
        #cv2.imwrite(wr_c, mask_bb)
        #wr_c = "/Users/BRLAB/others/img3/z_img_b" + str(x) + str(name) + ".jpg"
        #cv2.imwrite(wr_c, img_bb)
        for a in range(len(hull)): # 凸包（hull）を拡大させる、中心から遠いほど伸びは大きくなる（足先やかかと）
            [p1] =  hull[a]
            x1_sa = p1[0] - chushin_x 
            y1_sa = p1[1] - chushin_y
            x1 = p1[0] + (x1_sa)/20
            y1 = p1[1] + (y1_sa)/23
            if x1 < 0:
                x1 = 0
            elif x1 > haba:
                x1 = haba
            if y1 < 0:
                y1 = 0
            elif y1 > takasa:
                y1 = takasa
            p1[0] = x1
            p1[1] = y1
            hull[a] = [p1]
        mask_dd = cv2.fillConvexPoly(mask_b, hull, (0,0,255))
        mask_d = cv2.cvtColor(mask_dd, cv2.COLOR_BGR2HSV)
        mask_H, mask_S, mask_V = cv2.split(mask_d)
        _thre, mask_e = cv2.threshold(mask_S, 100, 255, cv2.THRESH_BINARY_INV)
        wr_e = "/Users/BRLAB/others/img3/z_mask_e" + str(x) + ".jpg"
        cv2.imwrite(wr_e, mask_e)
        mask_e = cv2.imread(wr_e)
        # マスクを重ねてノイズ除去____________________________________________________________
        rows,cols,channels = mask_e.shape
        roi = img_d[0:rows, 0:cols ]

        img2gray = cv2.cvtColor(mask_e,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 120, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        wr_im = "/Users/BRLAB/others/img3/z_mask_im"+ str(x) + str(name) + ".jpg"
        cv2.imwrite(wr_im, img1_bg)
        img = cv2.imread(wr_im)

        os.remove(wr1_L)
        os.remove(wr_b)
        #os.remove(wr_e)

    #繋ぐ___________________________________________________________
    x = 1    
    wr1 = "/Users/BRLAB/others/img3/z_mask_im" + str(x) + str(name) +  ".jpg"
    i01_img = cv2.imread(wr1)
    os.remove(wr1)   
    x = 2
    wr2 = "/Users/BRLAB/others/img3/z_mask_im" + str(x) + str(name) +  ".jpg"
    i02_img = cv2.imread(wr2)
    os.remove(wr2)
    im_himg = cv2.hconcat([i01_img, i02_img])
    res = "/Users/BRLAB/others/img3/result_3area_conL_" + str(name) + ".jpg"
    cv2.imwrite(res, im_himg)
    os.remove(res_L)
    return im_himg



def rgb(img, name): # 引数は、画像と保存名

    numbe = name

    # point_1
    # BGR 取得_____________________________________________________________________________________________________
    (takasa,haba,nannka) = img.shape
    ta_yo = int(takasa/10)
    ha_yo = int(haba/10)
    yoso = ta_yo * ha_yo

    bgr_list_1 = [[0 for i in range(yoso)] for j in range(5)] # x,y,b,g,r
    # 20 刻みに変更
    # green:20 刻み、red:20刻み,260 まで、
    grh_list_1 = [[0 for i in range(169)] for j in range(5)] # name,kazu,b,g,r,,,,,,,,,,green,red の順、０なし
    grh_lista_1 =  [[0 for i in range(169)] for j in range(5)] # name,kazu,b,g,r,,,,,,,,,green,red の順、０あり
    # 初期値をゼロにする
    gt = 0 # name,kazu,r,g,b
    for i in range(5):
        rt = 0 # 169 種の色
        for j in range(169):
            grh_list_1[gt][rt] = 0
            grh_lista_1[gt][rt] = 0
            rt += 1
        gt += 1
    # 0 入りの list の name を入れる
    gh = 0
    f = 0
    s = 0
    for i in range(169):
        grh_lista_1[0][gh] = (20+f*20)+(0.02+s*0.02) # 20.020, 20.040, 20.060, ..., 20.260, 40.020, 40.040, ..., 260.240, 260.260 (g.r)
        gh += 1 
        s += 1
        if s == 13:
            f += 1
            s = 0           
    #bgr_list に BGR 取得, 0 入りの list に書き込む
    ta = 10
    ai = 0
    while ta < takasa : 
        ha = 10
        while ha < haba :
            [bl,gr,re] = img[ta,ha,:]
            bgr_list_1[0][ai] = ha
            bgr_list_1[1][ai] = ta
            bgr_list_1[2][ai] = bl
            bgr_list_1[3][ai] = gr
            bgr_list_1[4][ai] = re
            g = 0
            gh = 0
            while g <= 240:
                if g < gr <= (g+20):
                    r = 0
                    while r <= 240: # r = 0,20,40,...240 で13回周る
                        if r < re <= (r+20):
                            grh_lista_1[1][gh] += 1
                            grh_lista_1[2][gh] += bl
                            grh_lista_1[3][gh] += gr
                            grh_lista_1[4][gh] += re
                            #print("gt = ",gt,"rt = ",rt)
                        gh += 1
                        r += 20
                        #print("rt = ",rt)
                else :
                    gh += 13 
                g += 20
                #print("gt = ",gt)
            #print("ta = ", bgr_list[1][ai], "ha = ", bgr_list[0][ai])
            ai += 1
            ha += 10
        ta += 10

    # RGBを平均値化する, 0 入りの list 
    gh = 0
    for i in range(169):
        if not grh_lista_1[1][gh] == 0:
            grh_lista_1[2][gh] = int(grh_lista_1[2][gh] / grh_lista_1[1][gh]) # blue
            grh_lista_1[3][gh] = int(grh_lista_1[3][gh] / grh_lista_1[1][gh]) # green
            grh_lista_1[4][gh] = int(grh_lista_1[4][gh] / grh_lista_1[1][gh]) # red
        gh += 1
    # 0 の項目を省いたlist を作る
    # G_R
    gh = 0
    g = 0
    r = 0
    count_1 = 0
    co1 = 0
    for i in range(169): # gh = 0 ~ 168
        # (g.r) 順のlist 
        if not grh_lista_1[1][gh] == 0:
            grh_list_1[0][co1]= (20+g*20)+(0.02+r*0.02)
            grh_list_1[1][co1] = grh_lista_1[1][gh]
            grh_list_1[2][co1] = grh_lista_1[2][gh]
            grh_list_1[3][co1] = grh_lista_1[3][gh]
            grh_list_1[4][co1] = grh_lista_1[4][gh]
            co1 += 1
            count_1 += 1
        r += 1
        if r == 13:
            g += 1
            r = 0
        gh += 1

    #_________________________________________________________________________________
    # GRgraph
    # 1
    grgrh = "/Users/BRLAB/others/R_G_grh.jpg"
    grh_img = cv2.imread(grgrh)
    co1 = 0
    ave_x = 0
    ave_y = 0
    add = 0
    for i in range(count_1): # gh = 0 ~ 168
        green_a = int(grh_list_1[0][co1]) - 10 # BGR のGの値を取得、20~40 の場合、けつの40で記録してるので、-10 する。
        red_a = (grh_list_1[0][co1] - int(grh_list_1[0][co1]) - 0.01 )*1000 # BGR のRの値を取得
        text = str(grh_list_1[1][co1]) # 表示にせる数値
        if 1000 <= grh_list_1[1][co1]:
            ad_x = 10
        elif 100 <= grh_list_1[1][co1] < 1000:
            ad_x = 20
        elif 10 <= grh_list_1[1][co1] < 100:
            ad_x = 30
        elif grh_list_1[1][co1] < 10:
            ad_x = 40
        x_p =int( ad_x + ( red_a*5 + 50 )) # 数値を表示する座標
        y_p = int( 1460 - ( green_a*5 + 50 )) # 数値を表示する座標
        # 色の平均値
        ave_x += grh_list_1[4][co1] * grh_list_1[1][co1] # red 
        ave_y += grh_list_1[3][co1] * grh_list_1[1][co1] # green
        # 書き込み
        cv2.putText(grh_img,text, (x_p,y_p), cv2.FONT_HERSHEY_PLAIN,2, (0,0,0),3, cv2.LINE_8)
        add += int(text)
        co1 += 1
    ave_red = int(ave_x/add)
    ave_green = int(ave_y/add)
    text_a = '(' + str(ave_red) + ', ' + str(ave_green) + ')'
    ave_x = ave_red*5 + 100
    ave_y = 1450 - (ave_green*5 + 50)
    cv2.drawMarker(grh_img, (ave_x, ave_y), (255,0,0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1, line_type=cv2.LINE_8)
    cv2.putText(grh_img,text_a, (1180,1380), cv2.FONT_HERSHEY_PLAIN,2, (0,0,0),3, cv2.LINE_8)
    grgrh_1 = "/Users/BRLAB/others/photo_GRgrh/"+str(name)+"_1_grh.jpg"
    cv2.imwrite(grgrh_1,grh_img)
    #____________________________________________________________________________________________________________

    return img ,ave_red ,ave_green


def area_w(im, name, ikiti):


    x = 0

    while x <= 2:
        x += 1
        
        if x == 1:
            img = im[0:3350,0:1180]
        elif x == 2: 
            img = im[0:3350,1180:2360]
        elif x == 3:
            break

        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)# HSV 色相(Hue)彩度(Saturation)明度(Value・Brightness)
        img_HSV = cv2.GaussianBlur(img_HSV, (3, 3), 0)

        img_H, img_S, img_V = cv2.split(img_HSV) # ＢＧＲの分割(今回はHSV)


        # 65 (足裏)__________________________________________________________________
        i = 65
        _thre, img_mk = cv2.threshold(img_V, i, 255, cv2.THRESH_BINARY) # 二値化
        wr_i = "/Users/BRLAB/others/img3/zz" + str(i) + str(x) + ".jpg"
        cv2.imwrite(wr_i, img_mk)# そのままでは色が塗れない。一度保存してからインプットした画像なら色が塗れる。（画像空間の問題）
        img_u= cv2.imread(wr_i)# img_V(明度のみ取得した画像)を一度保存してからインプットすることにより、画像空間をBGRで取れて、色を塗れるようになる。

        # 塗り替え
        img_uc = cv2.cvtColor(img_u, cv2.COLOR_BGR2HSV)

        # H(色相) は0～180、0,180(赤),75(緑っぽい),100(薄青),179(赤),255(緑っぽい)=255-180=75 !,
        h=img_uc[:,:,(0)]
        h=np.where((h<10) & (h>=0),120,h)
        img_uc[:,:,(0)]=h
        # S(彩度) は0～255、256=0
        s=img_uc[:,:,(1)]
        s=np.where((s<10) & (s>=0),255,s)
        img_uc[:,:,(1)]=s
        # V(明度) は0～255、256=0
        v=img_uc[:,:,(2)]
        v =np.where((v<=255) & (v>=100),255,v)
        img_uc[:,:,(2)]=v

        img_65=cv2.cvtColor(img_uc, cv2.COLOR_HSV2BGR)
        cv2.imwrite(wr_i, img_65)
        area1 = int(cv2.countNonZero(img_mk))
        #print("足裏：",area1,"/",img_mk.size)


        # ikiti（接地面）_____________________________________________________________________
        
        wr_ik = "/Users/BRLAB/others/img3/zz" + str(ikiti) + str(x) + ".jpg"
        #cv2.imwrite(wr_ik, img_V)
        #img_u= cv2.imread(wr_ik)
        img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        #cc=img_Lab[1000,500,:]
        #print(cc)
        img_Lab_L, img_Lab_a, img_Lab_b = cv2.split(img_Lab)


        #L*a*b* 色空間は、ヒトの感覚に合わせて考案された色空間である。
        # L* は明度で、0 ≤ L* ≤ 100 の値を取りうる。a* および b* は色味を表している。
        # a* は緑色と赤色を表し、a* が正の大きな値になるほど赤みが強くなり、負の大きな値になるほど緑みが強くなり、また 0 であれば無彩色となる。
        # また、b* は青色と黄色を表し、a* が正の大きな値になるほど黄色みが強くなり、負の大きな値になるほど青みが強くなり、また 0 であれば無彩色となる。
        # a* および b* の最小値と最大値は、明度 L の値によって異なる
        # L*a*b* 色空間において、L* は 0 ≤ L* ≤ 100、a* および b* はマイナスからプラスまでの値を取りうる。
        # OpenCV においては、0 ≤ L ≤ 100、-127 ≤ a ≤ 127 および -127 ≤ b ≤ 127 で定義されている。
        # ただし、OpenCV の cv2.cvtColor メソッドを用いて変換された場合は、0 ≤ L ≤ 255、0 ≤ a ≤ 255 および 0 ≤ b ≤ 255 となる。
        # OpenCV で L*a*b* 色空間を取り扱うときは、ここを注意する必要がある。
        _thre, img_mk2 = cv2.threshold(img_Lab_b, ikiti, 255, cv2.THRESH_BINARY) # 二値化
        cv2.imwrite(wr_ik, img_mk2)
        img_u= cv2.imread(wr_ik)
        # 塗り替え
        img_uc = cv2.cvtColor(img_u, cv2.COLOR_BGR2HSV)

        # H(色相) は0～180、
        h=img_uc[:,:,(0)]
        h=np.where((h<10) & (h>=0),87,h)
        img_uc[:,:,(0)]=h
        # S(彩度) は0～255、256=0
        s=img_uc[:,:,(1)]
        s=np.where((s<10) & (s>=0),150,s)
        img_uc[:,:,(1)]=s
        # V(明度) は0～255、256=0
        v=img_uc[:,:,(2)]
        v =np.where((v<=255) & (v>=100),255,v)
        img_uc[:,:,(2)]=v

        img_iki =cv2.cvtColor(img_uc, cv2.COLOR_HSV2BGR)
        cv2.imwrite(wr_ik, img_iki)

        area2 = int(cv2.countNonZero(img_mk2))
        #print("接地面：",area2,"/",img_mk.size)
        
        ratio = area2 / area1 *100
        ra1 = format(ratio,'.1f')
        #print("接地面/足裏：",area2,"/",area1,"...",ra1,"%")


        # 重ねる______________________________________________________

        rows,cols,channels = img_iki.shape
        roi = img_65[0:rows, 0:cols ]

        img2gray = cv2.cvtColor(img_iki,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 120, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        img2_fg = cv2.bitwise_and(img_iki,img_iki,mask = mask)
        img2_fg_src = cv2.cvtColor(img2_fg, cv2.COLOR_BGR2RGB)
        plt.imshow(img2_fg_src)



        dst = cv2.add(img1_bg,img2_fg)
        img_65[0:rows, 0:cols ] = dst
        wr_iik = "/Users/BRLAB/others/img3/z" + str(ikiti) + str(x) + ".jpg"
        cv2.imwrite(wr_iik, img_65)

        src3 = cv2.cvtColor(img_65, cv2.COLOR_BGR2RGB)
        #plt.imshow(src3)
        #plt.show()
        
    # ________________________________________________________________\
        
        area1 = int(area1)
        area2 = int(area2)
        
        #tx1 = 'blue : ' + str(area1) + '  lightblue : ' + str(area2)
        tx1 = "Contact Area"
        ra = format(ratio,'.5f') # ratio 1の桁数指定
        if x == 1:
            rera1 = ratio
        elif x == 2:
            rera2 = ratio

        tx2 = 'ratio :' + str(ra)
        ratio1 = area2 / area1
        ra1 = format(ratio1*100,'.1f')
        tx3 = 'lightblue / blue = ' + str(ra1) + '%' 
        tx4 = 'threshold = ' + str(ikiti)
        cv2.putText(img_65,tx1, (100, 100), cv2.FONT_HERSHEY_PLAIN,3, (255, 255, 255),4, cv2.LINE_8)
        #cv2.putText(img_65,tx2, (100, 200), cv2.FONT_HERSHEY_PLAIN,3, (255, 255, 255),4, cv2.LINE_8)
        cv2.putText(img_65,tx3, (100, 200), cv2.FONT_HERSHEY_PLAIN,3, (255, 255, 255),4, cv2.LINE_8)
        #画像データ、テキスト、テキスト右下座標、フォント、文字の縮尺、色（青, 緑, 赤）、文字の太さ、タイプライン（アルゴリズム）、
        cv2.putText(img_65,tx4, (100, 300), cv2.FONT_HERSHEY_PLAIN,3, (255, 255, 255),4, cv2.LINE_8)
        cv2.imwrite(wr_iik, img_65)


    #繋ぐ___________________________________________________________

    x = 1
    wr1 = "/Users/BRLAB/others/img3/z" + str(ikiti) + str(x) + ".jpg"
    i01 = cv2.imread(wr1)
    wr_i = "/Users/BRLAB/others/img3/zz" + str(i) + str(x) + ".jpg"
    wr_ik = "/Users/BRLAB/others/img3/zz" + str(ikiti) + str(x) + ".jpg"
    wr_e = "/Users/BRLAB/others/img3/z_mask_e" + str(x) + ".jpg"
    os.remove(wr_i)
    os.remove(wr_ik)
    os.remove(wr1)
    os.remove(wr_e)
    x = 2
    wr2 = "/Users/BRLAB/others/img3/z" + str(ikiti) + str(x) + ".jpg"
    i02 = cv2.imread(wr2)
    wr_i = "/Users/BRLAB/others/img3/zz" + str(i) + str(x) + ".jpg"
    wr_ik = "/Users/BRLAB/others/img3/zz" + str(ikiti) + str(x) + ".jpg"
    wr_e = "/Users/BRLAB/others/img3/z_mask_e" + str(x) + ".jpg"
    os.remove(wr_i)
    os.remove(wr_ik)
    os.remove(wr2)
    os.remove(wr_e)
    im_h = cv2.hconcat([i01, i02])
    res = "/Users/BRLAB/others/img3/result_3area_" + str(name) + ".jpg"
    cv2.imwrite(res, im_h)

    sole = "/Users/BRLAB/others/img3/sole" + str(name) +".jpg"
    os.remove(sole)

    rera = (rera1 + rera2)/2

    return im_h,rera

def area(im, name):
    pix = 3001
    img = maskAT(name, im, pix)
    im_hs ,red ,green= rgb(img, name)
    para = red + green
    if para < 175 :
        ikiti = 161
    elif para > 174 :
        ikiti = 162

    print('num = ', name, ', threshold = ' + str(ikiti))
    re_img, rera = area_w(img, name, ikiti)

    return re_img, rera

'''
name = 3
pho = "3_1_7519101_20220719.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
img = area(im, name)

#misete = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plt.imshow(misete)
#plt.show()

name = 51
pho = "51_1_7521803_20220803.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
img = area(im, name)

name = 5
pho = "5_1_7519097_20220719.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
img = area(im, name)


pho = "23_1_123456789_20220316.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
img = area(im, name)


pho = "24_2_7518084_20211119.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
img = area(im, name)


pho = "25_2_7520509_20211119.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
img = area(im, name)


pho = "31_4_7521504_20211119.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
img = area(im, name)


pho = "39_2_7518084_20211125.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
img = area(im, name)


pho = "41_1_7281_20220728.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
img = area(im, name)


pho = "58_1_80910_20220809.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
img = area(im, name)


pho = "61_1_7510000_20221010.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
img = area(im, name)


pho = "74_1_1235698_20220718.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
img = area(im, name)


pho = "75_1_7519038_20220718.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
img = area(im, name)


pho = "119_1_7520005_20211216.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
img = area(im, name)


pho = "133_1_2613_20211217.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
img = area(im, name)


pho = "137_1_2617_20211220.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
img = area(im, name)


#ratiot = format(ratio,'.1f')
#print("湿潤面：",ratiot, "%")

#grc0 = cv2.cvtColor(im_hs, cv2.COLOR_BGR2RGB)
#plt.imshow(grc0)
#plt.show()
'''
