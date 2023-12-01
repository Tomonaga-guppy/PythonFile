# エッジ検出（しわ）、Canny処理
# 足裏形状の形の外、足裏以外の部分を最後にくり抜き、ゼロにしてる

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# ただの白黒画像より、HSV空間画像として明度（V）のみ分割・抽出した画像の方がしわがくっきりしてる


def maskAT(name,im, pix):
    im = cv2.flip(im, 1) # 反転
    # numbe を画像名の最初の数字にしたい

    x = 0
    while x <= 2:
        x += 1
        
        if x == 1:
            img = im[95:3445,55:1235]
        elif x == 2: 
            img = im[95:3445,1300:2480]
        elif x == 3:
            break
        

        wrs = "/Users/BRLAB/others/img4/sole" + str(x) + ".jpg"
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

        res = "/Users/BRLAB/others/img4/result_3area_L_" + str(name) + str(x) + ".jpg"
        cv2.imwrite(res, img_L_r)

    #繋ぐ___________________________________________________________
    x = 1
    wr1_L = "/Users/BRLAB/others/img4/result_3area_L_" + str(name) + str(x) + ".jpg"
    i01_L = cv2.imread(wr1_L)
    wrs_1 = "/Users/BRLAB/others/img4/sole" + str(x) + ".jpg"
    i01 = cv2.imread(wrs_1)
    x = 2
    wr2_L = "/Users/BRLAB/others/img4/result_3area_L_" + str(name) + str(x) + ".jpg"
    i02_L = cv2.imread(wr2_L)
    wrs_2 = "/Users/BRLAB/others/img4/sole" + str(x) + ".jpg"
    i02 = cv2.imread(wrs_2)

    im_h_L = cv2.hconcat([i01_L, i02_L])
    res_L = "/Users/BRLAB/others/img4/result_3area_L" + str(name) + str(pix) + ".jpg"
    cv2.imwrite(res_L, im_h_L)
    im_h = cv2.hconcat([i01, i02])
    res = "/Users/BRLAB/others/img4/sole" + str(name) +".jpg"
    cv2.imwrite(res, im_h)
    os.remove(wrs_1)
    os.remove(wrs_2)

    x = 0
    imgdate = "/Users/BRLAB/others/img4/sole" + str(name) +".jpg"
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
        wr1_L = "/Users/BRLAB/others/img4/result_3area_L_" + str(name) + str(x) + ".jpg"

        mask_b = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # imgcope_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)# HSV 色相(Hue)彩度(Saturation)明度(Value・Brightness)
        #imgcope_HSV = cv2.GaussianBlur(imgcope_HSV, (5, 5), 0)
        #imgco_H, imgco_S, imgco_V = cv2.split(imgcope_HSV)
        #mask_b = cv2.dilate(mask_b,np.ones((10,10),np.uint8),iterations = 1)
        wr_b = "/Users/BRLAB/others/img4/z_mask_b" + str(x) + ".jpg"
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
        #wr_c = "/Users/BRLAB/others/img4/z_mask_b" + str(x) + str(name) + ".jpg"
        #cv2.imwrite(wr_c, mask_bb)
        #wr_c = "/Users/BRLAB/others/img4/z_img_b" + str(x) + str(name) + ".jpg"
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
        wr_e = "/Users/BRLAB/others/img4/z_mask_e" + str(x) + ".jpg"
        cv2.imwrite(wr_e, mask_e)
        mask_e = cv2.imread(wr_e)
        # マスクを重ねてノイズ除去____________________________________________________________
        rows,cols,channels = mask_e.shape
        roi = img_d[0:rows, 0:cols ]

        img2gray = cv2.cvtColor(mask_e,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 120, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        wr_im = "/Users/BRLAB/others/img4/z_mask_im"+ str(x) + str(name) + ".jpg"
        cv2.imwrite(wr_im, img1_bg)
        img = cv2.imread(wr_im)

        os.remove(wr1_L)
        os.remove(wr_b)
        #os.remove(wr_e)

    #繋ぐ___________________________________________________________
    x = 1    
    wr1 = "/Users/BRLAB/others/img4/z_mask_im" + str(x) + str(name) +  ".jpg"
    i01_img = cv2.imread(wr1)
    os.remove(wr1)   
    x = 2
    wr2 = "/Users/BRLAB/others/img4/z_mask_im" + str(x) + str(name) +  ".jpg"
    i02_img = cv2.imread(wr2)
    os.remove(wr2)
    im_himg = cv2.hconcat([i01_img, i02_img])
    res = "/Users/BRLAB/others/img4/result_sole" + str(name) + ".jpg"
    cv2.imwrite(res, im_himg)
    os.remove(res_L)

    ####これ用
    x = 1  
    wr_e = "/Users/BRLAB/others/img4/z_mask_e" + str(x) + ".jpg"
    mask_e_1 = cv2.imread(wr_e)
    x = 2
    wr_e = "/Users/BRLAB/others/img4/z_mask_e" + str(x) + ".jpg"
    mask_e_2 = cv2.imread(wr_e)
    return mask_e_1,mask_e_2



def edg_at(im, name,mask_e1,mask_e2):

    im = cv2.flip(im, 1) # 反転

    x = 0

    while x <= 2:
        x += 1
        
        if x == 1:
            img = im[95:3445,55:1235]
            mask_e = mask_e1
        elif x == 2: 
            img = im[95:3445,1300:2480]
            mask_e = mask_e2
        elif x == 3:
            break
 

        wrs = "/Users/BRLAB/others/img4/sole" + str(x) + ".jpg"
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
        pix = 19
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


        #ガウシアンフィルタ (Gaussian filter) 
        img_L_r  = cv2.GaussianBlur(img_L_r , (11,11), 1)
        #ガウス分布 (正規分布)で重み付けしたカーネルを使用するフィルタ
    

        res = "/Users/BRLAB/others/img4/result_3area_L_" + str(name) + str(x) + ".jpg"
        cv2.imwrite(res, img_L_r)
        img_at_b = cv2.imread(res)
        img_at_bg = cv2.cvtColor(img_at_b, cv2.COLOR_BGR2GRAY)
        _thre, img_bz = cv2.threshold(img_at_bg, 180, 255, cv2.THRESH_BINARY) 


        kernel = np.ones((3,3),np.uint8)
        img_bz = cv2.morphologyEx(img_bz, cv2.MORPH_CLOSE, kernel)
        dst = cv2.morphologyEx(img_bz, cv2.MORPH_OPEN, kernel)
        # = np.ones((4,4),np.uint8)
        #dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel2)

        #grc0 = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
        #plt.imshow(grc0)
        #plt.show()

        res = "/Users/BRLAB/others/img4/result_3area_L_" + str(name) + str(x) + ".jpg"
        cv2.imwrite(res, dst)
        dst = cv2.imread(res)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(res, dst)
        

        
        # 65 (明度による足裏を用いて足裏上のしわのみを検出)*************************************************************
        imgs_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)# HSV 色相(Hue)彩度(Saturation)明度(Value・Brightness)
        imgs_HSV = cv2.GaussianBlur(imgs_HSV, (3, 3), 0)
        imgs_H, imgs_S, imgs_V = cv2.split(imgs_HSV) # ＢＧＲの分割(今回はHSV)
        _thre, imgs_mk = cv2.threshold(imgs_V, 65, 255, cv2.THRESH_BINARY_INV)
        wr = "/Users/BRLAB/others/img4/mask" + str(x) + ".jpg"
        cv2.imwrite(wr, imgs_mk)
        imgs_65= cv2.imread(wr,cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.uint8)
        imgs = cv2.morphologyEx(imgs_65, cv2.MORPH_OPEN, kernel) # オープニングでノイズ除去
        #imgs = cv2.erode(imgs,kernel,iterations = 1) # 縮小, 足裏が黒、周りが白なので、これで膨張
        imgs = cv2.dilate(imgs,kernel,iterations = 1) # 膨張

        cv2.imwrite(wr, imgs)
        imgs_65= cv2.imread(wr)
        
        # (失敗：初めに足型のみ切り抜くと、足型上にエッジが強く出てしまう)
        #_thre, img = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO) # 二値化
        #cv2.imwrite("/Users/BRLAB/others/edg/00.jpg", img)
        
        # 重ねる_________________________________
        # 足裏と思われる領域上のしわのみ表示

        rows,cols,channels = imgs_65.shape
        roi = dst[0:rows, 0:cols ]

        img2gray = cv2.cvtColor(imgs_65,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 120, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        wr_shi = "/Users/BRLAB/others/img4/z_mask_shiwa"+ str(x) + ".jpg"
        cv2.imwrite(wr_shi, img1_bg)
        dst_ni= cv2.imread(wr_shi)

        
        # マスクを重ねてノイズ除去____________________________________________________________
        rows,cols,channels = mask_e.shape
        roi = dst_ni[0:rows, 0:cols ]

        mask_e_gray = cv2.cvtColor(mask_e,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(mask_e_gray, 120, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        shiwa_mask = cv2.bitwise_and(roi,roi,mask = mask_inv)
        cv2.imwrite(wr_shi, shiwa_mask)
        img_edg = cv2.imread(wr_shi)


        img_edg = cv2.cvtColor(img_edg, cv2.COLOR_BGR2GRAY)

        shiwa = int(cv2.countNonZero(img_edg))

        #print(shiwa)
        # **************************************************************************************


    
        # しわ画像に足裏形の背景をいれる！！__________________________________________________________________
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_HSV = cv2.GaussianBlur(img_HSV, (3, 3), 0)
        img_H, img_S, img_V = cv2.split(img_HSV)
        _thre, img_mkbg = cv2.threshold(img_V, 65, 255, cv2.THRESH_BINARY)
        wr_bg = "/Users/BRLAB/others/img4/zbg65" + str(x) + ".jpg"
        cv2.imwrite(wr_bg, img_mkbg)
        img_65= cv2.imread(wr_bg)
        kernel = np.ones((10,10),np.uint8)
        img_65 = cv2.dilate(img_65,kernel,iterations = 1) # 膨張

         # 塗り替え
        img_uc = cv2.cvtColor(img_65, cv2.COLOR_BGR2HSV)

        # H(色相) は0～180、0,180(赤),75(緑っぽい),100(薄青),179(赤),255(緑っぽい)=255-180=75 !,
        h=img_uc[:,:,(0)]
        h=np.where((h<10) & (h>=0),10,h)
        img_uc[:,:,(0)]=h
        # S(彩度) は0～255、256=0
        s=img_uc[:,:,(1)]
        s=np.where((s<10) & (s>=0),200,s)
        img_uc[:,:,(1)]=s
        # V(明度) は0～255、256=0
        v=img_uc[:,:,(2)]
        v =np.where((v<=255) & (v>=100),255,v)
        img_uc[:,:,(2)]=v

        img_j=cv2.cvtColor(img_uc, cv2.COLOR_HSV2BGR)
        cv2.imwrite(wr_bg, img_j)
        img_bg65 = cv2.imread(wr_bg)

        # マスクを重ねてノイズ除去____________________________________________________________
        rows,cols,channels = mask_e.shape
        roi = img_bg65[0:rows, 0:cols ]

        mask_e_gray = cv2.cvtColor(mask_e,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(mask_e_gray, 120, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        mask_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        wr_im = "/Users/BRLAB/others/img4/z_mask_65"+ str(x) + ".jpg"
        cv2.imwrite(wr_im, mask_bg)
        img_65 = cv2.imread(wr_im)
        # ___________________________________________________________________________________

        bunyou = cv2.imread(wr_im)
        bunyou = cv2.cvtColor(bunyou,cv2.COLOR_BGR2GRAY)
        ret, bunyou = cv2.threshold(bunyou, 100, 255, cv2.THRESH_BINARY)
        bun = int(cv2.countNonZero(bunyou))
        #wr_bun = "/Users/BRLAB/others/img4/z_bun"+ str(x) + ".jpg"
        #cv2.imwrite(wr_bun, bunyou)
        #____________________________________________________________________________________

        # 重ねる_____________________________________________________
        img_edg = cv2.cvtColor(img_edg, cv2.COLOR_GRAY2BGR)
        rows,cols,channels = img_edg.shape
        roi = img_65[0:rows, 0:cols ]

        img2gray = cv2.cvtColor(img_edg,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 120, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        img2_fg = cv2.bitwise_and(img_edg,img_edg,mask = mask)
            
        d_st = cv2.add(img1_bg,img2_fg)
        img_65[0:rows, 0:cols ] = d_st
        wr = "/Users/BRLAB/others/img4/bg65"+ str(x) + ".jpg"
        cv2.imwrite(wr, img_65)

        img11_bg = cv2.imread(wr)
        
        # しわ算出、テキスト書き込み_____________________________________________________________________
        ratio = 100 * shiwa / bun
        ra = format(ratio,'.2f')
        if x == 1:
            shiwa1 = ratio
        elif x == 2:
            shiwa2 = ratio
        text = 'edge : ' + str(ra) + '%'
        text0 = 'Edge'
        cv2.putText(img11_bg,text0, (100, 100), cv2.FONT_HERSHEY_PLAIN,3, (255, 255, 255),4, cv2.LINE_8)
        cv2.putText(img11_bg,text, (100, 200), cv2.FONT_HERSHEY_PLAIN,3, (255, 255, 255),4, cv2.LINE_8)
        
        #print()
        #if x == 1:
            #print("面積比(左)")
        #elif x == 2:
            #print("面積比(右)")
        #print("しわ率：",shiwa,"(しわ) /",bun,"(足裏形状)")
        #print("shiwa : ",ra,"%")
    
        wr = "/Users/BRLAB/others/img4/edge"+ str(name) + str(x) + ".jpg"
        cv2.imwrite(wr, img11_bg)


    x = 1
    wr1 = "/Users/BRLAB/others/img4/edge"+ str(name) + str(x) + ".jpg"
    i01 = cv2.imread(wr1)
    img_shiwa = "/Users/BRLAB/others/img4/z_mask_shiwa"+ str(x) + ".jpg"
    im_s1 = cv2.imread(img_shiwa)

    x = 2
    wr2 = "/Users/BRLAB/others/img4/edge"+ str(name) + str(x) + ".jpg"
    i02 = cv2.imread(wr2)
    img_shiwa = "/Users/BRLAB/others/img4/z_mask_shiwa"+ str(x) + ".jpg"
    im_s2 = cv2.imread(img_shiwa)

    im_h = cv2.hconcat([i01, i02])
    im_s = cv2.hconcat([im_s1, im_s2])
 
    res = "/Users/BRLAB/others/img4/result_4edge_" + str(name)+ ".jpg"
    cv2.imwrite(res, im_h)


    # 要らないものを消す

    imgdate = "/Users/BRLAB/others/img4/sole" + str(name) +".jpg"
    os.remove(imgdate)

    x = 1
    sole = "/Users/BRLAB/others/img4/sole" + str(x) + ".jpg"
    os.remove(sole)
    wr = "/Users/BRLAB/others/img4/edge"+ str(name) + str(x) + ".jpg"
    os.remove(wr)
    wr_e = "/Users/BRLAB/others/img4/z_mask_e" + str(x) + ".jpg"
    os.remove(wr_e)
    img_shi = "/Users/BRLAB/others/img4/result_3area_L_" + str(name) + str(x) + ".jpg"
    os.remove(img_shi)
    img_shiwa = "/Users/BRLAB/others/img4/z_mask_shiwa"+ str(x) + ".jpg"
    os.remove(img_shiwa)
    wr_im = "/Users/BRLAB/others/img4/z_mask_65"+ str(x) + ".jpg"
    os.remove(wr_im)
    wr_im = "/Users/BRLAB/others/img4/bg65"+ str(x) + ".jpg"
    os.remove(wr_im)
    wr_bg = "/Users/BRLAB/others/img4/zbg65" + str(x) + ".jpg"
    os.remove(wr_bg)
    wr_m = "/Users/BRLAB/others/img4/mask" + str(x) + ".jpg"
    os.remove(wr_m)

    x = 2
    sole = "/Users/BRLAB/others/img4/sole" + str(x) + ".jpg"
    os.remove(sole)
    wr = "/Users/BRLAB/others/img4/edge"+ str(name) + str(x) + ".jpg"
    os.remove(wr)
    wr_e = "/Users/BRLAB/others/img4/z_mask_e" + str(x) + ".jpg"
    os.remove(wr_e)
    img_shi= "/Users/BRLAB/others/img4/result_3area_L_" + str(name) + str(x) + ".jpg"
    os.remove(img_shi)
    img_shiwa = "/Users/BRLAB/others/img4/z_mask_shiwa"+ str(x) + ".jpg"
    os.remove(img_shiwa)
    wr_im = "/Users/BRLAB/others/img4/z_mask_65"+ str(x) + ".jpg"
    os.remove(wr_im)
    wr_im = "/Users/BRLAB/others/img4/bg65"+ str(x) + ".jpg"
    os.remove(wr_im)
    wr_bg = "/Users/BRLAB/others/img4/zbg65" + str(x) + ".jpg"
    os.remove(wr_bg)
    wr_m = "/Users/BRLAB/others/img4/mask" + str(x) + ".jpg"
    os.remove(wr_m)

    #shiwa0 = (shiwa1 + shiwa2)/2
    shiwa0 = 1
    return im_h, shiwa0


def  edg(im,name):
    pix = 3001
    mask1,mask2 = maskAT(name,im, pix)
    img_edg, re_shiwa = edg_at(im,name, mask1,mask2)
    
    return img_edg, re_shiwa


'''
name = 94
pho = "94_1_75190018_20221115.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)
#im = cv2.rotate(im, cv2.ROTATE_180)
img_edge, re_shiwa = edg(im,name)
'''

#reshiwa = format(re_shiwa,'.2f')
#print("しわ：",re_shiwa, "%")


#dst_tr = cv2.resize(res_edg, (450, 600))
#cv2.imshow('dst', dst_tr)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
