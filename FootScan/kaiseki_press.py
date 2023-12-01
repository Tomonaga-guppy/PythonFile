# 足裏形状取得には明度を用いる
# 接地面取得やその後の圧力分布にはグレースケールによる二値化を利用

import matplotlib.pyplot as plt
import cv2
import numpy as np


def press(im,name): # 引数は、画像と保存名

    im = cv2.flip(im, 1) # 反転

    res = "/Users/BRLAB/others/img2/" + str(name) + ".jpg"

    x = 0

    while x <= 2:
        x += 1
        
        if x == 1:
            img = im[95:3445,55:1235]
        elif x == 2: 
            img = im[95:3445,1300:2480]
        elif x == 3:
            break


        wrs = "/Users/BRLAB/others/img2/sole" + str(x) + ".jpg"
        cv2.imwrite(wrs, img)
        imgcope = cv2.imread(wrs)

        # 足裏周りのノイズ除去_____________________________________________________
        # 足裏型凸包の外側を真っ黒にし、そのマスクを取得、元の画像に合成
        (takasa, haba, nanka) = imgcope.shape
        chushin_y = int(takasa/2)
        chushin_x = int(haba/2)
        # print("中心は、", chushin_x,chushin_y)
        imgcope_HSV = cv2.cvtColor(imgcope, cv2.COLOR_BGR2HSV)# HSV 色相(Hue)彩度(Saturation)明度(Value・Brightness)
        imgcope_HSV = cv2.GaussianBlur(imgcope_HSV, (5, 5), 0)
        imgco_H, imgco_S, imgco_V = cv2.split(imgcope_HSV)
        _thre, mask_a = cv2.threshold(imgco_V, 65, 255, cv2.THRESH_BINARY) # 二値化
        mask_b = cv2.dilate(mask_a,np.ones((10,10),np.uint8),iterations = 1)
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

        for a in range(len(hull)): # 凸包（hull）を拡大させる、中心から遠いほど伸びは大きくなる（足先やかかと）
            [p1] =  hull[a]
            x1_sa = p1[0] - chushin_x 
            y1_sa = p1[1] - chushin_y
            x1 = p1[0] + (x1_sa)/12
            y1 = p1[1] + (y1_sa)/13
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
        mask_c = cv2.drawContours(mask_b,[hull],0,(0,0,255),3) # 凸包
        wr_c = "/Users/BRLAB/others/img2/z_mask_c" + str(x) + ".jpg"
        cv2.imwrite(wr_c, mask_c)
        mask_d = cv2.fillConvexPoly(mask_b, hull, (0,0,255)) # 凸包内の塗りつぶす
        wr = "/Users/BRLAB/others/img2/z_mask_d" + str(x) + ".jpg"
        cv2.imwrite(wr, mask_d)
        mask_d = cv2.cvtColor(mask_d, cv2.COLOR_BGR2HSV)
        mask_H, mask_S, mask_V = cv2.split(mask_d) # 彩度
        _thre, mask_e = cv2.threshold(mask_S, 100, 255, cv2.THRESH_BINARY_INV) # 彩度を用いて、塗りつぶした部分を黒、周りを白に
        wr_e = "/Users/BRLAB/others/img2/z_mask_e" + str(x) + ".jpg"
        cv2.imwrite(wr_e, mask_e)
        mask_e = cv2.imread(wr_e)
        # マスクを重ねてノイズ除去____________________________________________________________
        rows,cols,channels = mask_e.shape
        roi = imgcope[0:rows, 0:cols ]

        img2gray_e = cv2.cvtColor(mask_e,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray_e, 120, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg_e = cv2.bitwise_and(roi,roi,mask = mask_inv)
        wr_im = "/Users/BRLAB/others/img2/z_mask_im"+ str(x) + ".jpg"
        cv2.imwrite(wr_im, img1_bg_e)
        img = cv2.imread(wr_im)

        # ___________________________________________________________________________________


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)# HSV 色相(Hue)彩度(Saturation)明度(Value・Brightness)
        img_HSV = cv2.GaussianBlur(img_HSV, (3, 3), 0)
        img_H, img_S, img_V = cv2.split(img_HSV) # ＢＧＲの分割(今回はHSV)

        # 65 (足裏のみ明度を用いて検出)__________________________________________________________________
        ib = 65
        _thre, img_mk = cv2.threshold(img_V, ib, 255, cv2.THRESH_BINARY) # 二値化
        wr = "/Users/BRLAB/others/img2/zzg" + str(ib) + str(x) + ".jpg"
        cv2.imwrite(wr, img_mk)
        img_u= cv2.imread(wr)

        # 塗り替え
        img_uc = cv2.cvtColor(img_u, cv2.COLOR_BGR2HSV)
        
        # H(色相) は0～180、
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
        cv2.imwrite(wr, img_65)
        
        area65 = int(cv2.countNonZero(img_mk))
        
        # 85 _____________________________________________________________________
        ilb = 85
        _thre, img_mk2 = cv2.threshold(gray, ilb, 255, cv2.THRESH_BINARY) # 二値化
        wr = "/Users/BRLAB/others/img2/zzg" + str(ilb) + str(x) + ".jpg"
        cv2.imwrite(wr, img_mk2)
        img_u= cv2.imread(wr)
        img_uc = cv2.cvtColor(img_u, cv2.COLOR_BGR2HSV)
        # H(彩度) は0～180、
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

        img_80 =cv2.cvtColor(img_uc, cv2.COLOR_HSV2BGR)
        cv2.imwrite(wr, img_80)
        
        # 面積比
        area80 = int(cv2.countNonZero(img_mk2))
        ratio1 = 100 * area80 / area65
        ra1 = format(ratio1,'.1f')
        tx1 = 'lightblue/blue (%) :' + str(ra1) + '%' 
        #print()
        #if x == 1:
            #print("面積比(左)")
        #elif x == 2:
            #print("面積比(右)")
        #print(tx1)
    
        # 重ねる______________________________________________________

        rows,cols,channels = img_80.shape
        roi = img_65[0:rows, 0:cols ]

        img2gray = cv2.cvtColor(img_80,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 120, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        img2_fg = cv2.bitwise_and(img_80,img_80,mask = mask)
            
        dst = cv2.add(img1_bg,img2_fg)
        img_65[0:rows, 0:cols ] = dst
        
        img_65 = cv2.cvtColor(img_65, cv2.COLOR_BGR2RGB)
        plt.imshow(img_65)
        plt.show
        
        wr = "/Users/BRLAB/others/img2/zzzg" + str(65) + str(x) + ".jpg"
        img_65 = cv2.cvtColor(img_65, cv2.COLOR_RGB2BGR)
        cv2.imwrite(wr, img_65)
    
        # 黄緑、黄、オレンジ、赤、90～105_____________________________________________________________
        i = 85
        a = 0
        
        while i <= 105 :
            i += 5 
            a += 1
            _thre, img_mk = cv2.threshold(gray, i, 255, cv2.THRESH_BINARY) # 二値化
            wr = "/Users/BRLAB/others/img2/zzg" + str(i) + str(x) + ".jpg"
            cv2.imwrite(wr, img_mk)
            img_u= cv2.imread(wr)

            # 塗り替え
            img_uc = cv2.cvtColor(img_u, cv2.COLOR_BGR2HSV)

            # H(色相) は0～180、0,180(赤),75(緑っぽい),100(薄青),179(赤),255(緑っぽい)=255-180=75 !,
            h=img_uc[:,:,(0)]
            h=np.where((h<10) & (h>=0),(i-25-a*20),h)
            img_uc[:,:,(0)]=h
            # S(彩度) は0～255、256=0
            s=img_uc[:,:,(1)]
            s=np.where((s<10) & (s>=0),255,s)
            img_uc[:,:,(1)]=s
            # V(明度) は0～255、256=0
            v=img_uc[:,:,(2)]
            v =np.where((v<=255) & (v>=100),255,v)
            img_uc[:,:,(2)]=v

            img_j=cv2.cvtColor(img_uc, cv2.COLOR_HSV2BGR)
            cv2.imwrite(wr, img_j)
            
            # 面積比
            area_j = int(cv2.countNonZero(img_mk))
            ratio_j = 100 * area_j / area65
            ra_j = format(ratio_j,'.1f')
            
            if i == 90:
                col = 'green'
            elif i == 95:
                col = 'yellow'
            elif i == 100:
                col = 'orange'
            elif i == 105:
                col = 'red'
            tx_j = col +'/blue (%) :' + str(ra_j) + '%'
            #print(tx_j)

            
            # 重ねる______________________________________________________

            wr1 = "/Users/BRLAB/others/img2/zzzg" + str(65) + str(x) + ".jpg"
            img_65 = cv2.imread(wr1)
            
            rows,cols,channels = img_j.shape
            roi = img_65[0:rows, 0:cols ]

            img2gray = cv2.cvtColor(img_j,cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
            img2_fg = cv2.bitwise_and(img_j,img_j,mask = mask)
            
            dst = cv2.add(img1_bg,img2_fg)
            img_65[0:rows, 0:cols ] = dst
            
            cv2.imwrite(wr1, img_65)


            if i == 105 :
                break

        # 桃
        iw = 112
        _thre, img_mk = cv2.threshold(gray, iw, 255, cv2.THRESH_BINARY) # 二値化
        wre = "/Users/BRLAB/others/img2/zzg" + str(iw) + str(x) + ".jpg"
        cv2.imwrite(wre, img_mk)
        img_m= cv2.imread(wre)
        img_uc = cv2.cvtColor(img_m, cv2.COLOR_BGR2HSV)
        
        # H(色相) は0～180、
        h=img_uc[:,:,(0)]
        h=np.where((h<10) & (h>=0),1,h)
        img_uc[:,:,(0)]=h
        # S(彩度) は0～255、256=0
        s=img_uc[:,:,(1)]
        s=np.where((s<10) & (s>=0),120,s)
        img_uc[:,:,(1)]=s
        # V(明度) は0～255、256=0
        v=img_uc[:,:,(2)]
        v =np.where((v<=255) & (v>=100),255,v)
        img_uc[:,:,(2)]=v

        img_110=cv2.cvtColor(img_uc, cv2.COLOR_HSV2BGR)
        cv2.imwrite(wre, img_110)
        
        img_110 = cv2.imread(wre)
        
        # 面積比
        area115 = int(cv2.countNonZero(img_mk))
        ratio3 = 100 * area115 / area65
        ra3 = format(ratio3,'.1f')
        tx3 = 'momo/blue (%) :' + str(ra3) + '%' 
        #print(tx3)

        # 重ねる______________________________________________________

        wr = "/Users/BRLAB/others/img2/zzzg" + str(65) + str(x) + ".jpg"
        img_65 = cv2.imread(wr)
    
        rows,cols,channels = img_110.shape
        roi = img_65[0:rows, 0:cols ]

        img2gray = cv2.cvtColor(img_110,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 120, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        img2_fg = cv2.bitwise_and(img_110,img_110,mask = mask)
            
        dst = cv2.add(img1_bg,img2_fg)
        img_65[0:rows, 0:cols ] = dst
        cv2.imwrite(wr, img_65)

        # 白_____________________________________________________________________

        iw = 119
        _thre, img_mk = cv2.threshold(gray, iw, 255, cv2.THRESH_BINARY) # 二値化
        wre = "/Users/BRLAB/others/img2/zzg" + str(iw) + str(x) + ".jpg"
        cv2.imwrite(wre, img_mk)
        img_120 = cv2.imread(wre)
        
        # 面積比
        area120 = int(cv2.countNonZero(img_mk))
        ratio4 = 100 * area120 / area65
        ra4 = format(ratio4,'.1f')
        tx4 = 'white/blue (%) :' + str(ra4) + '%' 
        #print(tx4)

        # 重ねる______________________________________________________

        wr = "/Users/BRLAB/others/img2/zzzg" + str(65) + str(x) + ".jpg"
        img_65 = cv2.imread(wr)
    
        rows,cols,channels = img_120.shape
        roi = img_65[0:rows, 0:cols ]

        img2gray = cv2.cvtColor(img_120,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 120, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        img2_fg = cv2.bitwise_and(img_120,img_120,mask = mask)
            
        dst = cv2.add(img1_bg,img2_fg)
        img_65[0:rows, 0:cols ] = dst
        cv2.imwrite(wr, img_65)

        text = 'Pressure Distribution'
        #text1 = 'Grounded Area = '+ str(ra1) + '%'
        if x == 1:
            ra_red1 = ratio_j
        elif x == 2:
            ra_red2 = ratio_j
        text2 = 'Red / Blue = ' + str(ra_j) + '%'
        cv2.putText(img_65,text, (100, 100), cv2.FONT_HERSHEY_PLAIN,3, (255, 255, 255),4, cv2.LINE_8)
        #cv2.putText(img_65,text1, (100, 200), cv2.FONT_HERSHEY_PLAIN,3, (255, 255, 255),4, cv2.LINE_8)
        cv2.putText(img_65,text2, (100, 200), cv2.FONT_HERSHEY_PLAIN,3, (255, 255, 255),4, cv2.LINE_8)
        cv2.imwrite(wr, img_65)
        
    #繋ぐ___________________________________________________________

    i = 65
    x = 1
    wr1 = "/Users/BRLAB/others/img2/zzzg" + str(i) + str(x) + ".jpg"
    i01 = cv2.imread(wr1)
    x = 2
    wr2 = "/Users/BRLAB/others/img2/zzzg" + str(i) + str(x) + ".jpg"
    i02 = cv2.imread(wr2)
    im_h = cv2.hconcat([i01, i02])
    res = "/Users/BRLAB/others/img2/result_2press_" + str(name) + ".jpg"
    cv2.imwrite(res, im_h)

    ra_red = (ra_red1 + ra_red2)/2
    
    return (im_h,ra_red)


'''
name = 27
pho = "27_1_1009_20220722.jpg"
date = "/Users/BRLAB/others/imgdate/" + str(pho)
im = cv2.imread(date)

im_hs , ratio = press(im,119)
#ratiot = format(ratio,'.1f')
#print("湿潤面：",ratiot, "%")

grc0 = cv2.cvtColor(im_hs, cv2.COLOR_BGR2RGB)
plt.imshow(grc0)
plt.show()

'''

#print("結果を",res,"として保存しました。")


#img_trimmim = cv2.resize(im_hs, (500, 666))
#cv2.imshow(name,img_trimmim)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

