# 図形の外接矩形、凸包を求める

from ctypes import sizeof
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import csv
import os
#%matplotlib inline

#name = input("保存するときの名前：")
#print("入力する画像を下のフォルダに入れてください。")
#print("/Users/BRLAB/others/cont/")
#print("入力する画像のデータ（例　119_1_7520005_20211216.jpg ）")
#pho = input()
#print()


def juangl(im, name, file_name): # 足裏画像に重心軌跡プロットし、外反母趾あたりの角度を書き込む関数、im(足裏画像), name(頭の番号), file_name(ファイル名)

    im = cv2.flip(im, 1) # 左右反転（右足を右側に、左足を左側に表示）

    # jushin plot _________________________________________________

    img_jushin = im # 足裏画像
    jushinlist = [[0 for i in range(3001)] for j in range(2)] # 重心軌跡座標取得用行列定義

    # 中心 = (  1287 ,  1477  ) # 画像上でのロードセル座標におけるゼロ点の座標
    # 倍率 = (  10.82246285605723 ,  12.16265402653278  ) # 重心座標の変化が画像上で何ピクセル分かの倍率
    ans_ox = 1263.0 # 中心、反転前は、1287
    ans_oy = 1477.0 # 中心
    ans_rax = -11 #10.82246 倍率
    ans_ray = 12 #12.16265 倍率

    # csv ファイルの読み込み ↓ 書き換える必要あり！！！！このファイル内はこれだけ
    #excel_file = '//192.168.1.149/Users/BRLAB/Desktop/jushin/csv_filter/' + str(file_name) + '_filter.csv' # 書き換える必要あり！！！！これだけ
    excel_file = '/Users/BRLAB/Desktop/jushin/csv_filter/' + str(file_name) + '_filter.csv' # 書き換える必要あり！！！！これだけ
    csvdate = open(excel_file, 'r') # csvファイルを開く
    reader = csv.reader(csvdate) # csvファイルを読み込む(行列), 下で開く

    charlie = 0
    dalta = 0
    i = 0
    for row in reader: # 重心位置取得分＋１（3001回?）繰り返す、最初はラベルなので0,0とする。
        #各行の内容の表示alfa(行),bravo(時間(*0.1)),charlie( X 座標),dalta( Y 座標)
        alfa,bravo,charlie,dalta = row

        if i == 0:
            jushinlist[0][i] = 0
            jushinlist[1][i] = 0
        else :
            jushinlist[0][i] = float(charlie)
            jushinlist[1][i] = float(dalta)

        pox = ans_ox + ans_rax*jushinlist[0][i] # (x座標) 中心座標＋倍率*重心座標(csvファイルより取得した値)
        poy = ans_oy + ans_ray*jushinlist[1][i] # (y座標) 中心座標＋倍率*重心座標(csvファイルより取得した値)
        pox = int(pox)
        poy = int(poy)
        # ↓ 画像に重心位置をプロット
        cv2.drawMarker(img_jushin, (pox,poy), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=5, line_type=cv2.LINE_8)
        i += 1

    csvdate.close() # csv ファイル閉じる
    #  ↓ 書き換えなくていい
    resju = "/Users/BRLAB/others/img5/jushin_" + str(file_name) + ".jpg"
    cv2.imwrite(resju, img_jushin)

    # ________________________________________________________________

    x_ = 0

    while x_ <= 2: # 右と左で計二回
        x_ += 1
        
        if x_ == 1: # 左
            img = im[95:3445,55:1235] # 画像を左側のみにする
            img_ju = img_jushin[95:3445,55:1235]

        elif x_ == 2: # 右
            img = im[95:3445,1300:2480] # 画像を右側のみにする
            img_ju = img_jushin[95:3445,1300:2480]
        elif x_ == 3:
            break

        #img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)# HSV 色相(Hue)彩度(Saturation)明度(Value・Brightness)
        #img_HSV = cv2.GaussianBlur(img_HSV, (9, 9), 0)
        #img_H, img_S, img_V = cv2.split(img_HSV) # ＢＧＲの分割(今回はHSV)

        # 65 (足裏)__________________________________________________________________
        i = 65 # 二値化閾値
        img_mk = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 画像をグレースケールにする
        #threshold=220
        _thre, img_mk = cv2.threshold(img_mk, i, 255, cv2.THRESH_BINARY) # 二値化
        wr = "/Users/BRLAB/others/img1/zz" + str(i) + str(x_) + ".jpg"
        cv2.imwrite(wr, img_mk)# そのままでは色が塗れない。一度保存してからインプットした画像なら色が塗れる。（画像空間の問題）
        img_gray0 = cv2.imread(wr) # 二値化した画像を読み込む、再度読み込めばBGR空間の画像となっている
        #img_gray = cv2.imread(wr,cv2.IMREAD_GRAYSCALE) 
        img_gray = cv2.morphologyEx(img_mk, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8)) # 膨張・収縮により穴埋め
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, np.ones((40, 40), np.uint8))# 収縮・膨張によりノイズ除去
        
        # Closing...2値化した画像の点々の隙間ノイズ(画素値0のノイズ)を除去、物体の隙間を補完、別れた部分を結合する役割( 膨張の後に収縮)
        # 1...2値化された画像の、値が1の部分をモルフォロジー変換で膨張・拡大させる,ここで物体の隙間が埋まったり結合される
        # 2...1.で膨張・拡大した画像をモルフォロジー変換で収縮・縮小して元の大きさに戻す
        # cv2.morphologyEx...膨張処理，収縮処理

        #wr = "/Users/BRLAB/others/img1/zzzzzz" + str(x_) + ".jpg"
        #cv2.imwrite(wr, img_gray)

        # img_grayを平均化領域9x9で平均化処理しimg_blurに代入
        #img_blur = cv2.blur(img_gray,(9,9)) 

        # オブジェクトimg_blurを閾値threshold(220)で反転二値化しimg_binaryに代入
        #ret, img_binary= cv2.threshold(img_blur, threshold, 255, cv2.THRESH_BINARY_INV) 
        # img_binaryを輪郭抽出
    
        # contours, 外接矩形取得
        contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        img7 = img

        # 作成できる外接矩形を全て試し、足裏形状を囲うもののみ選択
        for i in range(len(contours)):

            # 最も大きい輪郭に外接矩形を作成し、画像に重ね書き
            rect = cv2.minAreaRect(contours[i])
            ((x,y),(w, h),th) = cv2.minAreaRect(contours[i])

            
            if w*h > 1000000 : # >750000 は大きい輪郭のみを残すため

                #     box = cv2.minAreaRect(contour) ... 輪郭の回転を考慮した外接矩形を計算する
                # 引数 ... contour : 輪郭
                # 戻り値 ... box : ((X座標, Y座標)，(幅, 高さ)，回転角)
                # 描画には4隅の頂点が必要なので、cv2.boxPoints()で変換

                #     points = cv2.boxPoints(box)、
                # 引数 ... box : ((左上X座標, 左上Y座標)，(幅, 高さ)，回転角)
                # 戻り値 ... points : 4隅の頂点

                box = cv2.boxPoints(rect) # 外接矩形は描かないことにしたよ。
                box = np.int0(box)
                hull=cv2.convexHull(contours[i]) #  凸包

                # 色付き画像に外接矩形と凸包を書いていく
                #img = cv2.drawContours(img,[hull],0,(0,0,255),3) # 凸包
                #img = cv2.drawContours(img,[box],0,(0,255,0),7) # 外接矩形
                #img = cv2.line(img,pt1=(xt1,yt1),pt2=(xt2,yt2),color=(0, 255, 0),thickness=3)
                result_img1 = img
                # 色黒画像に外接矩形と凸包を書いていく
                result_imggray1 = cv2.drawContours(img_gray0,[hull],0,(0,0,255),3) # 色黒に凸包
                #result_imggray1 = cv2.drawContours(result_imggray1,[box],0,(0,255,0),3)

        wr = "/Users/BRLAB/others/img1/zzangle" + str(x_) + ".jpg"
        cv2.imwrite(wr, result_img1)
        #wr = "/Users/BRLAB/others/img1/zzanglegray" + str(x_) + ".jpg"
        #cv2.imwrite(wr, result_imggray1)

        saizu = hull.size
        #print("hull の要素の数は、",saizu)


        # 凸包(holl) をおおざっぱなものにする！
        # hull(凸包)の囲うポイントが前のポイントとの距離があまりにも小さいとき消してみる。
        # 上記の条件かつ、角度変化が小さいものも削除してみる
        aa  = 0
        hull2 = hull   # 凸包座標の行列
        anglea2 = -180 # 適当な初期値
        for a in range(len(hull)): # 凸包座標行列の座標の数だけ下を回す
            [p1] =  hull[a]  # a番目の座標[p1]=[x,y]
            x1 = p1[0]
            y1 = p1[1]
            [p2] =  hull[a-1] # a-1番目の座標
            x2 = p2[0]
            y2 = p2[1]
            x = x2-x1 # x2 と x1 の距離
            y = y2-y1 # y2 と y1 の距離
            anglea1 = (math.degrees(math.atan2(x,y))) # x,yの値からa番目の座標とa-1番目の座標を結んだ線の角度を導く
            anglea = (anglea1-anglea2)**2 # 上の角度とanglea2との差の二乗
            if x**2 < 40000 and y**2 < 40000 and anglea < 45 : # x,yの距離と前の座標との角度が小さいときに該当座標を 凸包座標行列の中から排除する
                if (x**2 < 3000 and y**2 < 3000) or anglea< 4 :
                    # numpy.delete(arr, obj, axis=None)
                    # arr: 入力配列
                    # obj: 削除する行番号や列番号を整数、スライス、リスト（配列）で指定
                    # axis: 削除対象となる軸（次元）
                    hull2 = np.delete(hull2,a-aa,0)
                    aa += 1
            anglea2 = (math.degrees(math.atan2(x,y)))

        #saizu = hull2.size
        #print("アレンジ後の hull の要素の数は、",saizu)
        hull = hull2

        # maxlwngth の初期設定！
        [p01] =  hull[0]
        x1 = p01[0]
        y1 = p01[1]
        [p02] =  hull[1]
        x2 = p02[0]
        y2 = p02[1]
        maxlength = (x2-x1)**2 + (y2-y1)**2


        # 外反母趾の角度を示す線の作成！
        #con = 0
        a = 1
        # 足裏内側の二点接線作成、一番長いものを緑色の線として残すため
        hullsaizu = (hull.size / 2) -1 # ( hull の要素数/2 - 1 ), (x,y)で2とカウントさせるため 2 で割ってやる
        while  a < hullsaizu :
            # 点と点の距離を計算していく
            [p1] =  hull[a]
            x1 = p1[0]
            y1 = p1[1]
            [p2] =  hull[a-1]
            x2 = p2[0]
            y2 = p2[1]
            # hull[a]のポイントと順番の確認用
            #con += 1
            #cv2.putText(result_img1,str(con), (x1, y1), cv2.FONT_HERSHEY_PLAIN,5, (0, 230, 255),6, cv2.LINE_8)
            
            if x_ == 1: # 左に写された足
                x = x1-x2
                y = y1-y2
                length3 = x**2 + y**2
                # 足裏内側の二点接線 = ある領域内で一番長いもの、length3 > maxlength、左に写された足は母指球側は右側なので半分より右に座標があることを条件にする
                if all((y1 > 1000, y > 800, x1> 500, x2 > 500, length3 > 10000,  length3 > maxlength )): 
                    maxp1 = (x1,y1)
                    maxp2 = (x2,y2)
                    maxlength = (x2-x1)**2 + (y2-y1)**2

            elif x_ == 2: # 右に写された足
                x = x2-x1 # x > 0 の場合で、身体の「内側」から「外側」方向と決める
                y = y2-y1 # y > 0 の場合で、「かかと」から「方向」と決める
                length3 = x**2 + y**2
                # 足裏内側の二点接線 = ある領域内で一番長いもの、length3 > maxlength、右に写された足は母指球側は左側なので半分より左に座標があることを条件にする
                if all(( y2 > 1000, y > 800, x1< 500,x2 < 500, length3>10000 , length3 > maxlength)): 
                    maxp1 = (x1,y1)
                    maxp2 = (x2,y2)
                    maxlength = (x2-x1)**2 + (y2-y1)**2
            a += 1


        # 主軸角度求める！！線を描く！！（青線の角度）
        # 足裏内側の最長二点接線から、足裏内側のy軸に対する角度を求める
        xm1 = maxp1[0]
        xm2 = maxp2[0] 
        ym1 = maxp1[1] 
        ym2 = maxp2[1] 
        if x_ == 1:
            xt = xm2-xm1 # 内股だと - 、基本（開いているので）+
            yt = ym1-ym2 # 基本 + 
        elif x_ == 2:
            xt = xm2-xm1 # 内股だと - 、基本（開いているので）+
            yt = ym2-ym1 # 基本 + 
        angle0 = (math.degrees(math.atan2(xt,yt))) # atan2(xt,yt) = arctan(xt/yt) (-pi < ans < pi)
        #angle00 = format(angle0,'.2f')
        #print("angle0 = ",angle00)

        # 主軸
        #cv2.line(img_ju,pt1=(xm1,ym1),pt2=(xm2,ym2),color=(255, 0, 0),thickness=7)


        # 足先の座標(ysaki)を取得！
        # 母指球角度検出の時に、足先付近のを除去するのに使うよ
        ysaki = 0
        for a in range(len(hull)):
            [p2] =  hull[a-1]
            x2 = p2[0]
            y2 = p2[1]
            if ysaki > y2 :
                ysaki = y2
        #print("ysaki =", ysaki)


        # 角度がわかるよう描く！！
        # 母指球当たりの曲がりにマーキング, 扇形を書いていく
        # 今回、hull は、時計回りでポイントを打つもよう,
        count = 0
        for a in range(len(hull)):
            # 点と点の距離を計算していく
            [p1] =  hull[a]
            x1 = p1[0]
            y1 = p1[1]
            [p2] =  hull[a-1]
            x2 = p2[0]
            y2 = p2[1]

            if x_ == 1:
                x = x1-x2 # x > 0 の場合で、身体の「内側」から「外側」方向と決める
                y = y1-y2 # y > 0 の場合で、「かかと」から「方向」と決める
            elif x_ == 2:
                x = x1-x2 # x > 0 の場合で、身体の「内側」から「外側」方向と決める
                y = y2-y1 # y > 0 の場合で、「かかと」から「方向」と決める
            length1 = x**2 + y**2 # あまりにも小さい距離感のものは削除
            #cv2.line(result_img1,pt1=(x1,y1),pt2=(x1+10,y1+10),color=(55, 255, 1),thickness=5)
            # ラインの角度求める
            angle1 = (math.degrees(math.atan2(x,y))) # ラインとy軸のなす角度
            angle = angle0 + angle1 # 足裏内側の角度を考慮した
            # 角度がわかるように描く
            if all(( (ysaki+100)< y1 < 2000,  (ysaki+100)< y2 < 2000, x >-100, y > 100, length1 > 3000, angle>0)) : 
                count += 1
                # 小数点第２位までに限定
                angle11 = format(angle1,'.2f')
                angle_ = format(angle,'.2f')
                #print(count," : ", angle11, "...",angle_)
                angle_t = format(angle,'.1f')
                #円弧の作図用、angle, startAngle, endAngle用
                an = int(angle)
                an1 = int(angle1)
                # 半径
                if length1 <= 80000:
                    r = 70
                elif 160000>= length1 >80000:
                    r = 100
                elif 360000>= length1 > 160000:
                    r = 200
                elif 640000>= length1 > 360000:
                    r = 300
                elif 1000000>= length1 > 640000:
                    r = 450
                elif length1 > 1000000:
                    r = 600
                
                # テキスト
                angletext = str(angle_t) + "[deg]"
                #print("r = ",r)

                # 扇形用
                if x_ == 1: # 左
                    x3 = int(x1-((length1**0.5)*math.tan(-xt/yt)))
                    y3 = int(y1 - (length1**0.5))
                    # 書き込む（線：cv2.line、円弧：cv2.ellipse、文字：cv2.putText）
                    cv2.line(img_ju, pt1=(x1,y1),pt2=(x2,y2),color=(255, 255, 100),thickness=7)
                    cv2.ellipse(img_ju, (x1,y1), (r,r), an, -90-an1, -90-an1-an, (255, 255, 100), thickness=7, lineType=cv2.LINE_8, shift=0)
                    cv2.line(img_ju, pt1=(x1,y1),pt2=(x3,y3),color=(255, 255, 100),thickness=7)
                    cv2.putText(img_ju, angletext, (x2-500, y2+80), cv2.FONT_HERSHEY_PLAIN,6, (255, 255, 255),6, cv2.LINE_8)
                

                elif x_ == 2: # 右
                    x3 = int(x2+((length1**0.5)*math.tan(-xt/yt)))
                    y3 = int(y2 - (length1**0.5))

                    # 書き込む（線：cv2.line、円弧：cv2.ellipse、文字：cv2.putText）
                    cv2.line(img_ju, pt1=(x1,y1),pt2=(x2,y2),color=(255, 255, 100),thickness=7)
                    cv2.ellipse(img_ju, (x2,y2), (r,r), -an, -90+an1, -90+an1+an, (255, 255, 100), thickness=7, lineType=cv2.LINE_8, shift=0)
                    cv2.line(img_ju,pt1=(x2,y2),pt2=(x3,y3),color=(255, 255, 100),thickness=7)
                    cv2.putText(img_ju,angletext, (x1+20, y1+80), cv2.FONT_HERSHEY_PLAIN,6, (255, 255, 255),6, cv2.LINE_8)
                    
            wr = "/Users/BRLAB/others/img1/zzangle" + str(x_) + ".jpg"
            cv2.imwrite(wr, img_ju)


    x_ = 1
    wr1 = "/Users/BRLAB/others/img1/zzangle"+ str(x_) + ".jpg"
    i1 = cv2.imread(wr1)

    x_ = 2
    wr2 = "/Users/BRLAB/others/img1/zzangle"+ str(x_) + ".jpg"
    i2 = cv2.imread(wr2)
    
    img_jushin[95:3445,70:1250] = i1
    img_jushin[95:3445,1315:2495] = i2

    res = "/Users/BRLAB/others/img1/result_1angle_" + str(name) + ".jpg"
    cv2.imwrite(res, img_jushin)

    return img_jushin

'''
name = 133
pho = "133_1_2613_20211217"
date = "/Users/BRLAB/others/imgdate/" + str(pho) + ".jpg"
im = cv2.imread(date)

im_h = juangl(im,name,pho)
im_hs = cv2.cvtColor(im_h, cv2.COLOR_BGR2RGB)
plt.imshow(im_hs)
plt.show()
'''

#basename = os.path.basename(date)
#print(basename)
#base = basename.strip(".jpg")
#print(base)

