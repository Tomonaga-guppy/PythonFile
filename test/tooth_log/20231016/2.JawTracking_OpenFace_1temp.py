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

seal_template = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/seal_template/seal_2.png"

#depth_scale = mm/depth_data
depth_scale = 1.0000000474974513

def OpenFace(root_dir):
    pattern = os.path.join(root_dir, '*/RGB_image')
    RGB_dirs = glob.glob(pattern, recursive=True)
    # print('mp4files=',RGB_dirs)
    for i,RGB_dir in enumerate(RGB_dirs):
        print(f"{i+1}/{len(RGB_dirs)}  {RGB_dir}")
        dir_path = os.path.dirname(RGB_dir) + '/'
        video_out_path = dir_path + 'SealDetection.mp4'

        if not glob.glob(RGB_dir + "/*"):  #bagファイルの読み取りが出来ていない場合は飛ばす
            continue

        if os.path.isfile(video_out_path) == False:
            # OpenFaceで顔の推定
            command = 'C:/OpenFace_2.2.0_win_x64/FeatureExtraction.exe -2Dfp -3Dfp -tracked '
            inputpath = '-fdir ' + RGB_dir + ' '
            outpath = '-out_dir ' + dir_path
            subprocess.run (command + inputpath + outpath )
            # print(f"subprocess = {command + inputpath + outpath}")

        # # OpenFaceで顔の推定
        # command = 'C:/OpenFace_2.2.0_win_x64/FeatureExtraction.exe -2Dfp -3Dfp -tracked '
        # inputpath = '-fdir ' + RGB_dir + ' '
        # outpath = '-out_dir ' + dir_path
        # # print ('OpenFace =',command + inputpath + outpath)
        # subprocess.run (command + inputpath + outpath )

        # OpenFace_result 0 frame, 5-72 x_pixel, 73-140 y_pixel
        csv_file = dir_path + 'RGB_image.csv'
        if os.path.isfile(csv_file) == False:
            csv_file = dir_path + 'OpenFace.csv'

        OpenFace_result = []
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                OpenFace_result.append(row)

        #OpenFace実行しに作成されるファイル名を"RGB_image" → "OpenFace"に変更
        before_files = glob.glob(dir_path + "*RGB_image*.*")
        for before_file in before_files:
            after_file = before_file.replace("RGB_image", "OpenFace")
            if os.path.exists(after_file):
                os.remove(after_file)
            os.rename(before_file,after_file)


        # 動画を読み込む
        width, height = 1280, 720
        fps = 30
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter(video_out_path,fourcc, fps, (width, height))

        frame_count = 1
        seal=[[0,0]]
        List=[]

        while True:

            # img_depth (y,x) = (720, 1280)
            img_path = dir_path + "RGB_image/{:04d}.png".format(frame_count)
            img_depth_path = dir_path + "Depth_image/{:04d}.png".format(frame_count)

            if os.path.isfile(img_path) == False or os.path.isfile(img_depth_path)== False:
                break

            img = cv2.imread(img_path)
            img_depth = cv2.imread(img_depth_path, cv2.IMREAD_ANYDEPTH)

            # x,y,z = float(OpenFace_result[frame_count][141]),float(OpenFace_result[frame_count][209]),float(OpenFace_result[frame_count][277])
            # OpenFace座標格納(mm)
            landmark_List=[]
            try:
                for i in range(68):
                    # print(frame_count, i, i+141, i+209, i+277)
                    x,y = float(OpenFace_result[frame_count][i+141]),float(OpenFace_result[frame_count][i+209])
                    depthi = img_depth[int(float(OpenFace_result[frame_count][i+73])),int(float(OpenFace_result[frame_count][i+5]))] #整数型
                    z = depthi * depth_scale
                    landmark_List.append([i,x,y,z])
            except IndexError:
                break  #OpenFaceの解析したframe数と合わなくなったら終了

            # シールを見つける処理 参照点 48, 54のx, 8のy   シール検出の範囲指定のためintのまま処理
            mask1_x=int(float(OpenFace_result[frame_count][53]))    #48x
            mask1_y=int(float(OpenFace_result[frame_count][121]))    #48y
            mask2_x=int(float(OpenFace_result[frame_count][59]))    #54x
            mask2_y=int(float(OpenFace_result[frame_count][81]))    #8y

            seal_position, img, ratio, template_shape = SealDetection(height, width, img, mask1_x, mask2_x, mask1_y, mask2_y ,frame_count)

            cv2.circle(img, (seal_position[0],seal_position[1]), 1, (255, 120, 255), -1)
            seal_x = seal_position[0]
            seal_y = seal_position[1]
            seal.append([seal_x,seal_y])

            # シールの単位換算 (pixel → mm) 参照点 x:36-45 y:27-33 3D/2D
            # xpixel = template_shape[0]
            # ypixel = template_shape[1]
            # xpixel_scale = 20 / (xpixel*ratio/100)
            # ypixel_scale = 20 / (ypixel*ratio/100)
            # print(f"x2pixel_scale = {xpixel_scale}")
            # print(f"y2pixel_scale = {ypixel_scale}")
            # x = seal_x*xpixel_scale
            # y = seal_y*ypixel_scale

            xpixel_scale = (float(OpenFace_result[frame_count][186])-float(OpenFace_result[frame_count][177]))/(float(OpenFace_result[frame_count][50])-float(OpenFace_result[frame_count][41]))
            ypixel_scale = (float(OpenFace_result[frame_count][242])-float(OpenFace_result[frame_count][236]))/(float(OpenFace_result[frame_count][106])-float(OpenFace_result[frame_count][100]))
            # X68 = X33 + (x68 - x33)*scale
            x = float(OpenFace_result[frame_count][174]) + (seal_x - float(OpenFace_result[frame_count][38]))*xpixel_scale
            y = float(OpenFace_result[frame_count][242]) + (seal_y - float(OpenFace_result[frame_count][106]))*ypixel_scale

            # print(f"x_scale = {xpixel_scale} mm/pixel")
            # print(f"y_scale = {ypixel_scale} mm/pixel")

            # シールの奥行 depthから計算
            depth68 = img_depth[seal_y,seal_x]
            z = depth68 * depth_scale

            landmark_List.append([68,x,y,z])
            cv2.circle(img, (int(seal_x),int(seal_y)), 5, (255, 0, 255), -1)    #整数型
            List.append(landmark_List)

            # print(frame_count)
            frame_count +=1
            # writer.write(img)
            writer.write(img)

        writer.release()
        # print("Movie is saved in " + RGB_image)

        NumpyList = np.array(List)
        print(f"np.shape(NumpyList) = {np.shape(NumpyList)}")
        path = dir_path + "result.npy"
        np.save(path,NumpyList)
        #作製したnumpy配列は[フレーム数-1[landmark number[number, x, y, z]]]


# def SealDetection(height,width,img):
def SealDetection(height,width,img,mask1_x,mask2_x,mask1_y,mask2_y ,frame_count):
    np.value = []
    seal_position = (0, 0)

    # 矩形のマスク画像の生成
    mask = np.zeros((height,width,3), dtype = np.uint8)
    #矩形検出範囲の設定
    mask = cv2.rectangle(mask, (mask1_x,mask1_y), (mask2_x,mask2_y), (255, 255, 255), -1)
    #検出範囲をマスク
    mask_img = cv2.bitwise_and(img, mask)
    #ガウガシアンフィルタで平滑化
    blur_img = cv2.GaussianBlur(mask_img, (5,5), 0)
    template = cv2.imread(seal_template)
    template_shape = template.shape
    image_gray = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    global rot_angle, ratio
    global max_Val ; 0

    if frame_count == 1:  #テンプレートマッチングする時の最適な回転量、縮尺を算出
        #一番マッチする回転量を計算
        result_list = []
        for i in range(0, 180, 5):
            rot_temp = rotate_template(template, i)
            h, w = rot_temp.shape[0:2]
            gray_rot_temp = cv2.cvtColor(rot_temp, cv2.COLOR_RGB2GRAY)
            # template matching
            result = cv2.matchTemplate(image_gray, gray_rot_temp, cv2.TM_CCOEFF_NORMED)  #ZNCC
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
            result_list.append([i,maxVal,maxLoc,w,h])

        maxVal = -1
        for i, max_Val, max_Loc, w, h in result_list:
            if max_Val > maxVal:
                rot_angle = i
                maxVal = max_Val
                maxLoc_rot = max_Loc
                w_rot = w
                h_rot = h

        global rot_template_gray
        rot_template = rotate_template(template, rot_angle)
        rot_template_gray = cv2.cvtColor(rot_template, cv2.COLOR_BGR2GRAY)

        #一番マッチする倍率を計算
        result_list = []
        for i in range(50,131,1):
            arrenge_temp = cv2.resize(rot_template_gray.copy(), None, fx=i/100, fy=i/100, interpolation=cv2.INTER_CUBIC)
            h, w = arrenge_temp.shape
            result = cv2.matchTemplate(image_gray, arrenge_temp, cv2.TM_CCOEFF_NORMED)
            # 最も類似度が高い位置を取得する。
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
            result_list.append([i,maxVal,maxLoc,w,h])

        maxVal = -1
        for i, max_Val, max_Loc, w, h in result_list:
            if max_Val > maxVal:
                ratio = i
                maxVal = max_Val
                maxLoc = max_Loc
                w_result = w
                h_result = h

        img = cv2.rectangle(img.copy(), (mask1_x,mask1_y), (mask2_x,mask2_y), (255, 255, 255), 3)
        img = cv2.rectangle(img, maxLoc, (maxLoc[0] + w_result, maxLoc[1] + h_result), (255, 255, 255), 3)

    elif frame_count != 1:
        arrenge_temp = cv2.resize(rot_template_gray.copy(), None, fx = ratio/100, fy = ratio/100, interpolation=cv2.INTER_CUBIC)
        h_result, w_result = arrenge_temp.shape
        result = cv2.matchTemplate(image_gray, arrenge_temp, cv2.TM_CCOEFF_NORMED)
        # 最も類似度が高い位置を取得する。
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

        img = cv2.rectangle(img.copy(), (mask1_x,mask1_y), (mask2_x,mask2_y), (255, 255, 255), 3)
        img = cv2.rectangle(img, maxLoc, (maxLoc[0] + w_result, maxLoc[1] + h_result), (255, 255, 255), 3)

    print(f"frame = {frame_count}, maxval = {maxVal}, rot_angle = {rot_angle}, ratio = {ratio}")

    # 描画する。
    center_x = int(maxLoc[0]+w_result/2)   #整数型
    center_y = int(maxLoc[1]+h_result/2)
    radius = int((w_result+h_result)/4)

    seal_position = center_x, center_y
    # seal_position = int(maxLoc[0]), int(maxLoc[1]) #整数型

    return seal_position, img, ratio, template_shape

def rotate_template(temp, angle):
    height, width = temp.shape[0:2]
    #回転中心と倍率
    center = (int(width/2), int(height/2))  #整数型
    scale = 1.0
    #回転行列を求める
    trans = cv2.getRotationMatrix2D(center, angle, scale)
    #画像に対してアフィン変換を行う
    rot_image = cv2.warpAffine(temp, trans, (width, height))
    return rot_image

OpenFace(root_dir)
