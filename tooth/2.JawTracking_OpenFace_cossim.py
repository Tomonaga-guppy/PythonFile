import os
import glob
import math
import subprocess
import csv
import cv2
import numpy as np
import sys
import random
import pyrealsense2 as rs
import pandas as pd
import matplotlib.pyplot as plt

# root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_12_demo"
root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_12_20"

seal_temp = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/seal_template/seal.png"

#depth_scale = mm/depth_data
depth_scale = 1.0000000474974513

ply = False  #plyファイルを作成するかどうか(Trueは作成する)

def OpenFace(root_dir):
    pattern = os.path.join(root_dir, '*/RGB_image')  #RGB_imageがあるディレクトリを検索
    RGB_dirs = glob.glob(pattern, recursive=True)
    for i,RGB_dir in enumerate(RGB_dirs):
        print(f"{i+1}/{len(RGB_dirs)}  {RGB_dir}")
        dir_path = os.path.dirname(RGB_dir) + '/'
        video_out_path = dir_path + 'SealDetection.mp4'

        #mkgの結果を取得
        id = os.path.basename(os.path.dirname(dir_path))
        print(f"ID {id}")

        seal_template = seal_temp
        if id == "20231218_d":
            seal_template = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/seal_template/seal_2023_1218_d.png"
            print(f"seal_template = {seal_template}")

        #root_dirの2つ前のディレクトリパスを取得
        bagfile = os.path.dirname(os.path.dirname(dir_path)) + '/'+ id + '.bag'
        print(f"bagfile = {bagfile}")

        if not glob.glob(RGB_dir + "/*"):  #bagファイルの読み取りが出来ていない場合は飛ばす
            continue

        if os.path.isfile(video_out_path) == False:
        # if True:
            # OpenFaceで顔の推定
            command = 'C:/OpenFace_2.2.0_win_x64/FeatureExtraction.exe -2Dfp -tracked '
            inputpath = '-fdir ' + RGB_dir + ' '
            outpath = '-out_dir ' + dir_path
            subprocess.run (command + inputpath + outpath)
            # print(f"subprocess = {command + inputpath + outpath}")

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

        List=[]

        ply_list_all = []
        ply_list2_all = []

        right_eye_bec = pd.DataFrame()
        left_eye_bec = pd.DataFrame()

        save_frame_count = []


        error_bagfiles = []  #読み取れなかったbagファイル記録用
        config = rs.config()
        config.enable_device_from_file(bagfile)

        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(config)
        except RuntimeError:
            error_bagfiles.append(bagfile)
            continue

        # create Align Object
        align_to = rs.stream.color
        align = rs.align(align_to)

        device = profile.get_device()
        playback = device.as_playback()
        playback.set_real_time(False)  #リアルタイム再生をオフにするとフレームごとに処理が終わるまで待機してくれる

        # hole_filling_filterのパラメータ
        hole_filling = rs.hole_filling_filter(2)

        try:
            pre_time = 0
            frame_count = 1
            eye_rfvec_list = []
            cossim_list = []
            while True:
                frames = pipeline.wait_for_frames()

                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

                depth_frame = aligned_frames.get_depth_frame()
                filter_frame = hole_filling.process(depth_frame)
                result_frame = filter_frame.as_depth_frame()

                cur_time = playback.get_position()  #再生時間の取得 単位はナノ秒

                if cur_time < pre_time:  #前フレームより再生時間が進んでいなかったら終了
                    break

                #内部パラメータの取得（色情報）
                color_intr = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()

                img = color_image.copy()
                imgcopy = img.copy()

                try:
                    # シールを見つける処理 参照点 48, 54のx, 8のy   シール検出の範囲指定のためintのまま処理
                    mask1_x=int(float(OpenFace_result[frame_count][53]))    #48x
                    mask1_y=int(float(OpenFace_result[frame_count][121]))    #48y
                    mask2_x=int(float(OpenFace_result[frame_count][59]))    #54x
                    mask2_y=int(float(OpenFace_result[frame_count][81]))    #8y
                except IndexError:
                    break  #OpenFaceの解析したframe数と合わなくなったら終了


                seal_position, imgcopy, ratio, template_shape = SealDetection(height, width, imgcopy, mask1_x, mask2_x, mask1_y, mask2_y ,frame_count,seal_template)

                cv2.circle(imgcopy, (seal_position[0],seal_position[1]), 1, (255, 120, 255), -1)
                seal_x_pixel = seal_position[0]
                seal_y_pixel = seal_position[1]
                seal_z = result_frame.get_distance(seal_x_pixel, seal_y_pixel)
                seal_position = np.array(rs.rs2_deproject_pixel_to_point(color_intr , [seal_x_pixel,seal_y_pixel], seal_z))*1000  #pixel座標を(m→)mm座標に変換
                seal_x, seal_y, seal_z = seal_position[0], seal_position[1], seal_position[2]*depth_scale

                # OpenFace座標格納(mm)
                landmark_List=[]
                for i in range(68):
                    xpix = int(float(OpenFace_result[frame_count][i+5]))
                    ypix = int(float(OpenFace_result[frame_count][i+73]))
                    z = result_frame.get_distance(xpix, ypix)
                    point_pos = np.array(rs.rs2_deproject_pixel_to_point(color_intr , [xpix,ypix], z))*1000
                    x,y,z = point_pos[0], point_pos[1], point_pos[2]*depth_scale
                    landmark_List.append([i,x,y,z])

                #30→36,30→45のコサイン類似度を計算
                vec_30 = np.array(landmark_List[30][:])  #[i,x,y,z]の順で格納
                vec_36 = np.array(landmark_List[36][:])
                vec_45 = np.array(landmark_List[45][:])
                vec_r = (vec_36 - vec_30)[1:]  #iはいらないので消す
                vec_l = (vec_45 - vec_30)[1:]
                eye_rfvec_list.append([vec_r,vec_l])
                # print(f"vec_r = {vec_r}")
                # print(f"eye_rfvec_list = {np.shape(eye_rfvec_list)}")  #(frame,2,4)

                # if frame_count > 1:
                #     refvec = np.array(eye_rfvec_list[0][:][:])  #1フレーム目の値を参照ベクトルとして類似度を算出
                #     # print(f"refvec = {np.shape(refvec)}")  #(2,3)
                #     cossim_r = np.dot(refvec[0][:], eye_rfvec_list[frame_count-1][0][:3])/(np.linalg.norm(refvec[0][:])*np.linalg.norm(eye_rfvec_list[frame_count-1][0][:3]))
                #     cossim_l = np.dot(refvec[1][:], eye_rfvec_list[frame_count-1][1][:3])/(np.linalg.norm(refvec[1][:])*np.linalg.norm(eye_rfvec_list[frame_count-1][1][:3]))
                #     # print(f"cossim_r = {cossim_r}")
                #     # print(f"cossim_l = {cossim_l}")
                #     cossim_list.append((cossim_r,cossim_l))


                if ply:
                    # if frame_count==60 or frame_count == 308:  #a用
                    # if frame_count==150 or frame_count == 514:  #b用
                    # if frame_count == 150 or frame_count == 287:  #c用
                    # if frame_count == 120 or frame_count == 367:  #d用
                    # if frame_count == 120 or frame_count == 410:  #e用
                    if frame_count==150 or frame_count == 456:  #f用
                        save_frame_count.append(frame_count)
                        xpix_max = int(max([float(OpenFace_result[frame_count][i+5]) for i in range(68)]))
                        xpix_min = int(min([float(OpenFace_result[frame_count][i+5]) for i in range(68)]))
                        ypix_max = int(max([float(OpenFace_result[frame_count][i+73]) for i in range(68)]))
                        ypix_min = int(min([float(OpenFace_result[frame_count][i+73]) for i in range(68)]))

                        # print([OpenFace_result[frame_count][i+73] for i in range(68)])
                        # print(max([OpenFace_result[frame_count][i+73] for i in range(68)]))

                        point_num = 100000
                        ply_list  = []
                        try:
                            for i in range(point_num):
                                xpix = random.randint(xpix_min, xpix_max)
                                ypix = random.randint(ypix_min, ypix_max)
                                # xpix = random.randint(xpix_min-100, xpix_max+100)
                                # ypix = random.randint(ypix_min-50, ypix_max+50)
                                z = result_frame.get_distance(xpix, ypix)
                                point_pos = np.array(rs.rs2_deproject_pixel_to_point(color_intr , [xpix,ypix], z))*1000
                                x,y,z = point_pos[0], point_pos[1], point_pos[2]*depth_scale
                                color = (img[ypix,xpix])
                                color = color[::-1]
                                ply_list.append([x,y,z,color[0],color[1],color[2]])
                        except IndexError:
                            break  #OpenFaceの解析したframe数と合わなくなったら終了
                        ply_list_all.append(ply_list)

                        ply_list2 = []
                        try:
                            for i in range(68):
                                xpix = int(float(OpenFace_result[frame_count][i+5]))
                                ypix = int(float(OpenFace_result[frame_count][i+73]))

                                z = result_frame.get_distance(xpix, ypix)
                                point_pos = np.array(rs.rs2_deproject_pixel_to_point(color_intr , [xpix,ypix], z))*1000
                                x,y,z = point_pos[0], point_pos[1], point_pos[2]*depth_scale

                                ply_list2.append([x,y,z])
                            ply_list2.append([seal_x,seal_y,seal_z])  #シールの座標を追加
                        except IndexError:
                            break  #OpenFaceの解析したframe数と合わなくなったら終了
                        ply_list2_all.append(ply_list2)

                landmark_List.append([68,seal_x,seal_y,seal_z])
                cv2.circle(imgcopy, (int(seal_x_pixel),int(seal_y_pixel)), 5, (255, 0, 255), -1)    #整数型
                List.append(landmark_List)

                #imgcopyにフレーム数を書き込む
                cv2.putText(imgcopy, str(frame_count), (100, 100), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 2, cv2.LINE_AA)

                frame_count +=1
                # writer.write(imgcopy)

        finally:
            # writer.release()
            NumpyList = np.array(List)
            path = dir_path + "landmark.npy"
            # np.save(path,NumpyList)
            #作製したnumpy配列は[フレーム数-1[landmark number[number, x, y, z]]]
            pipeline.stop()


            #refvec
            refvec = np.median(eye_rfvec_list, axis=0)  #各フレームのベクトルの中央値を参照ベクトルとする
            for frame_count in range(1,frame_count-1):
                # refvec = np.array(eye_rfvec_list[0][:][:])  #1フレーム目の値を参照ベクトルとして類似度を算出
                # print(f"refvec = {np.shape(refvec)}")  #(2,3)
                cossim_r = np.dot(refvec[0][:], eye_rfvec_list[frame_count-1][0][:3])/(np.linalg.norm(refvec[0][:])*np.linalg.norm(eye_rfvec_list[frame_count-1][0][:3]))
                cossim_l = np.dot(refvec[1][:], eye_rfvec_list[frame_count-1][1][:3])/(np.linalg.norm(refvec[1][:])*np.linalg.norm(eye_rfvec_list[frame_count-1][1][:3]))
                print(f"cossim_r = {cossim_r}")
                print(f"cossim_l = {cossim_l}")
                cossim_list.append((cossim_r,cossim_l))


            cossim_list = np.array(cossim_list)
            print(f"cossim_list.shape = {cossim_list.shape}")
            #cossim_listからプロット
            fig = plt.figure()
            ax1 = fig.add_subplot(2,1,1)
            ax2 = fig.add_subplot(2,1,2)
            ax1.plot(range(1,frame_count+1), cossim_list[:,0], label="value")
            ax2.plot(range(1,frame_count+1), cossim_list[:,1], label="value")
            ax1.set_xticks(np.arange(0, frame_count, 30))
            ax2.set_xticks(np.arange(0, frame_count, 30))
            ax1.set_yticks(np.arange(0.99, 1))
            ax2.set_yticks(np.arange(0.99, 1))
            ax1.grid()
            ax2.grid()
            ax1.set_title(f'cossim_r {id}')
            ax2.set_title(f'cossim_l {id}')
            ax1.tick_params(labelsize=10)
            ax2.tick_params(labelsize=10)
            ax1.set_xlabel('frame [-]', fontsize=10)
            ax2.set_xlabel('frame [-]', fontsize=10)
            ax1.set_ylabel('cossim [-]', fontsize=10)
            ax2.set_ylabel('cossim [-]', fontsize=10)
            ax1.legend()
            ax2.legend()
            save_path = dir_path + 'cossim.png'
            plt.savefig(save_path, bbox_inches='tight')
            plt.show()


            if ply:
                # print(f"ply_list2_all.shape = {np.array(ply_list2_all).shape}")
                ply_list_all = np.array(ply_list_all)
                ply_list2_all = np.array(ply_list2_all)

                ply_path = dir_path + "plycam"
                if not os.path.exists(ply_path):
                    os.mkdir(ply_path)

                for i in range(ply_list_all.shape[0]):
                    frame_count = save_frame_count[i]
                    # PLYファイルのヘッダを書き込む
                    header = f"ply\nformat ascii 1.0\nelement vertex {point_num}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                    header_face = f"ply\nformat ascii 1.0\nelement vertex 69\nproperty float x\nproperty float y\nproperty float z\nend_header\n"

                    # PLYファイルに書き込む
                    with open(dir_path + f"plycam/random_cloud{frame_count}.ply", "w") as ply_file:
                        ply_file.write(header)
                        for vertex in ply_list_all[i,:,:]:
                            vertex = list(map(str, vertex[:3])) + list(map(str, map(int, vertex[3:])))
                            ply_file.write(" ".join(vertex) + "\n")

                    # PLYファイルに書き込む
                    with open(dir_path + f"plycam/face_cloud{frame_count}.ply", "w") as ply_file:
                        ply_file.write(header_face)
                        for vertex in ply_list2_all[i,:,:]:
                            ply_file.write(" ".join(map(str, vertex)) + "\n")


            # ear_df = pd.DataFrame(ear_list)
            # ear_df.index = range(1, len(ear_df) + 1)

            # threshold_ear = 0.05
            # blink_list = []
            # try:
            #     for frame in range(1,frame_count-5):
            #         # print(frame)
            #         if ear_df[0][frame] > ear_df[0][frame+1] and ear_df[0][frame] - min(ear_df[0][frame:frame+5]) > threshold_ear:  #5フレーム後までの最小値との差が0.05以上 0.05は適当
            #             min_index = ear_df[0][frame:frame+5].idxmin()
            #             # print("kouho")
            #             for add_frame in range(1,30):
            #                 if min_index < frame+add_frame and ear_df[0][frame+add_frame] - ear_df[0][min_index] > threshold_ear:  #EARが回復したとみなす条件
            #                     blink_start_index = frame
            #                     blink_end_index = frame + add_frame
            #                     blink_list.extend(range(blink_start_index,blink_end_index+1))
            #                     # print(f"min_index = {min_index}")
            #                     # print(f"blink_start={blink_start_index},blink_end={blink_end_index}")
            #                     break
            #             else: pass
            # except KeyError:
            #     pass

            # blink_list = sorted(list(set(blink_list))) #blink_listを重複削除して昇順にソート
            # #blink_listをnpyファイルに保存
            # print(f"blink_frame_list = {blink_list}")
            # print(f"len(blink_frame_list) = {len(blink_list)}")
            # np.save(dir_path + "blink_list.npy",blink_list)

            # # earをプロット
            # fig, ax = plt.subplots(figsize=(10, 5))
            # ax.scatter([i for i in blink_list], [ear_df[0][i] for i in blink_list], label="blink", color="r")
            # ax.plot(ear_df.index, ear_df[0], label="value")
            # ax.set_xticks(np.arange(0, frame_count, 30))
            # ax.set_yticks(np.arange(0.2, 0.75, 0.1))
            # ax.grid()
            # ax.set_title(f'EAR {id}')
            # ax.tick_params(labelsize=10)   #軸のフォントサイズを設定
            # # ax.set_xticklabels(ax.get_xticks(), rotation=-45)  #軸を斜めにする
            # ax.set_xlabel('frame [-]', fontsize=10)  #軸ラベルを設定
            # ax.set_ylabel('EAR [-]', fontsize=10)
            # ax.legend()
            # save_path = dir_path + 'EAR_mm.png'
            # plt.savefig(save_path, bbox_inches='tight')
            # plt.show()
            # plt.close(fig)



    if len(error_bagfiles) != 0:
        print(f"以下のbagファイルが読み取れませんでした\n{error_bagfiles}")

# def SealDetection(height,width,img):
def SealDetection(height,width,imgcopy,mask1_x,mask2_x,mask1_y,mask2_y ,frame_count,seal_template):
    np.value = []
    seal_position = (0, 0)
    seal_template = seal_template

    # 矩形のマスク画像の生成
    mask = np.zeros((height,width,3), dtype = np.uint8)
    #矩形検出範囲の設定
    mask = cv2.rectangle(mask, (mask1_x,mask1_y), (mask2_x,mask2_y), (255, 255, 255), -1)
    #検出範囲をマスク
    mask_img = cv2.bitwise_and(imgcopy, mask)
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
        for i in range(0, 180+1, 5):
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

        # imgcopy = cv2.rectangle(imgcopy, (mask1_x,mask1_y), (mask2_x,mask2_y), (255, 255, 255), 3)
        imgcopy = cv2.rectangle(imgcopy, maxLoc, (maxLoc[0] + w_result, maxLoc[1] + h_result), (255, 255, 255), 3)

    elif frame_count != 1:
        arrenge_temp = cv2.resize(rot_template_gray.copy(), None, fx = ratio/100, fy = ratio/100, interpolation=cv2.INTER_CUBIC)
        h_result, w_result = arrenge_temp.shape
        result = cv2.matchTemplate(image_gray, arrenge_temp, cv2.TM_CCOEFF_NORMED)
        # 最も類似度が高い位置を取得する。
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

        # imgcopy = cv2.rectangle(imgcopy, (mask1_x,mask1_y), (mask2_x,mask2_y), (255, 255, 255), 3)
        imgcopy = cv2.rectangle(imgcopy, maxLoc, (maxLoc[0] + w_result, maxLoc[1] + h_result), (255, 255, 255), 3)

    print(f"frame = {frame_count}, maxval = {maxVal}, rot_angle = {rot_angle}, ratio = {ratio}")

    # 描画する。
    center_x = int(maxLoc[0]+w_result/2)   #整数型
    center_y = int(maxLoc[1]+h_result/2)
    radius = int((w_result+h_result)/4)

    seal_position = center_x, center_y
    # seal_position = int(maxLoc[0]), int(maxLoc[1]) #整数型

    return seal_position, imgcopy, ratio, template_shape

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
