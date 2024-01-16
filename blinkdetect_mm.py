import numpy as np
import os
import glob
import csv
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from matplotlib.backends.backend_pdf import PdfPages


root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/2023_12_20"
pdf = PdfPages(os.path.join(root_dir,'EAR.pdf'))

blink_video = False  #Trueでまばたき検出動画を作成

def BlinkDetection(OpenFace_result,frame):
    eye_landmark_list  = [range(36,42),range(42,48)]
    ear_rl = np.array([])
    for eye in range(2):  #右目と左目のEARを計算
        eye_position = []
        for point in eye_landmark_list[eye]:
            eye_position.append([float(OpenFace_result[frame][point+5]), float(OpenFace_result[frame][point+73])])
        eye_position = np.array(eye_position)
        ver1 =  np.linalg.norm(eye_position[1]-eye_position[5])
        ver2 = np.linalg.norm(eye_position[2]-eye_position[4])
        hor = np.linalg.norm(eye_position[0]-eye_position[3])
        ear = (ver1 + ver2) / (2.0 * hor)
        ear_rl = np.append(ear_rl,ear)
    return ear_rl  #右目と左目のEARを返す

pattern = os.path.join(root_dir, '*/RGB_image')  #RGB_imageがあるディレクトリを検索
RGB_dirs = glob.glob(pattern, recursive=True)

EAR_all_list = []

for i,RGB_dir in enumerate(RGB_dirs):
    id = os.path.basename(os.path.dirname(RGB_dir))
    print(f"{i+1}/{len(RGB_dirs)}  {RGB_dir}")
    dir_path = os.path.dirname(RGB_dir) + '/'
    video_out_path = dir_path + 'SealDetection.mp4'
    # OpenFace_result 0 frame, 5-72 x_pixel, 73-140 y_pixel
    csv_file = dir_path + 'RGB_image.csv'
    if os.path.isfile(csv_file) == False:
        csv_file = dir_path + 'OpenFace.csv'
    OpenFace_result = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            OpenFace_result.append(row)

    ear_list = []
    frame_count = len(OpenFace_result)
    for frame in range(1,frame_count):  #1行目はヘッダーなので除く
        ear_rl = BlinkDetection(OpenFace_result,frame)
        ear_list.append(ear_rl)
    input_df = pd.DataFrame(ear_list)
    df = input_df
    df.index = range(1, len(df) + 1)
    print(f"df = {df}")
    # print(f"df.index = {df.index}")


    # # signal.argrelminによるピーク（極小値）検出
    # peaks = signal.argrelmin(df[0].values, order=30)  #わりとあり．連続したまばたきに弱い
    # print(f"peaks = {peaks}")
    # ax1.scatter(peaks[0], df.iloc[peaks[0]], label="peak", color="r") #peak
    # blink_frame_list = peaks[0].tolist()

    #挑戦
    threshold_down = 0.03
    threshold_up = 0.03

    blink_list = []

    for frame in range(1,frame_count-5):
        # print(frame)
        if df[0][frame] - min(df[0][frame:frame+5]) > threshold_down:  #5フレーム後までの最小値との差が0.05以上 0.05は適当
        # if df[0][frame] > df[0][frame+1] and df[0][frame] - min(df[0][frame:frame+5]) > threshold_down:  #5フレーム後までの最小値との差が0.05以上 0.05は適当
            min_index = df[0][frame:frame+5].idxmin()
            print(f"kouho={min_index}")
            for add_frame in range(1,30):
                if frame+add_frame > frame_count-1:
                    break
                if min_index < frame+add_frame and df[0][frame+add_frame] - df[0][frame] < threshold_up:  #EARが回復したとみなす条件
                # if min_index < frame+add_frame and df[0][frame+add_frame] - df[0][min_index] > threshold_up:  #EARが回復したとみなす条件
                    blink_start_index = frame
                    blink_end_index = frame + add_frame
                    blink_list.extend(range(blink_start_index,blink_end_index+1))
                    # print(f"min_index = {min_index}")
                    # print(f"blink_start={blink_start_index},blink_end={blink_end_index}")
                    break


    # threshold_ear = 0.05
    # blink_list = []
    # try:
    #     for frame in range(1,frame_count-5):
    #         # print(frame)
    #         if df[0][frame] > df[0][frame+1] and df[0][frame] - min(df[0][frame:frame+5]) > threshold_ear:  #5フレーム後までの最小値との差が0.05以上 0.05は適当
    #             min_index = df[0][frame:frame+5].idxmin()
    #             # print("kouho")
    #             for add_frame in range(1,30):
    #                 if min_index < frame+add_frame and df[0][frame+add_frame] - df[0][min_index] > threshold_ear:  #EARが回復したとみなす条件
    #                     blink_start_index = frame
    #                     blink_end_index = frame + add_frame
    #                     blink_list.extend(range(blink_start_index,blink_end_index+1))
    #                     # print(f"min_index = {min_index}")
    #                     # print(f"blink_start={blink_start_index},blink_end={blink_end_index}")
    #                     break
    # except KeyError:
    #     print("KeyError")
    #     pass


    # threshold_ear_down = 0.06
    # threshold_ear_recov = 0.02
    # blink_list = []
    # try:
    #     for frame in range(1,frame_count-5):
    #         # print(frame)
    #         if df[0][frame] > df[0][frame+1] and df[0][frame] - min(df[0][frame:frame+5]) > threshold_ear_down:  #5フレーム後までの最小値との差が0.05以上 0.05は適当
    #             min_index = df[0][frame:frame+5].idxmin()
    #             # print("kouho")
    #             for add_frame in range(1,30):
    #                 if min_index < frame+add_frame and df[0][frame] - df[0][frame+add_frame] < threshold_ear_recov:  #EARが回復したとみなす条件
    #                     blink_start_index = frame
    #                     blink_end_index = frame + add_frame
    #                     blink_list.extend(range(blink_start_index,blink_end_index+1))
    #                     print(f"min_index = {min_index}")
    #                     print(f"blink_start={blink_start_index},blink_end={blink_end_index}")
    #                     break
    # except KeyError:
    #     print("KeyError")
    #     pass


    blink_list = sorted(list(set(blink_list))) #blink_listを重複削除して昇順にソート
    # # blink_list.sort()  #blink_listを昇順にソート
    print(f"blink_list = {blink_list}")
    # #blinklistから60引いたものをprint
    # print(f"blink_list-60 = {[i-60 for i in blink_list]}")
    # print(f"blink_list-60 = {len([i-60 for i in blink_list])}")

    # median = df[0].median()
    # #第一四分位数
    # q1 = df[0].quantile(.25)
    # #第三四分位数
    # q3 = df[0].quantile(.75)
    # print(f"q1,median,q3 = {q1},{median},{q3}")
    # EAR_all_list.append([(df.index).tolist(),(df[0].values).tolist()])
    # # print(EAR_all_list[i])
    # print(f"EAR_all.shape = {len(EAR_all_list[i][0][:])}")

    fig = plt.figure(tight_layout=True, figsize=(10, 10))
    # fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2,1,1)

    ax1.scatter([i for i in blink_list], [df[0][i] for i in blink_list], label="blink", color="r") #blink
    # ax1.scatter([i-60 for i in blink_list], [df[0][i] for i in blink_list], label="blink", color="r") #blink

    # earをプロット
    ax1.plot(df.index, df[0], label="value")
    # ax1.plot(df.index-60, df[0], label="value")
    ax1.set_xticks(np.arange(0, frame_count, 30))
    ax1.set_yticks(np.arange(0, 0.5, 0.1))
    ax1.grid()
    # ax1.set_title(f'EAR_right {id}')
    ax1.tick_params(labelsize=10)   #軸のフォントサイズを設定
    # ax1.set_xticklabels(ax1.get_xticks(), rotation=-45)  #軸を斜めにする
    ax1.set_xlabel('frame [-]', fontsize=10)  #軸ラベルを設定
    ax1.set_ylabel('right EAR [-]', fontsize=10)
    # ax1.fill_between(df.index, df.iqr_lower, df.iqr_upper, alpha=0.2) # Upper Lower
    # ax1.scatter(df.index, df.iqr_outlier, label="outlier", color="r") # Outlier
    ax1.legend()


    ax2 = fig.add_subplot(2,1,2)

    ax2.scatter([i for i in blink_list], [df[1][i] for i in blink_list], label="blink", color="r") #blink
    # ax1.scatter([i-60 for i in blink_list], [df[0][i] for i in blink_list], label="blink", color="r") #blink

    # earをプロット
    ax2.plot(df.index, df[1], label="value")
    # ax1.plot(df.index-60, df[0], label="value")

    # ax2.grid(which = "major", axis = "x", color = "gray", alpha = 0.5, linestyle = "-", linewidth = 1)
    # ax2.grid(which = "major", axis = "y", color = "gray", alpha = 0.5, linestyle = "-", linewidth = 1)
    # ax2.grid(which = "minor", axis = "x", color = "gray", alpha = 0.1, linestyle = "-", linewidth = 1)
    # ax2.grid(which = "minor", axis = "y", color = "gray", alpha = 0.1, linestyle = "-", linewidth = 1)

    ax2.set_xticks(np.arange(0, frame_count, 30))
    ax2.set_yticks(np.arange(0, 0.5, 0.1))
    #薄いグリッド線を30フレームごとに表示
    # ax2.grid(which='major',color='lightgray',linestyle='-')
    # ax2.grid(which='minor',color='lightgray',linestyle='-')
    ax2.grid()
    # ax2.set_title(f'EAR_left {id}')
    ax2.tick_params(labelsize=10)   #軸のフォントサイズを設定
    # ax1.set_xticklabels(ax1.get_xticks(), rotation=-45)  #軸を斜めにする
    ax2.set_xlabel('frame [-]', fontsize=10)  #軸ラベルを設定
    ax2.set_ylabel('left EAR [-]', fontsize=10)
    # ax1.fill_between(df.index, df.iqr_lower, df.iqr_upper, alpha=0.2) # Upper Lower
    # ax1.scatter(df.index, df.iqr_outlier, label="outlier", color="r") # Outlier
    ax2.legend()

    # fig.text(0.5, 0.04, 'frame [-]', ha='center', va='center', fontsize=20)
    # fig.text(0.04, 0.5, 'EAR [-]', ha='center', va='center',rotation='vertical', fontsize=20)


    plt.tight_layout()
    save_path = dir_path + 'EAR.png'
    plt.savefig(save_path)
    # plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    pdf.savefig(fig)
    plt.close(fig)

    if blink_video:
        mp4_path = glob.glob(dir_path+'*original.mp4')[0]
        video_out_path = dir_path + 'blinkdetection.mp4'
        #まばたきがあるフレームは"Blink"とテキストで表示
        cap = cv2.VideoCapture(mp4_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video_out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
        frame = 0
        while(cap.isOpened()):
            ret, frame_img = cap.read()
            if ret == True:
                if frame in blink_list:
                    cv2.putText(frame_img, 'Blink', (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                video_out.write(frame_img)
                frame += 1
            else:
                break
        cap.release()
        video_out.release()
        cv2.destroyAllWindows()
        print(f"video_out_path = {video_out_path}")


# print(f"EAR_all_list = {EAR_all_list}")


pdf.close()
print("pdfsaved")
