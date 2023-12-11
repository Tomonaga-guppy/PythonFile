import numpy as np
import os
import glob
import csv

root_dir = "C:/Users/zutom/BRLAB/tooth/Temporomandibular_movement/movie/scale"

# def calculate_ear(eye_pixel):
#     ver1 = np.linalg.norm(eye_pixel[1]-eye_pixel[5])
#     ver2 = np.linalg.norm(eye_pixel[2]-eye_pixel[4])
#     hor = np.linalg.norm(eye_pixel[0]-eye_pixel[3])
#     return (ver1 + ver2) / (2.0 * hor)

# def get_eye_pixel(points, frame, OpenFace_result):
#     return np.array([[float(OpenFace_result[frame][point+5]), float(OpenFace_result[frame][point+73])] for point in points])

# def BlinkDetection(OpenFace_result, frame):
#     right_eye_pixel = get_eye_pixel(range(36,42), frame, OpenFace_result)
#     left_eye_pixel = get_eye_pixel(range(42,48), frame, OpenFace_result)

#     ear_right = calculate_ear(right_eye_pixel)
#     ear_left = calculate_ear(left_eye_pixel)

#     return ear_right + ear_left


def BlinkDetection(OpenFace_result,frame):
    eye_landmark_list  = [range(36,42),range(42,48)]
    ear_sum = 0
    for eye in range(2):  #右目と左目のEARを計算
        eye_pixel = []
        for point in eye_landmark_list[eye]:
            eye_pixel.append([float(OpenFace_result[frame][point+5]), float(OpenFace_result[frame][point+73])])
        eye_pixel = np.array(eye_pixel)
        ver1 =  np.linalg.norm(eye_pixel[1]-eye_pixel[5])
        ver2 = np.linalg.norm(eye_pixel[2]-eye_pixel[4])
        hor = np.linalg.norm(eye_pixel[0]-eye_pixel[3])
        ear_sum += (ver1 + ver2) / (2.0 * hor)
    return ear_sum  #右目と左目のEARの合計を返す

pattern = os.path.join(root_dir, '*B2*/RGB_image')  #RGB_imageがあるディレクトリを検索
RGB_dirs = glob.glob(pattern, recursive=True)
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
    #閾値より小さければまばたきと判定
    for frame in range(1,frame_count): #まばたきをしているフレームをリストに追加
        ear = BlinkDetection(OpenFace_result,frame)
        ear_list.append(ear)



    #ear_listをdataframeに変換して箱髭図を作成
    import pandas as pd
    import matplotlib.pyplot as plt

    input_df = pd.DataFrame(ear_list)
    df = input_df.copy()
    span = 12
    threshold = 3
    q1 = df.rolling(min_periods=1, window=span).quantile(0.25)
    q3 = df.rolling(min_periods=1, window=span).quantile(0.75)
    iqr = q3 - q1
    iqr_lower = q1 -(iqr * threshold)
    iqr_upper = q3 +(iqr * threshold)
    df["iqr"] = iqr
    df["iqr_lower"] = iqr_lower
    df["iqr_upper"] = iqr_upper
    df["iqr_outlier"] = df[0][(df[0] < df.iqr_lower) | (df[0] > df.iqr_upper)]

    # plt.plot(df.index, df[0], label="value") # Value
    # plt.xticks(np.arange(0, frame_count, 30))
    # plt.yticks(np.arange(0.2, 0.7, 0.1))
    # plt.grid()
    # #軸のフォントサイズを設定
    # plt.tick_params(labelsize=10)
    # #軸を斜めにする
    # plt.xticks(rotation=-45)
    # #軸ラベルを設定ｆ
    # plt.xlabel('frame [-]', fontsize=10)
    # plt.ylabel('EAR [-]', fontsize=10)
    # plt.fill_between(df.index, df.iqr_lower, df.iqr_upper, alpha=0.2) # Upper Lower
    # plt.scatter(df.index, df.iqr_outlier, label="outlier", color="r") # Outlier
    # plt.legend()
    # plt.show()
    # plt.close()


    def Hampel(x, k, thr=3):
        arraySize = len(x)
        idx = np.arange(arraySize)
        output_x = x.copy()
        output_Idx = np.zeros_like(x)

        for i in range(arraySize):
            mask1 = np.where( idx >= (idx[i] - k) ,True, False)
            mask2 = np.where( idx <= (idx[i] + k) ,True, False)
            kernel = np.logical_and(mask1, mask2)
            median = np.median(x[kernel])
            std = 1.4826 * np.median(np.abs(x[kernel] - median))

            if np.abs(x[i] - median) > thr * std:
                output_Idx[i] = 1
                # output_x[i] = median

        # return output_x, output_Idx.astype(bool)
        return output_x, output_Idx

    result = Hampel(input_df[0], 12, 3)
    #result[1]がtureのindexを取得
    blink_frame_list = []
    for i,flag in enumerate(result[1]):
        if flag == 1:
            blink_frame_list.append(i)
    print(f"blink_frame_list = {blink_frame_list}")

    df2 = df[result[1].astype(bool)]
    print(df2)
    # plt.scatter(df2.index, df2[0], marker="o", label = 'Filtered signal', color = "red")

    plt.plot(df2.index, df2[0], marker="o", label = 'Error signal', color = "red")
    plt.plot(df.index, df[0], marker="o", label = 'Input signal')
    # plt.plot(df.index, result[0], linewidth = 2.0,label = 'Filtered signal')
    #result[1]がfalseのときのdfのindexをデータフレームで取得
    plt.title('Hampel Filter')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    plt.grid()
    plt.show()



    # mp4_path = glob.glob(dir_path+'*original.mp4')[0]
    # video_out_path = dir_path + 'blinkdetection.mp4'
    # #まばたきがあるフレームは"Blink"とテキストで表示
    # import cv2
    # cap = cv2.VideoCapture(mp4_path)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # video_out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
    # frame = 0
    # while(cap.isOpened()):
    #     ret, frame_img = cap.read()
    #     if ret == True:
    #         if frame in blink_frame_list:
    #             cv2.putText(frame_img, 'Blink', (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    #         video_out.write(frame_img)
    #         frame += 1
    #     else:
    #         break
    # cap.release()
    # video_out.release()
    # cv2.destroyAllWindows()
    # print(f"video_out_path = {video_out_path}")