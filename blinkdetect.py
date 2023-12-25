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

pattern = os.path.join(root_dir, '*/RGB_image')  #RGB_imageがあるディレクトリを検索
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
    for frame in range(1,frame_count):
        ear = BlinkDetection(OpenFace_result,frame)
        ear_list.append(ear)
    input_df = pd.DataFrame(ear_list)
    df = input_df.copy()


    # # 四分位範囲による外れ値検出 https://data-analysis-stats.jp/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92/%E6%99%82%E7%B3%BB%E5%88%97%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E5%A4%96%E3%82%8C%E5%80%A4%E3%81%AE%E7%B5%B1%E8%A8%88%E7%9A%84%E3%81%AA%E6%B1%82%E3%82%81%E6%96%B9%EF%BC%881%E6%AC%A1%E5%85%83%EF%BC%89/
    # span = 11
    # threshold = 3
    # q1 = df.rolling(min_periods=1, window=span).quantile(0.25)
    # q3 = df.rolling(min_periods=1, window=span).quantile(0.75)
    # iqr = q3 - q1
    # iqr_lower = q1 -(iqr * threshold)
    # iqr_upper = q3 +(iqr * threshold)
    # df["iqr"] = iqr
    # df["iqr_lower"] = iqr_lower
    # df["iqr_upper"] = iqr_upper
    # df["iqr_outlier"] = df[0][(df[0] < df.iqr_lower) | (df[0] > df.iqr_upper)]

    # #df["iqr_outlier"]がNaNでない行を抽出
    # df_out = df.dropna(subset=["iqr_outlier"])
    # #df["iqr_outlier"]がNaNでない行のindexを抽出
    # blink_frame_list = df_out.index.values.tolist()
    # # print(blink_frame_list)
    # #blink_frame_listをnpyファイルとして保存
    # # np.save(dir_path + 'blink_frame_list.npy', blink_frame_list)

    # # earをプロット
    # plt.plot(df.index, df[0], label="value")
    # plt.xticks(np.arange(0, frame_count, 30))
    # plt.yticks(np.arange(0.2, 0.7, 0.1))
    # plt.grid()
    # plt.tick_params(labelsize=10)   #軸のフォントサイズを設定
    # plt.xticks(rotation=-45)  #軸を斜めにする
    # plt.xlabel('frame [-]', fontsize=10)  #軸ラベルを設定
    # plt.ylabel('EAR [-]', fontsize=10)
    # # plt.fill_between(df.index, df.iqr_lower, df.iqr_upper, alpha=0.2) # Upper Lower
    # plt.scatter(df.index, df.iqr_outlier, label="outlier", color="r") # Outlier
    # plt.legend()
    # save_path = dir_path + 'EAR.png'
    # plt.savefig(save_path, bbox_inches='tight')
    # plt.show()
    # plt.close()




    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    # signal.argrelminによるピーク（極小値）検出
    peaks = signal.argrelmin(df[0].values, order=30)  #わりとあり．連続したまばたきに弱い
    print(f"peaks = {peaks}")
    ax.scatter(peaks[0], df.iloc[peaks[0]], label="peak", color="r") #peak
    blink_frame_list = peaks[0].tolist()

    # earをプロット
    ax.plot(df.index, df[0], label="value")
    ax.set_xticks(np.arange(0, frame_count, 30))
    ax.set_yticks(np.arange(0.2, 0.7, 0.1))
    ax.grid()
    ax.set_title(f'EAR {id}')
    ax.tick_params(labelsize=10)   #軸のフォントサイズを設定
    ax.set_xticklabels(ax.get_xticks(), rotation=-45)  #軸を斜めにする
    ax.set_xlabel('frame [-]', fontsize=10)  #軸ラベルを設定
    ax.set_ylabel('EAR [-]', fontsize=10)
    # ax.fill_between(df.index, df.iqr_lower, df.iqr_upper, alpha=0.2) # Upper Lower
    # ax.scatter(df.index, df.iqr_outlier, label="outlier", color="r") # Outlier
    ax.legend()
    save_path = dir_path + 'EAR.png'
    plt.savefig(save_path, bbox_inches='tight')
    pdf.savefig(fig)
    # plt.show()
    plt.close(fig)



    # import scipy.signal as signal
    # from matplotlib.backends.backend_pdf import PdfPages

    # # 平滑化されたデータ用の空のDataFrameを作成
    # df_sg = pd.DataFrame(index=df.index)
    # # 各列データを平滑化して、結果をdf_sgに格納
    # for col in df.columns:
    #     df_sg[col] = signal.savgol_filter(df[col], window_length=11, polyorder=3)

    # # 平滑化前後の差分
    # delta_df = abs(df - df_sg)
    # # Hampel Identifierの適用
    # delta_diff = abs(delta_df - delta_df.median())
    # mad = 1.4826 * delta_diff.median()
    # cl = delta_df.median() + 3 * mad
    # delta_new = delta_df[delta_diff < 3 * mad]
    # delta_outliers = delta_df[delta_diff > 3 * mad]
    # # 新しいデータフレームを作成して平滑化データと外れ値を格納
    # df_sg_hampel = pd.DataFrame(index=delta_new.index)
    # df_outliers_sg_hampel = pd.DataFrame(index=delta_outliers.index)
    # for col in delta_new.columns:
    #     df_index = delta_new[col].dropna().index
    #     outlier_index = delta_outliers[col].dropna().index
    #     df_sg_hampel[col] = df[col].loc[df_index]
    #     df_outliers_sg_hampel[col] = df[col].loc[outlier_index]

    # #hampelフィルターによる外れ値検出
    # def Hampel(x, k, thr=3):
    #     arraySize = len(x)
    #     idx = np.arange(arraySize)
    #     output_x = x.copy()
    #     output_Idx = np.zeros_like(x)

    #     for i in range(arraySize):
    #         mask1 = np.where( idx >= (idx[i] - k) ,True, False)
    #         mask2 = np.where( idx <= (idx[i] + k) ,True, False)
    #         kernel = np.logical_and(mask1, mask2)
    #         median = np.median(x[kernel])
    #         std = 1.4826 * np.median(np.abs(x[kernel] - median))
    #         if np.abs(x[i] - median) > thr * std:
    #             output_Idx[i] = 1
    #             output_x[i] = median
    # # return output_x, output_Idx.astype(bool)
    #     return output_x, output_Idx

    # # result = Hampel(df[0], k=2, thr=3)
    # output_x, output_Idx = Hampel(df[0], k=2, thr=3)
    # print(output_x)
    # print(output_Idx)
    # #output_IdxがTrueの行を抽出
    # df_fillter = df[output_Idx]
    # plt.plot(df_fillter, df[0], label="value") # Value
    # plt.show()



    # mp4_path = glob.glob(dir_path+'*original.mp4')[0]
    # video_out_path = dir_path + 'blinkdetection.mp4'
    # #まばたきがあるフレームは"Blink"とテキストで表示
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

pdf.close()
print("pdfsaved")
