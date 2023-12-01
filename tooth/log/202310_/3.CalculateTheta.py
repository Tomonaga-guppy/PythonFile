import numpy as np
import os
import glob
import openpyxl as op

root_dir = 'C:/Users/Tomson/BRLAB/tooth/Temporomandibular_movement/movie/2023_06_06'

def CalculateTheta():
    pattern = os.path.join(root_dir, 'patient*/calibration/result.npy')
    npy_files = glob.glob(pattern, recursive=True)

    for i,npy_file in enumerate(npy_files):
        aa=np.load(npy_file, allow_pickle=True)
        theta_sum = 0

        for frame in range(aa.shape[0]):
            #e_XL (36 → 45のベクトル)
            bector_x = np.array([aa[frame][45][1]-aa[frame][36][1], aa[frame][45][2]-aa[frame][36][2], aa[frame][45][4]-aa[frame][36][4]])

            bector30_36 = np.array([aa[frame][36][1]-aa[frame][30][1],aa[frame][36][2]-aa[frame][30][2],aa[frame][36][4]-aa[frame][30][4]])
            c = - (np.dot(bector_x,bector30_36))/(np.linalg.norm(bector_x)**2)

            #e_yLnose
            #bector_Xと30からbectorXに下した垂線の交点をPとした、30とPを結ぶベクトルがbector_y_nose
            bector_y_nose = bector30_36 + c*bector_x
            bector_Pposition = [bector_y_nose[0]+aa[frame][30][1],bector_y_nose[1]+aa[frame][30][2],bector_y_nose[2]+aa[frame][30][4]]

            bector_y_nose2= np.array([bector_y_nose[0],bector_y_nose[1], bector_Pposition[2]])
            theta_sum += float(np.arccos(np.dot(bector_y_nose,bector_y_nose2)/(np.linalg.norm(bector_y_nose)*np.linalg.norm(bector_y_nose2))))
            theta = (theta_sum/aa.shape[0])

        #theta保存用のexcelファイル作成
        wb = op.Workbook()
        # ワークシートの有効化
        ws = wb.active
        ws['A1'] = theta
        excelpath = root_dir + '/patient' + str(i+1) + '/theta.xlsx'
        wb.save(excelpath)
        print('theta.xlsx is saved in',excelpath)

CalculateTheta()