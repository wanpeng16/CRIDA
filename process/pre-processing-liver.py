import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn import svm
import itertools
import openpyxl

hcc_ceus = pd.read_excel(os.path.join('/home/amax/Desktop/workspace/dataset/liver', 'HCC_CEUS.xlsx'))
hcc_ids = hcc_ceus['ID'].values
hcc_peaks = hcc_ceus['peak'].values
icc_ceus = pd.read_excel(os.path.join('/home/amax/Desktop/workspace/dataset/liver', 'ICC_CEUS.xlsx'))
icc_ids = icc_ceus['ID'].values
icc_peaks = icc_ceus['peak'].values
def find_peak(id,type):
    if type=='HCC':
        return hcc_peaks[np.where(hcc_ids==int(id))][0]
    else:
        return icc_peaks[np.where(icc_ids==int(id))][0]





if __name__ == "__main__":


    data_path = "/home/amax/Desktop/workspace/dataset/liver/liver_feats"
    scaler_ceus = StandardScaler()
    scaler_us = StandardScaler()
    ceus_data = []
    ceus_data_dict = {}
    us_data = None
    us_ceus_data = []
    for f in os.listdir(os.path.join(data_path,'HCC')):
        if f.endswith('.xlsx'):
            if f.startswith('CEUS'):
                xls_file = os.path.join(data_path,'HCC', f)
                sheet = pd.read_excel(xls_file, sheet_name='Sheet1')
                ceus_data.append(sheet.values)
                sample_id = f.split('.')[0].split('_')[1]
                peak = find_peak(sample_id,'HCC')
                ceus_data_dict['HCC_'+f.split('.')[0].split('_')[1]] = [peak, sheet.values]
    for f in os.listdir(os.path.join(data_path,'ICC')):
        if f.endswith('.xlsx'):
            if f.startswith('CEUS'):
                xls_file = os.path.join(data_path,'ICC', f)
                sheet = pd.read_excel(xls_file, sheet_name='Sheet1')
                ceus_data.append(sheet.values)
                sample_id = f.split('.')[0].split('_')[1]
                peak = find_peak(sample_id,'ICC')
                ceus_data_dict['ICC_'+f.split('.')[0].split('_')[1]] = [peak, sheet.values]
    xls_file = os.path.join(data_path, 'US_ALL.xlsx')
    sheet = pd.read_excel(xls_file, sheet_name='Sheet1')
    us_data = sheet.values


    ceus_data = np.concatenate(ceus_data, axis=0)
    ceus_data = ceus_data[:, 2:]
    scaler_ceus.fit(ceus_data)
    scaler_us.fit(us_data[:, 3:])
    rows = us_data.shape[0]

    for i in range(rows):
        sample_id = str(int(us_data[i, 1]))
        label = int(us_data[i,2])
        if label == 0:
            sample_id_label = 'HCC_'+sample_id
        else:
            sample_id_label = 'ICC_'+sample_id

        if sample_id_label in ceus_data_dict.keys():
            if sample_id == '98':
                print(1)
            excel_file_path = '/home/amax/Desktop/workspace/dataset/liver/norm_liver_feats/CEUS_' + sample_id_label + '.xlsx'
            if os.path.exists(excel_file_path):
                continue

            peak = ceus_data_dict[sample_id_label][0]
            ceus_feat = ceus_data_dict[sample_id_label][1]
            us_feat = us_data[i, 3:]
            ceus_tid = ceus_feat[:, 1]
            ceus_tid = ceus_tid.astype(int)
            # normalize
            us_ceus_feat = np.concatenate((
                scaler_us.transform(us_feat.reshape(1, -1)),
                scaler_ceus.transform(ceus_feat[ceus_tid == peak, 2:])),
                axis=1)
            us_ceus_feat = np.insert(us_ceus_feat, 0, float(sample_id))
            us_ceus_feat = np.insert(us_ceus_feat, 1, float(label))
            us_ceus_data.append(us_ceus_feat)
            ceus_feat_norm = scaler_ceus.transform(ceus_feat[:, 2:])
            idx = np.where(ceus_tid == peak)[0][0] + 1
            pd.DataFrame(ceus_feat_norm).to_excel(excel_file_path, sheet_name='feats', float_format='%.5f')

            workbook = openpyxl.load_workbook(excel_file_path)
            sheet2 = workbook.create_sheet('idx', index=2)
            sheet2['A1'] = idx
            # 保存修改后的 Excel 文件
            workbook.save(excel_file_path)
        else:
            raise ValueError('No sample_id found!')

    pd.DataFrame(us_ceus_data).to_excel('/home/amax/Desktop/workspace/dataset/liver/US_CEUS_feats_short.xlsx', float_format='%.5f')
