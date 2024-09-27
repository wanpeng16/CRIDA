import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import itertools
import openpyxl


def find_peak(path):
    peak = None
    for f in os.listdir(path):
        if f.__contains__('_p'):
            peak = int(f.split('.')[0].split('_')[0])
    if peak is None:
        raise ValueError('No peak found!')
    else:
        return peak


if __name__ == "__main__":
    xls = pd.read_excel('/home/amax/Desktop/workspace/dataset/breast/metastasis/乳腺超声组学统计.xls', sheet_name='Sheet1')
    params = xls.values
    data_path = "/home/amax/Desktop/workspace/dataset/breast/feats"
    ceus_data = []
    ceus_data_dict = {}
    us_data = []
    us_ceus_data = []
    scaler_ceus = StandardScaler()
    scaler_us = StandardScaler()
    sheet_us = pd.read_excel(os.path.join(data_path, 'US_ALL.xls'), sheet_name='Sheet1')
    for f in os.listdir(data_path):
        if f.endswith('.xls'):
            if f.startswith('CEUS'):
                sheet = pd.read_excel(os.path.join(data_path, f), sheet_name='Sheet1')
                sample_id = f.split('.')[0].split('_')[1]
                filtered_rows = xls[xls['序号'] == int(sample_id)]
                label = filtered_rows['淋巴结转移'].values[0]
                if os.path.exists(os.path.join('/home/amax/Desktop/workspace/dataset/breast/source_imaging', sample_id)):
                    peak = find_peak(os.path.join('/home/amax/Desktop/workspace/dataset/breast/source_imaging', sample_id, 'imgs'))
                else:
                    peak = find_peak(os.path.join('/home/amax/Desktop/workspace/dataset/breast/source_imaging', sample_id + '良', 'imgs'))
                if label == '有':
                    ceus_data_dict[f.split('.')[0].split('_')[1]] = [1, peak, sheet.values]
                    ceus_data.append(sheet.values)
                    us_data.append(np.reshape(sheet_us[sheet_us['ID'] == int(sample_id)].values[0], (1, -1)))
                elif label == '无':
                    ceus_data_dict[f.split('.')[0].split('_')[1]] = [0, peak, sheet.values]
                    ceus_data.append(sheet.values)
                    us_data.append(np.reshape(sheet_us[sheet_us['ID'] == int(sample_id)].values[0], (1, -1)))
                else:
                    continue

    ceus_data = np.concatenate(ceus_data, axis=0)
    ceus_data = ceus_data[:, 2:]
    scaler_ceus.fit(ceus_data)
    us_data = np.concatenate(us_data, axis=0)
    scaler_us.fit(us_data[:, 1:])
    rows = us_data.shape[0]

    for i in range(rows):
        sample_id = str(int(us_data[i, 0]))
        if sample_id in ceus_data_dict.keys():
            peak = ceus_data_dict[sample_id][1]
            ceus_feat = ceus_data_dict[sample_id][2]
            us_feat = us_data[i, 1:]
            ceus_tid = ceus_feat[:, 1]
            ceus_tid = ceus_tid.astype(int)
            # normalize
            us_ceus_feat = np.concatenate((
                scaler_us.transform(us_feat.reshape(1, -1)),
                scaler_ceus.transform(ceus_feat[ceus_tid == peak, 2:])),
                axis=1)
            us_ceus_feat = np.insert(us_ceus_feat, 0, float(sample_id))
            us_ceus_feat = np.insert(us_ceus_feat, 1, float(ceus_data_dict[sample_id][0]))
            us_ceus_data.append(us_ceus_feat)

            ceus_feat_norm = scaler_ceus.transform(ceus_feat[:, 2:])
            idx = np.where(ceus_tid == peak)[0][0] + 1

            excel_file_path = '/home/amax/Desktop/workspace/dataset/breast/norm_feats_metastasis/CEUS_' + sample_id + '.xlsx'
            pd.DataFrame(ceus_feat_norm).to_excel(excel_file_path, sheet_name='feats', float_format='%.5f')

            workbook = openpyxl.load_workbook(excel_file_path)
            sheet2 = workbook.create_sheet('idx', index=2)
            sheet2['A1'] = idx
            # 保存修改后的 Excel 文件
            workbook.save(excel_file_path)

        else:
            raise ValueError('No sample_id found!')

    pd.DataFrame(us_ceus_data).to_excel('/home/amax/Desktop/workspace/dataset/breast/US_CEUS_metastasis_short_label.xlsx', float_format='%.5f')
