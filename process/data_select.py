import os

import numpy as np
import pandas as pd
import tqdm


def hcc_icc_us():
    path = '/data1/dataset/liver/GE参量成像肝脏造影/raw'
    hcc_us = pd.read_excel(os.path.join(path, 'HCC_US.xlsx'))
    icc_us = pd.read_excel(os.path.join(path, 'ICC_US.xlsx'))
    # hcc_ceus = pd.read_excel(os.path.join(path, 'HCC_CEUS.xlsx'))
    # icc_ceus = pd.read_excel(os.path.join(path, 'ICC_CEUS.xlsx'))
    # hcc_us_ceus = pd.concat([hcc_us.set_index('ID'),hcc_ceus.set_index('ID')],axis=1,join='inner')
    # icc_us_ceus = pd.concat([icc_us.set_index('ID'),icc_ceus.set_index('ID')],axis=1,join='inner')
    hcc_us['label'] = 0
    icc_us['label'] = 1
    pd.concat([hcc_us,icc_us]).to_excel('US_ALL.xlsx')
def ceus_select():

    path = '/home/amax/Desktop/workspace/dataset/liver'
    hcc_ceus = pd.read_excel(os.path.join(path, 'HCC_CEUS.xlsx'))
    ids = hcc_ceus['ID'].values
    starts = hcc_ceus['start'].values
    peaks = hcc_ceus['peak'].values
    endings = hcc_ceus['ending'].values
    print(np.min(peaks-starts))

    dest_path = os.path.join(path,'liver_feats','HCC')
    os.makedirs(dest_path,exist_ok=True)
    print(ids)
    for id in tqdm.tqdm(ids):
        start = starts[np.where(ids==id)][0]
        ending = endings[np.where(ids==id)][0]
        ceus = pd.read_excel(os.path.join(path,'HCC',f'CEUS_{id}.xlsx'), sheet_name='Sheet1')
        ceus['ID'] = id
        ceus = ceus[(ceus['time']>=start) & (ceus['time']<=ending)]
        ceus=ceus[['ID']+[i for i in ceus.columns if i!='ID']]
        ceus.to_excel(os.path.join(dest_path,f'CEUS_{id}.xlsx'),index=False)
ceus_select()
