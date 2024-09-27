import os

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn import svm

from datasets.dataset import DynamicCEUS_Excel



def smoothSeq(seq):

    res = np.cumsum(seq, axis=1)
    seq_len = np.size(res, 1)
    res = res / np.expand_dims(np.linspace(1, seq_len, seq_len), 0)
    return res

def rootExpandKernelMap(data):

    element_sign=np.sign(data)
    nonlinear_value=np.sqrt(np.fabs(data))
    return np.vstack((nonlinear_value*(element_sign>0),nonlinear_value*(element_sign<0)))

def getNonLinearity(data,nonLin='ref'):

    # we don't provide the Chi2 kernel in our code
    if nonLin=='none':
        return data
    if nonLin=='ref':
        return rootExpandKernelMap(data)
    elif nonLin=='tanh':
        return np.tanh(data)
    elif nonLin=='ssr':
        return np.sign(data)*np.sqrt(np.fabs(data))
    else:
        raise("We don't provide {} non-linear transformation".format(nonLin))

def normalize(seq,norm='l2'):

    if norm=='l2':
        seq_norm = np.linalg.norm(seq, ord=2, axis=0)
        seq_norm[seq_norm == 0] = 1
        seq_norm = seq / np.expand_dims(seq_norm, 0)
        return seq_norm
    elif norm=='l1':
        seq_norm=np.linalg.norm(seq,ord=1,axis=0)
        seq_norm[seq_norm==0]=1
        seq_norm=seq/np.expand_dims(seq_norm,0)
        return seq_norm
    else:
        raise("We only provide l1 and l2 normalization methods")



def rank_pooling(time_seq,C = 1,NLStyle = 'ssr'):
    '''
    This function only calculate the positive direction of rank pooling.
    :param time_seq: D x T
    :param C: hyperparameter
    :param NLStyle: Nonlinear transformation.Including: 'ref', 'tanh', 'ssr'.
    :return: Result of rank pooling
    '''

    seq_smooth=smoothSeq(time_seq)
    seq_nonlinear=getNonLinearity(seq_smooth,NLStyle)
    seq_norm=normalize(seq_nonlinear)
    seq_len=np.size(seq_norm, 1)
    Labels=np.array(range(1,seq_len+1))
    seq_svr=scipy.sparse.csr_matrix(np.transpose(seq_norm))
    svr_model = svm.LinearSVR(epsilon=0.1, tol=0.001, C=C, loss='squared_epsilon_insensitive', fit_intercept=False, dual=False)
    svr_model.fit(seq_svr,Labels)
    return svr_model.coef_

def rank_pooling_early(dataset,output):
    if not os.path.exists(output):
        os.mkdir(output)
    wash_in_table = []
    wash_out_table = []
    for one_case in dataset:
        wash_in = one_case[0][1].transpose(1,0)
        wash_out = one_case[0][2].transpose(1,0)
        y = one_case[1]
        id = one_case[2]
        pooling_wash_in = rank_pooling(wash_in)
        pooling_wash_out = rank_pooling(wash_out)
        wash_in_table.append(np.concatenate([[id,y[0]],pooling_wash_in]))
        wash_out_table.append(np.concatenate([[id,y[0]],pooling_wash_out]))
    df = pd.DataFrame(wash_in_table)
    df.to_excel(os.path.join(output,'rank_pooling_early_wash_in.xlsx'))
    df = pd.DataFrame(wash_out_table)
    df.to_excel(os.path.join(output,'rank_pooling_early_wash_out.xlsx'))

def rank_pooling_once(dataset,output):
    if not os.path.exists(output):
        os.mkdir(output)
    dynamic = []
    for one_case in dataset:
        wash_in = one_case[0][1].transpose(1,0)
        wash_out = one_case[0][2].transpose(1,0)
        id = one_case[2]
        one_dynamic = np.concatenate([wash_in,wash_out])
        pooling_one_dynamic = rank_pooling(one_dynamic)
        dynamic.append(np.concatenate([[id],pooling_one_dynamic]))
    df = pd.DataFrame(dynamic)
    df.to_excel(os.path.join(output,'rank_pooling_once.xlsx'))


if __name__ == '__main__':
    type = 'breast'
    root_path = '/home/amax/Desktop/workspace/dataset/pyro_data'
    root_path = os.path.join(root_path,type)
    xls_path = os.path.join(root_path, f'US_CEUS.xlsx')
    sheet = pd.read_excel(xls_path, sheet_name='Sheet1')
    data = sheet.values
    x = data[:, 1:]
    y = data[:, 2]
    dataset = DynamicCEUS_Excel(np.concatenate((x, y.reshape(-1, 1)), axis=1), root=root_path,
                                subset='train',
                                type=type,
                                return_in_out=True)
    rank_pooling_early(dataset,os.path.join(root_path,'rank_pooling_early'))
    # rank_pooling_early(dataset, os.path.join(root_path, 'rank_pooling_once'))