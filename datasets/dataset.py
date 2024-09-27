# -*- coding=utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd


class DynamicCEUS_Excel(Dataset):
    def __init__(self, data, root='E:/data/breast', feats_dir_name=None, type=None, return_in_out=False,
                 subset='train'):
        super(DynamicCEUS_Excel, self).__init__()
        self.root = root
        self.data = data
        self.subset = subset
        self.fdynamic_mem = {}
        self.feats_dir_name = 'feats'
        self.type = type
        self.return_in_out = return_in_out

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        fdata = self.data[index]
        if self.type == 'liver':
            label = "HCC" if int(fdata[-1]) == 0 else 'ICC'
            path = os.path.join(self.root, self.feats_dir_name, f'CEUS_{label}_' + str(int(fdata[0])) + '.xlsx')
        else:
            path = os.path.join(self.root, self.feats_dir_name, 'CEUS_' + str(int(fdata[0])) + '.xlsx')

        sheet = pd.read_excel(path, sheet_name='feats')
        ceusdata = sheet.values[:, 1:]
        sheet = pd.read_excel(path, sheet_name='idx')
        points = sheet.columns.values[0]
        wash_in = ceusdata[:points, :]
        wash_out = ceusdata[points:, :]
        wash_out_T = wash_out.shape[0]

        # 保证wash_in至少有10帧
        # 保证wash_out至少有20帧
        if points >= 10:
            wash_in = wash_in[-10:]  # 取后10帧
        else:
            wash_in = np.vstack([wash_in, wash_in[points - 10:]])  # 补充至10帧

        indices = torch.linspace(0, wash_out_T - 1, steps=20, dtype=torch.long)

        wash_out = wash_out[indices]
        fdynamic = np.concatenate((wash_in, wash_out), axis=0)
        fdata = np.reshape(fdata, (1, -1))
        # X = std_scaler_us.transform(fdata[:, 2:-1])
        X = fdata[:, 2:-1]
        # fdynamic = std_scaler_ceus.transform(fdynamic)
        X = torch.Tensor.float(torch.from_numpy(X))
        y = torch.Tensor.long(torch.from_numpy(fdata[:, -1]))
        fdynamic = torch.Tensor.float(torch.from_numpy(fdynamic))
        if self.return_in_out:
            return (X, wash_in, wash_out), y, fdata[0][0]
        return (X, fdynamic), y, index


class DynamicCEUS_RankPoolingEarly(Dataset):
    def __init__(self, data, root='E:/data/breast', type=None, subset='train'):
        super(DynamicCEUS_RankPoolingEarly, self).__init__()
        self.root = root
        self.data = data
        self.subset = subset
        self.feats_dir_name = 'rank_pooling_early'
        self.type = type
        self.return_in_out = True
        wash_in_path = os.path.join(root, self.feats_dir_name, 'rank_pooling_early_wash_in.xlsx')
        self.wash_in_data = pd.read_excel(wash_in_path).values[:, 1:]
        wash_out_path = os.path.join(root, self.feats_dir_name, 'rank_pooling_early_wash_out.xlsx')
        self.wash_out_data = pd.read_excel(wash_out_path).values[:, 1:]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        fdata = self.data[index]
        label = int(fdata[-1])
        id = fdata[0]
        pooling_wash_in = self.wash_in_data[(self.wash_in_data[:, 0] == id) & (self.wash_in_data[:, 1] == label)][0, 2:]
        pooling_wash_out = self.wash_out_data[(self.wash_out_data[:, 0] == id) & (self.wash_out_data[:, 1] == label)][0,
                           2:]
        fdynamic = np.concatenate((pooling_wash_in, pooling_wash_out), axis=0)
        fdata = np.reshape(fdata, (1, -1))
        X = fdata[:, 2:-1]
        X = torch.Tensor.float(torch.from_numpy(X))
        y = torch.Tensor.long(torch.from_numpy(fdata[:, -1]))
        fdynamic = torch.Tensor.float(torch.from_numpy(fdynamic))
        return (X, fdynamic), y, index


class DynamicCEUS_LSSLEarly(Dataset):
    def __init__(self, data, root='E:/data/breast', type=None, subset='train'):
        super(DynamicCEUS_LSSLEarly, self).__init__()
        self.root = root
        self.data = data
        self.subset = subset
        self.feats_dir_name = 'lssl'
        self.type = type
        self.return_in_out = True
        lssl_path = os.path.join(root, self.feats_dir_name, 'lssl_early.xlsx')
        self.lssl_data = pd.read_excel(lssl_path).values[:, 1:]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        fdata = self.data[index]
        label = int(fdata[-1])
        id = fdata[0]
        fdynamic = self.lssl_data[(self.lssl_data[:, 0] == id) & (self.lssl_data[:, 1] == label)][0, 2:]
        fdata = np.reshape(fdata, (1, -1))
        X = fdata[:, 2:-1]
        X = torch.Tensor.float(torch.from_numpy(X))
        y = torch.Tensor.long(torch.from_numpy(fdata[:, -1]))
        fdynamic = torch.Tensor.float(torch.from_numpy(fdynamic))
        return (X, fdynamic), y, index


class DynamicCEUS_RankPoolingEarlyCCA(DynamicCEUS_RankPoolingEarly):
    def __init__(self, data, root='E:/data/breast', type=None, subset='train'):
        super(DynamicCEUS_RankPoolingEarlyCCA, self).__init__(data, root, type, subset)

    def __getitem__(self, index):
        fdata = self.data[index]
        label = int(fdata[-1])
        id = fdata[0]
        pooling_wash_in = self.wash_in_data[(self.wash_in_data[:, 0] == id) & (self.wash_in_data[:, 1] == label)][0, 2:]
        pooling_wash_out = self.wash_out_data[(self.wash_out_data[:, 0] == id) & (self.wash_out_data[:, 1] == label)][0,
                           2:]
        fdata = np.reshape(fdata, (1, -1))
        X = fdata[:, 2:-1]
        X = torch.Tensor.float(torch.from_numpy(X))
        y = torch.Tensor.long(torch.from_numpy(fdata[:, -1]))
        pooling_wash_in = torch.Tensor.float(torch.from_numpy(pooling_wash_in))
        pooling_wash_out = torch.Tensor.float(torch.from_numpy(pooling_wash_out))
        X = X[0]
        [us, ceus] = torch.split(X, [X.shape[0] - pooling_wash_in.shape[0], pooling_wash_in.shape[0]], dim=0)
        return {"views": [us, ceus, pooling_wash_in, pooling_wash_out], 'label': y, 'index': index}
