# -*- coding=utf-8 -*-

import numpy as np
from torch.utils.data import DataLoader
import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils.build import build_dataset




def create_excel_dataloader(opt):
    type = 'liver' if 'liver' in opt.dataset else 'breast'
    DataSet = build_dataset(opt.dataset)
    xls_path = os.path.join(opt.path, f'US_CEUS.xlsx')
    sheet = pd.read_excel(xls_path, sheet_name='Sheet1')
    data = sheet.values
    X = data[:, 1:]
    y = data[:, 2]

    # k折交叉验证方式
    if opt.train_type == 'k-folder':
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=opt.seed)
        folder_data = []
        for train_index, test_index in kf.split(X, y):
            train_data, val_data, train_y, val_y = train_test_split(X[train_index], y[train_index], test_size=0.25,
                                                                    stratify=y[train_index], random_state=opt.seed)
            train_fold = DataSet(np.concatenate((train_data,
                                                           train_y.reshape(-1, 1)), axis=1),
                                           root=opt.path,
                                           type=type,
                                           )

            val_fold = DataSet(np.concatenate((val_data,
                                                         val_y.reshape(-1, 1)), axis=1),
                                         root=opt.path, subset='val',
                                         type=type
                                         )
            test_fold = DataSet(np.concatenate((X[test_index],
                                                          y[test_index].reshape(-1, 1)), axis=1),
                                          root=opt.path, subset='test',
                                          type=type
                                          )
            train_loader = DataLoader(train_fold,
                                      batch_size=opt.batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=opt.num_workers)
            memory_loader = DataLoader(train_fold,
                                       batch_size=64,
                                       shuffle=False,
                                       drop_last=False,
                                       num_workers=opt.num_workers)
            test_loader = DataLoader(test_fold,
                                     batch_size=opt.batch_size,
                                     shuffle=False,
                                     drop_last=True,
                                     num_workers=opt.num_workers)
            val_loader = DataLoader(val_fold,
                                    batch_size=opt.batch_size,
                                    shuffle=False,
                                    drop_last=True,
                                    num_workers=opt.num_workers)
            folder_data.append((train_loader, val_loader, test_loader, memory_loader))
            return folder_data
        # 训练、验证、测试集划分方式
    elif opt.train_type == 'tt':
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, stratify=y, random_state=opt.seed)
        dset_train = DataSet(np.concatenate((train_x, train_y.reshape(-1, 1)), axis=1), root=opt.path,
                                       subset='train')
        train_loader = DataLoader(dset_train,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=opt.num_workers)

        memory_loader = DataLoader(dset_train,
                                   batch_size=64,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=opt.num_workers)

        dset_test = DataSet(np.concatenate((test_x, test_y.reshape(-1, 1)), axis=1), root=opt.path,
                                      subset='test')
        test_loader = DataLoader(dset_test,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=opt.num_workers)

        return train_loader, test_loader, memory_loader
    else:
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2,
                                                            stratify=y, random_state=opt.seed)

        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.25,
                                                          stratify=train_y, random_state=opt.seed)

        dset_train = DataSet(np.concatenate((train_x, train_y.reshape(-1, 1)), axis=1), root=opt.path,
                                       subset='train',
                                       type=type,
                                       )
        train_loader = DataLoader(dset_train,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=opt.num_workers,
                                  pin_memory=False)

        memory_loader = DataLoader(dset_train,
                                   batch_size=64,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=opt.num_workers,
                                   pin_memory=False)

        dset_test = DataSet(np.concatenate((test_x, test_y.reshape(-1, 1)), axis=1), root=opt.path,
                                      subset='test', type=type)
        test_loader = DataLoader(dset_test,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=opt.num_workers,
                                 pin_memory=False)

        dset_val = DataSet(np.concatenate((val_x, val_y.reshape(-1, 1)), axis=1), root=opt.path, subset='val',
                                     type=type)
        val_loader = DataLoader(dset_val,
                                batch_size=opt.batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=opt.num_workers,
                                pin_memory=False)
        return train_loader, val_loader, test_loader, memory_loader

def create_dataloader(opt):
    if 'image' not in opt.dataset:
        return create_excel_dataloader(opt)
    type = 'liver' if 'liver' in opt.dataset else 'breast'

    DataSet = build_dataset(opt.dataset)
    dset_train = DataSet(root=opt.dset_dir,subset='train',type=type,seed=opt.seed)
    train_loader = DataLoader(dset_train,
                              shuffle=True,
                              batch_size=opt.batch_size,
                              num_workers=opt.num_workers
                              )

    memory_loader = DataLoader(dset_train,
                               batch_size=64,
                               shuffle=False,
                               num_workers=opt.num_workers)

    dset_test = DataSet(root=opt.dset_dir,subset='test',

                        type=type,seed=opt.seed)
    test_loader = DataLoader(dset_test,
                             batch_size=opt.batch_size,
                             num_workers=opt.num_workers)

    dset_val = DataSet(root=opt.dset_dir, subset='val', type=type,seed=opt.seed)
    val_loader = DataLoader(dset_val,
                            batch_size=opt.batch_size,
                            num_workers=opt.num_workers)
    return train_loader, val_loader, test_loader, memory_loader

if __name__ == '__main__':
    type = 'liver'
    root_path = '/home/amax/Desktop/workspace/dataset/pyro_data'
    root_path = os.path.join(root_path, type)
    xls_path = os.path.join(root_path, f'US_CEUS_{type}_short.xlsx')
    sheet = pd.read_excel(xls_path, sheet_name='Sheet1')
    data = sheet.values
    x = data[:, 1:]
    y = data[:, 2]
    dataset= DynamicCEUS_RankPoolingEarly(np.concatenate((x, y.reshape(-1, 1)), axis=1), root=root_path,
                                subset='train',
                                type=type)
    for one_case in dataset:
        print(one_case)