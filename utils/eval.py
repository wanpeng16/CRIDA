import torch
from train import build
from datasets.data_loader import create_dataloader
from module.network import ClassifierBase
from utils.train import val_epoch
import pandas as pd
import os
import numpy as np

if __name__ == "__main__":
    config, logger = build()
    df = pd.read_excel('optuna_results.xlsx', sheet_name='Sheet1')
    params = df.values
    runs = params.shape[0]
    res = []

    for i in range(runs):
        if params[i, -1] == 'COMPLETE':
            config.lamnda0 = params[i, 5]
            config.lr = params[i, 6]
            config.num_clusters = params[i, 7]
            config.seed = int(params[i, 8])
            config.sigma = params[i, 9]
            config.ckpt_path = os.path.join(config.ckpt_dir, config.dset_name, str(config.seed))
            train_loader, val_loader, test_loader, memory_loader = create_dataloader(config)
            hidden_dims = [config.hidden_dim, config.hidden_dim // 2]
            classifier = ClassifierBase(num_classes=config.num_class, us_dim=config.us_dim, ceus_dim=config.ceus_dim,
                                        hidden_dims=hidden_dims).cuda()
            classifier.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'Classifier.pth'))['model'])
            test_acc, test_sen, test_spe, test_f1, test_auc = val_epoch(classifier, config, [val_loader, test_loader])
            if np.abs(test_acc - params[i, 1]) > 1e-3:
                print( str(i) + ': error')
            else:
                res.append([config.seed, config.lr, config.lamnda0, config.num_clusters, config.sigma,
                            test_acc, test_sen, test_spe, test_f1, test_auc])

    data = np.array(res)

    # 将NumPy数组转换为Pandas数据框
    df = pd.DataFrame(data, columns=['seed', 'lr', 'lambda0', 'clusters', 'sigma', 'acc', 'sen', 'spe', 'f1', 'auc'])

    # 将数据框保存到Excel文件
    df.to_excel('full_results.xlsx', index=False)  # 如果不需要保存索引, 可以指定 index=False
