# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: feature-data-enhancement
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/5/7
# @Time        : 下午12:15
# @Description :
import argparse
import datetime
import time

import pandas as pd

from config import config
from utils.build import build_trainer_tester
import os
from utils.params import BaseArgs

class Args(BaseArgs):
    def __init__(self):
        super(Args, self).__init__()

        self.is_train = True
        self.split = 'train'



def build():
    opt, log = Args().parse()
    default_config = config.copy()
    opt = argparse.Namespace(**default_config)
    opt.path = os.path.join(opt.dset_dir, opt.dset_name)
    return opt, log


if __name__ == '__main__':
    opt, logger = build()
    TrainerTesterClass = build_trainer_tester(opt)
    if 'seeds' in opt:
        acc_list = []
        sen_list = []
        spe_list= []
        f1_list=[]
        auc_list = []
        for seed in opt.seeds:
            opt.seed = seed
            opt.ckpt_path = os.path.join(opt.ckpt_dir, str(opt.seed))
            model = TrainerTesterClass(opt)
            acc, sen, spe, f1, auc = model.run(val_only=True)
            acc_list.append(acc)
            sen_list.append(sen)
            spe_list.append(spe)
            f1_list.append(f1)
            auc_list.append(auc)
        df= pd.DataFrame({'seed':opt.seeds,'acc': acc_list, 'sen':sen_list, 'spe':spe_list, 'f1':f1_list, 'auc':auc_list})
        target_dir = os.path.join('out','evaluate',opt.running_config)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        df.to_excel(os.path.join(target_dir,'result.xlsx'))
