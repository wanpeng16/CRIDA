import argparse

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
    print(opt)
    TrainerTesterClass = build_trainer_tester(opt)
    print(f'Will train use TrainerTester class: {TrainerTesterClass.__name__}.')
    if 'seeds' in opt:
        for seed in opt.seeds:
            opt.seed = seed
            opt.ckpt_path = os.path.join(opt.ckpt_dir, str(opt.seed))
            model = TrainerTesterClass(opt)
            acc, sen, spe, f1, auc = model.run()


