import argparse

import optuna
import gc

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


def objective(trial):
    opt, logger = build()
    # lambda0 = trial.suggest_categorical('lambda0', [0.1, 0.25, 0.5, 0.75, 1.0])
    # sigma = trial.suggest_int('sigma', 5, 20)
    seed = trial.suggest_int('seed', 1, 100000)
    # num_clusters = trial.suggest_categorical('num_clusters', [2, 3, 4, 5])
    # lr = trial.suggest_categorical('lr', [1e-3, 5e-4, 1e-4])
    # opt.lambda0 = lambda0
    # opt.sigma = sigma
    opt.seed = seed
    # opt.num_clusters = num_clusters
    # opt.lr = lr
    opt.ckpt_path = os.path.join(opt.ckpt_dir, 'optuna',str(opt.seed))
    os.makedirs(opt.ckpt_path, exist_ok=True)


    try:
        TrainerTesterClass = build_trainer_tester(opt)
        model = TrainerTesterClass(opt)
        acc, sen, spe, f1, auc = model.run()
        trial.report(acc, step=0)
        trial.report(sen, step=1)
        trial.report(spe, step=2)
        trial.report(f1, step=3)
        trial.report(auc, step=4)
    except Exception as e:
        print(f"Exception in trial: {str(e)}")
        raise optuna.TrialPruned()
    finally:
        gc.collect()
    return acc


if __name__ == '__main__':
    opt, logger = build()
    type = opt.running_config
    study_name = type
    storage = f"sqlite:///out/{type}.db"
    study = optuna.create_study(storage=storage, study_name=study_name, direction="maximize",
                                pruner=optuna.pruners.MedianPruner(), load_if_exists=True)
    trials_df = study.trials_dataframe()
    file_path =f"./out/{type}_optuna_results.xlsx"  # 指定保存的文件名
    trials_df.to_excel(file_path, index=False, engine="openpyxl")
    print(study.best_params)
    print(study.best_value)
    exit(0)
    study.optimize(objective, n_trials=1500)
    trials_df = study.trials_dataframe()
    file_path = f"./out/{type}_optuna_results.xlsx"  # 指定保存的文件名
    trials_df.to_excel(file_path, index=False, engine="openpyxl")
    print(study.best_params)
    print(study.best_value)

