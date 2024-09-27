import os
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter

from config import config
from datasets.data_loader import create_dataloader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.build import build_model
import torch

from utils.params import BaseArgs
from utils.utils_vgg import memory_module_init_update, Calculate_mean_cv, setup_seed
from module.loss import SACoFALoss
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import copy
from utils.utils_vgg import annealing
from utils.utils_vgg import weighted_cv
import tqdm
import numpy as np


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def val_epoch(model, args, dataloaders):
    # switch to train mode
    model.eval()
    loss_fun = torch.nn.CrossEntropyLoss(reduction="none").cuda()
    start_test = True
    with torch.no_grad():
        for dataloader in dataloaders:
            for i, ((us_img, ceus_img, wash_in_images, wash_out_images), target, index) in enumerate(dataloader):
                target = Variable(target).cuda()
                ceus_img = ceus_img.cuda()
                us_img = us_img.cuda()
                wash_in_images = wash_in_images.cuda()
                wash_out_images = wash_out_images.cuda()
                output, _ = model(us_img, ceus_img, wash_in_images, wash_out_images)

                # get logit outputs
                loss = loss_fun(output, target.flatten())
                if start_test:
                    all_output = output.float()
                    all_label = target.float()
                    all_loss = loss.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, output.float()), 0)
                    all_label = torch.cat((all_label, target.float()), 0)
                    all_loss = torch.cat((all_loss, loss.float()), 0)

        all_output = torch.nn.functional.softmax(all_output, dim=1)
        _, y_pred = torch.max(all_output, 1)

        y_pred = y_pred.detach().cpu()
        y_true = all_label.detach().cpu().long()
        all_output = all_output.detach().cpu()
        all_loss = all_loss.detach().cpu()

        # split

        # MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAUROC
        m_acc = accuracy_score(y_true, y_pred)
        m_f1 = f1_score(y_true, y_pred, pos_label=1, average='binary')
        m_auc = roc_auc_score(y_true, all_output[:, 1])
        cm = confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cm.ravel()
        m_spe = TN / (TN + FP)
        m_sen = TP / (TP + FN)

    return m_acc, m_sen, m_spe, m_f1, m_auc


def pre_train_epoch(model, optimizer, train_loader):
    model.train()
    epoch_loss = 0.
    loss = nn.CrossEntropyLoss().cuda()
    for ind, ((us_img,ceus_img,wash_in_images,wash_out_images), labels, index) in enumerate(train_loader):
        labels = Variable(labels).cuda()
        ceus_img = ceus_img.cuda()
        us_img = us_img.cuda()
        wash_in_images = wash_in_images.cuda()
        wash_out_images = wash_out_images.cuda()
        predictions, ( x_us, x_ceus, wash_in, wash_out) = model(us_img, ceus_img, wash_in_images, wash_out_images)
        cls_loss =loss (predictions, labels)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()


        epoch_loss += cls_loss.item()

    epoch_loss = epoch_loss / (ind + 1)

    return epoch_loss


def train_epoch(model, optimizer, args, cls_criterion, train_loader, memory_static, memory_dynamic,
                memory_labels, memory_cluster, ratio):
    model.train()
    epoch_loss = 0.
    class_num = args.num_class
    memory_labels = memory_labels.flatten()

    for ind, ((us_img,ceus_img,wash_in_images,wash_out_images), labels, index) in enumerate(train_loader):
        labels = Variable(labels).cuda()
        ceus_img = ceus_img.cuda()
        us_img = us_img.cuda()
        wash_in_images = wash_in_images.cuda()
        wash_out_images = wash_out_images.cuda()
        predictions, ( x_us, x_ceus, wash_in, wash_out) = model(us_img, ceus_img, wash_in_images, wash_out_images)


        bs = us_img.shape[0]
        # compute output
        feats_us, feats_ceus = x_us, x_ceus
        feats_us_data = feats_us.detach()
        dynamics = torch.cat([wash_in, wash_out], dim=1)
        x_ceus_dynamics = torch.cat((x_ceus, dynamics), dim=1)
        feats_ceus_data = x_ceus_dynamics.detach()
        features = torch.cat([feats_us, x_ceus_dynamics], dim=1)

        # update the memory module
        feats_us_data = feats_us_data.cpu()
        feats_ceus_data = feats_ceus_data.cpu()
        memory_static[index] = feats_us_data.cpu()
        memory_dynamic[index] = feats_ceus_data.cpu()

        # US--> CEUS
        mean_ceus, cv_ceus, cluster_size_ceus = Calculate_mean_cv(args, memory_dynamic, memory_static, memory_labels,
                                                                  memory_cluster[:, 0],
                                                                  class_num)
        mean_us, cv_us, cluster_size_us = Calculate_mean_cv(args, memory_static, memory_dynamic, memory_labels,
                                                            memory_cluster[:, 1],
                                                            class_num)

        # 计算当前样本到不同类别中心的距离，以类别的协方差矩阵加权和为参数，生成新的样本
        if args.policy=='ISDA_cluster':
            cv_us = cv_us.view((cv_us.shape[0]*cv_us.shape[1],cv_us.shape[2],cv_us.shape[3]))
            cv_ceus = cv_ceus.view((cv_ceus.shape[0]*cv_ceus.shape[1],cv_ceus.shape[2],cv_ceus.shape[3]))
            CoVariance = torch.stack([torch.block_diag(cv_us[i], cv_ceus[i]) for i in range(cv_us.shape[0])])
            cls_loss = cls_criterion.loss(model.head, features, predictions, labels, CoVariance, ratio)

        else:
            weighted_cv_ceus = torch.zeros(bs,
                                           feats_ceus_data.shape[1],
                                           feats_ceus_data.shape[1]).cuda()

            weighted_cv_us = torch.zeros(bs, feats_us.shape[1], feats_us.shape[1]).cuda()
            for i in range(class_num):
                feats_ceus_c = feats_ceus_data[labels.cpu() == i]
                feats_us_c = feats_us_data[labels.cpu() == i]
                weighted_cv_ceus_c = weighted_cv(args=args, cluster_nums=args.num_clusters, feats=feats_us_c,
                                                 cluster_means=mean_ceus[i], cluster_cvs=cv_ceus[i],
                                                 cluster_size=cluster_size_ceus[i])
                weighted_cv_us_c = weighted_cv(args=args, cluster_nums=args.num_clusters, feats=feats_ceus_c,
                                               cluster_means=mean_us[i],
                                               cluster_cvs=cv_us[i], cluster_size=cluster_size_us[i])
                weighted_cv_ceus[labels.cpu() == i] = weighted_cv_ceus_c.cuda()
                weighted_cv_us[labels.cpu() == i] = weighted_cv_us_c.cuda()

            CoVariance = torch.stack([torch.block_diag(weighted_cv_us[i], weighted_cv_ceus[i]) for i in range(bs)])

            cls_loss = cls_criterion(model.head, features, predictions, labels, CoVariance, ratio)

        assert not torch.isnan(cls_loss), 'Model diverged with loss = NaN'

        # compute gradient and do SGD step
        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()

        # sys.stdout.write(
        #     '\r == == >  Batch[%d/%d] Training loss: %.4f' % (ind, len(train_loader), cls_loss))
        #
        epoch_loss += cls_loss.item()

    epoch_loss = epoch_loss / (ind + 1)

    return epoch_loss





class TrainerTesterVGG(object):
    def __init__(self, config):

        self.tboard = None
        self.model_save_path = None
        self.config = config
        self.gpu = config.gpu


    def run_over(self, train_loader, val_loader, test_loader, memory_loader):
        config = self.config
        self.model_save_path = self.config.ckpt_path
        self.tboard = SummaryWriter(log_dir=self.model_save_path)
        hidden_dims = config.hidden_dims
        Classifier = build_model(config.model)
        classifier = Classifier(num_classes=config.num_class, us_dim=config.us_dim, ceus_dim=config.ceus_dim,
                                hidden_dims=hidden_dims).cuda()
        classifier = DDP(classifier, device_ids=[self.gpu])
        optimizer = torch.optim.Adam(classifier.parameters(), lr=config.lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
        pretrain_epoch = tqdm.tqdm(range(100))
        pretrain_epoch.set_description_str("Pretraining model:")
        for current_epoch in pretrain_epoch:
            train_loader.sampler.set_epoch(current_epoch)
            pre_train_epoch(classifier,optimizer,train_loader)
            val_acc, val_sen, val_spe, val_f1, val_auc = val_epoch(classifier, config,
                                                                   [val_loader])


            test_acc, test_sen, test_spe, test_f1, test_auc = val_epoch(classifier, config,
                                                                        [test_loader])

            self.tboard.add_scalar('val acc', val_acc, current_epoch)
            self.tboard.add_scalar('val auc', val_auc, current_epoch)
            self.tboard.add_scalar('test acc', test_acc, current_epoch)
            self.tboard.add_scalar('test auc', test_auc, current_epoch)
            train_acc, _, _, _, train_auc = val_epoch(classifier, config, [train_loader])
            self.tboard.add_scalar('train acc', train_acc, current_epoch)
            self.tboard.add_scalar('train auc', train_auc, current_epoch)
        train_acc, _, _, _, _ = val_epoch(classifier, config,[train_loader])
        val_acc, _, _, _, _ = val_epoch(classifier, config,[val_loader])
        test_acc, _, _, _, _ = val_epoch(classifier, config,[test_loader])

        print(f"Pretrain model finished: train_acc  {train_acc:.4f}; val_acc {val_acc:.4f}; test_acc {test_acc:.4f}")

        s_len = len(memory_loader.dataset)
        memory_us = torch.zeros(s_len, config.us_dim)

        memory_ceus = torch.zeros(s_len, config.ceus_dim * (config.dynamics_num+1))
        memory_labels = torch.zeros(s_len).long()
        memory_cluster = torch.zeros(s_len, 2).long()
        # 初始化，聚类
        memory_module_init_update(memory_loader, classifier, memory_us, memory_ceus, memory_labels, memory_cluster,
                                  config)

        best_acc = 0.
        best_test_acc = 0
        best_model = None
        best_test_model = None
        result = {}
        ratios = annealing(epochs=config.epochs * len(train_loader), anneal_end_dynamics=config.lambda0)
        if config.policy == 'ISDA_cluster':
            from experiment.train_isda import ISDALoss
            Loss = ISDALoss(classifier.features_dim,class_num=config.num_class).cuda()
        else:
            Loss = SACoFALoss(class_num=config.num_class).cuda()
        epochs_tqdm = tqdm.tqdm(range(config.epochs))

        for current_epoch in epochs_tqdm:
            cls_loss = train_epoch(classifier, optimizer,
                                   config,
                                   Loss,
                                   train_loader,
                                   memory_us, memory_ceus,
                                   memory_labels, memory_cluster,
                                   ratios[current_epoch]
                                   )
            scheduler.step()

            if self.tboard:
                self.tboard.add_scalar("cls_loss", cls_loss, current_epoch)

            if current_epoch > 0 and (current_epoch + 1) % config.every_epoch == 0:

                val_acc, val_sen, val_spe, val_f1, val_auc = val_epoch(classifier, config,
                                                                       [val_loader])
                if best_acc < val_acc:
                    best_acc = val_acc
                    best_model = copy.deepcopy(classifier)

                test_acc, test_sen, test_spe, test_f1, test_auc = val_epoch(classifier, config,
                                                                            [test_loader])
                if best_test_acc < test_acc:
                    best_test_model = copy.deepcopy(classifier)
                    best_test_acc = test_acc
                    # sys.stdout.write("\n Best validation: [{}/{:02d}]	Acc:{:.3f} F1-score:{:.3f} auc:{:.3f}".format(
                    #     current_epoch, config.epochs, test_acc, test_f1, test_auc))

                if self.tboard:
                    self.tboard.add_scalar('val acc', val_acc, current_epoch)
                    self.tboard.add_scalar('val auc', val_auc, current_epoch)
                    self.tboard.add_scalar('test acc', test_acc, current_epoch)
                    self.tboard.add_scalar('test auc', test_auc, current_epoch)
                    train_acc, _, _, _, train_auc = val_epoch(classifier, config,[train_loader])
                    self.tboard.add_scalar('train acc', train_acc, current_epoch)
                    self.tboard.add_scalar('train auc', train_auc, current_epoch)
                epochs_tqdm.set_postfix(val_best_acc=best_acc, test_best_acc=best_test_acc)

        classifier.load_state_dict(best_model.state_dict())
        test_acc, test_sen, test_spe, test_f1, test_auc = val_epoch(classifier, config, [val_loader])
        result['model'] = best_model.state_dict()
        result['acc'] = test_acc
        result['sen'] = test_sen
        result['spe'] = test_spe
        result['f1'] = test_f1
        result['auc'] = test_auc
        result['seed'] = config.seed
        # print("\n Test Result: Acc:{:.3f} F1-score:{:.3f} auc:{:.3f}".format(test_acc, test_f1, test_auc))

        torch.save(result,
                   os.path.join(self.model_save_path, 'Classifier_val.pth'))

        classifier.load_state_dict(best_test_model.state_dict())
        test_acc, test_sen, test_spe, test_f1, test_auc = val_epoch(classifier, config, [test_loader])
        result['model'] = best_model.state_dict()
        result['acc'] = test_acc
        result['sen'] = test_sen
        result['spe'] = test_spe
        result['f1'] = test_f1
        result['auc'] = test_auc
        result['seed'] = config.seed
        # print("\n Test Result: Acc:{:.3f} F1-score:{:.3f} auc:{:.3f}".format(test_acc, test_f1, test_auc))

        torch.save(result,
                   os.path.join(self.model_save_path, 'Classifier_test.pth'))

        self.tboard.close()
        return max(best_test_acc, best_acc), test_sen, test_spe, test_f1, test_auc

    def run(self):
        setup_seed(self.config.seed)
        config = self.config
        # os.environ['CUDA_VISIBLE_DEVICES'] = self.config.gpu

        test_acc_list, test_sen_list, test_spe_list, test_f1_list, test_auc_list = [], [], [], [], []

        if config.train_type == 'k-folder':
            for train_loader, val_loader, test_loader, memory_loader in create_dataloader(config):
                test_acc, test_sen, test_spe, test_f1, test_auc = self.run_over(train_loader, val_loader, test_loader,
                                                                                memory_loader)
                test_acc_list.append(test_acc)
                test_sen_list.append(test_sen)
                test_spe_list.append(test_spe)
                test_f1_list.append(test_f1)
                test_auc_list.append(test_auc)
        elif config.train_type == 'tvt':
            train_loader, val_loader, test_loader, memory_loader = create_dataloader(config)
            test_acc, test_sen, test_spe, test_f1, test_auc = self.run_over(train_loader, val_loader, test_loader,
                                                                            memory_loader)
            test_acc_list.append(test_acc)
            test_sen_list.append(test_sen)
            test_spe_list.append(test_spe)
            test_f1_list.append(test_f1)
            test_auc_list.append(test_auc)
        elif config.train_type == 'tt':
            train_loader, val_loader, memory_loader = create_dataloader(config)
            test_loader = val_loader
            test_acc, test_sen, test_spe, test_f1, test_auc = self.run_over(train_loader, val_loader, test_loader,
                                                                            memory_loader)
            test_acc_list.append(test_acc)
            test_sen_list.append(test_sen)
            test_spe_list.append(test_spe)
            test_f1_list.append(test_f1)
            test_auc_list.append(test_auc)
        return (np.mean(test_acc_list), np.mean(test_sen_list),
                np.mean(test_spe_list), np.mean(test_f1_list), np.mean(test_auc_list))



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


def main(rank: int, world_size: int):
    ddp_setup(rank, world_size)
    opt, logger = build()
    print(opt)
    TrainerTesterClass = TrainerTesterVGG
    if 'seeds' in opt:
        for seed in opt.seeds:
            opt.seed = seed
            opt.gpu = rank
            opt.ckpt_path = os.path.join(opt.ckpt_dir, str(opt.seed))
            model = TrainerTesterClass(opt)
            acc, sen, spe, f1, auc = model.run()
    destroy_process_group()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')

    args = parser.parse_args()


    world_size = torch.cuda.device_count()
    mp.spawn(main,args=(world_size,),  nprocs=world_size)

