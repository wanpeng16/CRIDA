# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: feature-data-enhancement
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/5/5
# @Time        : 下午7:11
# @Description :


import os
from abc import abstractmethod

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datasets.data_loader import create_dataloader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from utils.build import build_model
import torch
from utils.utils_vgg import memory_module_init_update, Calculate_mean_cv, setup_seed
from module.loss import SACoFALoss
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import copy
from utils.utils_vgg import annealing
from utils.utils_vgg import weighted_cv
import tqdm
import numpy as np


def pre_train_epoch(model, optimizer, train_loader):
    model.train()
    epoch_loss = 0.
    loss = nn.CrossEntropyLoss().cuda()
    acc_list = []
    for ind, ((us_img, ceus_img, wash_in_images, wash_out_images), labels, index) in enumerate(train_loader):
        labels = Variable(labels).cuda()
        ceus_img = ceus_img.cuda()
        us_img = us_img.cuda()
        wash_in_images = wash_in_images.cuda()
        wash_out_images = wash_out_images.cuda()
        predictions, (x_us, x_ceus, wash_in, wash_out) = model(us_img, ceus_img, wash_in_images, wash_out_images)
        cls_loss = loss(predictions, labels)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()
        epoch_loss += cls_loss.item()
        predictions = torch.nn.functional.softmax(predictions, dim=1)
        _, y_pred = torch.max(predictions, 1)
        m_acc = accuracy_score(labels.detach().cpu(), y_pred.detach().cpu())
        acc_list.append(m_acc)

    epoch_loss = epoch_loss / (ind + 1)

    return epoch_loss, np.mean(acc_list)


class TrainerTesterBase(object):
    def __init__(self, config):
        self.test_loader = None
        self.val_loader = None
        self.train_loader = None
        self.tboard = None
        self.model_save_path = None
        self.config = config
        config = self.config
        self.model_save_path = self.config.ckpt_path
        self.tboard = SummaryWriter(log_dir=self.model_save_path)
        self.Classifier = build_model(config.model)
        print(f'Build modal class: {self.Classifier.__name__}.')

        self.best_test_acc = 0
        self.best_acc = 0

    def evaluate(self, classifier):
        best_val_model = torch.load(os.path.join(self.model_save_path, 'Classifier.pth'))['model']
        classifier.load_state_dict(best_val_model)
        test_acc, test_sen, test_spe, test_f1, test_auc = self.val_epoch(classifier, [self.test_loader],
                                                                         only_acc=False)
        return test_acc, test_sen, test_spe, test_f1, test_auc

    def val_epoch(self, model, dataloaders, only_acc=True):
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
            m_acc = accuracy_score(y_true, y_pred)
            if only_acc:
                return m_acc, None, None, None, None
            m_f1 = f1_score(y_true, y_pred, pos_label=1, average='binary')
            m_auc = roc_auc_score(y_true, all_output[:, 1])
            cm = confusion_matrix(y_true, y_pred)
            TN, FP, FN, TP = cm.ravel()
            m_spe = TN / (TN + FP)
            m_sen = TP / (TP + FN)

        return m_acc, m_sen, m_spe, m_f1, m_auc

    def log_epoch(self, model, current_epoch, train_acc, train_loss):
        self.tboard.add_scalar("train/cls_loss", train_loss, current_epoch)
        val_acc, _, _, _, _ = self.val_epoch(model, [self.val_loader])
        if self.best_acc < val_acc:
            self.best_acc = val_acc
            self.best_model = copy.deepcopy(model)
        test_acc, _, _, _, _ = self.val_epoch(model, [self.test_loader])
        if self.tboard:
            self.tboard.add_scalar('train/val', val_acc, current_epoch)
            self.tboard.add_scalar('train/test', test_acc, current_epoch)
            # train_acc, _, _, _, train_auc = val_epoch(classifier, config,[train_loader])
            self.tboard.add_scalar('train/train', train_acc, current_epoch)

    @abstractmethod
    def run_over(self, train_loader, val_loader, test_loader, memory_loader, val_only):
        pass

    def save_train(self, classifier, best_val_model):
        classifier.load_state_dict(best_val_model.state_dict())
        result = {}
        val_acc, val_sen, val_spe, val_f1, val_auc = self.val_epoch(classifier, [self.val_loader],
                                                                    only_acc=False)
        result['model'] = best_val_model.state_dict()
        result['seed'] = self.config.seed

        torch.save(result,
                   os.path.join(self.model_save_path, 'Classifier.pth'))

        self.tboard.close()
        return val_acc, val_sen, val_spe, val_f1, val_auc

    def run(self, val_only=False):
        setup_seed(self.config.seed)
        config = self.config
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config.gpu

        test_acc_list, test_sen_list, test_spe_list, test_f1_list, test_auc_list = [], [], [], [], []

        if config.train_type == 'k-folder':
            for train_loader, val_loader, test_loader, memory_loader in create_dataloader(config):
                self.train_loader = train_loader
                self.val_loader = val_loader
                self.test_loader = test_loader
                test_acc, test_sen, test_spe, test_f1, test_auc = self.run_over(train_loader, val_loader, test_loader,
                                                                                memory_loader, val_only)
                test_acc_list.append(test_acc)
                test_sen_list.append(test_sen)
                test_spe_list.append(test_spe)
                test_f1_list.append(test_f1)
                test_auc_list.append(test_auc)
        elif config.train_type == 'tvt':
            train_loader, val_loader, test_loader, memory_loader = create_dataloader(config)
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            test_acc, test_sen, test_spe, test_f1, test_auc = self.run_over(train_loader, val_loader, test_loader,
                                                                            memory_loader, val_only)
            test_acc_list.append(test_acc)
            test_sen_list.append(test_sen)
            test_spe_list.append(test_spe)
            test_f1_list.append(test_f1)
            test_auc_list.append(test_auc)
        elif config.train_type == 'tt':
            train_loader, val_loader, memory_loader = create_dataloader(config)
            test_loader = val_loader
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            test_acc, test_sen, test_spe, test_f1, test_auc = self.run_over(train_loader, val_loader, test_loader,
                                                                            memory_loader, val_only)
            test_acc_list.append(test_acc)
            test_sen_list.append(test_sen)
            test_spe_list.append(test_spe)
            test_f1_list.append(test_f1)
            test_auc_list.append(test_auc)
        return (np.mean(test_acc_list), np.mean(test_sen_list),
                np.mean(test_spe_list), np.mean(test_f1_list), np.mean(test_auc_list))
