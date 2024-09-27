import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import torch

from utils.train_base import TrainerTesterBase
from utils.utils_vgg import memory_module_init_update, Calculate_mean_cv
from module.loss import SACoFALoss
from sklearn.metrics import accuracy_score
from utils.utils_vgg import annealing
from utils.utils_vgg import weighted_cv
import tqdm
import numpy as np


class TrainerTesterDeep(TrainerTesterBase):
    def __init__(self, config):
        super().__init__(config)


    def run_over(self, train_loader, val_loader, test_loader, memory_loader,val_only):
        config = self.config
        hidden_dims = config.hidden_dims
        classifier = self.Classifier(num_classes=config.num_class, us_dim=config.us_dim, ceus_dim=config.ceus_dim,
                                     hidden_dims=hidden_dims).cuda()
        if val_only:
            return self.evaluate(classifier)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=config.lr,
                                     weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
        pretrain_epoch = tqdm.tqdm(range(config.pretrain_epochs))
        pretrain_epoch.set_description_str("Pretraining model")
        for current_epoch in pretrain_epoch:
            cls_loss, train_acc = self.pre_train_epoch(classifier, optimizer, train_loader)
            self.tboard.add_scalar("pretrain/cls_loss", cls_loss, current_epoch)

            val_acc, _, _, _, _ = self.val_epoch(classifier, [val_loader])

            test_acc, _, _, _, _ = self.val_epoch(classifier, [test_loader])

            self.tboard.add_scalar('pretrain/val_acc', val_acc, current_epoch)
            self.tboard.add_scalar('pretrain/test_acc', test_acc, current_epoch)
            self.tboard.add_scalar('pretrain/train_acc', train_acc, current_epoch)
        train_acc, _, _, _, _ = self.val_epoch(classifier, [train_loader])
        val_acc, _, _, _, _ = self.val_epoch(classifier, [val_loader])
        test_acc, _, _, _, _ = self.val_epoch(classifier, [test_loader])

        print(f"Pretrain model finished: train_acc  {train_acc:.4f}; val_acc {val_acc:.4f}; test_acc {test_acc:.4f}")

        s_len = len(memory_loader.dataset)
        memory_us = torch.zeros(s_len, config.us_dim)

        memory_ceus = torch.zeros(s_len, config.ceus_dim * (config.dynamics_num + 1))
        memory_labels = torch.zeros(s_len).long()
        memory_cluster = torch.zeros(s_len, 2).long()
        # 初始化，聚类
        memory_module_init_update(memory_loader, classifier, memory_us, memory_ceus, memory_labels, memory_cluster,
                                  config)

        self.best_acc = 0.
        self.best_test_acc = 0
        self.best_model = None
        ratios = annealing(epochs=config.epochs * len(train_loader), anneal_end_dynamics=config.lambda0)
        if config.policy == 'ISDA_cluster':
            from experiment.train_isda import ISDALoss
            Loss = ISDALoss(classifier.features_dim, class_num=config.num_class).cuda()
        else:
            Loss = SACoFALoss(class_num=config.num_class).cuda()
        epochs_tqdm = tqdm.tqdm(range(config.epochs))
        for current_epoch in epochs_tqdm:
            cls_loss, train_acc = self.train_epoch(classifier, optimizer,
                                                   config,
                                                   Loss,
                                                   train_loader,
                                                   memory_us, memory_ceus,
                                                   memory_labels, memory_cluster,
                                                   ratios[current_epoch]
                                                   )
            scheduler.step()
            if current_epoch > 0 and (current_epoch + 1) % config.every_epoch == 0:
                self.log_epoch(classifier, current_epoch, train_acc, cls_loss)
            epochs_tqdm.set_postfix(val_best_acc=self.best_acc, test_best_acc=self.best_test_acc)

        return self.save_train(classifier, self.best_model)

    def train_epoch(self, model, optimizer, args, cls_criterion, train_loader, memory_static, memory_dynamic,
                    memory_labels, memory_cluster, ratio):
        model.train()
        epoch_loss = 0.
        class_num = args.num_class
        memory_labels = memory_labels.flatten()
        acc_list = []
        for ind, ((us_img, ceus_img, wash_in_images, wash_out_images), labels, index) in enumerate(train_loader):
            labels = Variable(labels).cuda()
            ceus_img = ceus_img.cuda()
            us_img = us_img.cuda()
            wash_in_images = wash_in_images.cuda()
            wash_out_images = wash_out_images.cuda()

            predictions, (x_us, x_ceus, wash_in, wash_out) = model(us_img, ceus_img, wash_in_images, wash_out_images)

            bs = us_img.shape[0]
            # compute output
            feats_us, feats_ceus = x_us, x_ceus
            feats_us_data = feats_us.detach()
            if wash_out is None:
                dynamics = wash_in
            else:
                dynamics = torch.cat([wash_in, wash_out], dim=1)
            if dynamics is None:
                x_ceus_dynamics = x_ceus
            else:
                x_ceus_dynamics = torch.cat((x_ceus, dynamics), dim=1)
            feats_ceus_data = x_ceus_dynamics.detach()
            features = torch.cat([feats_us, x_ceus_dynamics], dim=1)

            # update the memory module
            feats_us_data = feats_us_data.cpu()
            feats_ceus_data = feats_ceus_data.cpu()
            memory_static[index] = feats_us_data.cpu()
            memory_dynamic[index] = feats_ceus_data.cpu()

            # US--> CEUS
            mean_ceus, cv_ceus, cluster_size_ceus = Calculate_mean_cv(args, memory_dynamic, memory_static,
                                                                      memory_labels,
                                                                      memory_cluster[:, 0],
                                                                      class_num)
            mean_us, cv_us, cluster_size_us = Calculate_mean_cv(args, memory_static, memory_dynamic,
                                                                memory_labels,
                                                                memory_cluster[:, 1],
                                                                class_num)
            # 计算当前样本到不同类别中心的距离，以类别的协方差矩阵加权和为参数，生成新的样本
            if args.policy == 'ISDA_cluster':
                cv_us = cv_us.view((cv_us.shape[0] * cv_us.shape[1], cv_us.shape[2], cv_us.shape[3])).cuda()
                cv_ceus = cv_ceus.view((cv_ceus.shape[0] * cv_ceus.shape[1], cv_ceus.shape[2], cv_ceus.shape[3])).cuda()
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

            optimizer.zero_grad()
            cls_loss.backward()
            optimizer.step()
            predictions = torch.nn.functional.softmax(predictions, dim=1)
            _, y_pred = torch.max(predictions, 1)
            m_acc = accuracy_score(labels.detach().cpu(), y_pred.detach().cpu())
            acc_list.append(m_acc)
            epoch_loss += cls_loss.item()
        epoch_loss = epoch_loss / (ind + 1)
        return epoch_loss, np.mean(acc_list)

    def pre_train_epoch(self, model, optimizer, train_loader):
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
