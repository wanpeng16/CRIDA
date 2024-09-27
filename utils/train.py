import os

from torch.utils.tensorboard import SummaryWriter

from datasets.data_loader import create_dataloader

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import torch

from utils.train_base import TrainerTesterBase
from utils.utils import memory_module_init_update, Calculate_mean_cv, setup_seed
from module.loss import SACoFALoss
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import copy
from utils.utils import annealing
from utils.utils import weighted_cv
import tqdm


class TrainerTesterExcel(TrainerTesterBase):
    def __init__(self, config):

        super().__init__(config)

    def run_over(self, train_loader, val_loader, test_loader, memory_loader, val_only):
        config = self.config

        hidden_dims = config.hidden_dims
        classifier = self.Classifier(num_classes=config.num_class, us_dim=config.us_dim, ceus_dim=config.ceus_dim,
                                     hidden_dims=hidden_dims).cuda()
        if val_only:
            return self.evaluate(classifier)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=config.lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
        s_len = len(memory_loader.dataset)
        memory_us = torch.zeros(s_len, classifier.features_dim).cuda()

        memory_ceus = torch.zeros(s_len, classifier.features_dim * (config.dynamics_num + 1)).cuda()
        memory_labels = torch.zeros(s_len, 1).long().cuda()
        memory_cluster = torch.zeros(s_len, 2).long().cuda()
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
            Loss = ISDALoss(classifier.features_dim, class_num=config.num_class).cuda()
        else:
            Loss = SACoFALoss(class_num=config.num_class).cuda()
        epochs_tqdm = tqdm.tqdm(range(config.epochs))
        for current_epoch in epochs_tqdm:
            cls_loss = self.train_epoch(classifier, optimizer,
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

                val_acc, val_sen, val_spe, val_f1, val_auc = self.val_epoch(classifier,
                                                                            [val_loader])
                if best_acc < val_acc:
                    best_acc = val_acc
                    best_model = copy.deepcopy(classifier)

                test_acc, test_sen, test_spe, test_f1, test_auc = self.val_epoch(classifier,
                                                                                 [test_loader])
                if best_test_acc < test_acc:
                    best_test_model = copy.deepcopy(classifier)
                    best_test_acc = test_acc

                if self.tboard:
                    self.tboard.add_scalar('val acc', val_acc, current_epoch)
                    self.tboard.add_scalar('val auc', val_auc, current_epoch)
                    self.tboard.add_scalar('test acc', test_acc, current_epoch)
                    self.tboard.add_scalar('test auc', test_auc, current_epoch)

            epochs_tqdm.set_postfix(val_best_acc=best_acc, test_best_acc=best_test_acc)

        return self.save_train(classifier, best_model, best_test_model)

    def val_epoch(self, model, dataloaders, only_acc=True):
        # switch to train mode
        model.eval()
        loss_fun = torch.nn.CrossEntropyLoss(reduction="none").cuda()
        start_test = True
        with torch.no_grad():
            for dataloader in dataloaders:
                for i, (images, target, _) in enumerate(dataloader):
                    images, dynamics = images
                    images, dynamics = images.cuda(), dynamics.cuda()
                    images = Variable(images)
                    x_dynamics = Variable(dynamics)
                    target = Variable(target.cuda())
                    images = torch.squeeze(images, dim=1)
                    x_us, x_ceus = torch.split(images, [self.config.us_dim, self.config.ceus_dim], dim=1)

                    # get logit outputs
                    output, _ = model(x_us, x_ceus, x_dynamics)
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

    def train_epoch(self, model, optimizer, args, cls_criterion, train_loader, memory_static, memory_dynamic,
                    memory_labels, memory_cluster, ratio):
        model.train()
        epoch_loss = 0.
        class_num = args.num_class
        memory_labels = memory_labels.flatten()

        for ind, (images, labels, index) in enumerate(train_loader):
            images, dynamics = images
            images, dynamics = images.cuda(), dynamics.cuda()
            labels = labels.cuda()
            images = torch.squeeze(images, dim=1)
            images = Variable(images)
            labels = Variable(labels)
            labels = labels.flatten()
            x_dynamics = Variable(dynamics)
            bs = images.shape[0]
            # compute output
            x_us, x_ceus = torch.split(images, [args.us_dim, args.ceus_dim], dim=1)
            predictions, features = model(x_us, x_ceus, x_dynamics)
            feats_us, feats_ceus = features[0], features[1]
            feats_us_data = feats_us.detach()
            feats_ceus_data = feats_ceus.detach()
            features = torch.cat([feats_us, feats_ceus], dim=1)

            # update the memory module
            memory_static[index] = feats_us_data.cuda()
            memory_dynamic[index] = feats_ceus_data.cuda()

            # US--> CEUS
            mean_ceus, cv_ceus, cluster_size_ceus = Calculate_mean_cv(args, memory_dynamic, memory_static,
                                                                      memory_labels,
                                                                      memory_cluster[:, 0],
                                                                      class_num)
            mean_us, cv_us, cluster_size_us = Calculate_mean_cv(args, memory_static, memory_dynamic, memory_labels,
                                                                memory_cluster[:, 1],
                                                                class_num)

            # 计算当前样本到不同类别中心的距离，以类别的协方差矩阵加权和为参数，生成新的样本
            if args.policy == 'ISDA_cluster':
                cv_us = cv_us.view((cv_us.shape[0] * cv_us.shape[1], cv_us.shape[2], cv_us.shape[3]))
                cv_ceus = cv_ceus.view((cv_ceus.shape[0] * cv_ceus.shape[1], cv_ceus.shape[2], cv_ceus.shape[3]))
                CoVariance = torch.stack([torch.block_diag(cv_us[i], cv_ceus[i]) for i in range(cv_us.shape[0])])
                cls_loss = cls_criterion.loss(model.head, features, predictions, labels, CoVariance, ratio)

            else:
                weighted_cv_ceus = torch.zeros(bs,
                                               feats_ceus.shape[1],
                                               feats_ceus.shape[1]).cuda()

                weighted_cv_us = torch.zeros(bs, model.features_dim, model.features_dim).cuda()
                for i in range(class_num):
                    feats_ceus_c = feats_ceus_data[labels == i]
                    feats_us_c = feats_us_data[labels == i]
                    weighted_cv_ceus_c = weighted_cv(args=args, cluster_nums=args.num_clusters, feats=feats_us_c,
                                                     cluster_means=mean_ceus[i], cluster_cvs=cv_ceus[i],
                                                     cluster_size=cluster_size_ceus[i])
                    weighted_cv_us_c = weighted_cv(args=args, cluster_nums=args.num_clusters, feats=feats_ceus_c,
                                                   cluster_means=mean_us[i],
                                                   cluster_cvs=cv_us[i], cluster_size=cluster_size_us[i])
                    weighted_cv_ceus[labels == i] = weighted_cv_ceus_c
                    weighted_cv_us[labels == i] = weighted_cv_us_c

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
