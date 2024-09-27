# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: pyro-search-metastasis-rank-pooling
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/5/1
# @Time        : 下午8:45
# @Description :
import argparse
import pickle

import tqdm
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from config import running_config, config, dataset_config, model_config
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from utils.build import build_model, build_dataset
import torch

from utils.train_deep import TrainerTesterDeep
from utils.utils_vgg import memory_module_init_update, Calculate_mean_cv, setup_seed
from module.loss import SACoFALoss
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from utils.utils_vgg import annealing
from utils.utils_vgg import weighted_cv

import os
import shutil

import numpy as np

from visual.stylegan import get_model


class NoiseGenerator(nn.Module):
    def __init__(self, batch_size, z_dim, init_value=None):
        super(NoiseGenerator, self).__init__()
        self.batch_size = batch_size
        self.z_dim = z_dim
        if init_value is None:
            self.noise = torch.randn(1, z_dim, requires_grad=True)
        else:
            self.noise = init_value
        self.noise = torch.nn.Parameter(self.noise.expand(batch_size, z_dim).clone())

        # self.linear = nn.Linear(z_dim, z_dim)

    def forward(self):
        # noise = torch.randn(1, self.z_dim, requires_grad=True)
        # noise = noise.expand(self.batch_size, self.z_dim)
        # noise = torch.nn.Parameter(noise).cuda()
        output = self.noise

        n_mean = output.mean(dim=1).unsqueeze(1).expand(output.size(0),
                                                        output.size(1))
        n_std = output.std(dim=1).unsqueeze(1).expand(output.size(0),
                                                      output.size(1))
        output = (output - n_mean) / n_std
        return torch.nn.Parameter(torch.randn(self.batch_size, self.z_dim)).cuda()


def save_data_to_file(file_path, index_matrix, weighted_cv_us, weighted_cv_ceus):
    with open(file_path, 'wb') as file:
        data = {
            'index_matrix': index_matrix,
            'weighted_cv_us': weighted_cv_us,
            'weighted_cv_ceus': weighted_cv_ceus
        }
        pickle.dump(data, file)


# 从文件中读取数据
def load_data_from_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        index_matrix = data['index_matrix']
        weighted_cv_us = data['weighted_cv_us']
        weighted_cv_ceus = data['weighted_cv_ceus']

        return index_matrix, weighted_cv_us, weighted_cv_ceus


def val_epoch(model, args, dataloaders, only_acc=True):
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
        if only_acc:
            return m_acc, None, None, None, None
        m_f1 = f1_score(y_true, y_pred, pos_label=1, average='binary')
        m_auc = roc_auc_score(y_true, all_output[:, 1])
        cm = confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cm.ravel()
        m_spe = TN / (TN + FP)
        m_sen = TP / (TP + FN)

    return m_acc, m_sen, m_spe, m_f1, m_auc


def train_epoch(model, optimizer, args, cls_criterion, train_loader, memory_static, memory_dynamic,
                memory_labels, memory_cluster, ratio, log_epoch=False):
    model.train()
    epoch_loss = 0.
    class_num = args.num_class
    memory_labels = memory_labels.flatten()
    acc_list = []
    all_index = None
    all_wight_cv_us = None
    all_wight_cv_ceus = None
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
        mean_ceus, cv_ceus, cluster_size_ceus = Calculate_mean_cv(args, memory_dynamic, memory_static,
                                                                  memory_labels,
                                                                  memory_cluster[:, 0],
                                                                  class_num)
        mean_us, cv_us, cluster_size_us = Calculate_mean_cv(args, memory_static, memory_dynamic,
                                                            memory_labels,
                                                            memory_cluster[:, 1],
                                                            class_num)
        # 计算当前样本到不同类别中心的距离，以类别的协方差矩阵加权和为参数，生成新的样本

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
        if log_epoch:
            if all_index is None:
                all_index = index.cpu()
                all_wight_cv_us = weighted_cv_us.cpu()
                all_wight_cv_ceus = weighted_cv_ceus.cpu()
            else:
                all_index = torch.cat([all_index.cpu(), index.cpu()])
                all_wight_cv_us = torch.cat([all_wight_cv_us.cpu(), weighted_cv_us.cpu()])
                all_wight_cv_ceus = torch.cat([all_wight_cv_ceus.cpu(), weighted_cv_ceus.cpu()])
        CoVariance = torch.stack([torch.block_diag(weighted_cv_us[i], weighted_cv_ceus[i]) for i in range(bs)])

        cls_loss = cls_criterion(model.head, features, predictions, labels, CoVariance, ratio)

        assert not torch.isnan(cls_loss), 'Model diverged with loss = NaN'

        # compute gradient and do SGD step
        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()

        predictions = torch.nn.functional.softmax(predictions, dim=1)
        _, y_pred = torch.max(predictions, 1)
        m_acc = accuracy_score(labels.detach().cpu(), y_pred.detach().cpu())
        acc_list.append(m_acc)
        epoch_loss += cls_loss.item()

    epoch_loss = epoch_loss / (ind + 1)
    if log_epoch:
        if not os.path.exists(opt.ckpt_path):
            os.makedirs(opt.ckpt_path)
        save_data_to_file(os.path.join(args.ckpt_path, 'cov.pkl'), all_index, all_wight_cv_us, all_wight_cv_ceus)

    return epoch_loss, np.mean(acc_list)


class Visualization(TrainerTesterDeep):
    def __init__(self, config, log_epoch):
        super().__init__(config)
        self.log_epoch = log_epoch

    def run_over(self, train_loader, memory_loader,*args):
        config = self.config
        self.model_save_path = self.config.ckpt_path
        hidden_dims = config.hidden_dims
        Classifier = build_model(config.model)
        classifier = Classifier(num_classes=config.num_class, us_dim=config.us_dim, ceus_dim=config.ceus_dim,
                                hidden_dims=hidden_dims).cuda()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=config.lr,
                                     weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
        pretrain_epoch = tqdm.tqdm(range(config.pretrain_epochs))
        pretrain_epoch.set_description_str("Pretraining model")
        for current_epoch in pretrain_epoch:
            cls_loss, train_acc = self.pre_train_epoch(classifier, optimizer, train_loader)
        train_acc, _, _, _, _ = val_epoch(classifier, config, [train_loader])

        print(f"Pretrain model finished: train_acc  {train_acc:.4f}")

        s_len = len(memory_loader.dataset)
        memory_us = torch.zeros(s_len, config.us_dim)

        memory_ceus = torch.zeros(s_len, config.ceus_dim * (config.dynamics_num + 1))
        memory_labels = torch.zeros(s_len).long()
        memory_cluster = torch.zeros(s_len, 2).long()
        # 初始化，聚类
        memory_module_init_update(memory_loader, classifier, memory_us, memory_ceus, memory_labels, memory_cluster,
                                  config)

        ratios = annealing(epochs=config.epochs * len(train_loader), anneal_end_dynamics=config.lambda0)

        Loss = SACoFALoss(class_num=config.num_class).cuda()
        epochs_tqdm = range(self.log_epoch + 1)
        for current_epoch in epochs_tqdm:
            cls_loss, train_acc = train_epoch(classifier, optimizer,
                                              config,
                                              Loss,
                                              train_loader,
                                              memory_us, memory_ceus,
                                              memory_labels, memory_cluster,
                                              ratios[current_epoch],
                                              current_epoch == self.log_epoch
                                              )

            print("Epoch %d: train Acc %.4f" % (current_epoch, train_acc))
            if self.log_epoch == current_epoch:
                test_acc, test_sen, test_spe, test_f1, test_auc = val_epoch(classifier, config,
                                                                            [train_loader])
                result = {}
                result['model'] = classifier.state_dict()
                result['acc'] = test_acc
                result['sen'] = test_sen
                result['spe'] = test_spe
                result['f1'] = test_f1
                result['auc'] = test_auc
                result['seed'] = self.config.seed
                # print("\n Test Result: Acc:{:.3f} F1-score:{:.3f} auc:{:.3f}".format(test_acc, test_f1, test_auc))
                if not os.path.exists(opt.ckpt_path):
                    os.makedirs(opt.ckpt_path)
                torch.save(result,
                           os.path.join(self.config.ckpt_path, f'Classifier_resnet_epoch_{current_epoch}.pth'))
            scheduler.step()

    def run_train(self):
        setup_seed(self.config.seed)
        config = self.config
        DataSet = build_dataset(opt.dataset)
        dset_train = DataSet(root=opt.dset_dir, subset='all', type=type, seed=opt.seed)
        train_loader = DataLoader(dset_train,
                                  shuffle=True,
                                  batch_size=opt.batch_size,
                                  num_workers=opt.num_workers
                                  )

        memory_loader = DataLoader(dset_train,
                                   batch_size=64,
                                   shuffle=False,
                                   num_workers=opt.num_workers)

        self.run_over(train_loader, memory_loader)

    def load_model(self):
        config = self.config
        us_generator = get_model(self.config.us_gan_model)
        ceus_generator = get_model(self.config.ceus_gan_model)

        feature_extractor_wight = torch.load(self.config.ckpt_wight_model)

        Classifier = build_model(config.model)
        classifier = Classifier(num_classes=config.num_class, us_dim=config.us_dim, ceus_dim=config.ceus_dim,
                                hidden_dims=config.hidden_dims).cuda()
        classifier.load_state_dict(feature_extractor_wight['model'])
        return classifier, us_generator, ceus_generator

    def visual(self):
        config = self.config
        config.ckpt_wight_model = os.path.join(self.config.ckpt_path, f'Classifier_resnet_epoch_{log_epoch}.pth')
        config.cov_path = os.path.join(self.config.ckpt_path, 'cov.pkl')
        tf = SummaryWriter(log_dir=config.output_path)

        # Step 1: Load dataset and pretrain model

        DataSet = build_dataset(config.dataset)
        dataset = DataSet(root=config.dset_dir, subset='all', type=type, seed=config.seed, transform=False,
                          cache=False)

        index = dataset.get_by_id(config.raw_image_id, config.class_id)
        assert len(
            index) != 0, f"There is no image with id {config.raw_image_id} and label {config.class_id} in train set"
        index = index[0]
        (us_img, ceus_img, wash_in_images, wash_out_images), label, index = dataset[index]
        save_real_img = dataset.get_original_image(us_img)
        save_real_img.save(os.path.join(config.output_path, f'real_us.png'))
        save_real_img_ceus = dataset.get_original_image(ceus_img)
        save_real_img_ceus.save(os.path.join(config.output_path, f'real_ceus.png'))
        tf.add_images("real", np.hstack([np.array(save_real_img), np.array(save_real_img_ceus)]), dataformats='HWC')

        inputs_us = torch.cat([us_img.unsqueeze(0), us_img.unsqueeze(0)], dim=0)
        inputs_ceus = torch.cat([ceus_img.unsqueeze(0), ceus_img.unsqueeze(0)], dim=0)

        classifier, us_generator, ceus_generator = self.load_model()
        classifier = classifier.cpu()
        for module in classifier.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eps = 1e-3
        us_generator.train()
        for param in us_generator.parameters():
            param.requires_grad = False

        ceus_generator.train()
        for param in ceus_generator.parameters():
            param.requires_grad = False
        classifier.train()
        # for param in classifier.parameters():
        #     param.requires_grad = False
        classifier.us_encoder = classifier.us_encoder.cuda()
        classifier.ceus_encoder = classifier.ceus_encoder.cuda()
        features_ini = classifier.us_encoder(inputs_us.cuda())
        ceus_features_ini = classifier.ceus_encoder(inputs_ceus.cuda())
        features_ini = features_ini.cuda()
        ceus_features_ini = ceus_features_ini.cuda()
        resize_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        label = torch.zeros([1, us_generator.c_dim]).cuda()
        label[:, config.gan_class_id] = 1
        # Step 2: Search for input noise similar to the original image

        min_loss_samples = {}
        min_loss_ceus_samples = {}

        ceus_noise_vector = NoiseGenerator(2, us_generator.z_dim).cuda()
        ceus_noise_vector.train()
        noise_vector = NoiseGenerator(2, us_generator.z_dim).cuda()
        noise_vector.train()
        batch_label = label.expand(2, us_generator.c_dim).float().cuda()

        mse_loss = torch.nn.MSELoss(reduction='mean')
        params = list(noise_vector.parameters()) + list(ceus_noise_vector.parameters())
        opt1 = optim.Adam(params, lr=opt.noise_find_epoch, weight_decay=1e-4)


        epoch_tqdm = tqdm.tqdm(range(config.noise_find_epoch))
        epoch_tqdm.set_description_str("Step 1 Fine Tune :")
        save_dir = os.path.join(config.output_path, "Step1",'US')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        best_dir = os.path.join(config.output_path, "Step1",'best','US')
        if not os.path.exists(best_dir):
            os.makedirs(best_dir)

        ceus_save_dir = os.path.join(config.output_path, "Step1", 'CEUS')
        if not os.path.exists(ceus_save_dir):
            os.makedirs(ceus_save_dir)
        ceus_best_dir = os.path.join(config.output_path, "Step1", 'best', 'CEUS')
        if not os.path.exists(ceus_best_dir):
            os.makedirs(ceus_best_dir)
        for epoch in epoch_tqdm:
            fake_img = us_generator(noise_vector(), batch_label, truncation_psi=1, noise_mode='const')
            ceus_fake_img = ceus_generator(ceus_noise_vector(), batch_label, truncation_psi=1, noise_mode='const')

            fake_img_224 = resize_transform((fake_img * 127.5 + 128).clamp(0, 255).div(255))
            ceus_fake_img_224 = resize_transform((ceus_fake_img * 127.5 + 128).clamp(0, 255).div(255))

            fake_img_224.require_grad = True
            ceus_fake_img_224.require_grad = True

            feature_fake_img = classifier.us_encoder(fake_img_224)
            ceus_feature_fake_img = classifier.ceus_encoder(ceus_fake_img_224)

            loss_f_us = mse_loss(feature_fake_img, features_ini)
            loss_f_ceus = mse_loss(ceus_feature_fake_img, ceus_features_ini)
            loss_a = loss_f_us + loss_f_ceus
            epoch_tqdm.set_postfix(loss_f_us=loss_f_us.item(), loss_f_ceus=loss_f_ceus.item(),
                                  loss_a=loss_a.item())
            tf.add_scalar("loss/loss_a", loss_a.item(), epoch)
            tf.add_scalar("loss/loss_f_us", loss_f_us.item(), epoch)
            tf.add_scalar("loss/loss_f_ceus", loss_f_ceus.item(), epoch)
            opt1.zero_grad()
            loss_a.backward(retain_graph=True)  # retain_graph=True
            opt1.step()
            if len(min_loss_samples) < config.visual_num or loss_f_us < max(min_loss_samples.keys()):
                loss_f_us = loss_f_us.cpu()
                filenames = []
                save_fake_img_us = []
                fake_img_224 = fake_img_224.cpu()
                for i in range(fake_img_224.size(0)):
                    filename = f"sample_loss{loss_f_us}_epoch{epoch}_ind{i}.png"
                    save_fake_img = dataset.get_original_image(fake_img_224[i])
                    save_fake_img.save(os.path.join(save_dir, filename))
                    filenames.append(filename)
                    save_fake_img_us.append(np.array(save_fake_img))
                tf.add_images("loss_images", np.hstack(save_fake_img_us), epoch, dataformats='HWC')
                # current_min_loss = loss_f_us.item()
                min_loss_samples[loss_f_us.item()] = filenames
                if len(min_loss_samples) > config.visual_num:
                    max_loss = max(min_loss_samples.keys())
                    min_loss_samples.pop(max_loss)
            if len(min_loss_ceus_samples) < config.visual_num or loss_f_ceus < max(min_loss_ceus_samples.keys()):
                loss_f_ceus = loss_f_ceus.cpu()
                filenames = []
                save_fake_img_us = []
                fake_img_224 = ceus_fake_img_224.cpu()
                for i in range(fake_img_224.size(0)):
                    filename = f"sample_loss{loss_f_ceus}_epoch{epoch}_ind{i}.png"
                    save_fake_img = dataset.get_original_image(fake_img_224[i])
                    save_fake_img.save(os.path.join(ceus_save_dir, filename))
                    filenames.append(filename)
                    save_fake_img_us.append(np.array(save_fake_img))
                tf.add_images("loss_images", np.hstack(save_fake_img_us), epoch, dataformats='HWC')
                min_loss_ceus_samples[loss_f_ceus.item()] = filenames
                if len(min_loss_ceus_samples) > config.visual_num:
                    max_loss = max(min_loss_ceus_samples.keys())
                    min_loss_ceus_samples.pop(max_loss)

        for loss,filenames in min_loss_samples.items():
            for i,filename in enumerate(filenames):
                src_path = os.path.join(save_dir, filename)
                filename = f"sample_loss{loss}_ind{i}.png"
                shutil.copy(src_path,os.path.join(best_dir,filename))
        for loss,filenames in min_loss_ceus_samples.items():
            for i,filename in enumerate(filenames):
                src_path = os.path.join(ceus_save_dir, filename)
                filename = f"sample_loss{loss}_ind{i}.png"
                shutil.copy(src_path,os.path.join(ceus_best_dir,filename))
        # Step 3: Enhance features with covariance
        all_index, all_wight_cv_us, all_wight_cv_ceus = load_data_from_file(
            os.path.join(self.config.ckpt_path, 'cov.pkl'))
        all_index = all_index.cpu()
        all_wight_cv_us = all_wight_cv_us.cpu()
        all_wight_cv_ceus = \
        torch.split(torch.split(all_wight_cv_ceus.cpu(), [config.ceus_dim, config.ceus_dim, config.ceus_dim], dim=1)[0],
                    [config.ceus_dim, config.ceus_dim, config.ceus_dim], dim=2)[0]
        wight_cv_us = all_wight_cv_us[np.where(all_index == index)][0]

        wight_cv_ceus = all_wight_cv_ceus[np.where(all_index == index)][0]

        features_ini = features_ini[0]
        feature_num = features_ini.shape[0]
        batch_features_ini = features_ini.repeat(config.batch_size, 1).float().cuda()
        feature_objective_batch = batch_features_ini.clone()

        ceus_features_ini = ceus_features_ini[0]
        feature_num = ceus_features_ini.shape[0]
        ceus_batch_features_ini = ceus_features_ini.repeat(config.batch_size, 1).float().cuda()
        ceus_feature_objective_batch = ceus_batch_features_ini.clone()

        min_loss_samples = {}
        min_loss_ceus_samples = {}
        save_dir = os.path.join(config.output_path, "Step2",'US')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        best_dir = os.path.join(config.output_path, "Step2",'best','US')
        if not os.path.exists(best_dir):
            os.makedirs(best_dir)

        ceus_save_dir = os.path.join(config.output_path, "Step2", 'CEUS')
        if not os.path.exists(ceus_save_dir):
            os.makedirs(ceus_save_dir)
        ceus_best_dir = os.path.join(config.output_path, "Step2", 'best', 'CEUS')
        if not os.path.exists(ceus_best_dir):
            os.makedirs(ceus_best_dir)

        for i in range(config.batch_size):
            # Sampling enhancement based on covariance
            aug_np = np.random.multivariate_normal([0 for _ in range(feature_num)], config.alpha * wight_cv_us.cpu())
            aug = torch.Tensor(aug_np).float().cuda()
            batch_features_ini[i] = (batch_features_ini[i] + aug).detach()

            aug_np = np.random.multivariate_normal([0 for _ in range(feature_num)], config.alpha * wight_cv_ceus.cpu())
            aug = torch.Tensor(aug_np).float().cuda()
            ceus_batch_features_ini[i] = (ceus_batch_features_ini[i] + aug).detach()

        # Step 4: Search for input noise with similar features
        batch_noice_vector = NoiseGenerator(config.batch_size, us_generator.z_dim, noise_vector.noise[0].clone()).cuda()
        ceus_batch_noice_vector = NoiseGenerator(config.batch_size, us_generator.z_dim, noise_vector.noise[0].clone()).cuda()

        batch_label = label.expand(config.batch_size, us_generator.c_dim).float().cuda()
        mse_loss = torch.nn.MSELoss(reduction='mean')
        params = list(batch_noice_vector.parameters())+list(ceus_batch_noice_vector.parameters())
        opt2 = optim.Adam(params, lr=opt.noise_find_epoch, weight_decay=1e-4)

        search_epoch_tqdm = tqdm.tqdm(range(config.search_epoch))
        for epoch in search_epoch_tqdm:
            fake_img_batch = us_generator(batch_noice_vector(), batch_label, truncation_psi=1, noise_mode='const')
            ceus_fake_img_batch = ceus_generator(ceus_batch_noice_vector(), batch_label, truncation_psi=1, noise_mode='const')

            batch_fake_img_224 = resize_transform((fake_img_batch * 127.5 + 128).clamp(0, 255).div(255))
            ceus_batch_fake_img_224 = resize_transform((ceus_fake_img_batch * 127.5 + 128).clamp(0, 255).div(255))


            batch_fake_img_224.require_grad = True

            feature_fake_img = classifier.us_encoder(batch_fake_img_224)
            ceus_feature_fake_img = classifier.ceus_encoder(ceus_batch_fake_img_224)

            loss_us = mse_loss(feature_fake_img, feature_objective_batch)
            loss_ceus = mse_loss(ceus_feature_fake_img, ceus_feature_objective_batch)
            loss = loss_us+loss_ceus
            opt2.zero_grad()
            loss.backward(retain_graph=True)
            opt2.step()
            search_epoch_tqdm.set_postfix(loss=loss.item())
            if len(min_loss_samples) < config.visual_num or loss_us < max(min_loss_samples.keys()):
                loss_us = loss_us.cpu()
                filenames = []
                save_fake_img_us = []
                fake_img_224 = batch_fake_img_224.cpu()
                for i in range(fake_img_224.size(0)):
                    filename = f"sample_loss{loss_us}_epoch{epoch}_ind{i}.png"
                    save_fake_img = dataset.get_original_image(fake_img_224[i])
                    save_fake_img.save(os.path.join(save_dir, filename))
                    filenames.append(filename)
                    save_fake_img_us.append(np.array(save_fake_img))
                tf.add_images("loss_images", np.hstack(save_fake_img_us), epoch, dataformats='HWC')
                # current_min_loss = loss_f_us.item()
                min_loss_samples[loss_us.item()] = filenames
                if len(min_loss_samples) > config.visual_num:
                    max_loss = max(min_loss_samples.keys())
                    min_loss_samples.pop(max_loss)
            if len(min_loss_ceus_samples) < config.visual_num or loss_ceus < max(min_loss_ceus_samples.keys()):
                loss_ceus = loss_ceus.cpu()
                filenames = []
                save_fake_img_us = []
                fake_img_224 = ceus_batch_fake_img_224.cpu()
                for i in range(fake_img_224.size(0)):
                    filename = f"sample_loss{loss_ceus}_epoch{epoch}_ind{i}.png"
                    save_fake_img = dataset.get_original_image(fake_img_224[i])
                    save_fake_img.save(os.path.join(ceus_save_dir, filename))
                    filenames.append(filename)
                    save_fake_img_us.append(np.array(save_fake_img))
                tf.add_images("loss_images", np.hstack(save_fake_img_us), epoch, dataformats='HWC')
                min_loss_ceus_samples[loss_ceus.item()] = filenames
                if len(min_loss_ceus_samples) > config.visual_num:
                    max_loss = max(min_loss_ceus_samples.keys())
                    min_loss_ceus_samples.pop(max_loss)

        for loss, filenames in min_loss_samples.items():
            for i, filename in enumerate(filenames):
                src_path = os.path.join(save_dir, filename)
                filename = f"sample_loss{loss}_ind{i}.png"
                shutil.copy(src_path, os.path.join(best_dir, filename))
        for loss, filenames in min_loss_ceus_samples.items():
            for i, filename in enumerate(filenames):
                src_path = os.path.join(ceus_save_dir, filename)
                filename = f"sample_loss{loss}_ind{i}.png"
                shutil.copy(src_path, os.path.join(ceus_best_dir, filename))


def build(config):
    default_config = config.copy()
    opt = argparse.Namespace(**default_config)
    opt.path = os.path.join(opt.dset_dir, opt.dset_name)
    return opt


if __name__ == '__main__':
    dataset = 'liver_full'
    config.update(running_config[f'pyro_{dataset}'])
    config.update(dataset_config[config['dataset']])
    config.update(model_config[config['model']])
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
    config['seed'] = 0
    opt = build(config)
    opt.ckpt_path = os.path.join(opt.ckpt_dir, str(opt.seed))
    # for breast
    opt.us_gan_model = './ckpt/gan/us_network-snapshot-000140.pkl'
    opt.ceus_gan_model = './ckpt/gan/ceus_network-snapshot-000380.pkl'
    opt.output_path = './out/breast'
    opt.batch_size = 2
    opt.raw_image_id = 47
    opt.class_id = 0

    # for liver
    if dataset == 'liver_full':
        opt.us_gan_model = './ckpt/gan/liver/us_network-snapshot-000280.pkl'
        opt.ceus_gan_model = './ckpt/gan/liver/ceus_network-snapshot-000140.pkl'
        opt.output_path = './out/liver'
        opt.raw_image_id = 10
        opt.class_id = 1
    opt.output_path = os.path.join(opt.output_path,str(opt.class_id),str(opt.raw_image_id))
    print(opt.output_path)
    log_epoch = 50
    opt.noise_find_epoch = 500
    opt.noise_find_lr = 0.003

    opt.noise_find_schedule = range(0, 50, opt.noise_find_epoch)
    opt.print_freq = 5


    # opt.raw_image_id = 165
    # opt.class_id = 1
    opt.gan_class_id = 1
    opt.eta = 0
    opt.visual_num = 100
    opt.alpha = 0.2

    opt.search_epoch = 10000
    opt.search_lr = 0.0003

    visual = Visualization(opt, log_epoch=log_epoch)
    # Stage 1, Train model and get covariance for every train data
    if not os.path.exists(os.path.join(opt.ckpt_path, 'cov.pkl')):
        print("Run Stage 1")
        visual.run_train()
    else:
        print("Skip Stage 1")
    # Stage 2, Fine tune gan model and feature model, generate fake image
    visual.visual()
