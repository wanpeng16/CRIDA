import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import math
from torch.nn.utils import spectral_norm
import gc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utils.build import build_pooling_method

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
class EncoderBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_act='leaky_relu', num_conv=1):
        super(EncoderBlock, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif conv_act == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)
        if num_conv == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(out_channels),
                conv_act_layer
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(out_channels),
                conv_act_layer,
                nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(out_channels),
                conv_act_layer,
                nn.MaxPool2d(2)
            )

        self.init_model()

    def init_model(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        init.kaiming_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_act='leaky_relu', num_conv=1):
        super(DecoderBlock, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif conv_act == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        if num_conv == 1:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels,
                                   out_channels,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=(1, 1),
                                   output_padding=(1, 1)
                                   ),
                nn.BatchNorm2d(out_channels),
                conv_act_layer
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(out_channels),
                conv_act_layer,
                nn.ConvTranspose2d(out_channels,
                                   out_channels,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=(1, 1),
                                   output_padding=(1, 1)
                                   ),
                nn.BatchNorm2d(out_channels),
                conv_act_layer
            )
        self.init_model()

    def init_model(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        init.kaiming_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, image_size=(240, 296), hidden_dims=None, is_wasserstein=True):
        super(Discriminator, self).__init__()
        # Build Discriminator
        if hidden_dims is None:
            hidden_dims = [16, 16, 16, 16]

        modules = []
        for h_dim in hidden_dims:
            if is_wasserstein:
                modules.append(
                    spectral_norm(
                        nn.Conv2d(in_channels, h_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)))
            else:
                modules.append(
                    nn.Conv2d(in_channels, h_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False))
            modules.append(nn.BatchNorm2d(h_dim))
            modules.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = h_dim

        flatten_dim = hidden_dims[-1] * (image_size[0] // pow(2, len(hidden_dims))) * (
                image_size[1] // pow(2, len(hidden_dims)))

        modules.append(nn.Flatten(start_dim=1))
        if is_wasserstein:
            modules.append(spectral_norm(nn.Linear(flatten_dim, 1)))
            modules.append(nn.Tanh())
        else:
            modules.append(nn.Linear(flatten_dim, 1))
            modules.append(nn.Sigmoid())

        self.discriminator = nn.Sequential(*modules)

    def forward(self, x):
        return self.discriminator(x)


class Contrastive_loss(nn.Module):
    def __init__(self, label_varience=1.0):
        super(Contrastive_loss, self).__init__()
        self.label_varience = label_varience

    def safe_div(self, a, b):
        out = a / b
        out[torch.isnan(out)] = 0
        return out

    def embedding_sim(self, embedding, point, tic):
        distance = tic[:, point].unsqueeze(1) - tic
        pos_weight = torch.exp(-torch.square(distance) / (2 * self.label_varience)).type_as(tic)
        target_weight = self.safe_div(pos_weight, torch.sum(pos_weight, dim=1, keepdim=True))
        anchor = embedding[:, point].unsqueeze(1).repeat(1, tic.shape[1], 1)
        sc = nn.functional.cosine_similarity(anchor, embedding, dim=2) + 1.
        input_weight = self.safe_div(sc, torch.sum(sc, dim=1, keepdim=True))
        sim = F.kl_div(torch.log(input_weight + 1e-6), target_weight, reduction='batchmean')
        return sim

    def forward(self, embedding, tic):
        """
        :param embedding: [B, T, -1]
        :param tic: [B, T]
        :return:
        """
        tic = tic / torch.max(tic, dim=1, keepdim=True)[0]
        sim = [self.embedding_sim(embedding, t, tic) for t in range(tic.shape[1])]
        intra_sim_loss = torch.mean(torch.stack(sim))

        return intra_sim_loss


def annealing(epochs=200, anneal_frac_dynamics=1, anneal_start_dynamics=0.00001, anneal_end_dynamics=1):
    temp = np.arange(0, math.ceil(epochs // anneal_frac_dynamics))
    anneals_dynamics = anneal_start_dynamics + (anneal_end_dynamics - anneal_start_dynamics) * \
                       (np.sin(np.pi * temp / (epochs / anneal_frac_dynamics) + np.pi / 2) + 1) / 2
    anneals_dynamics = np.flip(np.concatenate((
        np.ones((anneal_frac_dynamics - 1) * epochs // anneal_frac_dynamics) * anneal_end_dynamics,
        anneals_dynamics)))

    return anneals_dynamics


# 计算每个类别的特征均值和特征协方差矩阵
def CalculateMean(features, labels, class_num):
    N = features.size(0)
    C = class_num
    A = features.size(1)

    avg_CxA = torch.zeros(C, A).cuda()
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C).cuda()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1.0

    # del onehot
    # gc.collect()
    for c in range(class_num):
        c_temp = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        c_temp = torch.sum(c_temp, dim=0)
        avg_CxA[c] = c_temp / Amount_CxA[c]
    return avg_CxA.detach()


def Calculate_CV(features, labels, ave_CxA, class_num):
    N = features.size(0)
    C = class_num
    A = features.size(1)

    var_temp = torch.zeros(C, A, A).cuda()
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C).cuda()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1
    Amount_CxAxA = Amount_CxA.view(C, A, 1).expand(C, A, A)
    # del Amount_CxA, onehot
    # gc.collect()

    avg_NxCxA = ave_CxA.expand(N, C, A)
    for c in range(C):
        features_by_sort_c = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        avg_by_sort_c = avg_NxCxA[:, c, :].mul(NxCxA_onehot[:, c, :])
        var_temp_c = features_by_sort_c - avg_by_sort_c
        var_temp[c] = torch.mm(var_temp_c.permute(1, 0), var_temp_c).div(Amount_CxAxA[c])
    return var_temp.detach()


def Calculate_mean_cv(args, memory_student, memory_teacher, memory_labels, memory_cluster, class_num):
    mean_ = []
    cv_ = []
    cluster_size_ = []
    # 计算每个类别的簇中心和协方差矩阵
    for i in range(class_num):
        memory_student_c = memory_student[memory_labels == i]
        memory_teacher_c = memory_teacher[memory_labels == i]
        # 利用超声图像的语义信息，对训练集进行聚类，得到K个类别
        # 计算每个类别的中心图像，作为该类别的代表；计算每个类别的协方差矩阵，表征类内增强模式变化
        labels_c = memory_cluster[memory_labels == i]
        mean_teacher_c = CalculateMean(memory_teacher_c, labels_c.long(), args.num_clusters)
        mean_student_c = CalculateMean(memory_student_c, labels_c.long(), args.num_clusters)

        cv_student_c = Calculate_CV(memory_student_c, labels_c.long(), mean_student_c, args.num_clusters)
        if memory_teacher.shape[-1] < memory_student.shape[-1]: # US 引导 CEUS
            cv_student_c_peak = cv_student_c[:, :memory_teacher.shape[-1], :memory_teacher.shape[-1]]
            cv_student_c_wash_in = cv_student_c[:, memory_teacher.shape[-1]:memory_teacher.shape[-1]*2, memory_teacher.shape[-1]:memory_teacher.shape[-1]*2]
            cv_student_c_wash_out = cv_student_c[:, memory_teacher.shape[-1]*2:, memory_teacher.shape[-1]*2:]
            cv_student_c = torch.stack([torch.block_diag(cv_student_c_peak[i], cv_student_c_wash_in[i], cv_student_c_wash_out[i]) for i in range(cv_student_c.shape[0])])

        cluster_size_.append(torch.bincount(labels_c, minlength=args.num_clusters))

        mean_.append(mean_teacher_c)
        cv_.append(cv_student_c)

    mean_ = torch.stack(mean_)  # [class_num, num_clusters, dim]
    cv_ = torch.stack(cv_)  # [class_num, num_clusters, dim, dim]

    return mean_, cv_, cluster_size_



# 初始化memory：计算source的256维特征
def memory_module_init_update(memory_loader, classifier, memory_static, memory_dynamic, memory_labels, memory_cluster,
                              config):
    classifier.eval()
    s_len = len(memory_loader.dataset)
    X_us = torch.zeros(s_len, config.us_dim).cuda()
    X_us = X_us.to(torch.float)

    X_ceus = None

    pooling = build_pooling_method(config.pooling)



    with torch.no_grad():
        for _, (images, labels, index) in enumerate(memory_loader):
            images, dynamics = images
            images, dynamics = images.cuda(), dynamics.cuda()
            labels = labels.cuda()
            images = torch.squeeze(images, dim=1)
            x_us, x_ceus = torch.split(images, [config.us_dim, config.ceus_dim], dim=1)
            X_us[index] = x_us
            pooling_dynamics = pooling(dynamics)
            x_ceus_dynamics = torch.cat((x_ceus, pooling_dynamics), dim=1)
            if X_ceus is None:
                X_ceus = torch.zeros(s_len, x_ceus_dynamics.shape[1]).cuda()
                X_ceus = X_ceus.to(torch.float)
            X_ceus[index] = x_ceus_dynamics
            flag = False

            if images.size(0) == 1:
                temp_iter = iter(memory_loader)
                (images_a, dynamics_a), _, _ = next(temp_iter)
                images_a, dynamics_a = images_a.cuda(), dynamics_a.cuda()
                images = torch.cat((images, images_a), dim=0)
                dynamics = torch.cat((dynamics, dynamics_a), dim=0)
                x_us, x_ceus = torch.split(images, [config.us_dim, config.ceus_dim], dim=1)
                flag = True

            _, features = classifier(x_us, x_ceus, dynamics)
            if flag:
                f_static = features[0]
                f_static = f_static[0]
                memory_static[index] = f_static.unsqueeze(0)
                f_dynamic = features[1]
                f_dynamic = f_dynamic[0]
                memory_dynamic[index] = f_dynamic.unsqueeze(0)
                memory_labels[index] = labels
            else:
                memory_static[index] = features[0]
                memory_dynamic[index] = features[1]
                memory_labels[index] = labels

        # 计算每个样本每个模态所属的类别
        y = memory_labels.flatten()
        for i in range(config.num_class):
            X_us_c = X_us[y == i]
            X_ceus_c = X_ceus[y == i]
            kmeans_static = KMeans(n_clusters=config.num_clusters)
            kmeans_dynamic = KMeans(n_clusters=config.num_clusters)
            kmeans_static.fit(X_us_c.detach().cpu().numpy())
            kmeans_dynamic.fit(X_ceus_c.detach().cpu().numpy())
            cluster_c = torch.cat((torch.from_numpy(kmeans_static.labels_).reshape(-1, 1),
                                   torch.from_numpy(kmeans_dynamic.labels_).reshape(-1, 1)), dim=1)
            memory_cluster[y == i] = cluster_c.to(torch.int64).cuda()


def weighted_cv(args, cluster_nums, feats, cluster_means, cluster_cvs, cluster_size):
    # 计算每个类别的权重系数
    dim = cluster_cvs.shape[-1]
    feats = feats.detach()
    weights = torch.exp(-1 * (torch.cdist(feats, cluster_means, p=2) / (2 * args.sigma))) * cluster_size
    cv_ = torch.matmul(weights, cluster_cvs.reshape(cluster_nums, -1))
    weights_norm = torch.sum(weights, dim=1, keepdim=True)
    cv_norm = cv_ / weights_norm
    cv_norm = cv_norm.reshape(-1, dim, dim)
    return cv_norm


if __name__ == "__main__":
    plt.plot(annealing(50, anneal_frac_dynamics=1, anneal_start_dynamics=0.001, anneal_end_dynamics=0.25))
    plt.show()

    # design a dynamic coefficient relied on the number of samples, the larger number of samples, the smaller
    # coefficient
    #
