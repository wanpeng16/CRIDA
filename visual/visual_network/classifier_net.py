# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: pyro-search-metastasis-rank-pooling
# @Author      : Shukang Zhang  
# @Owner       : fusheng
# @Data        : 2024/5/2
# @Time        : 下午3:45
# @Description :
import torch
import torch.nn as nn
import torch.optim as optim

from module.arp import RankPooling


class VGGNet(nn.Module):
    def __init__(self,out_features):
        super(VGGNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, out_features)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ClassifierVGGNet(nn.Module):
    def __init__(self, num_classes: int, us_dim: int, ceus_dim: int, hidden_dims: List = None):
        super(ClassifierVGGNet, self).__init__()

        self.num_classes = num_classes
        self.us_dim = us_dim
        self.ceus_dim = ceus_dim

        # build the FC layers
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self._features_dim = hidden_dims[-1]

        modules_us = []
        in_dim = us_dim

        for h_dim in hidden_dims:
            modules_us.append(VGGNet(out_features=self._features_dim))
            in_dim = h_dim

        self.us_encoder = nn.Sequential(*modules_us)

        modules_ceus = []
        in_dim = ceus_dim

        for h_dim in hidden_dims:
            modules_ceus.append(VGGNet(out_features=self._features_dim))
            in_dim = h_dim

        self.ceus_encoder = nn.Sequential(*modules_ceus)
        self.head = nn.Linear(self._features_dim * 4, num_classes)
        self.pooling = RankPooling()

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def encode_feature(self, x_us, x_ceus, x_dynamics):
        x_us = self.us_encoder(x_us)
        x_ceus = self.ceus_encoder(x_ceus)
        bs, T, _ = x_dynamics.shape
        x_dynamics = self.ceus_encoder(torch.reshape(x_dynamics, [bs * T, -1]))
        x_dynamics = torch.reshape(x_dynamics, [bs, T, -1])
        wash_in = self.pooling(x_dynamics[:, :10, :])
        wash_out = self.pooling(x_dynamics[:, 10:, :])
        return x_us, x_ceus, wash_in, wash_out

    def classifier(self, x_us, x_ceus):
        logits = self.head(torch.cat((x_us, x_ceus), dim=1))
        return logits, (x_us, x_ceus)

    def forward(self, x_us: torch.Tensor, x_ceus: torch.Tensor, x_dynamics: torch.Tensor):
        x_us, x_ceus, wash_in, wash_out = self.encode_feature(x_us, x_ceus, x_dynamics)
        x_ceus = torch.cat((x_ceus, wash_in, wash_out), dim=1)
        logits, (x_us, x_ceus) = self.classifier(x_us, x_ceus)
        return logits, (x_us, x_ceus)

    @features_dim.setter
    def features_dim(self, value):
        self._features_dim = value
