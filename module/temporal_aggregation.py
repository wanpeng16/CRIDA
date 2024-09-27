from typing import List

import torch
from torch import nn

from module.arp import RankPooling
from module.modules import NONLocalBlock1D
from module.network import ClassifierBase


class ClassifierRankPoolingEarly(ClassifierBase):
    def __init__(self, num_classes: int, us_dim: int, ceus_dim: int, hidden_dims: List = None):
        """
         For Rank Pooling Early Classifier
        :param num_classes:
        :param us_dim:
        :param ceus_dim:
        :param hidden_dims:
        """
        super(ClassifierRankPoolingEarly, self).__init__(num_classes, us_dim, ceus_dim, hidden_dims)
        self.head = nn.Linear(self._features_dim * 4, num_classes)

    def forward(self, x_us: torch.Tensor, x_ceus: torch.Tensor, dynamics):
        x_us = self.us_encoder(x_us)
        x_ceus = self.ceus_encoder(x_ceus)
        wash_in, wash_out = torch.split(dynamics, [self.ceus_dim, self.ceus_dim], dim=1)
        wash_in = self.ceus_encoder(wash_in)
        wash_out = self.ceus_encoder(wash_out)
        x_ceus = torch.cat((x_ceus, wash_in, wash_out), dim=1)
        logits = self.head(torch.cat((x_us, x_ceus), dim=1))
        return logits, (x_us, x_ceus)


class ClassifierRankPoolingOnce(ClassifierBase):
    def __init__(self, num_classes: int, us_dim: int, ceus_dim: int, hidden_dims: List = None):
        super(ClassifierRankPoolingOnce, self).__init__(num_classes, us_dim, ceus_dim, hidden_dims)
        self.head = nn.Linear(self._features_dim * 3, num_classes)
        self.pooling = RankPooling()

    def forward(self, x_us: torch.Tensor, x_ceus: torch.Tensor, dynamics):
        x_us = self.us_encoder(x_us)
        x_ceus = self.ceus_encoder(x_ceus)
        bs, T, _ = dynamics.shape
        x_dynamics = self.ceus_encoder(torch.reshape(dynamics, [bs * T, -1]))  # batch * T , feature_size
        x_dynamics = torch.reshape(x_dynamics, [bs, T, -1])
        rank_pooling_dynamics = self.pooling(x_dynamics)
        x_ceus = torch.cat((x_ceus, rank_pooling_dynamics), dim=1)
        logits = self.head(torch.cat((x_us, x_ceus), dim=1))
        return logits, (x_us, x_ceus)


class ClassifierAveragePooling(ClassifierBase):
    def __init__(self, num_classes: int, us_dim: int, ceus_dim: int, hidden_dims: List = None):
        super(ClassifierAveragePooling, self).__init__(num_classes, us_dim, ceus_dim, hidden_dims)
        self.head = nn.Linear(self._features_dim * 3, num_classes)

    def forward(self, x_us: torch.Tensor, x_ceus: torch.Tensor, x_dynamics: torch.Tensor):
        x_us = self.us_encoder(x_us)
        x_ceus = self.ceus_encoder(x_ceus)
        bs, T, _ = x_dynamics.shape
        x_dynamics = self.ceus_encoder(torch.reshape(x_dynamics, [bs * T, -1]))
        x_dynamics = torch.reshape(x_dynamics, [bs, T, -1])
        mean_dynamics = torch.mean(x_dynamics, dim=1)
        x_ceus = torch.cat((x_ceus, mean_dynamics), dim=1)
        logits = self.head(torch.cat((x_us, x_ceus), dim=1))
        return logits, (x_us, x_ceus)


class ClassifierLSSLEarly(ClassifierBase):
    def __init__(self, num_classes: int, us_dim: int, ceus_dim: int, hidden_dims: List = None):
        super(ClassifierLSSLEarly, self).__init__(num_classes, us_dim, ceus_dim, hidden_dims)

        self.head = nn.Linear(self._features_dim * 3, num_classes)

    def forward(self, x_us: torch.Tensor, x_ceus: torch.Tensor, dynamics):
        x_us = self.us_encoder(x_us)
        x_ceus = self.ceus_encoder(x_ceus)
        dynamics = self.ceus_encoder(dynamics)
        x_ceus = torch.cat((x_ceus, dynamics), dim=1)
        logits = self.head(torch.cat((x_us, x_ceus), dim=1))
        return logits, (x_us, x_ceus)


class ClassifierSelfAttention(ClassifierBase):
    def __init__(self, num_classes: int, us_dim: int, ceus_dim: int, hidden_dims: List = None):
        super(ClassifierSelfAttention, self).__init__(num_classes, us_dim, ceus_dim, hidden_dims)
        self.head = nn.Linear(self._features_dim * 3, num_classes)
        self.self_attention = NONLocalBlock1D(self._features_dim)

    def forward(self, x_us: torch.Tensor, x_ceus: torch.Tensor, x_dynamics: torch.Tensor):
        x_us = self.us_encoder(x_us)
        x_ceus = self.ceus_encoder(x_ceus)
        bs, T, _ = x_dynamics.shape
        x_dynamics = self.ceus_encoder(torch.reshape(x_dynamics, [bs * T, -1]))
        x_dynamics = torch.reshape(x_dynamics, [bs, T, -1])
        x_dynamics = x_dynamics.permute(0, 2, 1)
        dynamics = self.self_attention(x_dynamics)  # batch * feature_size
        x_ceus = torch.cat((x_ceus, dynamics), dim=1)
        logits = self.head(torch.cat((x_us, x_ceus), dim=1))
        return logits, (x_us, x_ceus)
