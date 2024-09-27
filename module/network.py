import torch
import torch.nn as nn
from typing import List
from module.arp import RankPooling


class EncoderBlock(nn.Module):
    """(FC => [BN] => ReLU) * 2"""

    def __init__(self, in_dim, out_dim, act='relu'):
        super(EncoderBlock, self).__init__()
        if act == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act == 'leaky_relu':
            act_layer = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError('No implementation of ', act)

        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            act_layer
        )

    def init_model(self):
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_uniform_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        return self.fc(x)


class ClassifierBase(nn.Module):
    def __init__(self, num_classes: int, us_dim: int, ceus_dim: int, hidden_dims: List = None):
        super(ClassifierBase, self).__init__()

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
            modules_us.append(EncoderBlock(in_dim=in_dim, out_dim=h_dim))
            in_dim = h_dim

        self.us_encoder = nn.Sequential(*modules_us)

        modules_ceus = []
        in_dim = ceus_dim

        for h_dim in hidden_dims:
            modules_ceus.append(EncoderBlock(in_dim=in_dim, out_dim=h_dim))
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
