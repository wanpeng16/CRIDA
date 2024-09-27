from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, resnet50
from torchvision.models import alexnet
from torchvision.models import resnet18
from torchvision.models import vit_b_16
from module.arp import RankPooling
from module.network import ClassifierBase

class AlexNetEncoder(nn.Module):
    def __init__(self,output_feature=512,dropout=0.5):
        super(AlexNetEncoder, self).__init__()

        self.features = alexnet(pretrained=False).features[:-1]  # 使用预训练的 VGG16 模型的特征提取部分
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.mlp = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_feature),
        )
        self.fc = nn.Linear(7 * 7 * 512, 512)  # 添加一个全连接层将特征映射到 512 维度

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x

class VGGEncoder(nn.Module):
    def __init__(self,output_feature=512,p=0.5):
        super(VGGEncoder, self).__init__()

        self.features = vgg16(pretrained=True).features[:-1]
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.mlp = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=p),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=p),
            nn.Linear(4096, output_feature),
        )
        self.fc = nn.Linear(7 * 7 * 512, 512)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x

class ResnetEncoder(nn.Module):
    def __init__(self,output_feature=512):
        super(ResnetEncoder, self).__init__()

        self.net = resnet18(weights= models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        # self.layer2.requires_grad_(False)

        self.layer3 = self.net.layer3
        # self.layer3.requires_grad_(False)

        self.layer4 = self.net.layer4
        # self.layer4.requires_grad_(False)

        self.avgpool = self.net.avgpool
        self.fc = nn.Linear(self.net.fc.in_features, out_features=output_feature)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x
    def forward(self,x):
        return self._forward_impl(x)

class Resnet50Encoder(ResnetEncoder):
    def __init__(self,output_feature=512,p=0.5):
        super(Resnet50Encoder, self).__init__()

        self.net = resnet50(pretrained=True)
        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2

        self.layer3 = self.net.layer3

        self.layer4 = self.net.layer4
        self.avgpool = self.net.avgpool

class Vit16Encoder(nn.Module):
    def __init__(self,output_feature=512,p=0.5):
        super(Vit16Encoder, self).__init__()

        self.net = vit_b_16(pretrained=True)
        self._process_input = self.net._process_input
        self.class_token = self.net.class_token
        self.encoder = self.net.encoder
        self.heads = self.net.heads
        self.fc = nn.Linear(1000, out_features=output_feature)

    def _forward_impl(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)
        x = self.fc(x)

        return x
    def forward(self,x):
        return self._forward_impl(x)
class ClassifierVGGBase(nn.Module):
    def __init__(self, num_classes: int, us_dim: int, ceus_dim: int, hidden_dims: List = None):
        super(ClassifierVGGBase, self).__init__()

        self.num_classes = num_classes
        self.us_dim = us_dim
        self.ceus_dim = ceus_dim

        # build the FC layers
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self._features_dim = hidden_dims[-1]

        modules_us = []
        Encode = ResnetEncoder
        modules_us.append(Encode(us_dim))
        modules_us.append(nn.ReLU())
        modules_us.append(nn.BatchNorm1d(us_dim))
        modules_us.append(nn.Dropout(p=0.5))
        self.us_encoder = nn.Sequential(*modules_us)
        modules_ceus = []

        modules_ceus.append(Encode(ceus_dim))
        modules_ceus.append(nn.ReLU())
        # modules_ceus.append(nn.BatchNorm1d(ceus_dim))
        modules_ceus.append(nn.Dropout(p=0.5))
        self.ceus_encoder = nn.Sequential(*modules_ceus)

        # modules_ceus = []
        # modules_ceus.append(Encode(ceus_dim))
        # modules_ceus.append(nn.ReLU())
        # # modules_ceus.append(nn.BatchNorm1d(ceus_dim))
        # modules_ceus.append(nn.Dropout(p=0.5))
        self.dynamics_encoder = nn.Sequential(*modules_ceus)
        self.head = nn.Linear(self.us_dim+self.ceus_dim * 3, num_classes)
        self._features_dim = self.us_dim
        self.pooling = RankPooling()

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def encode_feature(self, x_us, x_ceus, wash_in,wash_out):
        x_us = self.us_encoder(x_us)
        x_ceus = self.ceus_encoder(x_ceus)

        bs, T, c,h,w = wash_in.shape
        x_wash_in = self.ceus_encoder(torch.reshape(wash_in, [bs * T, c,h,w]))
        x_wash_in = torch.reshape(x_wash_in, [bs, T, -1])
        wash_in = self.pooling(x_wash_in)

        bs, T, c,h,w = wash_out.shape
        x_wash_out = self.ceus_encoder(torch.reshape(wash_out, [bs * T, c, h, w]))
        x_wash_out = torch.reshape(x_wash_out, [bs, T, -1])
        wash_out = self.pooling(x_wash_out)
        return x_us, x_ceus, wash_in, wash_out

    def classifier(self, x_us, x_ceus, wash_in, wash_out):
        x_ceus_dynamics = torch.cat((x_ceus, wash_in, wash_out), dim=1)
        logits = self.head(torch.cat((x_us, x_ceus_dynamics), dim=1))
        return logits, ( x_us, x_ceus, wash_in, wash_out)

    def classifier_us_ceus(self, x_us, x_ceus):
        logits = self.head(torch.cat((x_us, x_ceus), dim=1))
        return logits, (x_us, x_ceus)
    def forward(self,x_us, x_ceus, wash_in_images, wash_out_images):
        x_us, x_ceus, wash_in, wash_out = self.encode_feature(x_us, x_ceus, wash_in_images,wash_out_images)
        logits, ( x_us, x_ceus, wash_in, wash_out) = self.classifier(x_us, x_ceus, wash_in, wash_out)
        return logits, ( x_us, x_ceus, wash_in, wash_out)

    @features_dim.setter
    def features_dim(self, value):
        self._features_dim = value


class ClassifierVGGUSBase(nn.Module):
    def __init__(self, num_classes: int, us_dim: int, ceus_dim: int, hidden_dims: List = None):
        super(ClassifierVGGUSBase, self).__init__()

        self.num_classes = num_classes
        self.us_dim = us_dim
        self.ceus_dim = ceus_dim

        # build the FC layers
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self._features_dim = hidden_dims[-1]

        modules_us = []
        Encode = ResnetEncoder
        modules_us.append(Encode(us_dim))
        modules_us.append(nn.ReLU())
        modules_us.append(nn.BatchNorm1d(us_dim))
        modules_us.append(nn.Dropout(p=0.5))
        self.us_encoder = nn.Sequential(*modules_us)


        self.head = nn.Linear(self.us_dim+self.ceus_dim * 3, num_classes)
        self._features_dim = self.us_dim
        self.pooling = RankPooling()

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def encode_feature(self, x_us, x_ceus, wash_in,wash_out):
        x_us = self.us_encoder(x_us)
        return x_us, x_ceus, wash_in, wash_out

    def classifier(self, x_us, x_ceus, wash_in, wash_out):
        logits = self.head(x_us)
        return logits, ( x_us, x_ceus, wash_in, wash_out)

    def classifier_us_ceus(self, x_us, x_ceus):
        logits = self.head(torch.cat((x_us, x_ceus), dim=1))
        return logits, (x_us, x_ceus)
    def forward(self,x_us, x_ceus, wash_in_images, wash_out_images):
        x_us, x_ceus, wash_in, wash_out = self.encode_feature(x_us, x_ceus, wash_in_images,wash_out_images)
        logits, ( x_us, x_ceus, wash_in, wash_out) = self.classifier(x_us, x_ceus, wash_in, wash_out)
        return logits, ( x_us, x_ceus, wash_in, wash_out)

    @features_dim.setter
    def features_dim(self, value):
        self._features_dim = value

if __name__ == '__main__':

    model = ClassifierBase(2,128,256).cuda()
    us = torch.randn((16,3,256,256)).cuda()
    ceus = torch.randn((16,3,256,256)).cuda()
    dynamics = torch.randn((16,10,3,256,256)).cuda()
    logits, (x_us, x_ceus) = model(us, ceus, dynamics)
    print(logits.shape, (x_us.shape, x_ceus.shape)) # batch * class; batch * feature_size; batch*feature_size*3