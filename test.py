# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: feature-data-enhancement
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/5/5
# @Time        : 下午10:58
# @Description :
import torch

print(torch.cat([torch.zeros((5,128)),torch.empty((5,128))],dim=1))