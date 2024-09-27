# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: pyro-search-metastasis-rank-pooling
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/4/30
# @Time        : 下午2:23
# @Description :
import torch

from module.arp import RankPooling
from module.modules import NONLocalBlock1D


def Rank_Pooling(dynamics):
    rank_pooling = RankPooling()
    wash_in = rank_pooling(dynamics[:, :10, :])
    wash_out = rank_pooling(dynamics[:, 10:, :])
    return torch.cat((wash_in, wash_out), dim=1)
def rank_pooling_once(dynamics):
    rank_pooling = RankPooling()
    dynamics = rank_pooling(dynamics)
    return dynamics

def early_pooling(dynamics):
    return dynamics
def average_pooling(dynamics):
    return torch.mean(dynamics,dim=1)

def self_attention_pooling(dynamics):
    self_attention = NONLocalBlock1D(dynamics.shape[2])
    dynamics.permute(0, 2, 1)
    dynamics = self_attention(dynamics)
    return dynamics


