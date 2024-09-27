import numpy as np
import torch
import torch.nn as nn


class RankPooling(nn.Module):
    """approximate rank pooling"""

    def __init__(self, device="cuda:0"):
        super(RankPooling, self).__init__()
        self.device = device
    def coef_(self, seq_len):
        # 2t − T − 1
        coef_ = torch.linspace(start=1, end=seq_len, steps=seq_len, device=self.device)
        return 2 * coef_ - seq_len - 1

    def forward(self, x):
        """
        :param x: [B, T, -1]
        :return:
        """
        # temporal weights
        coef_ = self.coef_(x.shape[1]).unsqueeze(1).unsqueeze(0)  # [1, T, 1]
        # the unit time-varying mean vector
        x = nn.functional.normalize(x, p=2.0, dim=2)
        res = torch.sum(x * coef_, dim=1)
        return res



if __name__ == "__main__":
    seq = torch.randn((4, 30, 64), device='cuda:0')
    layer = RankPooling()
    wash_in = layer(seq[:, :10, :])
    wash_out = layer(seq[:, 10:, :])
    fdynamics = torch.cat((wash_in, wash_out), dim=0)
    print(fdynamics)
