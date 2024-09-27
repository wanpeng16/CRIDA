import torch
import torch.nn as nn


class SACoFALoss(nn.Module):
    def __init__(self, class_num):
        super(SACoFALoss, self).__init__()

        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss(torch.tensor([1., 0.5]).cuda())

    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]

        NxW_ij = weight_m.expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = cv_matrix[labels]

        sigma2 = torch.bmm(torch.bmm(NxW_ij - NxW_kj,
                                     CV_temp),
                           (NxW_ij - NxW_kj).permute(0, 2, 1))

        sigma2 = sigma2.mul(torch.eye(C).cuda()
                            .expand(N, C, C)).sum(2).view(N, C)

        aug_result = y + 0.5 * ratio * sigma2

        return aug_result

    def forward(self, fc, features, y, target_x, CoVariance, ratio):
        isda_aug_y = self.isda_aug(fc, features, y, target_x, CoVariance, ratio)
        loss = self.cross_entropy(isda_aug_y, target_x)
        return loss
