import os

import numpy as np
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader

from datasets.dataset import DynamicCEUS_Excel



class LSSL(nn.Module):
    def __init__(self, in_size, gpu='None'):
        super(LSSL, self).__init__()
        self.encoder = nn.Linear(in_features=in_size, out_features=in_size)
        self.decoder = nn.Linear(in_features=in_size, out_features=in_size)
        self.direction = nn.Linear(1, in_size)
        self.gpu = gpu

    def forward(self, img1, img2):
        bs = img1.shape[0]
        zs = self.encoder(torch.cat([img1, img2], 0))
        recons = self.decoder(zs)
        zs_flatten = zs.view(bs * 2, -1)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]
        recon1, recon2 = recons[:bs], recons[bs:]
        return [z1, z2], [recon1, recon2]

    # reconstruction loss
    def compute_recon_loss(self, x, recon):
        return torch.mean((x - recon) ** 2)

    # direction loss
    def compute_direction_loss(self, zs):
        z1, z2 = zs[0], zs[1]
        bs = z1.shape[0]
        delta_z = z2 - z1
        delta_z_norm = torch.norm(delta_z, dim=1) + 1e-12
        d_vec = self.direction(torch.ones(bs, 1).cuda())
        d_vec_norm = torch.norm(d_vec, dim=1) + 1e-12
        cos = torch.sum(delta_z * d_vec, 1) / (delta_z_norm * d_vec_norm)
        return (1. - cos).mean()
    def get_direction(self):
        return self.direction(torch.ones(1, 1).cuda())

def train_and_process():
    type = 'breast'
    root_path = '/home/amax/Desktop/workspace/dataset/pyro_data'
    root_path = os.path.join(root_path, type)
    xls_path = os.path.join(root_path, f'US_CEUS.xlsx')
    sheet = pd.read_excel(xls_path, sheet_name='Sheet1')
    data = sheet.values
    x = data[:, 1:]
    y = data[:, 2]
    max_epoch = 10
    directions = []
    processed_ids = []

    dataset = DynamicCEUS_Excel(np.concatenate((x, y.reshape(-1, 1)), axis=1), root=root_path,
                                subset='train',
                                type=type,
                                return_in_out=True)
    loader = DataLoader(dataset,
                              batch_size=32,
                              shuffle=True,
                              drop_last=False,
                              num_workers=2)
    for i, (images, target, id) in enumerate(loader):
        _, wash_in,wash_out = images
        dynamics = torch.cat([wash_in,wash_out],dim=1)
        dynamics = dynamics.cuda()
        for one_id,y,one_case in zip(id,target,dynamics):
            # if one_id in processed_ids:
            #     continue
            # processed_ids.append(one_id)
            model = LSSL(152)
            model = model.cuda()
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5, amsgrad=True)
            for epoch in range(max_epoch):
                loss = 0
                for index in range(one_case.shape[0]-1):
                    img1 = one_case[index].to(torch.float)
                    img2 = one_case[index+1].to(torch.float)
                    zs, recon= model(img1.unsqueeze(0),img2.unsqueeze(0))
                    loss = model.compute_recon_loss(img1,recon[0])
                    loss += model.compute_recon_loss(img1,recon[1])
                    loss += model.compute_direction_loss(zs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            model.eval()
            direction = model.get_direction()
            direction = np.concatenate([np.array([one_id.cpu().numpy(),y.cpu().numpy()[0]]),direction.detach().cpu().numpy()[0]],axis=0)
            directions.append(direction)
    df = pd.DataFrame(directions)
    if not os.path.exists(os.path.join(root_path,'lssl')):
        os.mkdir(os.path.join(root_path,'lssl'))
    df.to_excel(os.path.join(os.path.join(root_path,'lssl'),'lssl_early.xlsx'))

if __name__ == '__main__':
    train_and_process()