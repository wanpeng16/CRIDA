# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: feature-data-enhancement
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/5/2
# @Time        : 下午7:06
# @Description :
import random
import re
import time

import numpy as np
import torch
import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision import transforms


class DynamicCEUS_Images(Dataset):
    def __init__(self, data=None, root='E:/data/breast', type=None, return_in_out=False, subset='train',
                 seed=0,
                 cache=True,
                 transform=True):
        super(DynamicCEUS_Images, self).__init__()
        self.root = root
        self.image_path = os.path.join(root, 'images')


        self.us_image_paths = []
        self.ceus_image_paths = []
        self.wash_in_paths = []
        self.wash_out_paths = []
        self.labels = []
        self.case_id = []
        self.data = data
        self.subset = subset
        if self.subset == 'train' and transform:
            self.transform = transforms.Compose([
                # transforms.RandomChoice([
                #     transforms.RandomCrop(224, padding=16),
                #     transforms.RandomCrop(224, padding=(16, 64)),
                #     transforms.RandomCrop(224, padding=64, padding_mode='edge')
                # ]),
                transforms.RandomChoice([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(45),
                ]),
                # transforms.RandomChoice([
                #     transforms.ColorJitter(brightness=0.5),
                #     transforms.ColorJitter(contrast=0.5),
                #     transforms.ColorJitter(saturation=0.5),
                #     transforms.ColorJitter(hue=0.3)
                # ]),
            ])

        else:
            self.transform = transforms.Compose([

            ])
        self.resize_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])
        # self.ceus_transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # ])
        self.type = type
        self.return_in_out = return_in_out
        self.__list_files()
        # self.balance_classes()
        assert len(self.us_image_paths) == len(self.ceus_image_paths) == len(self.wash_in_paths) == len(
            self.wash_out_paths)
        # 合并数据列表
        data = list(
            zip(self.us_image_paths, self.ceus_image_paths, self.wash_in_paths, self.wash_out_paths, self.labels, self.case_id ))

        # 提取类别标签
        labels = self.labels
        # print(np.sum(np.array(labels)==0))

        if self.subset != 'all':

            train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2,
                                                                                stratify=labels, random_state=seed)
            train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.25,
                                                                              stratify=train_labels,
                                                                              random_state=seed)  # 0.25 x 0.8 = 0.2 = 20%

            # 解压缩划分后的数据
            if subset == 'train':
                sub_data = train_data
            elif subset == 'val':
                sub_data = val_data
            else:
                sub_data = test_data
            self.us_image_paths, self.ceus_image_paths, self.wash_in_paths, self.wash_out_paths, self.labels, self.case_id  = zip(
                *sub_data)
        self.cache = {}
        if cache:
            self.pre_load()
        self.random = random.Random(seed)

    def balance_classes(self):
        count_1 = sum(self.labels)
        count_0 = len(self.labels) - count_1
        diff = abs(count_1 - count_0)
        delete_indices = []

        if count_1 > count_0:
            for i in range(len(self.labels)):
                if self.labels[i] == 1:
                    delete_indices.append(i)
        else:
            for i in range(len(self.labels)):
                if self.labels[i] == 0:
                    delete_indices.append(i)

        delete_indices = random.sample(delete_indices, diff)

        for index in sorted(delete_indices, reverse=True):
            del self.us_image_paths[index]
            del self.ceus_image_paths[index]
            del self.wash_in_paths[index]
            del self.wash_out_paths[index]
            del self.labels[index]
            del self.case_id[index]
    def __list_files(self):
        for class_dir in os.listdir(self.image_path):
            class_path = os.path.join(self.image_path, class_dir)
            for case_dir in os.listdir(class_path):
                self.case_id.append(self.__extract_number_from_filename(case_dir))
                self.labels.append(int(class_dir))
                case_dir = os.path.join(class_path, case_dir)
                self.us_image_paths.append(os.path.join(case_dir, 'US.png'))
                self.ceus_image_paths.append(os.path.join(case_dir, 'CEUS.png'))
                one_wash_in_paths, one_wash_out_paths = self.__sample_files_by_p(os.path.join(case_dir, 'dynamics'))
                self.wash_in_paths.append(one_wash_in_paths)
                self.wash_out_paths.append(one_wash_out_paths)

    def __extract_number_from_filename(self, filename):
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        else:
            return 0

    def __sample_files_by_p(self, folder_path):
        file_names = sorted(os.listdir(folder_path), key=lambda x: self.__extract_number_from_filename(x))

        p_indices = [i for i, name in enumerate(file_names) if 'p' in name]

        if len(p_indices) == 0:
            return [], []

        central_p_index = p_indices[len(p_indices) // 2]

        # 计算前后文件数量
        num_files_before = central_p_index
        num_files_after = len(file_names) - central_p_index - 1

        # 计算间隔数量
        # interval_before = max(num_files_before // 5, 1)  # 前面文件的间隔数量
        # interval_after = max(num_files_after // 5, 1)  # 后面文件的间隔数量
        interval_before = 1
        interval_after = 1
        # 从中心向两边等间隔采样文件
        left_file_indices = list(range(central_p_index - interval_before * 10, central_p_index, interval_before))
        right_file_indices = list(
            range(central_p_index + interval_after, central_p_index + interval_after * 11, interval_after))

        # 处理索引越界的情况
        left_file_indices = [idx for idx in left_file_indices if 0 <= idx < len(file_names)]
        right_file_indices = [idx for idx in right_file_indices if 0 <= idx < len(file_names)]

        left_files = [os.path.join(folder_path, file_names[idx]) for idx in left_file_indices]
        right_files = [os.path.join(folder_path, file_names[idx]) for idx in right_file_indices]
        if len(left_files) < 10:
            repeat_count = 10 - len(left_files)
            repeat_left_files = [left_files[-1]] * repeat_count  # 从最后一个元素开始重复
            left_files = left_files + repeat_left_files
        if len(right_files) < 10:
            repeat_count = 10 - len(right_files)
            repeat_right_files = [right_files[0]] * repeat_count  # 从第一个元素开始重复
            right_files = right_files + repeat_right_files

        return left_files, right_files

    def __len__(self):
        return len(self.us_image_paths)
    def get_original_image(self,normal_image):
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        return transforms.ToPILImage()(self.unnormalize(normal_image,mean,std,False))
    def pre_load(self):
        data_len_tqdm = tqdm.tqdm(range(len(self.us_image_paths)))
        data_len_tqdm.set_description_str("Preload Data To Cache")
        for index in data_len_tqdm:
            us_image_path = self.us_image_paths[index]
            ceus_image_path = self.ceus_image_paths[index]
            wash_in_image_paths = self.wash_out_paths[index]
            wash_out_image_paths = self.wash_out_paths[index]
            us_img = Image.open(us_image_path)
            ceus_img = Image.open(ceus_image_path)
            wash_in_images = []
            for one_wash_in_image_path in wash_in_image_paths:
                one_wash_in_image = Image.open(one_wash_in_image_path)
                # one_wash_in_image = self.ceus_transform(one_wash_in_image)
                wash_in_images.append(one_wash_in_image)
            wash_out_images = []
            for one_wash_out_image_path in wash_out_image_paths:
                one_wash_out_image = Image.open(one_wash_out_image_path)
                # one_wash_out_image = self.ceus_transform(one_wash_out_image)
                wash_out_images.append(one_wash_out_image)
            label = self.labels[index]

            us_img = self.resize_transform(us_img)
            ceus_img = self.resize_transform(ceus_img)
            wash_in_images = [self.resize_transform(one_wash_in_image) for one_wash_in_image in wash_in_images]
            wash_out_images = [self.resize_transform(one_wash_out_image) for one_wash_out_image in wash_out_images]
            # wash_in_images = torch.stack(wash_in_images)
            # wash_out_images = torch.stack(wash_out_images)
            self.cache[index] = us_img,ceus_img,wash_in_images,wash_out_images,label
    def get_by_id(self,_id,_label):
        index = [i for i, (label, id) in enumerate(zip(self.labels, self.case_id)) if label == _label and id == _id]
        return index

    def unnormalize(self,tensor, mean, std, inplace) :
        """Unnormalize a tensor image with mean and standard deviation.

        Args:
            tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
            inplace(bool,optional): Bool to make this operation inplace.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

        if tensor.ndim < 3:
            raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                             '{}.'.format(tensor.size()))

        if not inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        tensor.mul_(std).add_(mean)
        return tensor

    def __getitem__(self, index):
        # s = time.time()
        # print(self.cache.keys())
        if index not in self.cache.keys():
            seed = self.random.randint(0, 100000)
            us_image_path = self.us_image_paths[index]
            ceus_image_path = self.ceus_image_paths[index]
            wash_in_image_paths = self.wash_out_paths[index]
            wash_out_image_paths = self.wash_out_paths[index]
            us_img = Image.open(us_image_path)
            us_img = self.resize_transform(us_img)
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            us_img = self.transform(us_img)
            ceus_img = Image.open(ceus_image_path)
            ceus_img = self.resize_transform(ceus_img)
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            ceus_img = self.transform(ceus_img)
            wash_in_images = []
            for one_wash_in_image_path in wash_in_image_paths:
                one_wash_in_image = Image.open(one_wash_in_image_path)
                one_wash_in_image = self.resize_transform(one_wash_in_image)
                random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                one_wash_in_image = self.transform(one_wash_in_image)
                wash_in_images.append(one_wash_in_image)
            wash_in_images = torch.stack(wash_in_images)
            wash_out_images = []
            for one_wash_out_image_path in wash_out_image_paths:
                one_wash_out_image = Image.open(one_wash_out_image_path)
                one_wash_out_image = self.resize_transform(one_wash_out_image)
                random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                one_wash_out_image = self.transform(one_wash_out_image)
                wash_out_images.append(one_wash_out_image)
            wash_out_images = torch.stack(wash_out_images)
            label = self.labels[index]
            # print(time.time() - s)
        else:
            _us_img,_ceus_img,_wash_in_images,_wash_out_images,label = self.cache[index]
            seed = self.random.randint(0,100000)
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            us_img = self.transform(_us_img)
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            ceus_img = self.transform(_ceus_img)

            transform_wash_in_images = []
            for i in range(len(_wash_in_images)):
                random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                transform_wash_in_images.append(self.transform(_wash_in_images[i]))
            wash_in_images = torch.stack(transform_wash_in_images)

            transform_wash_out_images = []
            for i in range(len(_wash_out_images)):
                random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                transform_wash_out_images.append(self.transform(_wash_out_images[i]))
            wash_out_images = torch.stack(transform_wash_out_images)
        if self.return_in_out:
            return (us_img, ceus_img, wash_in_images, wash_out_images), label
        return (us_img, ceus_img, wash_in_images, wash_out_images), label, index

if __name__ == '__main__':
    _dataset = DynamicCEUS_Images(root=os.path.expanduser('~/Desktop/workspace/dataset/pyro_data/breast/'))
    for i in range(len(_dataset)):
        (us_img,ceus_img,wash_in_images,wash_out_images), label, index= _dataset[i]
        # print(us_img.shape,ceus_img.shape,wash_in_images.shape,wash_out_images.shape,label,index)
