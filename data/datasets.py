import os
import os.path as osp
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data import create_transform
import torch

import torchvision
import numpy as np
from data.dataloader import MultiTrans


def build_transform(image_size):
    transform = create_transform(
        input_size=image_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
    )
    return transform


class Datasets(Dataset):
    def __init__(self, root_path, split='train', **args):
        dataset_name=root_path
        DATASET_DIR ='/dataset/'+root_path
        
        if split == 'train':
            THE_PATH = osp.join(DATASET_DIR, 'base')
        elif split == 'test':
            THE_PATH = osp.join(DATASET_DIR, 'novel')
        elif split == 'val':
            THE_PATH = osp.join(DATASET_DIR, 'val')
        else:
            raise ValueError('Unkown setname.')
        
        if root_path in ['CIFAR-FS', 'FC100']:
            mean = (0.5071, 0.4866, 0.4409)
            std = (0.2009, 0.1984, 0.2023)
            norm_params = {"mean": mean, 
                        "std": std}
        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            norm_params = {"mean": mean, 
                        "std": std}
        norm = transforms.Normalize(**norm_params)
        
        # Transform #
        image_size = args['image_size']
        self.transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    norm])
        if split=='train':
            if 'augment' not in args:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    norm
                ])
            else:
                self.transform = build_transform(image_size)
        else:
            self.transform = transforms.Compose([transforms.Resize(int(image_size * 1.1)),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                norm])
            
        self.dataset = torchvision.datasets.ImageFolder(THE_PATH, self.transform)

        # visualization of raw images #
        def convert_raw(x):
                mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
                std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
                return x * std + mean
        self.convert_raw = convert_raw


    def __getitem__(self, i):
        image, label = self.dataset[i]

        return image, label


    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    dataset= Datasets('CIFAR-FS', split='train')
    #print(dataset)
    print(len(dataset.dataset.classes))