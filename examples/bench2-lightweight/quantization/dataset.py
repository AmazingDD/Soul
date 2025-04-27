import torch
import torchvision
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import warnings
import os
import numpy as np
from os.path import isfile, join
import pickle
from PIL import Image 
import math

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)

class NMNIST(Dataset):
    def __init__(self, data_path='data/nmnist/frames_number_20_split_by_number',
                 train=True, transform=False):
        if train:
            self.filepath = os.path.join(data_path, 'train')
        else:
            self.filepath = os.path.join(data_path, 'test')
        self.clslist = os.listdir(self.filepath)

        self.dvs_filelist = []
        self.targets = []

        # self.resize = transforms.Resize(
        #     size=(64, 64), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        
        for cls in self.clslist:
            file_list = os.path.join(self.filepath, cls)
            for file in os.listdir(file_list):
                self.dvs_filelist.append(os.path.join(file_list, file))
                self.targets.append(int(cls))
        
        self.classes = range(10)
        self.data_num = len(self.targets)

    def __getitem__(self, index):
        file_pth = self.dvs_filelist[index]
        label = self.targets[index]

        data = torch.from_numpy(np.load(file_pth)['frames']).float()
        # data = self.resize(data)

        return data, label
    
    def __len__(self):
        return self.data_num


class CIFAR10DVS(Dataset):
    def __init__(self, data_path='data/cifar10_dvs/frames_number_20_split_by_number',
                 train=True):
        
        self.filepath = os.path.join(data_path)
        self.clslist = os.listdir(self.filepath)
        self.clslist.sort()

        self.dvs_filelist = []
        self.targets = []

        # self.resize = transforms.Resize(
        #     size=(48, 48), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        for i, cls in enumerate(self.clslist):
            file_list = os.listdir(os.path.join(self.filepath, cls))
            num_file = len(file_list)

            cut_idx = int(num_file * 0.9)
            train_file_list = file_list[:cut_idx]
            test_split_list = file_list[cut_idx:]
            for file in file_list:
                if train:
                    if file in train_file_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, cls, file))
                        self.targets.append(i)
                else:
                    if file in test_split_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, cls, file))
                        self.targets.append(i)

        self.data_num = len(self.dvs_filelist)
        self.data_type = 'train' if train else 'test'

        self.classes = range(10)

    def __getitem__(self, index):
        file_pth = self.dvs_filelist[index]
        label = self.targets[index]
        data = torch.from_numpy(np.load(file_pth)['frames']).float()
        # data = self.resize(data)

        return data, label

    def __len__(self):
        return self.data_num



class NCaltech101(Dataset):
    def __init__(self, data_path='data/ncaltech/frames_number_8_split_by_number',
                 data_type='train'):
        self.filepath = os.path.join(data_path)
        self.clslist = os.listdir(self.filepath)
        self.clslist.sort()

        self.dvs_filelist = []
        self.targets = []
        self.resize = transforms.Resize(
            size=(48, 48), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        for i, cls in enumerate(self.clslist):
            # print (i, cls)
            file_list = os.listdir(os.path.join(self.filepath, cls))
            num_file = len(file_list)

            cut_idx = int(num_file * 0.9)
            train_file_list = file_list[:cut_idx]
            test_split_list = file_list[cut_idx:]
            for file in file_list:
                if data_type == 'train':
                    if file in train_file_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, cls, file))
                        self.targets.append(i)
                else:
                    if file in test_split_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, cls, file))
                        self.targets.append(i)

        self.data_num = len(self.dvs_filelist)
        self.data_type = data_type

        self.classes = range(101)


    def __getitem__(self, index):
        file_pth = self.dvs_filelist[index]
        label = self.targets[index]
        data = torch.from_numpy(np.load(file_pth)['frames']).float()
        data = self.resize(data)

        return data, label

    def __len__(self):
        return self.data_num
    
class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root='./tiny-imagenet-200', train=True, transform=None):
        super().__init__()

        self.root = root
        self.transform = transform
        self.train = train

        if self.train:
            with open(os.path.join(self.root, 'train_dataset.pkl'), 'rb') as f:
                self.data, self.targets = pickle.load(f)
        else:
            with open(os.path.join(self.root, 'val_dataset.pkl'), 'rb') as f:
                self.data, self.targets = pickle.load(f)

        self.targets = self.targets.type(torch.LongTensor)

    def __getitem__(self, index):
        data = self.data[index].permute(1, 2, 0).numpy()
        data = Image.fromarray(data)
        if self.transform:
            data = self.transform(data)

        return data, self.targets[index] 
    
    def __len__(self):
        return len(self.targets)
