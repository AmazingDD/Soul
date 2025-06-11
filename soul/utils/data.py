import os
import math
import pickle
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
from torchvision.io import read_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from soul.utils.dvs_folder import CIFAR10DVS, DVS128Gesture

__all__ = ['load_data', 'get_loader']

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
            ``numpy.random.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(tqdm(origin_dataset, ncols=80)):
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

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def pickle_data(data, label, filename):
    outfile = open(filename, 'wb')
    pickle.dump((data, label), outfile)
    outfile.close()

class TinyImageNetDataset(Dataset):
    def __init__(self, root='./tiny-imagenet-200', train=True):
        super().__init__()

        self.NUM_CLASSES = 200
        self.IMGS_PER_CLASS = 500

        self.root = root
        self.train = train

        labels_str = [f.name for f in os.scandir(os.path.join(self.root, 'train')) if f.is_dir()]

        if self.train:
            file_path = os.path.join(self.root, 'train_dataset.pkl')
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    self.data, self.targets = pickle.load(f)
            else:
                print('Caching ImageNet Training dataset...')
                train_data = torch.Tensor().type(torch.ByteTensor)
                labels = []
                i = 1
                for root, dirs, files in os.walk(os.path.join(self.root, 'train')):
                    if root.find('images') != -1:
                        one_class = torch.Tensor().type(torch.ByteTensor)
                        for name in files:
                            img = read_image(root + os.sep + name)
                            if img.shape[0] == 1:
                                img = torch.tensor(cv.cvtColor(img.permute(1, 2, 0).numpy(), cv.COLOR_GRAY2RGB)).permute(2, 0, 1)
                            one_class = torch.cat((one_class, img), 0)
                            labels.append(i - 1)
                            # first_image = False

                        one_class = one_class.reshape(-1, 3, 64, 64)
                        print_progress_bar(i, self.NUM_CLASSES, prefix = 'Progress:', suffix = 'Complete')
                        i += 1
                        train_data = torch.cat((train_data, one_class), 0)

                pickle_data(train_data, torch.Tensor(labels), file_path)

                with open(file_path, 'rb') as f:
                    self.data, self.targets = pickle.load(f)
        else:
            file_path = os.path.join(self.root, 'val_dataset.pkl')
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    self.data, self.targets = pickle.load(f)
            else:
                print('Caching ImageNet Testing dataset...')
                val_data = torch.Tensor().type(torch.ByteTensor)
                
                labels = []
                val_annotations = pd.read_csv(os.path.join(self.root, 'val', 'val_annotations.txt'), sep='\t', names=['filename', 'label_str', 'x_min', 'y_min', 'x_max', 'y_max'])
                num_imgs = len(os.listdir(os.path.join(self.root, 'val', 'images')))

                i = 1
                for name in os.listdir(os.path.join(self.root, 'val', 'images')):
                    img = read_image(os.path.join(self.root, 'val', 'images') + os.sep + name)
                    if img.shape[0] == 1:
                        img = torch.tensor(cv.cvtColor(img.permute(1, 2, 0).numpy(), cv.COLOR_GRAY2RGB)).permute(2, 0, 1)
                    val_data = torch.cat((val_data, img), 0)
                    class_name = val_annotations.loc[val_annotations['filename'] == name]['label_str'].item()
                    labels.append(labels_str.index(class_name))
                    print_progress_bar(i, num_imgs, prefix = 'Progress:', suffix = 'Complete')
                    i += 1

                pickle_data(val_data.reshape(-1, 3, 64, 64), torch.Tensor(labels), file_path)

                with open(file_path, 'rb') as f:
                    self.data, self.targets = pickle.load(f)

        self.targets = self.targets.type(torch.LongTensor)

    def __getitem__(self, index):
        data = self.data[index].permute(1, 2, 0).numpy()
        data = Image.fromarray(data)

        return data, self.targets[index] 
    
    def __len__(self):
        return len(self.targets)

class DVStransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = torch.from_numpy(img).float()
        shape = [img.shape[0], img.shape[1]]
        img = img.flatten(0, 1)
        img = self.transform(img)
        shape.extend(img.shape[1:])
        return img.view(shape)
    
class DatasetWarpper(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.trasnform = transform

    def __getitem__(self, index):
        return self.trasnform(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)

def load_data(dataset_type, dataset_dir, T=4):
    dataset_type = dataset_type.lower()

    if dataset_type == 'cifar10':
        input_channels = 3
        H, W = 32, 32
        num_classes = 10

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
        ])

        dataset_train = torchvision.datasets.CIFAR10(
            root=os.path.join(dataset_dir), train=True, download=True
        )
        dataset_test = torchvision.datasets.CIFAR10(
            root=os.path.join(dataset_dir), train=False, download=True
        )
    elif dataset_type == 'cifar100':
        input_channels = 3
        H, W = 32, 32
        num_classes = 100

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]], std=[n / 255. for n in [68.2, 65.4, 70.4]]), 
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]], std=[n / 255. for n in [68.2, 65.4, 70.4]]), 
        ])

        dataset_train = torchvision.datasets.CIFAR100(
            root=os.path.join(dataset_dir), train=True, download=True
        )
        dataset_test = torchvision.datasets.CIFAR100(
            root=os.path.join(dataset_dir), train=False, download=True
        )

    elif dataset_type == 'cifar10dvs':
        input_channels = 2
        H, W = 64, 64
        num_classes = 10

        transform_train = DVStransform(
            transform=transforms.Compose([transforms.Resize(size=(H, W), antialias=True)])
        )
        transform_test = DVStransform(
            transform=transforms.Resize(size=(H, W), antialias=True)
        )

        dataset = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=T, split_by='number') # 10
        dataset_train, dataset_test = split_to_train_test_set(0.9, dataset, num_classes)
        del dataset

    elif dataset_type == 'imagenet':
        input_channels = 3
        H, W = 224, 224
        num_classes = 200

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        traindir = os.path.join(dataset_dir, 'train')
        valdir = os.path.join(dataset_dir, 'val')

        dataset_train = TinyImageNetDataset(dataset_dir, train=True)
        dataset_test = TinyImageNetDataset(dataset_dir, train=False)

    elif dataset_type == 'dvsgesture':
        input_channels = 2
        H, W = 64, 64
        num_classes = 11

        transform_train = DVStransform(transform=transforms.Compose([transforms.Resize(size=(H, W), antialias=True)]))
        transform_test = DVStransform(transform=transforms.Resize(size=(H, W), antialias=True))

        dataset_train = DVS128Gesture(dataset_dir, train=True, data_type='frame', frames_number=T, split_by='number')
        dataset_test = DVS128Gesture(dataset_dir, train=False, data_type='frame', frames_number=T, split_by='number')

    dataset_train = DatasetWarpper(dataset_train, transform_train)
    dataset_test = DatasetWarpper(dataset_test, transform_test)

    return dataset_train, dataset_test, input_channels, H, W, num_classes

def get_loader(train_dataset, test_dataset, train_sampler, config):
    train_shuffle = False if config['is_distributed'] else True
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=train_shuffle,
        sampler=train_sampler, 
        num_workers=config['workers'], 
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['workers'], 
        pin_memory=True
    )

    return train_loader, test_loader
