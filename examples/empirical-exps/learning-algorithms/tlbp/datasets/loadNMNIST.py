import os
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

import torchvision.transforms as transforms


class NMNIST(Dataset):
    def __init__(self, dataset_path, n_steps, transform=None):
        self.path = dataset_path
        self.samples = []
        self.labels = []
        self.transform = transform
        self.n_steps = n_steps
        for i in tqdm(range(10)):
            sample_dir = dataset_path + '/' + str(i) + '/'
            for f in listdir(sample_dir):
                filename = join(sample_dir, f)
                if isfile(filename):
                    self.samples.append(filename)
                    self.labels.append(i)

    def __getitem__(self, index):
        filename = self.samples[index]
        label = self.labels[index]

        # resize = transforms.Resize(size=(32, 32), interpolation=transforms.InterpolationMode.NEAREST)
        # # data = np.zeros((2, 34, 34, self.n_steps))
        # data = np.load(filename)['frames']  # (T, C, H, W)        
        # data = torch.from_numpy(data).float()
        # data = resize(data) # (T, C, H, W)     
        # data = data.numpy()
        # data = np.transpose(data, (1, 2, 3, 0)) # (T, C, H, W) -> (C, H, W, T)
        data = np.load(filename)['frames']  # (T, C, H, W)
        data = np.transpose(data, (1, 2, 3, 0)) # (T, C, H, W) -> (C, H, W, T)

        if self.transform:
            data = self.transform(data)
            data = data.type(torch.float32)
        else:
            data = torch.FloatTensor(data)

        # Input spikes are reshaped to ignore the spatial dimension and the neurons are placed in channel dimension.
        # The spatial dimension can be maintained and used as it is.
        # It requires different definition of the dense layer.
        return data, label

    def __len__(self):
        return len(self.samples)


def get_nmnist(data_path, network_config):
    n_steps = network_config['n_steps']
    batch_size = network_config['batch_size']
    print("loading NMNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    train_path = data_path + '/frames_number_10_split_by_number/train'
    test_path = data_path + '/frames_number_10_split_by_number/test'
    trainset = NMNIST(train_path, n_steps)
    testset = NMNIST(test_path, n_steps)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader
