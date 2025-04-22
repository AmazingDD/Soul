import os
import random
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not os.path.exists('./samples/'):
    os.makedirs('./samples/')

# cifar10 (3,32,32)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
])

test_dataset = torchvision.datasets.CIFAR10(
    root='/home/yudi/data/cifar10/', 
    train=False,
    download=True, 
    transform=transform_test)

data_loader_test = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=10,
        shuffle=True,
        num_workers=4,
        pin_memory=True, 
        drop_last=False)

sample = next(iter(data_loader_test))[0] # (B, C, H, W)
sample = sample.unsqueeze(1).repeat(1, 4, 1, 1, 1) # -> (B, T, C, H, W)
print('cifar10:', sample.shape)
torch.save(sample, 'cifar10-T4-size32.pt')


# tinyimagenet (3,224,224) (3, 64, 64)
for size in [224, 64]:
    tinyimagenet_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = torchvision.datasets.ImageFolder(root=os.path.join('/home/yudi/data/tiny-imagenet-200/', 'val'), transform=tinyimagenet_transform)

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=10,
        shuffle=True,
        num_workers=4,
        pin_memory=True, 
        drop_last=False)
    
    sample = next(iter(data_loader_test))[0]
    sample = sample.unsqueeze(1).repeat(1, 4, 1, 1, 1)
    print('tiny-imagenet-200:', sample.shape)
    torch.save(sample, f'imagenet-T4-size{size}.pt')


# DVSGesture (16, 2, 128, 128)
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
test_dataset = DVS128Gesture('/home/yudi/data/dvs128gesture/', train=False, data_type='frame', frames_number=16, split_by='number')

data_loader_test = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=10,
    shuffle=True,
    num_workers=4,
    pin_memory=True, 
    drop_last=False)

sample = next(iter(data_loader_test))[0] # (B, T, C, H, W)
print('DVSGesture:', sample.shape)
torch.save(sample, f'dvsgesture-T16-size128.pt')


# cifar10dvs (10, 2, 128, 128)
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets import split_to_train_test_set

dataset = CIFAR10DVS('/home/yudi/data/cifar10_dvs/', data_type='frame', frames_number=10, split_by='number')
_, test_dataset = split_to_train_test_set(0.9, dataset, 10)

data_loader_test = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=10,
    shuffle=True,
    num_workers=4,
    pin_memory=True, 
    drop_last=False)

sample = next(iter(data_loader_test))[0] # (B, T, C, H, W)
print('CIFAR10DVS:', sample.shape)
torch.save(sample, f'cifar10dvs-T10-size128.pt')
