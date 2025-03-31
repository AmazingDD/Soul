import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import MnistNet, NMnistNet, CifarNet
from spikingjelly.datasets.n_mnist import NMNIST

parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=3, help='random seed (default: 3)')
parser.add_argument('-b', '--batch_size', type=int, default=50, help='batch size')
parser.add_argument('-epochs', type=int, default=120, help='training epochs')
parser.add_argument('-T', '--time_step', type=int, default=5, help='time steps')
parser.add_argument('-gpu', type=int, default=0, help='GPU ID')
parser.add_argument('-dataset', type=str, default='.', help='dataset name')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

print('load data')

if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomResizedCrop(32, scale=(0.75,1.0), interpolation=PIL.Image.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(root='/home/yudi/data/cifar10', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='/home/yudi/data/cifar10', train=False, download=True, transform=transform_test)
elif args.dataset == 'mnist':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(root='/home/yudi/data', train=True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root='/home/yudi/data', train=False, transform=transform, download=True)
elif args.dataset == 'nmnist':
    train_set = NMNIST(
        root='/home/yudi/data/nmnist', train=True, data_type='frame', frames_number=args.time_step, split_by='number')
    test_set = NMNIST(
        root='/home/yudi/data/nmnist', train=False, data_type='frame', frames_number=args.time_step, split_by='number')
else:
    raise ValueError('Invalid')

if torch.cuda.is_available():
    device = f'cuda:{args.gpu}'
else:
    device = 'cpu'

batch_size = args.batch_size
train_loader = torch.utils.data.DataLoader(
    train_set, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_set, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True)

if args.dataset == 'mnist':
    net = MnistNet(args.time_step)
elif args.dataset == 'cifar10':
    net = CifarNet(args.time_step)
elif args.dataset == 'nmnist':
    net = NMnistNet(args.time_step)
else:
    raise ValueError('Invalid')
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

net.train()
for epoch in range(args.epochs):
    for images, labels in tqdm(train_loader, unit='batch'):
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        if args.dataset == 'nmnist':
            images = images.transpose(0, 1)
        else:
            images = images.unsqueeze(0).repeat(args.time_step, 1, 1, 1, 1)

        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}')

net.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
