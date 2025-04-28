import os
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn

import torchvision
from torchvision import transforms

from sew_resnet import SewResNet18

class DatasetWarpper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.trasnform = transform

    def __getitem__(self, index):
        return self.trasnform(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)
    
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

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_args():
    parser = argparse.ArgumentParser(description='preparation for deployment of heterogenous devices')
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-model_dir', type=str, default='./saved_models/', help='root dir for saving trained model')
    parser.add_argument('-data_dir', type=str, default='.', help='root dir of dataset')
    parser.add_argument('-gpu', type=int, default=0, help='gpu id')
    parser.add_argument('-workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-dataset', default='CIFAR10', help='dataset name')
    parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-lr', default=5e-4, type=float, help='learning rate')

    args = parser.parse_args()

    return args

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

def load_data(dataset_dir, dataset_type, T):
    if dataset_type == 'CIFAR10':
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

        train_dataset = torchvision.datasets.CIFAR10(
            root=dataset_dir, 
            train=True,
            download=True, 
            transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(
            root=dataset_dir, 
            train=False,
            download=True, 
            transform=transform_test)
        
        input_shape = (3, 32, 32)
        num_classes = 10

    elif dataset_type == 'TinyImageNet':
        tinyimagenet_transform = transforms.Compose([
            transforms.Resize((64, 64)), # (224, 224) 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(dataset_dir, 'train'), transform=tinyimagenet_transform)
        # test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(dataset_dir, 'val'), transform=tinyimagenet_transform)

        train_dataset = TinyImageNetDataset(dataset_dir, train=True, transform=tinyimagenet_transform)
        test_dataset = TinyImageNetDataset(dataset_dir, train=False, transform=tinyimagenet_transform)

        input_shape = (3, 64, 64)
        num_classes = 200

    elif dataset_type == 'DVSGesture':
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        transform_train = DVStransform(transform=transforms.Compose([transforms.Resize(size=(64, 64), antialias=True)]))
        transform_test = DVStransform(transform=transforms.Resize(size=(64, 64), antialias=True))

        train_dataset = DVS128Gesture(dataset_dir, train=True, data_type='frame', frames_number=T, split_by='number')
        test_dataset = DVS128Gesture(dataset_dir, train=False, data_type='frame', frames_number=T, split_by='number')

        train_dataset = DatasetWarpper(train_dataset, transform_train)
        test_dataset = DatasetWarpper(test_dataset, transform_test)

        input_shape = (2, 64, 64)
        num_classes = 11

    elif dataset_type == 'CIFAR10DVS':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        from spikingjelly.datasets import split_to_train_test_set

        transform_train = DVStransform(transform=transforms.Compose([transforms.Resize(size=(64, 64), antialias=True)]))
        transform_test = DVStransform(transform=transforms.Resize(size=(64, 64), antialias=True))

        dataset = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=T, split_by='number')
        train_dataset, test_dataset = split_to_train_test_set(0.9, dataset, 10)
        del dataset

        train_dataset = DatasetWarpper(train_dataset, transform_train)
        test_dataset = DatasetWarpper(test_dataset, transform_test)

        input_shape = (2, 64, 64)
        num_classes = 10

    else:
        raise ValueError(dataset_type)


    return train_dataset, test_dataset, input_shape, num_classes

if __name__ == '__main__':
    args = parse_args()
    ensure_dir(args.model_dir)

    print(str(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_train, dataset_test, input_shape, num_classes = load_data(args.data_dir, args.dataset, args.T)
    print(f'dataset_train: {len(dataset_train)}, dataset_test: {len(dataset_test)}')

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True, 
        drop_last=True)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True, 
        drop_last=False)
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = SewResNet18(num_classes=num_classes, T=args.T, input_shape=input_shape)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # for imagenet
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()

    max_test_acc = -1
    for epoch in range(args.epochs):
        model.train()
        for image, target in tqdm(data_loader_train, unit='batch', ncols=80):
            optimizer.zero_grad()
            image, target = image.to(device), target.to(device)

            output = model(image)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        model.eval()
        test_acc, test_loss, num_samples = 0., 0., 0

        with torch.no_grad():
            for image, target in data_loader_test:
                image, target = image.to(device), target.to(device)
                output = model(image)
                loss = criterion(output, target)

                num_samples += target.numel()
                test_loss += loss.item() * target.numel()
                test_acc += (output.argmax(1) == target).float().sum().item()

            test_loss /= num_samples
            test_acc /= num_samples

            print(f'Epoch [{epoch + 1}/{args.epochs}] Test Loss: {test_loss:.2f} Test Acc.: {test_acc * 100:.2f}%')

            if test_acc > max_test_acc:
                max_test_acc = test_acc
                torch.save(model.state_dict(), 
                           os.path.join(args.model_dir, f'SewResNet18_{args.dataset}_T{args.T}_ckpt_best.pth'))
                print(f'Best test_acc={test_acc:.4f}')
