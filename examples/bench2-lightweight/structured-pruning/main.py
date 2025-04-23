import os
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from vgg import SpikingVGG9

model_map = {
    "SpikingVGG9": SpikingVGG9,
    # "SewResNet18": SewResNet18
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_args():
    parser = argparse.ArgumentParser(description='Structured pruning')
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-data_dir', default='/home/yudi/data/cifar10/', help='dataset path')
    parser.add_argument('-model', default='SpikingVGG9', help='model')
    parser.add_argument('-gpu', default=0, help='device')
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('-epochs', default=200, type=int, metavar='N', help='number of total epochs')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-output_dir', default='./saved_models/', help='path where to save')
    parser.add_argument('-T', default=4, type=int, help='simulation steps')

    args = parser.parse_args()

    return args

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
            root=os.path.join(dataset_dir), 
            train=True,
            download=True, 
            transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(
            root=os.path.join(dataset_dir), 
            train=False,
            download=True, 
            transform=transform_test)
        
        input_shape = (3, 32, 32)
        num_classes = 10

    elif dataset_type == 'DVSGesture':
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        train_dataset = DVS128Gesture(dataset_dir, train=True, data_type='frame', frames_number=T, split_by='number')
        test_dataset = DVS128Gesture(dataset_dir, train=False, data_type='frame', frames_number=T, split_by='number')

        input_shape = (2, 128, 128)
        num_classes = 11
    else:
        raise ValueError(dataset_type)

    return train_dataset, test_dataset, input_shape, num_classes

if __name__ == '__main__':
    args = parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    max_test_acc1 = 0.
    output_dir = os.path.join(args.output_dir, f'{args.model}_T{args.T}')
    ensure_dir(output_dir)

    if torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'

    train_dataset, test_dataset, input_shape, num_classes = load_data(args.data_dir, args.dataset, args.T)

    print('Creating model')
    model = SpikingVGG9()
    model.to(device)

    manager = PruningNetworkManager(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    