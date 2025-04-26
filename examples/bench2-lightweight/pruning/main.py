import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from model.vgg import SpikingVGG9
from model.resnet import SewResNet18
from pruner import *

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_args():
    parser = argparse.ArgumentParser(description='Pruning')
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-dataset', default='CIFAR10', help='model')
    parser.add_argument('-data_dir', default='/home/yudi/data/cifar10/', help='dataset path')
    parser.add_argument('-model', default='SpikingVGG9', help='model')
    parser.add_argument('-gpu', default=0, help='device')
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('-epochs', default=70, type=int, metavar='N', help='number of total epochs')
    parser.add_argument('-ft_epochs', default=10, type=int, metavar='N', help='number of total epochs')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-output_dir', default='./saved_models/', help='path where to save')
    parser.add_argument('-T', default=4, type=int, help='simulation steps')
    # pruning options
    parser.add_argument('-thr', '--threshold', default=0.4, type=float, help='sparsity ratio')
    parser.add_argument('-target', '--granularity', default='weight', help='[optional] weight, channel')

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

def testing(model, test_loader, criterion, device):
    model.eval()
    test_loss, test_acc = 0., 0.
    test_samples = 0
    with torch.no_grad():
        for image, target in test_loader:
            image, target = image.to(device), target.to(device)

            output = model(image)
            loss = criterion(output, target)

            test_loss += loss.item() * target.numel()
            test_samples += target.numel()
            test_acc += (output.argmax(1) == target).float().sum().item()

    test_loss /= test_samples
    test_acc /= test_samples

    return test_acc * 100, test_loss

if __name__ == '__main__':
    args = parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ensure_dir(os.path.join(args.output_dir, 'raw'))
    ensure_dir(os.path.join(args.output_dir, f'{args.granularity}_sparse'))

    if torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'

    train_dataset, test_dataset, input_shape, num_classes = load_data(args.data_dir, args.dataset, args.T)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.workers, 
        shuffle=True, 
        pin_memory=True, 
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=True)
    
    model_map = {
        'spikingvgg9': SpikingVGG9,
        'sewresnet18': SewResNet18, 
    }

    pruner_map = {
        'weight': UnstructuredPruner,
        'channel': StructuredPruner
    }

    # training
    print('Start training')
    pruner = pruner_map[args.granularity.lower()](
        model_map[args.model.lower()](input_shape, args.T, num_classes), device)
    model = pruner.get_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("Building Initial Model")

    max_test_acc = 0.
    for epoch in range(args.epochs):
        train_acc, train_loss = 0., 0.
        train_samples = 0
        model.train()
        for image, target in tqdm(train_loader, unit='batch', ncols=80):
            optimizer.zero_grad()
            image, target = image.to(device), target.to(device)
            output = model(image)
                
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_samples += target.numel()
            train_loss += loss.item() * target.numel()
            train_acc += (output.argmax(1) == target).float().sum().item()

        lr_scheduler.step()

        train_loss /= train_samples
        train_acc /= train_samples

        test_acc, test_loss = testing(model, test_loader, criterion, device)

        print(f'Epoch [{epoch + 1}/{args.epochs}] Train Loss: {train_loss:.2f} Train Acc.: {train_acc * 100:.2f}% Test Loss: {test_loss:.2f} Test Acc.: {test_acc:.2f}%')

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(model.state_dict(), f'./{args.output_dir}/raw/best_{args.model}_{args.dataset}_T{args.T}.pth')

    # pruning
    model.load_state_dict(
        torch.load(f'./{args.output_dir}/raw/best_{args.model}_{args.dataset}_T{args.T}.pth', map_location='cpu'))
    model.to(device)

    test_acc, _ = testing(model, test_loader, criterion, device)
    print(f'Before pruning, Test Acc. {test_acc:.2f}%')

    pruner.apply_pruning(args.threshold)

    test_acc, _ = testing(model, test_loader, criterion, device)
    print(f'After pruning, Test Acc. {test_acc:.2f}%')

    # fine-tuning
    print('Finetuning model')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    pruner.freeze()
    model.train()
    max_test_acc = 0.
    for epoch in range(args.ft_epochs):
        for image, target in tqdm(train_loader, unit='batch', ncols=50):
            optimizer.zero_grad()
            image, target = image.to(device), target.to(device)

            output = model(image)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

        test_acc, test_loss = testing(model, test_loader, criterion, device)

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(model.state_dict(), f'./{args.output_dir}/{args.granularity}_sparse/best_sparse_{args.model}_{args.dataset}_T{args.T}_thr{args.threshold}.pth')

        print(f'Fine-Tune Epoch [{epoch + 1}/{args.ft_epochs}] Test Loss: {test_loss:.2f} Test Acc.: {test_acc:.2f}% Best Test Acc.: {max_test_acc:.2f}%')

    print('All Done!')