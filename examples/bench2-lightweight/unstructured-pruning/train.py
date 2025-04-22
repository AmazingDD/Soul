import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

import torchvision
from torchvision import transforms

from utils import *
from model import SpikingVGG9, SewResNet18

model_map = {
    "SpikingVGG9": SpikingVGG9,
    "SewResNet18": SewResNet18,
}

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
            download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            root=os.path.join(dataset_dir), 
            train=False,
            download=True)
        
        input_shape = (3, 32, 32)
        num_classes = 10

    elif dataset_type == 'DVSGesture':
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        transform_train = DVStransform(
            transform=transforms.Compose([transforms.Resize(size=(128, 128), antialias=True)])
        )
        transform_test = DVStransform(
            transform=transforms.Resize(size=(128, 128), antialias=True)
        )

        train_dataset = DVS128Gesture(dataset_dir, train=True, data_type='frame', frames_number=T, split_by='number')
        test_dataset = DVS128Gesture(dataset_dir, train=False, data_type='frame', frames_number=T, split_by='number')

        input_shape = (2, 128, 128)
        num_classes = 11
    else:
        raise ValueError(dataset_type)
    
    dataset_train = DatasetWarpper(train_dataset, transform_train)
    dataset_test = DatasetWarpper(test_dataset, transform_test)

    return dataset_train, dataset_test, input_shape, num_classes

def parse_args():
    parser = argparse.ArgumentParser(description='Unstructured weight pruning for SNNs')
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-model_dir', type=str, default='./saved_models/', help='root dir for saving trained model')
    parser.add_argument('-data_dir', type=str, default='.', help='root dir of dataset')
    parser.add_argument('-log_dir', type=str, default='./sparse_logs/', help='root dir of output logs')
    parser.add_argument('-gpu', type=int, default=0, help='gpu id')
    parser.add_argument('-workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-dataset', default='CIFAR10', help='dataset name')
    parser.add_argument('-model', default='SpikingVGG9', help='model name')
    parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-lr', default=1e-4, type=float, help='learning rate')
    # pruning parameter
    parser.add_argument('-thr', '--flat-width', type=float, default=0.1) # also known as threshold
    # parser.add_argument('-gradual', type=str, choices=['linear', 'sine'], default='linear', help="increase type of threshold") # if None, no change for threshold

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    logger = setup_logger(args.log_dir, args)
    logger.info(str(args))

    ensure_dir(args.model_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_train, dataset_test, input_shape, num_classes = load_data(args.data_dir, args.dataset, args.T)
    logger.info(f'dataset_train: {len(dataset_train)}, dataset_test: {len(dataset_test)}')

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
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'

    model = model_map[args.model](num_classes=num_classes, T=args.T, input_shape=input_shape, threshold=args.flat_width)
    # logger.info(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()

    max_test_acc = -1
    train_times, total_train_step = 0, len(data_loader_train) * args.epochs
    for epoch in range(args.epochs):
        model.train()
        for image, target in tqdm(data_loader_train, unit='batch'):
            optimizer.zero_grad()
            image, target = image.to(device), target.to(device)

            output = model(image)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            train_times += 1

            # STDS
            # if args.gradual is not None:
            #     for module in model.modules():
            #         if hasattr(module, 'setFlatWidth'):
            #             if args.gradual == 'linear':
            #                 module.setFlatWidth(linearInc(train_times, total_train_step) * args.flat_width)
            #             elif args.gradual == 'sine':
            #                 module.setFlatWidth(sineInc(train_times, total_train_step) * args.flat_width)
        
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

            if test_acc > max_test_acc:
                max_test_acc = test_acc
                torch.save(model.state_dict(), 
                           os.path.join(args.model_dir, f'{args.dataset}_{args.model}_T{args.T}_thr{args.flat_width}_seed{args.seed}_ckpt_best.pth'))
                logger.info(f'Best test_acc={test_acc:.4f}')
                
        # sparsity
        total_zerocnt = 0
        total_numel = 0
        for name, module in model.named_modules():
            if hasattr(module, "getSparsity"):
                zerocnt, numel = module.getSparsity()
                total_zerocnt += zerocnt
                total_numel += numel
                print(f'{name}: {zerocnt / numel * 100:.2f}%')
        logger.info(f' sparsity/total: {total_zerocnt / total_numel * 100:.2f}%')

        logger.info(f' Epoch[{epoch + 1}/{args.epochs}] test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, max_test_acc={max_test_acc:.4f}')
