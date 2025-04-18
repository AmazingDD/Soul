# from spikingjelly.datasets.es_imagenet import ESImageNet
from spikingjelly.activation_based import functional
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.n_caltech101 import NCaltech101 as sj_NCaltech101
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
# import numpy as np
import os
from tqdm import tqdm
from model import SpikingVGG9, Spikformer, SEWResNet18 # , SpikingVGG5, Spikformer
import argparse
from dataset import NCaltech101
from utils import set_seed, split_to_train_test_set
from torchvision import datasets

model_map = {
    # "SpikingVGG5": SpikingVGG5,
    "SpikingVGG9": SpikingVGG9,
    "Spikformer": Spikformer,
    "SEWResNet18": SEWResNet18
}

    
def get_dataloader(dataset_name, batch_size, data_dir, args):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    if dataset_name == "MNIST":
        mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
        trainset = torchvision.datasets.MNIST(root=data_dir, train=True, transform=mnist_transform, download=True)
        testset = torchvision.datasets.MNIST(root=data_dir, train=False, transform=mnist_transform, download=True)
    elif dataset_name == "CIFAR10":
        cifar10_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=cifar10_transform, download=True)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=cifar10_transform, download=True)
    elif dataset_name == "CIFAR100":
        cifar100_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, transform=cifar100_transform, download=True)
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, transform=cifar100_transform, download=True)
    elif dataset_name == "Caltech101":
        caltech101_transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为RGB
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ]
        )
        c_dataset = torchvision.datasets.Caltech101(root=data_dir, transform=caltech101_transform, download=True)
        train_size = int(0.8 * len(c_dataset))
        val_size = len(c_dataset) - train_size
        trainset, testset = random_split(c_dataset, [train_size, val_size])
    elif dataset_name == "NMNIST":
        trainset = NMNIST(root=data_dir, train=True, data_type='frame', frames_number=20, split_by='number')
        testset = NMNIST(root=data_dir, train=False, data_type='frame', frames_number=20, split_by='number')
    elif dataset_name == "CIFAR10DVS":
        c_dataset = CIFAR10DVS(root=data_dir, data_type='frame', frames_number=args.T, split_by='number')
        trainset, testset = split_to_train_test_set(0.9, c_dataset, 10)
    elif dataset_name == "NCaltech101":
        dummy_dataset = sj_NCaltech101(root=data_dir, data_type='frame', frames_number=args.T, split_by='number')
        trainset = NCaltech101(data_type="train")
        testset = NCaltech101(data_type="test")
    elif dataset_name == "TinyImageNet":
        tinyimagenet_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((64, 64))
        ])

        trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=tinyimagenet_transform)
        testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=tinyimagenet_transform)

    elif dataset_name == "DVSGesture":
        trainset = DVS128Gesture(root=data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
        testset = DVS128Gesture(root=data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
    else:
        raise ValueError(f"不支持的dataset_name: {dataset_name}")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

def train(args):
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    input_shape = None
    if args.dataset == "MNIST":
        input_shape = (1, 28, 28)
    elif args.dataset == "CIFAR10":
        input_shape = (3, 32, 32)
    elif args.dataset == "CIFAR100":
        input_shape = (3, 32, 32)
    elif args.dataset == "Caltech101":
        input_shape = (3, 64, 64)
    elif args.dataset == "NMNIST":
        input_shape = (2, 34, 34)
    elif args.dataset == "CIFAR10DVS":
        input_shape = (2, 128, 128)
    elif args.dataset == "NCaltech101":
        input_shape = (2, 48, 48)
    elif args.dataset == "TinyImageNet":
        input_shape = (3, 64, 64)
    elif args.dataset == "DVSGesture":
        input_shape = (2, 128, 128)
    # print(model_map[args.model_type])
    model = model_map[args.model_type](num_classes=args.num_classes, T=args.T, neuron_type=args.neuron_type, input_shape=input_shape).cuda()
    print(model)
    trainloader, testloader = get_dataloader(
        args.dataset,
        args.batch_size,
        args.data_dir,
        args
    )
    
    criterion = nn.CrossEntropyLoss()

    epochs = args.epochs
    # optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0  # 用于保存当前最佳准确率

    for epoch in range(epochs):  # 设置训练的 epoch 数
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(trainloader), unit='batch'):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            logit = model(inputs)
            loss = criterion(logit, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            functional.reset_net(model)
        scheduler.step() 
        print(f'Epoch [{epoch + 1}/ {epochs}], Loss: {running_loss / len(trainloader)}')

        # 在验证集上评估模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                logit = model(inputs)
                _, predicted = torch.max(logit.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                functional.reset_net(model)

        acc = 100 * correct / total
        print(f'Accuracy on the {args.dataset} test images: {acc:.2f}%')

        # 如果当前准确率超过最佳准确率，保存模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'./saved_models/best_{args.model_type}_{args.neuron_type}_{args.dataset}_{args.seed}.pth')
            print(f'Best model saved with accuracy: {best_acc:.2f}%')

    print('Finished Training')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--data_dir", type=str, default="./data/MNIST")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--neuron_type", type=str, default="ILIF")
    parser.add_argument("--model_type", type=str, default="SEWResNet18")
    
    args = parser.parse_args()
    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    set_seed(args.seed)
    train(args)
    
