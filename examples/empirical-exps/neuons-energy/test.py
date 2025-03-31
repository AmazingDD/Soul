import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torchvision
import torchvision.transforms as transforms
from spikingjelly.datasets.n_mnist import NMNIST

from model import SpikingVGG9, SpikingVGG5

model_map = {
    "SpikingVGG5": SpikingVGG5,
    "SpikingVGG9": SpikingVGG9
}

parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=3, help='random seed (default: 3)')
parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size')
parser.add_argument('-epochs', type=int, default=100, help='training epochs')
parser.add_argument('-T', '--time_step', type=int, default=4, help='time steps')
parser.add_argument('-gpu', type=int, default=0, help='GPU ID')
parser.add_argument('-dataset', type=str, default='mnist', help='dataset name')
parser.add_argument('-neuron', "--neuron_type", type=str, default="LIF")
parser.add_argument('-model', "--model_type", type=str, default="SpikingVGG5")
parser.add_argument('-lr', "--learning_rate", type=float, default=1e-3)
args = parser.parse_args()


# data_path = "/home/SD"
data_path = "/home/yudi/data"

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

if args.dataset == 'mnist':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(root=os.path.join(data_path), train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root=os.path.join(data_path), train=False, transform=transform, download=True)
    input_shape = (1, 28, 28)
    num_classes = 10
elif args.dataset == 'cifar10':
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
    train_dataset = torchvision.datasets.CIFAR10(root=os.path.join(data_path, 'cifar10'), train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root=os.path.join(data_path, 'cifar10'), train=False, download=True, transform=transform_test)
    input_shape = (3, 32, 32)
    num_classes = 10
elif args.dataset == 'nmnist':
    train_dataset = NMNIST(
        root=os.path.join(data_path, 'nmnist'), train=True, data_type='frame', frames_number=args.time_step, split_by='number')
    test_dataset = NMNIST(
        root=os.path.join(data_path, 'nmnist'), train=False, data_type='frame', frames_number=args.time_step, split_by='number')
    input_shape = (2, 34, 34)
    num_classes = 10
else:
    raise ValueError('Invalid')


if torch.cuda.is_available():
    device = f'cuda:{args.gpu}'
else:
    device = 'cpu'

batch_size = args.batch_size
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True)


model = model_map[args.model_type](num_classes=num_classes, T=args.time_step, neuron_type=args.neuron_type, input_shape=input_shape).to(device)

criterion = nn.CrossEntropyLoss()
epochs = args.epochs
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

best_acc = 0.0 
test_acc_rec = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, unit='batch', ncols=50):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        logit = model(inputs)
        loss = criterion(logit, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step() 
    print(f'Epoch [{epoch + 1}/ {epochs}], Loss: {running_loss / len(train_loader)}')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logit = model(inputs)
            _, predicted = torch.max(logit.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    
    test_acc_rec.append(acc)
    print(f'Accuracy on the {args.dataset} test images: {acc:.2f}%')

    if acc > best_acc:
        best_acc = acc
        # torch.save(model.state_dict(), f'./best_{args.model_type}_{args.neuron_type}_{args.dataset}.pth')
        # print(f'Best model saved with accuracy: {best_acc:.2f}%')

df = pd.DataFrame({
    'epochs': range(epochs),
    'accuracy': best_acc,
})
df.to_csv(f'./acc_{args.model_type}_{args.neuron_type}_{args.dataset}.csv', index=False)
print('Finished Training')

