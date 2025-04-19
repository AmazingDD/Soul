import os
import time
import argparse

# from spikingjelly.datasets.n_mnist import NMNIST
# from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
# from spikingjelly.datasets.n_caltech101 import NCaltech101 as sj_NCaltech101
from spikingjelly.activation_based import functional

import torch
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms

from dataset import NCaltech101, NMNIST, CIFAR10DVS
from model import SpikingVGG9
from model import Spikformer
from utils import set_seed, split_to_train_test_set

import power_check as pc

model_map = {
    # "SpikingVGG5": SpikingVGG5,
    "SpikingVGG9": SpikingVGG9,
    "Spikformer": Spikformer,
}

def get_dataloader(dataset_name, data_dir):
    if not os.path.exists(data_dir):
        raise ValueError(f'Invalid path: {data_dir}')
        
    if dataset_name == "MNIST":
        mnist_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        testset = torchvision.datasets.MNIST(root=data_dir, train=False, transform=mnist_transform, download=True)
    elif dataset_name == "CIFAR10":
        cifar10_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
        ])
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=cifar10_transform, download=True)

    elif dataset_name == "Caltech101":
        caltech101_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为RGB
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])

        c_dataset = torchvision.datasets.Caltech101(root=data_dir, transform=caltech101_transform, download=True)
        train_size = int(0.8 * len(c_dataset))
        val_size = len(c_dataset) - train_size
        _, testset = random_split(c_dataset, [train_size, val_size])

    elif dataset_name == "NMNIST":
        # testset = NMNIST(root=data_dir, train=False, data_type='frame', frames_number=20, split_by='number')
        testset = NMNIST(
            data_path=os.path.join(args.data_dir, 'nmnist/frames_number_20_split_by_number'), 
            train=False)
        # 
    elif dataset_name == "CIFAR10DVS":
        # c_dataset = CIFAR10DVS(root=data_dir, data_type='frame', frames_number=20, split_by='number')
        # _, testset = split_to_train_test_set(0.9, c_dataset, 10)
        testset = CIFAR10DVS(
            data_path=os.path.join(args.data_dir, 'cifar10_dvs/frames_number_20_split_by_number'), 
            train=False)
    elif dataset_name == "NCaltech101":
        # _ = sj_NCaltech101(root=data_dir, data_type='frame', frames_number=8, split_by='number')
        testset = NCaltech101(
            data_path=os.path.join(args.data_dir, 'ncaltech/frames_number_8_split_by_number'), 
            data_type="test")
    else:
        raise ValueError(f"Invalid dataset_name: {dataset_name}")
    
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    return testloader

def test(args, weight_dir):
    input_shape = None
    if args.dataset == "MNIST":
        input_shape = (1, 28, 28)
        num_classes = 10
        T = 4
    elif args.dataset == "CIFAR10":
        input_shape = (3, 32, 32)
        num_classes = 10
        T = 4
    elif args.dataset == "Caltech101":
        input_shape = (3, 64, 64)
        num_classes = 101
        T = 4
    elif args.dataset == "NMNIST":
        input_shape = (2, 34, 34)
        num_classes = 10
        T = 20
    elif args.dataset == "CIFAR10DVS":
        input_shape = (2, 128, 128)
        num_classes = 10
        T = 20
    elif args.dataset == "NCaltech101":
        input_shape = (2, 48, 48)
        num_classes = 101
        T = 8
    # print(model_map[args.model_type])
    model = model_map[args.model_type](num_classes=num_classes, T=T, neuron_type=args.neuron_type, input_shape=input_shape)
    model.load_state_dict(
        torch.load(os.path.join(
        weight_dir, f'best_{args.model_type}_{args.neuron_type}_{args.dataset}_{args.seed}.pth'), weights_only=True, map_location='cpu')
    )
    
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    testloader = get_dataloader(args.dataset, args.data_dir)

    with torch.no_grad():
        model.eval()

        pc.printFullReport(pc.getDevice())
        pl = pc.PowerLogger(interval=0.05)
        pl.start()
        time.sleep(5)
        pl.recordEvent(name='Process Start')

        start_time = time.time()
        cnt = 0
        for inputs, _ in testloader:
            if cnt == 10:
                break
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            else:
                inputs = inputs.cpu() 
            _ = model(inputs)
            functional.reset_net(model)
            cnt += 1

        print(f'inference time per sample: {(time.time() - start_time) / cnt:.3f}s')

        time.sleep(5)
        pl.stop()
        filename = f'./{args.model_type}/{args.neuron_type}/{args.dataset}/{args.seed}/test/'
        pl.showDataTraces(filename=filename)
        print(str(pl.eventLog))
        pc.printFullReport(pc.getDevice())
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="/home/orin/SD/")
    parser.add_argument("-model_type", type=str, default="SpikingVGG9")
    parser.add_argument("-dataset", type=str, default="MNIST")
    parser.add_argument("-seed", type=int, default=41) # 41, 42 , 43
    parser.add_argument("-neuron_type", type=str, default="LIF")

    args = parser.parse_args()

    set_seed(args.seed)
    test(args, weight_dir='/home/orin/SD/lif_weight_saved/')