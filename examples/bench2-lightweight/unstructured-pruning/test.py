import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision import transforms

from utils import *
from model import SpikingVGG9, SewResNet18

import power_check as pc

model_map = {
    "SpikingVGG9": SpikingVGG9,
    "SewResNet18": SewResNet18,
}

def load_data(dataset_dir, dataset_type, T):
    if dataset_type == 'CIFAR10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
        ])

        test_dataset = torchvision.datasets.CIFAR10(
            root=os.path.join(dataset_dir), 
            train=False,
            download=True)
        
        input_shape = (3, 32, 32)
        num_classes = 10

    elif dataset_type == 'DVSGesture':
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        transform_test = DVStransform(
            transform=transforms.Resize(size=(128, 128), antialias=True)
        )

        test_dataset = DVS128Gesture(dataset_dir, train=False, data_type='frame', frames_number=T, split_by='number')

        input_shape = (2, 128, 128)
        num_classes = 11
    else:
        raise ValueError(dataset_type)
    
    dataset_test = DatasetWarpper(test_dataset, transform_test)

    return dataset_test, input_shape, num_classes

def parse_args():
    parser = argparse.ArgumentParser(description='Unstructured weight pruning for SNNs')
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-model_dir', type=str, default='./saved_models/unstructured-pruning/', help='root dir for saving trained model')
    parser.add_argument('-data_dir', type=str, default='.', help='root dir of dataset')
    parser.add_argument('-workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-dataset', default='CIFAR10', help='dataset name')
    parser.add_argument('-model', default='SpikingVGG9', help='model name')
    parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
    # pruning parameter
    parser.add_argument('-thr', '--flat-width', type=float, default=0.1) # also known as threshold

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    dataset_test, input_shape, num_classes = load_data(args.data_dir, args.dataset, args.T)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=1,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True, 
        drop_last=False
    )

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # test
    model = model_map[args.model](num_classes=num_classes, T=args.T, input_shape=input_shape, threshold=args.flat_width)
    model.load_state_dict(
        torch.load(f'{args.model_dir}/{args.dataset}_{args.model}_T{args.T}_thr{args.flat_width}_seed{args.seed}_ckpt_best.pth', map_location='cpu'))
    model.to(device)
    
    model.eval()
    # latency & real energy cost 
    elapse = 0.
    with torch.no_grad():
        cnt = 0

        pc.printFullReport(pc.getDevice())
        pl = pc.PowerLogger(interval=0.05)
        pl.start()
        time.sleep(5)
        pl.recordEvent(name='Process Start')

        for image, target in data_loader_test:
            image, target = image.to(device), target.to(device)
            if cnt >= 10:
                break
            
            start_time = time.time()
            output = model(image)
            elapse += (time.time() - start_time)

            cnt += 1

        print(f'inference time per sample: {(time.time() - start_time) / cnt:.3f}s')

        time.sleep(5)
        pl.stop()
        filename = f'./{args.model_type}/{args.neuron_type}/{args.dataset}/{args.seed}/test/'
        pl.showDataTraces(filename=filename)
        print(str(pl.eventLog))
        pc.printFullReport(pc.getDevice())


    # SOPs theoretical energy cost
    # TODO


