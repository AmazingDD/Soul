import time
import argparse

import torch

from utils import *
from model import SpikingVGG9, SewResNet18

import power_check as pc

model_map = {
    "SpikingVGG9": SpikingVGG9,
    "SewResNet18": SewResNet18,
}

def parse_args():
    parser = argparse.ArgumentParser(description='Unstructured weight pruning for SNNs')
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-model_dir', type=str, default='./saved_models/', help='root dir for saving trained model')
    parser.add_argument('-dataset', default='CIFAR10', help='dataset name')
    parser.add_argument('-model', default='SpikingVGG9', help='model name')
    # pruning parameter
    parser.add_argument('-thr', '--flat-width', type=float, default=0.1) # also known as threshold
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    
    # make sure you have implemented sample_generator.py
    if args.dataset == 'CIFAR10':
        num_classes = 10
        T = 4
        input_shape = (3, 32, 32)
        sample_name = 'cifar10-T4-size32.pt'
    elif args.dataset == 'DVSGesture':
        num_classes = 11
        T = 16
        input_shape = (2, 128, 128)
        sample_name = 'cifar10-T16-size128.pt'
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    test_samples = torch.load(f'./samples/{sample_name}')
    test_samples = test_samples.to(device)

    # test
    model = model_map[args.model](num_classes=num_classes, T=T, input_shape=input_shape, threshold=args.flat_width)
    model.load_state_dict(
        torch.load(f'{args.model_dir}/{args.dataset}_{args.model}_T{args.T}_thr{args.flat_width}_seed{args.seed}_ckpt_best.pth', map_location='cpu'))
    model.to(device)
    
    model.eval()
    # latency & real energy cost 
    with torch.no_grad():
        cnt = test_samples.shape[0] # B

        pc.printFullReport(pc.getDevice())
        pl = pc.PowerLogger(interval=0.05)
        pl.start()
        time.sleep(5)
        pl.recordEvent(name='Process Start')

        start_time = time.time()
        for sample in test_samples:
            output = model(sample.unsqueeze(0))
        print(f'inference time per sample: {(time.time() - start_time) / cnt:.3f}s')

        time.sleep(5)
        pl.stop()
        filename = f'./{args.model_type}/{args.neuron_type}/{args.dataset}/{args.seed}/test/'
        pl.showDataTraces(filename=filename)
        print(str(pl.eventLog))
        pc.printFullReport(pc.getDevice())


    # SOPs theoretical energy cost
    # TODO


