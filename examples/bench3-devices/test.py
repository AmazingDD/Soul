import time
import argparse
import torch

from sew_resnet import SewResNet18

parser = argparse.ArgumentParser(description='Testing for heterogeneous devices')
parser.add_argument('-seed', default=42, type=int)
parser.add_argument('-model_dir', type=str, default='./saved_models/', help='root dir for saving trained model')
parser.add_argument('-dataset', default='CIFAR10', help='dataset name')
parser.add_argument('-model', default='SpikingVGG9', help='model name')
args = parser.parse_args()

if args.dataset == 'CIFAR10':
    num_classes = 10
    T = 4
    input_shape = (3, 32, 32)
    sample_name = 'cifar10-T4-size32.pt'
elif args.dataset == 'CIFAR10DVS':
    num_classes = 10
    T = 10
    input_shape = (2, 64, 64)
    sample_name = 'cifar10dvs-T10-size64.pt'
elif args.dataset == 'DVSGesture':
    num_classes = 11
    T = 16
    input_shape = (2, 64, 64)
    sample_name = 'dvsgesture-T16-size64.pt'
elif args.dataset == 'TinyImageNet':
    num_classes = 200
    T = 4
    input_shape = (3, 64, 64)
    sample_name = 'imagenet-T4-size64.pt'
else:
    raise ValueError(f'Invalid dataset: {args.dataset}')

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

test_samples = torch.load(f'../samples/{sample_name}')
test_samples = test_samples.to(device)

model = SewResNet18(num_classes=num_classes, T=T, input_shape=input_shape)
model.load_state_dict(torch.load(f'{args.model_dir}/{args.model}_{args.dataset}_T{T}_ckpt_best.pth', map_location='cpu'))
model.to(device)

model.eval()
cnt = test_samples.shape[0] # B
with torch.no_grad():
    start_time = time.time()
    for sample in test_samples:
        output = model(sample.unsqueeze(0))
    print(f'inference time per sample: {(time.time() - start_time) / cnt:.3f}s')
