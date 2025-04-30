import time
import argparse
import torch

from Q_SpikingVGG9 import SpikingVGG9

parser = argparse.ArgumentParser()
parser.add_argument("--weight_dir", type=str, default="./saved_models")
parser.add_argument("--dataset", type=str, default="TinyImageNet")
parser.add_argument("--seed", type=int, default=42) # 41, 42 , 43
parser.add_argument("--neuron_type", type=str, default="LIF")
parser.add_argument("--model_type", type=str, default="SpikingVGG9")
args = parser.parse_args()


input_shape = None
if args.dataset == "CIFAR10":
    input_shape = (3, 32, 32)
    num_classes = 10
    T = 4
    sample_name = 'cifar10-T4-size32.pt'

elif args.dataset == "CIFAR10DVS":
    input_shape = (2, 128, 128)
    num_classes = 10
    T = 16
    sample_name = 'cifar10dvs-T10-size64.pt'

elif args.dataset == "TinyImageNet":
    input_shape = (3, 64, 64)
    num_classes = 200
    T = 4
    sample_name = 'imagenet-T4-size64.pt'
elif args.dataset == "DVSGesture":
    input_shape = (2, 128, 128)
    num_classes = 11
    T = 16
    sample_name = 'dvsgesture-T16-size64.pt'

model_map = {
    "SpikingVGG9": SpikingVGG9
}

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

params = torch.load(f'{args.weight_dir}/quantized_best_{args.model_type}_{args.neuron_type}_{args.dataset}_{args.seed}.pth')
model = model_map[args.model_type](input_shape=input_shape, T=T, num_classes=num_classes, neuron_type=args.neuron_type)
model.load_state_dict(params)

model.to(device)
model.eval()

test_samples = torch.load(f'../../samples/{sample_name}')
test_samples = test_samples.to(device)
cnt = test_samples.shape[0] # B
with torch.no_grad():
    start_time = time.time()
    for sample in test_samples:
        output = model(sample.unsqueeze(0))
    print(f'inference time per sample: {(time.time() - start_time) / cnt:.3f}s')
