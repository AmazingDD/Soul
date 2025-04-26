import os
import numpy as np
from tqdm import tqdm
import random
from D3LIF import D3LIF_Quant
from l_quantize import Symmetric_Quantize, Aggregated_Spiking_Layer
import os
import copy
import argparse
from spikingjelly.activation_based import functional, layer, neuron
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.activation_based.neuron import LIFNode
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from dataset import NCaltech101, NMNIST, CIFAR10DVS, TinyImageNetDataset
from Q_SewResNet import SEWResNet18
from Q_SpikingVGG9 import SpikingVGG9
from typing import Optional

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility.")

model_map = {
    "SpikingVGG9": SpikingVGG9,
    "SEWResNet18": SEWResNet18
}

def get_dataloader(dataset_name, data_dir):
    if not os.path.exists(data_dir):
        raise ValueError(f'Invalid path: {data_dir}') 
    if dataset_name == "CIFAR10":
        cifar10_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
        ])
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=cifar10_transform, download=True)
    elif dataset_name == "TinyImageNet":
        tinyimagenet_transform = transforms.Compose([
            transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        testset = TinyImageNetDataset(data_dir, train=False, transform=tinyimagenet_transform)

    elif dataset_name == "DVSGesture":
        testset = DVS128Gesture(root=data_dir, train=False, data_type='frame', frames_number=16, split_by='number')
    else:
        raise ValueError(f"Invalid dataset_name: {dataset_name}")
    
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    return testloader

def aggregate_spikingvgg9_layers(original_model):
    # 创建新模型实例，使用相同的构造函数参数
    new_model = SpikingVGG9(
        num_classes=original_model.num_classes,
        T=original_model.T,
        neuron_type=original_model.neuron_type,
        input_shape=original_model.input_shape
    )
    
    # 处理features部分
    original_features = list(original_model.features.children())
    new_features_layers = []
    layer_id = 0
    i = 0
    while i < len(original_features):
        current_layer = original_features[i]
        # 检查是否是Conv-BN-LIF组合
        if isinstance(current_layer, layer.Conv2d) and i + 2 < len(original_features):
            next_bn = original_features[i + 1]
            next_lif = original_features[i + 2]
            if isinstance(next_bn, layer.BatchNorm2d) and isinstance(next_lif, neuron.LIFNode):
                # 深拷贝这三个层并合并
                aggregated_layer = Aggregated_Spiking_Layer(
                    copy.deepcopy(current_layer),
                    copy.deepcopy(next_bn),
                    copy.deepcopy(next_lif),
                    layer_id
                )
                new_features_layers.append(aggregated_layer)
                layer_id += 1
                i += 3
                continue
        # 非组合层，直接拷贝
        new_features_layers.append(copy.deepcopy(current_layer))
        i += 1
    new_model.features = nn.Sequential(*new_features_layers)
    
    # 处理mlp部分，直接深拷贝
    new_mlp_layers = []
    for original_layer in original_model.mlp.children():
        new_mlp_layers.append(copy.deepcopy(original_layer))
    new_model.mlp = nn.Sequential(*new_mlp_layers)
    
    # 处理fc层
    new_model.fc = copy.deepcopy(original_model.fc)
    
    return new_model

def test(args, model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            logit = model(inputs)
            _, predicted = torch.max(logit.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            functional.reset_net(model)

    acc = 100 * correct / total
    print(f'Accuracy on the {args.dataset} test images: {acc:.2f}%')
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../bench1-neuronal/data/CIFAR10")
    parser.add_argument("--weight_dir", type=str, default="./saved_models")
    parser.add_argument("--model_type", type=str, default="SpikingVGG9")
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--seed", type=int, default=42) # 41, 42 , 43
    parser.add_argument("--neuron_type", type=str, default="LIF")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    input_shape = None
    if args.dataset == "CIFAR10":
        input_shape = (3, 32, 32)
        num_classes = 10
        T = 4
    elif args.dataset == "TinyImageNet":
        input_shape = (3, 64, 64)
        num_classes = 200
        T = 4
    elif args.dataset == "DVSGesture":
        input_shape = (2, 128, 128)
        num_classes = 11
        T = 16
    # print(model_map[args.model_type])
    testloader = get_dataloader(args.dataset, args.data_dir)
    original_model = model_map[args.model_type](num_classes=num_classes, T=T, neuron_type=args.neuron_type, input_shape=input_shape).cuda()
    print(original_model)
    
    original_model.load_state_dict(
        torch.load(os.path.join(
        args.weight_dir, f'best_{args.model_type}_{args.neuron_type}_{args.dataset}_{args.seed}.pth'), weights_only=True, map_location='cpu')
    )
    
    print("Before Quantization")
    test(args, model=original_model, testloader=testloader)

    new_model = aggregate_spikingvgg9_layers(original_model)
    
    # # 验证参数一致性
    # with torch.no_grad():
    #     for (n1, p1), (n2, p2) in zip(original_model.named_parameters(), new_model.named_parameters()):
    #         print(f"Layer {n1} params match: {torch.allclose(p1, p2)}")
    print(new_model)
    
    # 已转换成功，TODO Need Debug
    new_model.features.apply(Symmetric_Quantize(Symmetric_Quantize.Target_Precision.INT8))
    weights = new_model.state_dict()
    
    for name, module in new_model.named_modules():
        if isinstance(module, D3LIF_Quant): # mem_update
            weights[f'{name}.v'] = module.v
            weights[f'{name}.bias'] = module.bias

    print("After Quantization")
    test(args, model=new_model, testloader=testloader)