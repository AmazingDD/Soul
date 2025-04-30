import time
import argparse
import torch
import torch.nn as nn
import copy

from Q_SpikingVGG9 import SpikingVGG9
from fuse import fuse_rateBatchNorm_xs
from l_quantize import Aggregated_Spiking_Layer

from spikingjelly.activation_based import layer, neuron

parser = argparse.ArgumentParser()
parser.add_argument("--weight_dir", type=str, default="/media/orin/57DC-987E/saved_models_quantization")
parser.add_argument("--dataset", type=str, default="TinyImageNet")
parser.add_argument("--seed", type=int, default=42) # 41, 42 , 43
parser.add_argument("--neuron_type", type=str, default="LIF")
parser.add_argument("--model_type", type=str, default="SpikingVGG9")
args = parser.parse_args()

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
    sample_name = 'cifar10dvs-T10-size128.pt'

elif args.dataset == "TinyImageNet":
    input_shape = (3, 64, 64)
    num_classes = 200
    T = 4
    sample_name = 'imagenet-T4-size64.pt'
elif args.dataset == "DVSGesture":
    input_shape = (2, 128, 128)
    num_classes = 11
    T = 16
    sample_name = 'dvsgesture-T16-size128.pt'

model_map = {
    "SpikingVGG9": SpikingVGG9
}

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

params = torch.load(f'{args.weight_dir}/quantized_best_{args.model_type}_{args.neuron_type}_{args.dataset}_{args.seed}.pth')
model = model_map[args.model_type](input_shape=input_shape, T=T, num_classes=num_classes, neuron_type=args.neuron_type)
new_model = aggregate_spikingvgg9_layers(model)
new_model.apply(fuse_rateBatchNorm_xs)


new_model.load_state_dict(params)

new_model.to(device)
new_model.eval()

test_samples = torch.load(f'../../samples/{sample_name}')
test_samples = test_samples.to(device)

cnt = test_samples.shape[0] # B
print(f'{args.dataset} sample num {cnt}')
with torch.no_grad():
    start_time = time.time()
    for sample in test_samples:
        output = new_model(sample.unsqueeze(0))
    print(f'inference time per sample: {(time.time() - start_time) / cnt * 1000:.3f}ms')
    

