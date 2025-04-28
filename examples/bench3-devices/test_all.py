import time
import argparse
import torch
import os

from sew_resnet import SewResNet18

parser = argparse.ArgumentParser(description='Testing for heterogeneous devices')
parser.add_argument('-seed', default=42, type=int)
parser.add_argument('-model_dir', type=str, default='./saved_models/', help='root dir for saving trained model')
args = parser.parse_args()

# ========== 数据集配置 ==========
dataset_config = {
    'CIFAR10': {'num_classes': 10, 'T': 4, 'input_shape': (3, 32, 32),
                'sample_name': 'cifar10-T4-size32.pt'},
    'CIFAR10DVS': {'num_classes': 10, 'T': 10, 'input_shape': (2, 64, 64),
                   'sample_name': 'cifar10dvs-T10-size64.pt'},
    'DVSGesture': {'num_classes': 11, 'T': 16, 'input_shape': (2, 64, 64),
                   'sample_name': 'dvsgesture-T16-size64.pt'},
    'TinyImageNet': {'num_classes': 200, 'T': 4, 'input_shape': (3, 64, 64),
                     'sample_name': 'imagenet-T4-size64.pt'}
}

current_dir = os.path.dirname(os.path.abspath(__file__))

for dataset_name, config in dataset_config.items():
    # ========== 数据加载 ==========
    samples_path = os.path.join(current_dir, "../samples", config['sample_name'])
    test_samples = torch.load(samples_path)  # 原始数据不加载到设备

    # ========== 模型加载 ==========
    weight_filename = f"SewResNet18_{dataset_name}_T{config['T']}_ckpt_best.pth"
    weight_path = os.path.join(current_dir, weight_filename)

    model = SewResNet18(
        num_classes=config['num_classes'],
        T=config['T'],
        input_shape=config['input_shape']
    )
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))  # 先加载到CPU
    model.eval()

    # ========== 多设备测试 ==========
    for device_type in ['cpu', 'cuda']:
        if device_type == 'cuda' and not torch.cuda.is_available():
            continue

        device = torch.device(device_type)
        device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else "CPU"
        print(f"\nTesting {dataset_name} on {device_name}")

        # 移动数据到目标设备
        samples = test_samples.to(device)
        model.to(device)

        # ========== 推理计时 ==========
        cnt = samples.shape[0]
        with torch.no_grad():
            start = time.time()
            for sample in samples:
                _ = model(sample.unsqueeze(0))
            elapsed = time.time() - start
            print(f"Inference time per sample: {elapsed / cnt:.3f}s")