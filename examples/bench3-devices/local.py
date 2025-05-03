import torch
import time
import argparse
import requests

parser = argparse.ArgumentParser(description='Testing for heterogeneous devices')
parser.add_argument('-dataset', default='CIFAR10', help='dataset name')
args = parser.parse_args()

if args.dataset == 'CIFAR10':
    sample_name = 'cifar10-T4-size32.pt'
elif args.dataset == 'CIFAR10DVS':
    sample_name = 'cifar10dvs-T10-size64.pt'
elif args.dataset == 'DVSGesture':
    sample_name = 'dvsgesture-T16-size64.pt'
elif args.dataset == 'TinyImageNet':
    sample_name = 'imagenet-T4-size64.pt'
else:
    raise ValueError(f'Invalid dataset: {args.dataset}')

SERVER_URL = "http://127.0.0.1:3000/predict"

test_samples = torch.load(f'../samples/{sample_name}')
print('test_samples shape: ', test_samples.shape)

cnt = test_samples.shape[0]
start_time = time.time()
for idx, test_sample in enumerate(test_samples):
    payload = {"image": test_sample}
    response = requests.post(SERVER_URL, json=payload)
    if response.status_code == 200:
        result = response.json()
        print(f"Sample {idx}: Prediction = {result['prediction']}")
    else:
        print(f"Sample {idx}: Failed with error {response.text}")
latency = time.time() / cnt
print(f'Latency per sample for cloud service: {latency:.4f}s')