import torch
import time
import argparse
import requests
import json

def send_with_bandwidth_limit(url, data, max_bandwidth_kbps):
    json_data = json.dumps(data)
    data_bytes = json_data.encode('utf-8')
    total_size = len(data_bytes)

    # size per sending
    chunk_size = 1024  # 1KB的块
    delay_per_chunk = chunk_size / (max_bandwidth_kbps * 1024)

    def generate():
        for i in range(0, total_size, chunk_size):
            chunk = data_bytes[i:i+chunk_size]
            yield chunk
            time.sleep(delay_per_chunk)
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=generate(), headers=headers)

    return response


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
    payload = {"image": test_sample.numpy().tolist()}
    # response = requests.post(SERVER_URL, json=payload)

    response = send_with_bandwidth_limit(
        SERVER_URL,
        payload,
        max_bandwidth_kbps=1000  # 1mbps for default edge bandwidth
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Sample {idx}: Prediction = {result['prediction']}")
    else:
        print(f"Sample {idx}: Failed with error {response.text}")
latency = (time.time() - start_time) / cnt
print(f'Latency per sample for cloud service: {latency:.4f}s')