import torch
import argparse
from flask import Flask, request, jsonify
import time

from sew_resnet import SewResNet18

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BandwidthLimiter:
    def __init__(self, app, max_bandwidth):
        self.app = app
        self.max_bandwidth = max_bandwidth  # 最大带宽（字节/秒）
    
    def __call__(self, environ, start_response):
        # 对于上传的数据进行带宽限制
        content_length = int(environ.get('CONTENT_LENGTH', 0))
        if content_length > 0:
            start_time = time.time()
            expected_time = content_length / self.max_bandwidth
            actual_time = time.time() - start_time
            if actual_time < expected_time:
                time.sleep(expected_time - actual_time)
        
        return self.app(environ, start_response)
    
app.wsgi_app = BandwidthLimiter(app.wsgi_app, max_bandwidth=1024 * 10000) # 10mbps for server bandwidth

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']
        image = torch.tensor(image_data) # (T, C, H, W)
        image = image.unsqueeze(0)  # (1, T,C,H,W)
        image = image.to(device)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = outputs.max(1) 

        return jsonify({"prediction": int(predicted.item())})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# this should be run on server side
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing for heterogeneous devices')
    parser.add_argument('-model_dir', type=str, default='./saved_models/', help='root dir for saving trained model')
    parser.add_argument('-dataset', default='CIFAR10', help='dataset name')
    parser.add_argument('-model', default='SewResNet18', help='model name')
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

    model = SewResNet18(T=T, num_classes=num_classes, input_shape=input_shape)
    model.load_state_dict(torch.load(f'{args.model_dir}/{args.model}_{args.dataset}_T{T}_ckpt_best.pth', map_location='cpu'))
    model.to(device)
    model.eval()


    app.run(host='0.0.0.0', port=3000)