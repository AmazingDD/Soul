import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from channel_selection import channel_selection
from spikingjelly.clock_driven import functional
from snnresnet6 import *
_seed_ = 2020
random.seed(2020)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import utils
import numpy as np

np.random.seed(_seed_)



device='cuda:1'
# Prune settings
#-1 preresnet
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=164,
                    help='depth of the resnet')
parser.add_argument('--percent', type=float, default=0.1,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, download=True,transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])),
        batch_size=128, shuffle=True,drop_last=True, **kwargs)


    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    correct = 0
    with torch.no_grad():

        for image, target in metric_logger.log_every(test_loader, print_freq=100, header='Test:'):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            #loss = criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]

            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(100. * correct / len(test_loader.dataset))

    acc1, acc5 = metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(acc1)

    '''for data, target in test_loader:

        data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        functional.reset_net(model)

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))'''
    return acc1

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = resnet().to(device)
modules = list(model.modules())
total = sum([param.nelement() for param in model.parameters()])
print('  + Number of params: %.2fM' % (total / 1e6))
'''for layer_id in range(len(modules)):
    #print(layer_id)
    print(modules[39].kernel_size)'''


model.to(device)

checkpoint = torch.load('./vgg16cifar10best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
#checkpoint = torch.load('./vggylbest.pt')
#model.load_state_dict(checkpoint)
test(model)
cfg_mask = []


import csv


with open('mask.csv', 'r') as file:
    
    csv_reader = csv.reader(file)
    
    cfg_mask = []
    
    for row in csv_reader:
       
        row=list(map(lambda x: int(float(x)) if x != '' else x, row))
        #print(row)
       
        row=torch.tensor(row)
        print(torch.sum(row == 1).item())
        
        cfg_mask.append(row)
# simple test model after Pre-processing prune (simple set BN scales to zeros)


# simple test model after Pre-processing prune (simple set BN scales to zeros)


acc = test(model)

print("Cfg:")

cfg=cfg = [[64, 64],[64-2, 64-12], [64-5,128-44],[128-23, 128-47] , [128-23,256-145],[256-86, 256-132] ,[256-159,512-490],[512-373,512-511],[512-252]]
newmodel = resnet(cfg=cfg)
print(cfg)
newmodel.to(device)

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(args.save, "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc))

old_modules = list(model.modules())
new_modules = list(newmodel.modules())
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
conv_count = 0


'''for layer_id in range(len(old_modules)):
    print(layer_id)
    print(new_modules[layer_id])'''

bn_count = 0

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, nn.BatchNorm2d):
        if bn_count == 0:
            print('1111')
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
            bn_count += 1
            continue
        print('22222')
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))

        if isinstance(old_modules[layer_id + 1], channel_selection):
            print('aaa')
            # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

            # We need to set the channel selection layer.
            m2 = new_modules[layer_id + 1]
            #print(m2.indexes.data.shape)
            m2.indexes.data.zero_()
            m2.indexes.data[idx1.tolist()] = 1.0

            layer_id_in_cfg += 1
            print(layer_id_in_cfg)
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                #print(len(cfg_mask))
                end_mask = cfg_mask[layer_id_in_cfg]
                print(end_mask)
        else:
            print('bbb')
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                print(len(cfg_mask))
                end_mask = cfg_mask[layer_id_in_cfg]
            print(layer_id_in_cfg)
    elif isinstance(m0, nn.Conv2d):
        if conv_count == 0:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        #print(old_modules[layer_id - 1])
        if isinstance(old_modules[layer_id-3], channel_selection) or isinstance(old_modules[layer_id-3], nn.BatchNorm2d) or isinstance(old_modules[layer_id-4], channel_selection):
            # This convers the convolutions in the residual block.
            # The convolutions are either after the channel selection layer or after the batch normalization layer.     
            print(layer_id_in_cfg)
            conv_count += 1
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            print(idx0)
            print(idx1)

            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            #print('aaa')
            print(w1.shape)
            #print(w1[idx1.tolist(), :, :, :].shape)
            # If the current convolution is not the last convolution in the residual block, then we can change the
            # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
            if conv_count % 2 != 1:
                w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            continue

        # We need to consider the case where there are downsampling convolutions.
        # For these convolutions, we just copy the weights.
        m1.weight.data = m0.weight.data.clone()
    elif isinstance(m0, nn.Linear):
        #print(layer_id)
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        #print('aaa')
        m1.weight.data = m0.weight.data[:, idx0].clone()
        #print(m1.weight.data.shape)
        m1.bias.data = m0.bias.data.clone()


torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
'''for layer_id in range(len(old_modules)):
    print(layer_id)
    print(new_modules[layer_id])'''
#print(newmodel)
model = newmodel
test(model)


from spikingjelly.clock_driven.monitor import Monitor
import numpy as np
from spikingjelly.clock_driven import functional, surrogate, layer, neuron
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from spikingjelly.clock_driven.monitor import Monitor
import numpy as np
try:
    from spikingjelly.cext import neuron as cext_neuron
except ImportError:
    cext_neuron = None
class my(Monitor):
    def __init__(self, net: nn.Module, device: str = None, backend: str = 'numpy'):
        super().__init__(net, device, backend)
        self.module_dict = dict()
        for name, module in net.named_modules():
            if (cext_neuron is not None and isinstance(module, cext_neuron.BaseNode)) or isinstance(module, neuron.BaseNode):
                self.module_dict[name] = module
                #setattr(module, 'monitor', self)

        # 'torch' or 'numpy'
        self.net = net
        self.backend = backend

        if isinstance(device, str) and self.backend == 'torch':
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise ValueError('Expected a cuda or cpu device, but got: {}'.format(device))
    def forward_hook(self, module, input, output):
        output=output[0]
        if module.__class__.__name__.startswith('MultiStep'):
            output_shape = output.shape
            data = output.view([-1,] + list(output_shape[2:])).clone() 
        else:
            data = output.clone()

        # Numpy
        if self.backend == 'numpy':
            data = data.cpu().numpy()
            if module.neuron_cnt is None:
                module.neuron_cnt = data[0].size 
            module.firing_time += np.sum(data) 
            module.cnt += data.size 
            fire_mask = (np.sum(data, axis=0) > 0)

        # PyTorch
        else:
            data = data.to(self.device)
            if module.neuron_cnt is None:
                module.neuron_cnt = data[0].numel()
            module.firing_time += torch.sum(data)
            module.cnt += data.numel()
            fire_mask = (torch.sum(data, dim=0) > 0)
        
        
        module.fire_mask = fire_mask | module.fire_mask 


    def cnt(self,all:bool=True,module_name:str=None) -> torch.Tensor or float:

        if all:
            ttl_firing_time = 0
            ttl_cnt = 0
            for name, module in self.module_dict.items():
                ttl_firing_time += module.firing_time
                ttl_cnt += module.cnt
            return ttl_firing_time
        else:
            if module_name not in self.module_dict.keys():
                raise ValueError(f'Invalid module_name \'{module_name}\'')
            module = self.module_dict[module_name]
            return module.firing_time / module.cnt
device='cuda:1'

dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

devset = torchvision.datasets.CIFAR10(root='./', train=False,
        download=True, transform=dev_transformer)
devloader = torch.utils.data.DataLoader(devset, batch_size=64,
        shuffle=True,  pin_memory='cpu',drop_last=True)
        
#cfg=[[64, 64-3],[64-10, 64-15], [64-11,128-55],[128-27, 128-60] , [128-36,256-172],[256-152, 256-194] ,[256-196,512-497],[512-462,512-509],[512-289]]
#net = resnet(cfg=cfg).to(device)

print('de11cba')


#checkpoint=torch.load('./pruned.pth.tar',map_location='cpu')
net=model.to(device)

#net.load_state_dict(checkpoint['state_dict'])
mon = my(net, device='cpu',backend='torch')
mon.enable()

r=0.0
with torch.no_grad():
    for data_batch, labels_batch in devloader:
        # move to GPU if available
        
    
        data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)
        # fetch the next evaluation batch
        
        #data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
    
        
        # compute model output
        net(data_batch)
        #print('a')
    
        temp = mon.cnt()
        #print('temp:' + str(temp))
    
        r = r + temp
    
        mon.reset()
        functional.reset_net(net)
sss=r/((devloader.__len__())*64)
print('sss:' + str(sss))
print((devloader.__len__())*64)
print('  + Number of params: %.2fK' % (sss / 1e3))