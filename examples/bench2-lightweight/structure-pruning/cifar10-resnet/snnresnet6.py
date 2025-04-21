from __future__ import absolute_import
import math
import torch.nn as nn
from channel_selection import channel_selection
from spikingjelly.clock_driven import layer,neuron
#from layers import *
from spikingjelly.clock_driven import functional, surrogate, layer, neuron
from typing import Callable, overload
from maskk import PruningLayer
import torch

__all__ = ['resnet']
print('aaabbbccc')
"""
preactivation resnet with bottleneck design.
"""
def check_backend(backend: str):
    if backend == 'torch':
        return
    elif backend == 'cupy':
        assert cupy is not None, 'CuPy is not installed! You can install it from "https://github.com/cupy/cupy".'
    elif backend == 'lava':
        assert slayer is not None, 'Lava-DL is not installed! You can install it from "https://github.com/lava-nc/lava-dl".'
    else:
        raise NotImplementedError(backend)

class myMultiStepIFNode(neuron.IFNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, backend='torch', lava_s_cale=1 << 6):

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)

        self.register_memory('v_seq', None)

        check_backend(backend)

        self.backend = backend

        self.lava_s_cale = lava_s_cale

        if backend == 'lava':
            self.lava_neuron = self.to_lava()
        else:
            self.lava_neuron = None


    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]

        if self.backend == 'torch':
            spike_seq = []
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
                self.v_seq.append(self.v.unsqueeze(0))
            spike_seq = torch.cat(spike_seq, 0)
            self.v_seq = torch.cat(self.v_seq, 0)
            return spike_seq,self.v_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)

            spike_seq, self.v_seq = neuron_kernel.MultiStepIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.v = self.v_seq[-1].clone()

            return spike_seq

        elif self.backend == 'lava':
            if self.lava_neuron is None:
                self.lava_neuron = self.to_lava()

            spike, self.v = lava_exchange.lava_neuron_forward(self.lava_neuron, x_seq, self.v)

            return spike

        else:
            raise NotImplementedError(self.backend)


    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'


    def to_lava(self):
        return lava_exchange.to_lava_neuron(self)


    def reset(self):
        super().reset()
        if self.lava_neuron is not None:
            self.lava_neuron.current_state.zero_()
            self.lava_neuron.voltage_state.zero_()


class Basicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Basicblock, self).__init__()
        self.prune1 = PruningLayer()
        self.prune2 = PruningLayer()
        self.bn1 = layer.SeqToANNContainer(nn.BatchNorm2d(inplanes))
        self.select = channel_selection(inplanes)
        self.relu1 = myMultiStepIFNode(detach_reset=True)
        self.conv1 = layer.SeqToANNContainer(nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride,padding=1,bias=False))
        self.bn2 = layer.SeqToANNContainer(nn.BatchNorm2d(cfg[1]))
        self.relu2 = myMultiStepIFNode(detach_reset=True)
        self.conv2 = layer.SeqToANNContainer(nn.Conv2d(cfg[1], planes * 1, kernel_size=3,stride=1,
                               padding=1, bias=False))


        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        #print(x.shape)
        out = self.bn1(x) #111
        #print(out.shape)
        out = self.select(out)
        #print(out.shape)
        out,v1 = self.relu1(out) #111
        v1 = v1.detach()
        out = self.prune1(out, v1)
        #print(out.shape)
        out = self.conv1(out)
        #print(out.shape)

        out = self.bn2(out) #222
        #print(out.shape)
        out,v2 = self.relu2(out)
        v2 = v2.detach()
        out = self.prune2(out, v2 )  #222
        #print(out.shape)
        out = self.conv2(out)




        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #print(residual.shape)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class resnet(nn.Module):
    def __init__(self, depth=164, dataset='cifar10', cfg=None):
        super(resnet, self).__init__()


        n =2
        block = Basicblock

        if cfg is None:
            # Construct config variable.
            #cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
            cfg = [[64, 64]*2 , [64,128],[128, 128] , [128,256],[256, 256] ,[256,512],[512,512],[512]]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 64

        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, padding=1,
                                bias=False)
        self.bn11 = nn.BatchNorm2d(64)

        self.relu11 = neuron.MultiStepIFNode(detach_reset=True)
        self.layer1 = self._make_layer(block, 64, n, cfg = cfg[0:2*n])
        self.layer2 = self._make_layer(block, 128, n, cfg = cfg[2*n:4*n], stride=2)
        self.layer3 = self._make_layer(block, 256, n, cfg = cfg[4*n:6*n], stride=2)
        self.layer4 = self._make_layer(block, 512, n, cfg=cfg[6* n:8* n], stride=2)
        self.bn = layer.SeqToANNContainer(nn.BatchNorm2d(512 * block.expansion))
        self.select = channel_selection(512 * block.expansion)
        self.relu = myMultiStepIFNode(detach_reset=True)
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.prune1 = PruningLayer()


        self.fc = nn.Linear(512, 10)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layer.SeqToANNContainer(nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False))
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:2], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[2*i: 2*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn11(self.conv11(x))
        x.unsqueeze_(0)
        x = x.repeat(4, 1, 1, 1, 1)
        x = self.relu11(x)
        #print(x.shape)
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.layer4(x)
        x = self.bn(x)
        x = self.select(x)
        x,v = self.relu(x)
        v = v.detach()
        x = self.prune1(x, v)

        x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)
        #print(x.shape)
        x = torch.flatten(x, 2)
        #print(x.shape)
        return self.fc(x.mean(dim=0))