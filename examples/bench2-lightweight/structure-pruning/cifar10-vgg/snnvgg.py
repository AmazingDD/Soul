
from maskk import PruningLayer
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from spikingjelly.clock_driven import functional, surrogate, layer, neuron
from typing import Callable, overload
__all__ = [
    'VGG', 'snnvgg11', 'snnvgg11_bn', 'snnvgg13', 'snnvgg13_bn', 'snnvgg16', 'snnvgg16_bn',
    'snnvgg19_bn', 'snnvgg19',
]
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




class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self,cfg):
        super(VGG, self).__init__()


        self.layer1=layer.SeqToANNContainer(nn.Conv2d(3, cfg[0], kernel_size=3, padding=1, bias=False),
                                            nn.BatchNorm2d( cfg[0]),
                                           )
        self.neuron1=myMultiStepIFNode(detach_reset=True)
        self.prune1 = PruningLayer()

        self.layer2 = layer.SeqToANNContainer(nn.Conv2d( cfg[0],  cfg[1], kernel_size=3, padding=1, bias=False),
                                              nn.BatchNorm2d( cfg[1]),
                                            )
        self.neuron2=myMultiStepIFNode(detach_reset=True)
        self.prune2 = PruningLayer()
        self.pool1=layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = layer.SeqToANNContainer(nn.Conv2d( cfg[1], cfg[3], kernel_size=3, padding=1, bias=False),
                                              nn.BatchNorm2d( cfg[3]),
                                              )
        self.neuron3=myMultiStepIFNode(detach_reset=True)
        self.prune3 = PruningLayer()

        self.layer4 = layer.SeqToANNContainer(nn.Conv2d( cfg[3],  cfg[4], kernel_size=3, padding=1, bias=False),
                                              nn.BatchNorm2d( cfg[4]),
                                             )
        self.neuron4=myMultiStepIFNode(detach_reset=True)
        self.prune4 = PruningLayer()
        self.pool2=layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = layer.SeqToANNContainer(nn.Conv2d( cfg[4], cfg[6], kernel_size=3, padding=1, bias=False),
                                              nn.BatchNorm2d(cfg[6]),
                                              )
        self.neuron5=myMultiStepIFNode(detach_reset=True)
        self.prune5 = PruningLayer()

        self.layer6 = layer.SeqToANNContainer(nn.Conv2d(cfg[6], cfg[7], kernel_size=3, padding=1, bias=False),
                                              nn.BatchNorm2d(cfg[7]),
                                              )
        self.neuron6=myMultiStepIFNode(detach_reset=True)
        self.prune6 = PruningLayer()

        self.layer7 = layer.SeqToANNContainer(nn.Conv2d(cfg[7], cfg[8], kernel_size=3, padding=1, bias=False),
                                              nn.BatchNorm2d(cfg[8]),
                                              )
        self.neuron7=myMultiStepIFNode(detach_reset=True)
        self.prune7 = PruningLayer()
        self.pool3=layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer8 = layer.SeqToANNContainer(nn.Conv2d(cfg[8], cfg[10], kernel_size=3, padding=1, bias=False),
                                              nn.BatchNorm2d(cfg[10]),
                                             )
        self.neuron8=myMultiStepIFNode(detach_reset=True)
        self.prune8 = PruningLayer()

        self.layer9 = layer.SeqToANNContainer(nn.Conv2d(cfg[10], cfg[11], kernel_size=3, padding=1, bias=False),
                                              nn.BatchNorm2d(cfg[11]),
                                             )
        self.neuron9=myMultiStepIFNode(detach_reset=True)
        self.prune9 = PruningLayer()

        self.layer10 = layer.SeqToANNContainer(nn.Conv2d(cfg[11], cfg[12], kernel_size=3, padding=1, bias=False),
                                              nn.BatchNorm2d(cfg[12]),
                                             )
        self.neuron10=myMultiStepIFNode(detach_reset=True)
        self.prune10 = PruningLayer()

        self.pool4=layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer11 = layer.SeqToANNContainer(nn.Conv2d(cfg[12], cfg[14], kernel_size=3, padding=1, bias=False),
                                              nn.BatchNorm2d(cfg[14]),
                                             )
        self.neuron11=myMultiStepIFNode(detach_reset=True)
        self.prune11 = PruningLayer()

        self.layer12 = layer.SeqToANNContainer(nn.Conv2d(cfg[14], cfg[15], kernel_size=3, padding=1, bias=False),
                                              nn.BatchNorm2d(cfg[15]),
                                              )
        self.neuron12=myMultiStepIFNode(detach_reset=True)
        self.prune12 = PruningLayer()

        self.layer13 = layer.SeqToANNContainer(nn.Conv2d(cfg[15], cfg[16], kernel_size=3, padding=1, bias=False),
                                              nn.BatchNorm2d(cfg[16]),
                                             )
        self.neuron13=myMultiStepIFNode(detach_reset=True)
        self.prune13 = PruningLayer()
        self.pool5=layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2))




        self.sn1 = neuron.MultiStepIFNode(detach_reset=True)
        self.sn2 = neuron.MultiStepIFNode(detach_reset=True)

        
        self.classifier1 = layer.SeqToANNContainer(
            nn.Linear(cfg[16], 512, bias=False),
            nn.BatchNorm1d(512))

        self.classifier2 = layer.SeqToANNContainer(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512))
        self.fc = nn.Linear(512, 10)

        



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)

    def forward(self, x):

        x.unsqueeze_(0)
        out = x.repeat(4, 1, 1, 1, 1)
        
        

        out=self.layer1(out)
        out, v1=self.neuron1(out)
        v1=v1.detach()
        out=self.prune1(out,v1)
        out = self.layer2(out)
        out, v2=self.neuron2(out)
        v2 = v2.detach()
        out = self.prune2(out, v2)
        out=self.pool1(out)

        out = self.layer3(out)
        out, v3=self.neuron3(out)
        v3 = v3.detach()
        out = self.prune3(out, v3)
        out = self.layer4(out)
        out, v4=self.neuron4(out)
        v4 = v4.detach()
        out = self.prune4(out, v4)
        out = self.pool2(out)
        #print(out.shape)

        out = self.layer5(out)
        out, v5=self.neuron5(out)
        v5 = v5.detach()
        out = self.prune5(out, v5)
        out = self.layer6(out)
        out, v6=self.neuron6(out)
        v6 = v6.detach()
        out = self.prune6(out, v6)
        out = self.layer7(out)
        out, v7=self.neuron7(out)
        v7 = v7.detach()
        out = self.prune7(out, v7)
        out = self.pool3(out)
        #print(out.shape)

        out = self.layer8(out)
        out, v8=self.neuron8(out)
        v8 = v8.detach()
        out = self.prune8(out, v8)
        out = self.layer9(out)
        out, v9=self.neuron9(out)
        v9 = v9.detach()
        out = self.prune9(out, v9)
        out = self.layer10(out)
        out, v10=self.neuron10(out)
        v10 = v10.detach()
        out = self.prune10(out, v10)
        out = self.pool4(out)
        #print(out.shape)

        out = self.layer11(out)
        out, v11=self.neuron11(out)
        v11= v11.detach()
        out = self.prune11(out, v11)
        out = self.layer12(out)
        out, v12=self.neuron12(out)
        v12 = v12.detach()
        out = self.prune12(out, v12)
        out = self.layer13(out)
        out, v13=self.neuron13(out)
        v13 = v13.detach()
        out = self.prune13(out, v13)
        out = self.pool5(out)


        # print(out.shape)
        ##out = self.features(out)
        out = torch.flatten(out, 2)
        #print(out.shape)
        out=self.classifier1(out)
        out=self.sn1(out)
        #print(out.shape)
        out = self.classifier2(out)
        #print(out.shape)
        out = self.sn2(out)




        return self.fc(out.mean(dim=0))





def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2))]
        else:
            conv2d = layer.SeqToANNContainer(nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False))
            prune_l=PruningLayer()
            if batch_norm:
                layers += [conv2d, layer.SeqToANNContainer(nn.BatchNorm2d(v)),
                           myMultiStepIFNode(detach_reset=True),prune_l]
            else:
                layers += [conv2d, neuron.MultiStepIFNode(detach_reset=True)]

            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def snnvgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def snnvgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def snnvgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def snnvgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def snnvgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))



def snnvgg16_bn(cfg=None):
    if cfg is None:
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        print('DDD')
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(cfg)


def snnvgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def snnvgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))
