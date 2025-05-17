import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from spikingjelly.activation_based import functional

__all__ = ['VGG', 'SpikingVGG5', 'SpikingVGG9', 'SpikingVGG11', 'SpikingVGG13', 'SpikingVGG16', 'SpikingVGG19']

cfgs = {
    'vgg5': [64, 'M', 128, 'M'],
    'vgg9': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def multi_time_forward(x_seq, stateless_module):
    y_shape = [x_seq.shape[0], x_seq.shape[1]] # [T, B]
    y = x_seq.flatten(0, 1)
    if isinstance(stateless_module, (list, tuple, nn.Sequential)):
        for m in stateless_module:
            y = m(y)
    else:
        y = stateless_module(y)
    
    y_shape.extend(y.shape[1:]) # [T, B] + [...] -> [T, B, ...]
    return y.view(y_shape)

class MaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, x):
        return multi_time_forward(x, super().forward)

class ConvBNLIF(nn.Module):
    def __init__(self, lif, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.sn = deepcopy(lif)

    def forward(self, x):
        x = multi_time_forward(x, self.conv)
        output = self.sn(x)

        return output
    
class ConvMLP(nn.Module):
    def __init__(self, lif, in_features=512, out_features=4096, kernel_size=7, mlp_ratio=1.0):
        super().__init__()

        self.input_kernel_size = kernel_size
        mid_features = int(out_features * mlp_ratio)

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, kernel_size, bias=False),
            nn.BatchNorm2d(mid_features),
        )
        self.sn1 = deepcopy(lif)

        self.fc2 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, bias=False),
            nn.BatchNorm2d(out_features),
        )
        self.sn2 = deepcopy(lif)

    def forward(self, x):
        # default multi-step mode, the input shape of x from Conv layers must be (T, B, C, H, W)
        T, B, C, H, W = x.shape
        if H < self.input_kernel_size or W < self.input_kernel_size:
            # keeep the input size >= 7*7
            output_size = (max(self.input_kernel_size, H), max(self.input_kernel_size, W))
            x = F.adaptive_avg_pool2d(x.flatten(0, 1), output_size) # -> (TB, C, output_size[0], output_size[1])
            x = x.reshape(T, B, C, output_size[0], output_size[1])

        x = multi_time_forward(x, self.fc1)
        x = self.sn1(x)
        x = multi_time_forward(x, self.fc2)
        x = self.sn2(x)

        return x


class VGG(nn.Module):
    def __init__(self, cfg, config):
        super(VGG, self).__init__()  

        self.num_classes = config['num_classes']
        self.T = config['time_step']

        prev_chs = config['input_channels']
        pool_layer = MaxPool2d
        layers = []
        for v in cfg:
            last_idx = len(layers) - 1
            if v == 'M':
                layers += [pool_layer(kernel_size=2, stride=2)]
            else:
                v = int(v)
                layers += [ConvBNLIF(config['neuron'], prev_chs, v)]
                prev_chs = v
            
        self.features = nn.Sequential(*layers)

        self.num_features = prev_chs
        self.head_hidden_size = config['mlp_hidden_dim']

        self.pre_logits = ConvMLP(
            config['neuron'],
            prev_chs,
            self.head_hidden_size,
            7,
            mlp_ratio=config['mlp_ratio'],
        )
        self.head = nn.Linear(self.head_hidden_size, config['num_classes'])

        self._initialize_weights()


    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False):
        x = self.pre_logits(x)
        x = x.flatten(2).mean(0) # -> (T, B. CHW) -> (B, CHW) TODO
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        assert len(x.shape) in [4, 5], f'Invalid input shape {x.shape}...'
        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1) # -> (B, T, C, H, W)
        x = x.transpose(0, 1)

        x = self.forward_features(x)
        x = self.forward_head(x)

        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def SpikingVGG5(config):
    return VGG(cfgs['vgg5'], config)

def SpikingVGG9(config):
    return VGG(cfgs['vgg9'], config)

def SpikingVGG11(config):
    return VGG(cfgs['vgg11'], config)

def SpikingVGG13(config):
    return VGG(cfgs['vgg13'], config)

def SpikingVGG16(config):
    return VGG(cfgs['vgg16'], config)

def SpikingVGG19(config):
    return VGG(cfgs['vgg19'], config)
