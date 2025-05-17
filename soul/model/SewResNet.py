import torch
import torch.nn as nn

from copy import deepcopy

from spikingjelly.activation_based import functional

__all__ = ['SEWResNet', 'SEWResNet18', 'SEWResNet34', 'SEWResNet50']

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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def sew_function(identity: torch.Tensor, out: torch.Tensor, cnf:str):
    if cnf == 'ADD':
        return identity + out
    elif cnf == 'AND':
        return identity * out
    elif cnf == 'IAND':
        return identity * (1. - out)
    else:
        raise NotImplementedError(cnf)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self, 
            lif, 
            inplanes, 
            planes, 
            stride=1, 
            downsample=None, 
            groups=1,
            base_width=64, 
            dilation=1, 
            norm_layer=None, 
            connect_f: str = 'ADD', 
        ):
        super(BasicBlock, self).__init__()

        self.downsample = downsample
        self.stride = stride
        self.connect_f = connect_f

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes)
        )
        self.sn1 = deepcopy(lif)

        self.conv2 = nn.Sequential(
            conv3x3(planes, planes),
            norm_layer(planes)
        )
        self.sn2 = deepcopy(lif)

        if self.downsample is not None:
            self.downsample_sn = deepcopy(lif)

    def forward(self, x):
        identity = x

        out = multi_time_forward(x, self.conv1)
        out = self.sn1(out)

        out = multi_time_forward(out, self.conv2)
        out = self.sn2(out)

        if self.downsample is not None:
            identity = self.downsample_sn(multi_time_forward(x, self.downsample))

        out = sew_function(identity, out, self.connect_f)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(
            self, 
            lif,
            inplanes, 
            planes, 
            stride=1, 
            downsample=None, 
            groups=1,
            base_width=64, 
            dilation=1, 
            norm_layer=None, 
            connect_f='ADD',
        ):
        super(Bottleneck, self).__init__()

        self.connect_f = connect_f
        self.downsample = downsample
        self.stride = stride

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Sequential(
            conv1x1(inplanes, width),
            norm_layer(width)
        )
        self.sn1 = deepcopy(lif)

        self.conv2 = nn.Sequential(
            conv3x3(width, width, stride, groups, dilation),
            norm_layer(width)
        )
        self.sn2 = deepcopy(lif)

        self.conv3 = nn.Sequential(
            conv1x1(width, planes * self.expansion),
            norm_layer(planes * self.expansion)
        )
        self.sn3 = deepcopy(lif)

        if self.downsample is not None:
            self.downsample_sn = deepcopy(lif)

    def forward(self, x):
        identity = x

        out = multi_time_forward(x, self.conv1)
        out = self.sn1(out)

        out = multi_time_forward(out, self.conv2)
        out = self.sn2(out)

        out = multi_time_forward(out, self.conv3)
        out = self.sn3(out)

        if self.downsample is not None:
            identity = self.downsample_sn(multi_time_forward(x, self.downsample))

        out = sew_function(identity, out, self.connect_f)

        return out
    
def zero_init_blocks(net: nn.Module, connect_f: str):
    for m in net.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.conv3.module[1].weight, 0)
            if connect_f == 'AND':
                nn.init.constant_(m.conv3.module[1].bias, 1)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.conv2.module[1].weight, 0)
            if connect_f == 'AND':
                nn.init.constant_(m.conv2.module[1].bias, 1)

class SEWResNet(nn.Module):
    def __init__(
            self, 
            config, 
            block, 
            layers, 
            replace_stride_with_dilation=None,
            norm_layer=None, 
        ):
        super().__init__()

        lif = config['neuron']
        num_classes = config['num_classes']
        self.T = config['time_step']
        in_channels = config['input_channels']
        connect_f = config['connect_function']
        self.groups = config['groups']
        self.base_width = config['width_per_group']
        zero_init_residual = config['zero_init_residual']

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}")
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes)
        )
        self.sn1 = deepcopy(lif)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(lif, block, 64, layers[0], connect_f=connect_f)
        self.layer2 = self._make_layer(lif, block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], connect_f=connect_f)
        self.layer3 = self._make_layer(lif, block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], connect_f=connect_f)
        self.layer4 = self._make_layer(lif, block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], connect_f=connect_f)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            zero_init_blocks(self, connect_f)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, lif, block, planes, blocks, stride=1, dilate=False, connect_f: str=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(lif, self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, connect_f)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(lif, self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, connect_f=connect_f)
            )

        return nn.Sequential(*layers)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = multi_time_forward(x, self.conv1)
        x = self.sn1(x)
        x = multi_time_forward(x, self.maxpool)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = multi_time_forward(x, self.avgpool)

        x = torch.flatten(x, 2).mean(0)
        x = self.head(x)

        return x

    def forward(self, x):
        functional.reset_net(self)

        assert len(x.shape) in [4, 5], f'Invalid input shape {x.shape}...'
        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1) # B, T, C, H, W
        x = x.transpose(0, 1) # [T, B, C, H, W]

        x = self.forward_features(x)
        x = self.forward_head(x)

        return x
    
def SEWResNet18(config):
    return SEWResNet(config, BasicBlock, [2, 2, 2, 2])

def SEWResNet34(config):
    return SEWResNet(config, BasicBlock, [3, 4, 6, 3])

def SEWResNet50(config):
    return SEWResNet(config, Bottleneck, [3, 4, 6, 3])
