from spikingjelly.activation_based import layer, functional, neuron, surrogate
from spikingjelly.activation_based.model.sew_resnet import sew_resnet18
import torch
import torch.nn as nn
import inspect
from .CLIF import MultiStepCLIFNeuron
from .GLIF import GatedLIFNode
from .ILIF import ILIF
import copy

neuron_map = {
    "LIF": neuron.LIFNode,
    "PLIF": neuron.ParametricLIFNode,
    "CLIF": MultiStepCLIFNeuron,
    "GLIF": GatedLIFNode,
    "ILIF": ILIF
}

def _get_neuron_layer(neuron_type, T=4, v_threshold=1.0):
        """根据神经元类型动态返回神经元层"""
        if neuron_type in ["PSN", "GLIF"]:
            return neuron_map[neuron_type](T=T, surrogate_function=surrogate.ATan(),v_threshold=v_threshold)
        elif neuron_type in ["LIF", "PLIF"]:
            return neuron_map[neuron_type](surrogate_function=surrogate.ATan(),v_threshold=v_threshold)
        elif neuron_type in ["ILIF"]:
            return neuron_map[neuron_type](v_threshold=v_threshold)
        else:
            return neuron_map[neuron_type](surrogate_function=surrogate.ATan(),v_threshold=v_threshold)  

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def sew_function(x: torch.Tensor, y: torch.Tensor, cnf:str):
    if cnf == 'ADD':
        return x + y
    elif cnf == 'AND':
        return x * y
    elif cnf == 'IAND':
        return x * (1. - y)
    else:
        raise NotImplementedError

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None, neuron_type: str="LIF", T:int=4):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = _get_neuron_layer(neuron_type, T=T)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = _get_neuron_layer(neuron_type, T=T)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = _get_neuron_layer(neuron_type, T=T)
        self.stride = stride
        self.cnf = cnf

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = sew_function(identity, out, self.cnf)

        return out

class SEWResNet18(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2,2,2,2], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, cnf: str = "ADD", neuron_type: str="LIF", T:int=4, input_shape=(3,64,64)):
        super().__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer
        self.T=T
        in_channels = input_shape[0]
        img_size_h = input_shape[1]
        img_size_w = input_shape[2]
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = _get_neuron_layer(neuron_type, T=T)
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], cnf=cnf, neuron_type=neuron_type, T=T)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], cnf=cnf, neuron_type=neuron_type, T=T)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], cnf=cnf, neuron_type=neuron_type, T=T)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], cnf=cnf, neuron_type=neuron_type, T=T)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        functional.set_step_mode(self, "m")

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, cnf: str=None, neuron_type: str="LIF", T:int=4):
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, cnf, neuron_type, T))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, cnf=cnf, neuron_type=neuron_type, T=T))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        # print("final.shape: ", x.shape)
        x = x.mean(0)
        x = self.fc(x)
        return x

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1) # B, T, C, H, W
        x = x.transpose(0, 1) # [T, B, C, H, W]
        # print("input shape: ", x.shape)
        return self._forward_impl(x)