import torch.nn as nn
from spikingjelly.activation_based import layer, functional

from .sparse import *

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, 
                 inplanes, 
                 planes, 
                 stride=1, 
                 downsample=None, 
                 groups=1, 
                 base_width=64, 
                 dilation=1, 
                 norm_layer=None, 
                 connect_f=None, 
                 flat_width=1.0):
        super(BasicBlock, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.conv1 = layer.SeqToANNContainer(
            PConv(flat_width, inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            norm_layer(planes),
        )
        self.lif1 = create_lif()

        self.conv2 = layer.SeqToANNContainer(
            PConv(flat_width, planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(planes),
        )
        self.lif2 = create_lif()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.lif1(self.conv1(x))
        out = self.lif2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.connect_f == 'ADD':
            out = out + identity
        elif self.connect_f == 'AND':
            out = out * identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out
    
def zero_init_blocks(net: nn.Module, connect_f: str):
    for m in net.modules():
        if isinstance(m, BasicBlock):
            nn.init.constant_(m.conv2.module[1].weight, 0)
            if connect_f == 'AND':
                nn.init.constant_(m.conv2.module[1].bias, 1)

class SewResNet18(nn.Module):
    def __init__(self, 
                 num_classes=10, 
                 T=4,
                 input_shape=(3, 32, 32),
                 zero_init_residual=False, 
                 groups=1, 
                 width_per_group=64, 
                 replace_stride_with_dilation=None, 
                 norm_layer: nn.Module = None,
                 connect_f = 'ADD', 
                 threshold=1.0):
        super().__init__()

        layers = [2, 2, 2, 2]
        block = BasicBlock

        self.T = T
        self.flat_width = threshold

        C, H, W = input_shape

        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}")

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = layer.SeqToANNContainer(
            PConv(
                self.flat_width,
                C, 
                self.inplanes, 
                kernel_size=7, 
                stride=2, 
                padding=3, bias=False
            ),
            norm_layer(self.inplanes)
        )
        self.lif1 = create_lif()
        self.maxpool = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0], connect_f=connect_f)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], connect_f=connect_f)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], connect_f=connect_f)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], connect_f=connect_f)
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))

        if num_classes * 10 < 512 * block.expansion:
            self.fc = nn.Sequential(
                PLinear(self.flat_width, 512 * block.expansion, num_classes * 10),
                nn.AvgPool1d(10, 10)
            )
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        if zero_init_residual:
            zero_init_blocks(self, connect_f)

        functional.set_step_mode(self, 'm')

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layer.SeqToANNContainer(
                    PConv(
                        self.flat_width, 
                        self.inplanes, 
                        planes * block.expansion, 
                        kernel_size=1,
                        stride=stride, 
                        bias=False
                    ),
                    norm_layer(planes * block.expansion),
                ),
                create_lif()
            )

        layers = []
        layers.append(
            block(
                self.inplanes, 
                planes, 
                stride, 
                downsample, 
                self.groups,
                self.base_width, 
                previous_dilation, 
                norm_layer, 
                connect_f, 
                self.flat_width
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, 
                    planes, 
                    groups=self.groups,
                    base_width=self.base_width, 
                    dilation=self.dilation,
                    norm_layer=norm_layer, 
                    connect_f=connect_f,
                    flat_width=self.flat_width
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.lif1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 2).mean(0) # -> (B, D)
        out = self.fc(x) # -> (B, num_cls)

        return out
    
    def forward(self, x):
        functional.reset_net(self)
        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1) # (B, T, C, H, W)
        x = x.transpose(0, 1)

        return self._forward_impl(x)
    

