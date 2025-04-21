import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, layer, functional
from typing import Union, List

from .layers import conv1x1, conv3x3, ConvBlock

def create_lif():
    return neuron.LIFNode(
        v_threshold=1., v_reset=0., surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m')

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample: ConvBlock = None, groups=1, base_width=64, dilation=1, norm_layer=None, activation=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = create_lif
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.downsample = downsample
        self.conv1 = ConvBlock(
            conv3x3(inplanes, planes, stride), 
            norm_layer(planes), 
            activation(), 
            sparse_weights=True, 
            sparse_neurons=True
        )
        self.conv2 = ConvBlock(
            conv3x3(planes, planes), 
            norm_layer(planes), 
            activation(), 
            sparse_weights=True, 
            sparse_neurons=True
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity

        return out

    def connects(self, sparse: torch.Tensor, dense: torch.Tensor):
        conn, total = 0, 0
        id_sparse, id_dense = sparse, dense
        with torch.no_grad():
            c, t, sparse, dense = self.conv1.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv2.connects(sparse, dense)
            conn, total = conn + c, total + t
            if self.downsample is not None:
                c, t, id_sparse, id_dense = self.downsample.connects(id_sparse, id_dense)
                conn, total = conn + c, total + t
            return conn, total, sparse + id_sparse, dense + id_dense

    def calc_c(self, x: torch.Tensor, prev_layers: List[ConvBlock] = []):
        ident = x
        with torch.no_grad():
            x = self.conv1.calc_c(x, prev_layers)
            x = self.conv2.calc_c(x, [self.conv1])
            if self.downsample is not None:
                ident = self.downsample.calc_c(ident, prev_layers)
            return x + ident
        
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample: ConvBlock = None, groups=1, base_width=64, dilation=1, norm_layer=None, activation=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = create_lif
        width = int(planes * (base_width / 64.)) * groups

        self.downsample = downsample
        self.conv1 = ConvBlock(
            conv1x1(inplanes, width), 
            norm_layer(width), 
            activation(), 
            sparse_weights=True, 
            sparse_neurons=True
        )
        self.conv2 = ConvBlock(
            conv3x3(width, width, stride, groups, dilation), 
            norm_layer(width), 
            activation(), 
            sparse_weights=True, 
            sparse_neurons=True
        )
        self.conv3 = ConvBlock(
            conv1x1(width, planes * self.expansion), 
            norm_layer(planes * self.expansion), 
            activation(), 
            sparse_weights=True, 
            sparse_neurons=True
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity

        return out

    def connects(self, sparse: torch.Tensor, dense: torch.Tensor):
        conn, total = 0, 0
        id_sparse, id_dense = sparse, dense
        with torch.no_grad():
            c, t, sparse, dense = self.conv1.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv2.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv3.connects(sparse, dense)
            conn, total = conn + c, total + t
            if self.downsample is not None:
                c, t, id_sparse, id_dense = self.downsample.connects(id_sparse, id_dense)
                conn, total = conn + c, total + t
            return conn, total, sparse + id_sparse, dense + id_dense

    def calc_c(self, prev_layers: List[ConvBlock] = []):
        ident = x
        with torch.no_grad():
            x = self.conv1.calc_c(x, prev_layers)
            x = self.conv2.calc_c(x, [self.conv1])
            x = self.conv3.calc_c(x, [self.conv2])
            if self.downsample is not None:
                ident = self.downsample.calc_c(ident, prev_layers)
            return x + ident
        
class SewResNet18(nn.Module):
    def __init__(self, 
                 block: Union[BasicBlock, Bottleneck] = BasicBlock, 
                 layers: List[int] = [2, 2, 2, 2], 
                 num_classes=10, 
                 zero_init_residual=False, 
                 groups=1, 
                 width_per_group=64, 
                 replace_stride_with_dilation=None, 
                 norm_layer: nn.Module = None, 
                 T=4,
                 input_shape=(3, 32, 32)):
        super().__init__()

        self.skip = ['conv1']

        self.T = T
        C, H, W = input_shape
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = ConvBlock(
            nn.Conv2d(C, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes), create_lif(), static=True, T=T, sparse_weights=False,
            sparse_neurons=False)
        self.maxpool = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))

        if num_classes * 10 < 512 * block.expansion:
            self.fc = nn.Sequential(
                nn.Linear(512 * block.expansion, num_classes * 10),
                nn.AvgPool1d(10, 10)
            )
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.init_weight()
        if zero_init_residual:
            self.zero_init_blocks()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def zero_init_blocks(self):
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.conv3.norm.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.conv2.norm.weight, 0)

    def _make_layer(self, 
                    block: Union[BasicBlock, Bottleneck], 
                    planes: int, 
                    blocks: int, 
                    stride=1,
                    dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvBlock(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion), 
                create_lif(),
                sparse_weights=True, 
                sparse_neurons=True
            )

        layers: List[Union[BasicBlock, Bottleneck]] = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2).mean(0) # -> (B, D)

        out = self.fc(x)

        return out
    
    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1) # (B, T, C, H, W)
        x = x.transpose(0, 1)

        return self._forward_impl(x)
    
    def connects(self, input_shape, device):
        conn, total = 0, 0
        C, H, W = input_shape
        with torch.no_grad():
            # static conv
            sparse = torch.ones(1, C, H, W).to(device)
            dense = torch.ones(1, C, H, W).to(device)
            c, t, sparse, dense = self.conv1.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.maxpool[0](sparse), self.maxpool[0](dense)

            def _connects(layer: List[Union[BasicBlock, Bottleneck]], conn: float, total: float, sparse: torch.Tensor, dense: torch.Tensor):
                for block in layer:
                    c, t, sparse, dense = block.connects(sparse, dense)
                    conn, total = conn + c, total + t
                return conn, total, sparse, dense

            conn, total, sparse, dense = _connects(self.layer1, conn, total, sparse, dense)
            conn, total, sparse, dense = _connects(self.layer2, conn, total, sparse, dense)
            conn, total, sparse, dense = _connects(self.layer3, conn, total, sparse, dense)
            conn, total, sparse, dense = _connects(self.layer4, conn, total, sparse, dense)
            # ignore fc

        return conn, total
    
    def calc_c(self, input_shape, device):
        C, H, W = input_shape
        with torch.no_grad():
            x = torch.ones(1, C, H, W).to(device)
            x = self.conv1.calc_c(x)
            x = self.maxpool[0](x)
            prev_layers = [self.conv1]

            def _calc_c(layer: List[Union[BasicBlock, Bottleneck]], x: torch.Tensor, prev_layers: List[ConvBlock]):
                for block in layer:
                    x = block.calc_c(x, prev_layers)
                    if block.downsample is None:
                        if isinstance(block, BasicBlock):
                            prev_layers.append(block.conv2)
                        elif isinstance(block, Bottleneck):
                            prev_layers.append(block.conv3)
                    else:
                        if isinstance(block, BasicBlock):
                            prev_layers = [block.conv2, block.downsample]
                        elif isinstance(block, Bottleneck):
                            prev_layers = [block.conv3, block.downsample]
                return x, prev_layers

            x, prev_layers = _calc_c(self.layer1, x, prev_layers)
            x, prev_layers = _calc_c(self.layer2, x, prev_layers)
            x, prev_layers = _calc_c(self.layer3, x, prev_layers)
            x, prev_layers = _calc_c(self.layer4, x, prev_layers)
            # ignore fc
        return
