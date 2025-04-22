import torch.nn as nn
from spikingjelly.activation_based import layer, functional

from .sparse import *


class SpikingVGG9(nn.Module):
    def __init__(self, num_classes, T, input_shape, threshold=0.1):
        super().__init__()

        C, H, W = input_shape
        in_features_dim = 256 * (H // 8) * (W // 8)

        self.T = T
        self.flat_width = threshold

        self.conv1 = PConv(
            self.flat_width,
            C, 
            64, 
            kernel_size=3, 
            padding=1, bias=False, 
        ) 
        self.bn1 = layer.BatchNorm2d(64)
        self.lif1 = create_lif()

        self.conv2 = PConv(
            self.flat_width,
            64, 
            64, 
            kernel_size=3, 
            padding=1, bias=False, 
        ) 
        self.bn2 = layer.BatchNorm2d(64)
        self.lif2 = create_lif()

        self.maxpool1 = layer.MaxPool2d(2, 2)

        self.conv3 = PConv(
            self.flat_width,
            64, 
            128, 
            kernel_size=3, 
            padding=1, bias=False, 
        ) 
        self.bn3 = layer.BatchNorm2d(128)
        self.lif3 = create_lif()

        self.conv4 = PConv(
            self.flat_width,
            128, 
            128, 
            kernel_size=3, 
            padding=1, bias=False, 
        ) 
        self.bn4 = layer.BatchNorm2d(128)
        self.lif4 = create_lif()

        self.maxpool2 = layer.MaxPool2d(2, 2)

        self.conv5 = PConv(
            self.flat_width,
            128, 
            256, 
            kernel_size=3, 
            padding=1, bias=False, 
        ) 
        self.bn5 = layer.BatchNorm2d(256)
        self.lif5 = create_lif()

        self.conv6 = PConv(
            self.flat_width,
            256, 
            256, 
            kernel_size=3, 
            padding=1, bias=False, 
        ) 
        self.bn6 = layer.BatchNorm2d(256)
        self.lif6 = create_lif()

        self.conv7 = PConv(
            self.flat_width,
            256, 
            256, 
            kernel_size=3, 
            padding=1, bias=False, 
        ) 
        self.bn7 = layer.BatchNorm2d(256)
        self.lif7 = create_lif()

        self.maxpool3 = layer.MaxPool2d(2, 2)

        self.ln1 = PLinear(
            self.flat_width,
            in_features_dim, 
            1024, 
            bias=False, 
        )
        self.lif8 = create_lif()

        self.fc = nn.Linear(1024, num_classes, bias=False)

        functional.set_step_mode(self, 'm')

    def forward(self, x):
        functional.reset_net(self)
        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1) # B, T, C, H, W
        x = x.transpose(0, 1)

        x = functional.seq_to_ann_forward(x, self.conv1)
        x = self.bn1(x)
        x = self.lif1(x)

        x = functional.seq_to_ann_forward(x, self.conv2)
        x = self.bn2(x)
        x = self.lif2(x)

        x = self.maxpool1(x)

        x = functional.seq_to_ann_forward(x, self.conv3)
        x = self.bn3(x)
        x = self.lif3(x)

        x = functional.seq_to_ann_forward(x, self.conv4)
        x = self.bn4(x)
        x = self.lif4(x)

        x = self.maxpool2(x)

        x = functional.seq_to_ann_forward(x, self.conv5)
        x = self.bn5(x)
        x = self.lif5(x)

        x = functional.seq_to_ann_forward(x, self.conv6)
        x = self.bn6(x)
        x = self.lif6(x)

        x = functional.seq_to_ann_forward(x, self.conv7)
        x = self.bn7(x)
        x = self.lif7(x)

        x = self.maxpool3(x)

        x = x.view(x.shape[0], x.shape[1], -1)  # -> (T, B, D)
        x = functional.seq_to_ann_forward(x, self.ln1)
        x = self.lif8(x).mean(0) # -> (B, D)

        x = self.fc(x) # -> (B, num_cls)

        return x
