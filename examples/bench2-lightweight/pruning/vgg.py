import torch
import torch.nn as nn

from spikingjelly.activation_based import layer, neuron, surrogate, functional


class SpikingVGG9(nn.Module):
    def __init__(self, input_shape, T, num_classes):
        super().__init__()

        self.model_type = 'vgg'

        self.C, self.H, self.W = input_shape
        self.T = T

        self.features = nn.Sequential(
            layer.Conv2d(self.C, 64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()),

            layer.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()),

            layer.MaxPool2d(2, 2),

            layer.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()),

            layer.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()),

            layer.MaxPool2d(2, 2),

            layer.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(256),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()),

            layer.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(256),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()),

            layer.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(256),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()),

            layer.MaxPool2d(2, 2),            
        )

        self.fc = nn.Sequential(
            layer.Linear(256 * (self.H // 8) * (self.W // 8), 1024, bias=False),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()),
            nn.Linear(1024, num_classes)
        )

        functional.set_step_mode(self, 'm')


    def forward(self, x):
        functional.reset_net(self)

        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1) # B, T, C, H, W
        x = x.transpose(0, 1)

        x = self.features(x) 
        x = torch.flatten(x, 2) # -> (T, B, D)
        x = self.fc(x.mean(0)) # -> (B, D) -> (B, num_cls)

        return x

