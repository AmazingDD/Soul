import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional

class BaseNet(nn.Module):
    def __init__(self, T=4):
        super().__init__()

        self.T = T

        self.features = nn.Sequential()

        self.classifier = nn.Identity()

    def forward(self, x):
        functional.reset_net(self)

        x = self.features(x).mean(0)
        logit = self.classifier(x)

        return logit

class MnistNet(BaseNet):
    def __init__(self, T=4):
        super().__init__(T=T)

        self.features = nn.Sequential(
            layer.Conv2d(1, 12, kernel_size=(5, 5), bias=False),
            layer.BatchNorm2d(12, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d((2, 2)),
            layer.Conv2d(12, 64, kernel_size=(5, 5), bias=False),
            layer.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d((2, 2)),
            layer.Flatten()
        )

        self.classifier = nn.Linear(in_features=1024, out_features=10, bias=False)

        functional.set_step_mode(self, 'm')

class NMnistNet(BaseNet):
    def __init__(self, T=4):
        super().__init__(T=T)

        self.features = nn.Sequential(
            layer.Conv2d(2, 12, kernel_size=(5, 5), bias=False),
            layer.BatchNorm2d(12, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d((2, 2)),
            layer.Conv2d(12, 64, kernel_size=(5, 5), bias=False),
            layer.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d((2, 2)),
            layer.Flatten()
        )

        self.classifier = nn.Linear(in_features=1600, out_features=10, bias=False)

        functional.set_step_mode(self, 'm')

class CifarNet(BaseNet):
    def __init__(self, T=4):
        super().__init__(T=T)

        self.features = nn.Sequential(
            layer.Conv2d(3, 96, kernel_size=(3, 3), bias=False),
            layer.BatchNorm2d(96, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0, surrogate_function=surrogate.ATan()),
            layer.Conv2d(96, 256, kernel_size=(3, 3), bias=False),
            layer.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d((2, 2)),
            layer.Conv2d(256, 384, kernel_size=(3, 3), bias=False),
            layer.BatchNorm2d(384, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0, surrogate_function=surrogate.ATan()),
            layer.Conv2d(384, 256, kernel_size=(3, 3), bias=False),
            layer.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0, surrogate_function=surrogate.ATan()),
            layer.Flatten(),
            layer.Linear(in_features=25600, out_features=1024, bias=False),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0, surrogate_function=surrogate.ATan()),
        )

        self.classifier = nn.Linear(in_features=1024, out_features=10, bias=False)

        functional.set_step_mode(self, 'm')

