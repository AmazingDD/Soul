import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional



class Net(nn.Module):
    def __init__(self, num_classes=10, C=3, H=32, W=32, T=4):
        super().__init__()

        # Spike-wise VGG9 example
        self.features = nn.Sequential(
            layer.Conv2d(C, 30, kernel_size=(5, 5), bias=False),
            layer.BatchNorm2d(30, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0, surrogate_function=surrogate.ATan()),
            
            layer.Conv2d(30, 250, kernel_size=(3, 3), bias=False),
            layer.BatchNorm2d(250, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0, surrogate_function=surrogate.ATan()),
            layer.Conv2d(250, 200, kernel_size=(5, 5), bias=False),
            layer.BatchNorm2d(200, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0, surrogate_function=surrogate.ATan()),
        )

        self.classifier = nn.Linear(in_features=7200, out_features=num_classes, bias=False)

        self.T = T
        functional.set_step_mode(self, 'm')

    def forward(self, x):
        functional.reset_net(self)

        x = self.features(x)
        x = self.fc(x).mean(0)

        logit = self.classifier(x)

        return logit

