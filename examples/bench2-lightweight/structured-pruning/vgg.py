import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional

from sparse import PruningLayer

class SpikingVGG9(nn.Module):
    def __init__(self, input_shape, T, num_classes):
        super().__init__()

        C, H, W = input_shape
        self.T = T

        self.layer1 = layer.SeqToANNContainer(
            nn.Conv2d(C, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.neuron1 = neuron.LIFNode(
            detach_reset=True, surrogate_function=surrogate.ATan(), store_v_seq=True)
        self.prune1 = PruningLayer()

        self.layer2 = layer.SeqToANNContainer(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.neuron2 = neuron.LIFNode(
            detach_reset=True, surrogate_function=surrogate.ATan(), store_v_seq=True)
        self.prune2 = PruningLayer()

        self.pool1 = layer.SeqToANNContainer(nn.MaxPool2d(2, 2))
        
        self.layer3 = layer.SeqToANNContainer(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.neuron3 = neuron.LIFNode(
            detach_reset=True, surrogate_function=surrogate.ATan(), store_v_seq=True)
        self.prune3 = PruningLayer()

        self.layer4 = layer.SeqToANNContainer(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.neuron4 = neuron.LIFNode(
            detach_reset=True, surrogate_function=surrogate.ATan(), store_v_seq=True)
        self.prune4 = PruningLayer()

        self.pool2 = layer.SeqToANNContainer(nn.MaxPool2d(2, 2))

        self.layer5 = layer.SeqToANNContainer(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.neuron5 = neuron.LIFNode(
            detach_reset=True, surrogate_function=surrogate.ATan(), store_v_seq=True)
        self.prune5 = PruningLayer()

        self.layer6 = layer.SeqToANNContainer(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.neuron6 = neuron.LIFNode(
            detach_reset=True, surrogate_function=surrogate.ATan(), store_v_seq=True)
        self.prune6 = PruningLayer()

        self.layer7 = layer.SeqToANNContainer(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.neuron7 = neuron.LIFNode(
            detach_reset=True, surrogate_function=surrogate.ATan(), store_v_seq=True)
        self.prune7 = PruningLayer()

        self.pool3=layer.SeqToANNContainer(nn.MaxPool2d(2, 2))

        self.ln1 = layer.SeqToANNContainer(nn.Linear(256 * (H // 8) * (W // 8), 1024, bias=False))
        self.sn1 = neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan())
        self.fc = nn.Linear(1024, num_classes)

        functional.set_step_mode(self, 'm')

    def forward(self, x):
        functional.reset_net(self)
        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1) # B, T, C, H, W
        x = x.transpose(0, 1)

        x = self.layer1(x)
        x = self.neuron1(x)
        v1 = self.neuron1.v_seq.detach()
        x = self.prune1(x, v1)

        x = self.layer2(x)
        x = self.neuron2(x)
        v2 = self.neuron2.v_seq.detach()
        x = self.prune2(x, v2)

        x = self.pool1(x)

        x = self.layer3(x)
        x = self.neuron3(x)
        v3 = self.neuron3.v_seq.detach()
        x = self.prune3(x, v3)

        x = self.layer4(x)
        x = self.neuron4(x)
        v4 = self.neuron4.v_seq.detach()
        x = self.prune4(x, v4)

        x = self.pool2(x)

        x = self.layer5(x)
        x = self.neuron5(x)
        v5 = self.neuron5.v_seq.detach()
        x = self.prune5(x, v5)

        x = self.layer6(x)
        x = self.neuron6(x)
        v6 = self.neuron6.v_seq.detach()
        x = self.prune6(x, v6)

        x = self.layer7(x)
        x = self.neuron7(x)
        v7 = self.neuron7.v_seq.detach()
        x = self.prune7(x, v7)

        x = self.pool3(x)

        x = torch.flatten(x, 2) # -> (T, B, D)
        x = self.ln1(x)
        x = self.sn1(x).mean(0) # -> (B, D)

        x = self.fc(x) # -> (B, num_cls)

        return x
