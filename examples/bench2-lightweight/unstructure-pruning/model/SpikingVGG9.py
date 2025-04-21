import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, functional
from .layers import conv1x1, conv3x3, ConvBlock

def create_lif():
    return neuron.LIFNode(
        v_threshold=1., v_reset=0., surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m')

class SpikingVGG9(nn.Module):
    def __init__(self, num_classes, T, input_shape):
        super().__init__()

        self.skip = ['conv1']

        self.T = T
        C, H, W = input_shape

        # the first conv cannot implement neuron pruning for receiving input
        self.conv1 = ConvBlock(
            conv3x3(C, 64), 
            nn.BatchNorm2d(64),
            create_lif(), 
            static=True, 
            T=T, 
            sparse_weights=True,
            sparse_neurons=False) 
        
        self.conv2 = ConvBlock(
            conv3x3(64, 64), 
            nn.BatchNorm2d(64),
            create_lif(), 
            sparse_weights=True, 
            sparse_neurons=True)
        
        self.maxpool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = ConvBlock(
            conv3x3(64, 128), 
            nn.BatchNorm2d(128),
            create_lif(), 
            sparse_weights=True, 
            sparse_neurons=True)
        
        self.conv4 = ConvBlock(
            conv3x3(128, 128), 
            nn.BatchNorm2d(128),
            create_lif(), 
            sparse_weights=True, 
            sparse_neurons=True)
        
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv5 = ConvBlock(
            conv3x3(128, 256), 
            nn.BatchNorm2d(256),
            create_lif(), 
            sparse_weights=True, 
            sparse_neurons=True)
        
        self.conv6 = ConvBlock(
            conv3x3(256, 256), 
            nn.BatchNorm2d(256),
            create_lif(), 
            sparse_weights=True, 
            sparse_neurons=True)
        
        self.conv7 = ConvBlock(
            conv3x3(256, 256), 
            nn.BatchNorm2d(256),
            create_lif(), 
            sparse_weights=True, 
            sparse_neurons=True)
        
        self.maxpool3 = nn.MaxPool2d(2, 2)

        in_features = (H // 2 // 2 // 2) * (W // 2 // 2 // 2)

        self.mlp = ConvBlock(
            conv1x1(256 * in_features * in_features, 1024), 
            None, 
            create_lif(), 
            sparse_neurons=True, 
            sparse_weights=True)
        
        self.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=False)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1) # B, T, C, H, W
        x = x.transpose(0, 1) # [T, B, C, H, W]

        x = self.conv1(x)
        x = self.conv2(x)
        x = functional.seq_to_ann_forward(x, self.maxpool1)

        x = self.conv3(x)
        x = self.conv4(x)
        x = functional.seq_to_ann_forward(x, self.maxpool2)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = functional.seq_to_ann_forward(x, self.maxpool3)

        x = x.view(x.shape[0], x.shape[1], -1, 1, 1) 
        x = self.mlp(x)
        x = x.flatten(2).mean(0) # -> (B, D)

        out = self.fc(x)

        return out
    
    def calc_c(self, input_shape, device):
        C, H, W = input_shape
        with torch.no_grad():
            x = torch.ones(1, C, H, W).to(device)
            x = self.conv1.calc_c(x)
            x = self.conv2.calc_c(x, [self.conv1])
            x = self.maxpool1(x)

            x = self.conv3.calc_c(x, [self.conv2])
            x = self.conv4.calc_c(x, [self.conv3])
            x = self.maxpool2(x)

            x = self.conv5.calc_c(x, [self.conv4])
            x = self.conv6.calc_c(x, [self.conv5])
            x = self.conv7.calc_c(x, [self.conv6])
            x = self.maxpool3(x)

            x = x.view(x.shape[0], x.shape[1], -1, 1, 1) 
            x = self.mlp.calc_c(x, [self.conv7])

        return
    
    def connects(self, input_shape, device):
        conn, total = 0, 0
        C, H, W = input_shape
        with torch.no_grad():
            sparse = torch.ones(1, C, H, W).to(device)
            dense = torch.ones(1, C, H, W).to(device)

            c, t, sparse, dense = self.conv1.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv2.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.maxpool1(sparse), self.maxpool1(dense)

            c, t, sparse, dense = self.conv3.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv4.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.maxpool2(sparse), self.maxpool2(dense)

            c, t, sparse, dense = self.conv5.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv6.connects(sparse, dense)
            conn, total = conn + c, total + t
            c, t, sparse, dense = self.conv7.connects(sparse, dense)
            conn, total = conn + c, total + t
            sparse, dense = self.maxpool3(sparse), self.maxpool3(dense)

            sparse = sparse.view(sparse.shape[0], sparse.shape[1], -1, 1, 1)
            dense = dense.view(dense.shape[0], dense.shape[1], -1, 1, 1)
            c, t, sparse, dense = self.mlp.connects(sparse, dense)
            conn, total = conn + c, total + t

        return conn, total