from spikingjelly.activation_based import layer, functional, neuron, surrogate
import torch
import torch.nn as nn
import inspect
from .CLIF import MultiStepCLIFNeuron
from .GLIF import GatedLIFNode
import copy

neuron_map = {
    "LIF": neuron.LIFNode,
    # "Izhikevich": neuron.IzhikevichNode,
    # "IF": neuron.IFNode,
    "PLIF": neuron.ParametricLIFNode,
    # "KLIF": neuron.KLIFNode,
    # "EIF": neuron.EIFNode,
    # "QIF": neuron.QIFNode,
    "CLIF": MultiStepCLIFNeuron,
    # "PSN": neuron.PSN,
    "GLIF": GatedLIFNode
}

class SpikingVGG9(nn.Module):
    def __init__(self, num_classes, T, neuron_type, input_shape):
        super(SpikingVGG9, self).__init__()
        self.T = T
        self.num_classes = num_classes
        self.neuron_type = neuron_type
        self.input_shape = input_shape  # 记录输入形状

        C, H, W = input_shape
        self.features = nn.Sequential(
            layer.Conv2d(C, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self._get_neuron_layer(),
            layer.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self._get_neuron_layer(),
            layer.MaxPool2d(kernel_size=2, stride=2, padding=0),
            layer.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self._get_neuron_layer(),

            layer.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self._get_neuron_layer(),
            layer.MaxPool2d(kernel_size=2, stride=2, padding=0),
            layer.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self._get_neuron_layer(),
            layer.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self._get_neuron_layer(),
            layer.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self._get_neuron_layer(),
            layer.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        # 自动计算flatten后的维度
        self.in_features = self._calculate_flatten_size(C, H, W)
        
        # self.input_scale = nn.Parameter(torch.tensor([2, 1.5, 1.25, 1], dtype=torch.float32))
        # self.input_scale = nn.Parameter(torch.tensor([0.25, 0.5, 0.75, 1], dtype=torch.float32))
        # self.input_scale = nn.Parameter(torch.tensor([1, 1, 1, 1], dtype=torch.float32))
        
        self.mlp = nn.Sequential(
            layer.Flatten(start_dim=1, end_dim=-1),
            layer.Linear(in_features=self.in_features, out_features=1024, bias=False),
            self._get_neuron_layer(),
        )
        
        self.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=False)
        functional.set_step_mode(self, "m")
    
    def _get_neuron_layer(self):
        """根据神经元类型动态返回神经元层"""
        if self.neuron_type in ["PSN", "GLIF"]:
            return neuron_map[self.neuron_type](T=self.T, surrogate_function=surrogate.ATan())
        elif self.neuron_type in ["LIF", "PLIF", "GLIF"]:
            return neuron_map[self.neuron_type](surrogate_function=surrogate.ATan())
        else:
            return neuron_map[self.neuron_type](surrogate_function=surrogate.ATan())    
    
    def _calculate_flatten_size(self, C, H, W):
        """
        计算特征层输出的flatten大小
        """
        functional.set_step_mode(self, "m")
        with torch.no_grad():
            dummy_input = torch.zeros(1, C, H, W)  # 单张图
            dummy_input = dummy_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # 模拟时序输入 [T, B, C, H, W]
            # print(dummy_input.shape)
            out = self.features(dummy_input)  # 过特征提取层
            out = out.flatten(start_dim=1)  # 展平成MLP输入
            return out.shape[-1]  # flatten的长度
        functional.reset_net(self)

    def forward(self, x):
        # print("input.shape: ", x.shape) # [B, C, H, W], or [B, T, C, H, W]
        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1) # B, T, C, H, W
        x = x.transpose(0, 1) # [T, B, C, H, W]
        
        # x = x * self.input_scale.view(self.T, 1, 1, 1, 1)  # 广播到 [T, B, C, H, W]
        
        x = self.features(x)
        x = self.mlp(x).mean(0)
        # print("x.shape: ", x.shape)
        logit = self.fc(x).squeeze()
        logit = logit.unsqueeze(0) if logit.dim() == 1 else logit
        # print("logit.shape: ", logit.shape)
        return logit