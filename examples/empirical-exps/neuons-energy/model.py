from spikingjelly.activation_based import layer, functional, neuron, surrogate
import torch
import torch.nn as nn

neuron_map = {
    "LIF": neuron.LIFNode,
    "Izhikevich": neuron.IzhikevichNode,
    "IF": neuron.IFNode,
    "PLIF": neuron.ParametricLIFNode
}

class SpikingVGG9(nn.Module):
    def __init__(self, num_classes, T, neuron_type, input_shape):
        super(SpikingVGG9, self).__init__()
        self.T = T
        self.num_classes = num_classes
        self.neuron = neuron_map[neuron_type]
        self.input_shape = input_shape  # 记录输入形状

        C, H, W = input_shape
            
        self.features = nn.Sequential(
            layer.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self.neuron(detach_reset=True, surrogate_function=surrogate.ATan()),
            layer.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self.neuron(detach_reset=True, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=2, stride=2, padding=0),
            layer.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self.neuron(detach_reset=True, surrogate_function=surrogate.ATan()),

            layer.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self.neuron(detach_reset=True, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=2, stride=2, padding=0),
            layer.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self.neuron(detach_reset=True, surrogate_function=surrogate.ATan()),
            layer.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self.neuron(detach_reset=True, surrogate_function=surrogate.ATan()),
            layer.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self.neuron(detach_reset=True, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        # 自动计算flatten后的维度
        self.in_features = self._calculate_flatten_size(C, H, W)
        
        self.mlp = nn.Sequential(
            layer.Flatten(start_dim=1, end_dim=-1),
            layer.Linear(in_features=self.in_features, out_features=1024, bias=False),
            self.neuron(detach_reset=True, surrogate_function=surrogate.ATan()),
        )
        
        self.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=False)
        functional.set_step_mode(self, "m")
        
    def _calculate_flatten_size(self, C, H, W):
        """
        计算特征层输出的flatten大小
        """
        functional.set_step_mode(self, "m")
        with torch.no_grad():
            dummy_input = torch.zeros(1, C, H, W)  # 单张图
            dummy_input = dummy_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # 模拟时序输入 [T, B, C, H, W]
            out = self.features(dummy_input)  # 过特征提取层
            out = out.flatten(start_dim=1)  # 展平成MLP输入
            return out.shape[-1]  # flatten的长度
        functional.reset_net(self)

    def forward(self, x):
        functional.reset_net(self)
        # print(x.shape) # [B, T, C, H, W], [96, 8, 2, 64, 64]
        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1) # B, T, C, H, W
        x = x.transpose(0, 1) # [T, B, C, H, W]
        x = self.features(x)
        # print(x.shape)
        x = self.mlp(x).mean(0)
        logit = self.fc(x)
        return logit

class SpikingVGG5(nn.Module):
    def __init__(self, num_classes, T, neuron_type, input_shape):
        """
        :param num_classes: 分类数
        :param T: 时序长度
        :param neuron_type: 脉冲神经元类型
        :param input_shape: (C, H, W)，输入图像的通道数、高度、宽度
        """
        super(SpikingVGG5, self).__init__()
        self.T = T
        self.num_classes = num_classes
        self.neuron = neuron_map[neuron_type]
        self.input_shape = input_shape  # 记录输入形状

        C, H, W = input_shape

        # 定义Spiking VGG5的卷积和池化层
        self.features = nn.Sequential(
            layer.Conv2d(C, 64, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(64),
            self.neuron(detach_reset=True, surrogate_function=surrogate.ATan()),

            layer.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(128),
            self.neuron(detach_reset=True, surrogate_function=surrogate.ATan()),

            layer.MaxPool2d(kernel_size=2, stride=2),

            layer.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(256),
            self.neuron(detach_reset=True, surrogate_function=surrogate.ATan()),

            layer.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(256),
            self.neuron(detach_reset=True, surrogate_function=surrogate.ATan()),

            layer.MaxPool2d(kernel_size=2, stride=2),
        )

        # 自动计算flatten后的维度
        self.in_features = self._calculate_flatten_size(C, H, W)

        # MLP和分类头
        self.mlp = nn.Sequential(
            layer.Flatten(start_dim=1, end_dim=-1),
            layer.Linear(self.in_features, 512, bias=False),
            self.neuron(detach_reset=True, surrogate_function=surrogate.ATan()),
        )

        self.fc = nn.Linear(512, num_classes, bias=False)

        # 设置step_mode
        functional.set_step_mode(self, "m")

    def _calculate_flatten_size(self, C, H, W):
        """
        计算特征层输出的flatten大小
        """
        functional.set_step_mode(self, "m")
        with torch.no_grad():
            dummy_input = torch.zeros(1, C, H, W)  # 单张图
            dummy_input = dummy_input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # 模拟时序输入 [T, B, C, H, W]
            out = self.features(dummy_input)  # 过特征提取层
            out = out.flatten(start_dim=1)  # 展平成MLP输入
            return out.shape[-1]  # flatten的长度
        functional.reset_net(self)

    def forward(self, x):
        functional.reset_net(self)
        # x: [B, C, H, W]
        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1) # B, T, C, H, W
        x = x.transpose(0, 1) # [T, B, C, H, W]
        x = self.features(x)
        x = self.mlp(x).mean(0)
        logit = self.fc(x)
        return logit