import math
import numpy as np

from typing import Callable
from spikingjelly.activation_based import surrogate

from . import base

import torch
import torch.nn as nn

class GatedLIFNode(base.MemoryModule):
    def __init__(self, T: int, inplane = None,
                 init_linear_decay = None, init_v_subreset = None, init_tau: float = 0.25, init_v_threshold: float = 0.5, init_conduct: float = 0.5, v_threshold: float=1.0,
                 surrogate_function: Callable = surrogate.Sigmoid(), step_mode='m', backend='torch'):
        """
        * :ref:`中文API <GatedLIFNode.__init__-cn>`

        .. _GatedLIFNode.__init__-cn:

        :param T: 时间步长
        :type T: int

        :param inplane: 输入tensor的通道数。不设置inplane，则默认使用layer-wise GLIF
        :type inplane: int

        :param init_linear_decay: 膜电位线性衰减常数初始值，不设置就默认为init_v_threshold/(T * 2)
        :type init_linear_decay: float

        :param init_v_subreset: 膜电位复位电压初始值
        :type init_v_subreset: float

        :param init_tau: 膜电位时间常数的初始值
        :type init_tau: float

        :param init_v_threshold: 神经元的阈值电压初始值
        :type init_v_threshold: float

        :param init_conduct: 膜电位电导率初始值
        :type init_conduct: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param step_mode: 步进模式，只支持 `'m'` (多步)
        :type step_mode: str

        :param backend: 使用哪种后端。不同的 ``step_mode`` 可能会带有不同的后端。可以通过打印 ``self.supported_backends`` 查看当前
            使用的步进模式支持的后端。在支持的情况下，使用 ``'cupy'`` 后端是速度最快的。gated-LIF只支持torch
        :type backend: str


        模型出处：`GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks <https://openreview.net/forum?id=UmFSx2c4ubT>`
        GLIF中所有的膜电位参数都是可学的，包括新引入的门控系数。

        * :ref:`API in English <GatedLIFNode.__init__-en>`

        .. _GatedLIFNode.__init__-en:

        :param T: time-step
        :type T: int

        :param inplane: input tensor channel number, default: None(layer-wise GLIF). If set, otherwise(channel-wise GLIF)
        :type inplane: int

        :param init_linear_decay: initial linear-decay constant，default: init_v_threshold/(T * 2)
        :type init_linear_decay: float

        :param init_v_subreset: initial soft-reset constant
        :type init_v_subreset: float

        :param init_tau: initial exponential-decay constant
        :type init_tau: float

        :param init_v_threshold: initial menbrane potential threshold
        :type init_v_threshold: float

        :param init_conduct: initial conduct
        :type init_conduct: float

        :param surrogate_function: surrogate gradient
        :type surrogate_function: Callable

        :param step_mode: step mode, only support `'m'` (multi-step)
        :type step_mode: str

        :param backend: backend fot this neuron layer, which can be "gemm" or "conv". This option only works for the multi-step mode
        :type backend: str


        Gated LIF neuron refers to `GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks <https://openreview.net/forum?id=UmFSx2c4ubT>`
        All membrane-related parameters are learnable, including the gates.
        """

        assert isinstance(init_tau, float) and init_tau < 1.
        assert isinstance(T, int) and T is not None
        assert isinstance(inplane, int) or inplane is None
        assert (isinstance(init_linear_decay, float) and init_linear_decay < 1.) or init_linear_decay is None
        assert (isinstance(init_v_subreset, float) and init_v_subreset < 1.) or init_v_subreset is None

        assert step_mode == 'm'
        super().__init__()
        self.surrogate_function = surrogate_function
        self.backend = backend
        self.step_mode = step_mode
        self.T = T
        self.register_memory('v', 0.)
        self.register_memory('u', 0.)
        self.channel_wise = inplane is not None
        if self.channel_wise: #channel-wise learnable params
            self.alpha, self.beta, self.gamma = [nn.Parameter(torch.tensor(0.2 * (np.random.rand(inplane) - 0.5), dtype=torch.float)) for i in range(3)]
            self.tau = nn.Parameter(- math.log(1 / init_tau - 1) * torch.ones(inplane, dtype=torch.float))
            self.v_threshold = nn.Parameter(- math.log(1 / init_v_threshold - 1) * torch.ones(inplane, dtype=torch.float))
            init_linear_decay = init_v_threshold / (T * 2) if init_linear_decay is None else init_linear_decay
            self.linear_decay = nn.Parameter(- math.log(1 / init_linear_decay - 1) * torch.ones(inplane, dtype=torch.float))
            init_v_subreset = init_v_threshold if init_v_subreset is None else init_v_subreset
            self.v_subreset = nn.Parameter(- math.log(1 / init_v_subreset - 1) * torch.ones(inplane, dtype=torch.float))
            self.conduct = nn.Parameter(- math.log(1 / init_conduct - 1) * torch.ones((T, inplane), dtype=torch.float))

        else:   #layer-wise learnable params
            self.alpha, self.beta, self.gamma = [nn.Parameter(torch.tensor(0.2 * (np.random.rand() - 0.5), dtype=torch.float)) for i in range(3)]
            self.tau = nn.Parameter(torch.tensor(- math.log(1 / init_tau - 1), dtype=torch.float))
            self.v_threshold = nn.Parameter(torch.tensor(- math.log(1 / init_v_threshold - 1), dtype=torch.float))
            init_linear_decay = init_v_threshold / (T * 2) if init_linear_decay is None else init_linear_decay
            self.linear_decay = nn.Parameter(torch.tensor(- math.log(1 / init_linear_decay - 1), dtype=torch.float))
            init_v_subreset = init_v_threshold if init_v_subreset is None else init_v_subreset
            self.v_subreset = nn.Parameter(torch.tensor(- math.log(1 / init_v_subreset - 1), dtype=torch.float))
            self.conduct = nn.Parameter(- math.log(1 / init_conduct - 1) * torch.ones(T, dtype=torch.float))

    @property
    def supported_backends(self):
        return 'torch'

    def extra_repr(self):
        with torch.no_grad():
            tau = self.tau
            v_subreset = self.v_subreset
            linear_decay = self.linear_decay
            conduct = self.conduct
        return super().extra_repr() + f', tau={tau}' + f', v_subreset={v_subreset}' + f', linear_decay={linear_decay}' + f', conduct={conduct}'

    def neuronal_charge(self, x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, t):
        input = x * (1 - beta * (1 - self.conduct[t].view(1, -1, 1, 1).sigmoid()))
        self.u = ((1 - alpha * (1 - self.tau.view(1, -1, 1, 1).sigmoid())) * self.v \
                  - (1 - alpha) * self.linear_decay.view(1, -1, 1, 1).sigmoid()) \
                 + input

    def neuronal_reset(self, spike, alpha: torch.Tensor, gamma: torch.Tensor):
        self.u = self.u - (1 - alpha * (1 - self.tau.view(1, -1, 1, 1).sigmoid())) * self.v * gamma * spike \
                 - (1 - gamma) * self.v_subreset.view(1, -1, 1, 1).sigmoid() * spike

    def neuronal_fire(self):
        return self.surrogate_function(self.u - self.v_threshold.view(1, -1, 1, 1).sigmoid())

    def multi_step_forward(self, x_seq: torch.Tensor):
        alpha, beta, gamma = self.alpha.view(1, -1, 1, 1).sigmoid(), self.beta.view(1, -1, 1, 1).sigmoid(), self.gamma.view(1, -1, 1, 1).sigmoid()
        y_seq = []
        spike = torch.zeros(x_seq.shape[1:], device=x_seq.device)
        for t in range(self.T):
            self.neuronal_charge(x_seq[t], alpha, beta, t)
            self.neuronal_reset(spike, alpha, gamma)
            spike = self.neuronal_fire()
            self.v = self.u
            y_seq.append(spike)
        return torch.stack(y_seq).squeeze()