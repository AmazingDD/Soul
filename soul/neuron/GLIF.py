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