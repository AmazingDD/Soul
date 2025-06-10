import math

import torch
import torch.nn as nn

from .LIF import BaseNode

class ParametricLIFNode(BaseNode):
    def __init__(self, config):
        
        init_tau = config['init_tau']
        v_threshold = config['v_threshold']
        v_reset = config['v_reset']
        surrogate_function = config['surrogate_function']
        detach_reset = config['detach_reset']
        step_mode = 'm'
        backend = 'torch'
        store_v_seq = config['store_v_seq']

        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.decay_input = config['decay_input']
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch', )
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. - self.w.sigmoid()
        return super().extra_repr() + f', tau={tau}, sg={self.surrogate_function}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) * self.w.sigmoid()
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()
        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - self.w.sigmoid()) + x
            else:
                self.v = self.v - (self.v - self.v_reset) * self.w.sigmoid() + x