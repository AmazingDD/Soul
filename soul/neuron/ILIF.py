import torch
import torch.nn as nn
from typing import Callable
from abc import abstractmethod
from soul.neuron.base import MemoryModule
from soul.utils.surrogate import Quant

class ILIF_BaseNode(MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = Quant(), detach_reset: bool = False):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.register_memory('v_threshold', v_threshold)
        self.register_memory('v_reset', v_reset)

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.v = self.v - spike_d * self.v_threshold

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike / 8


class ILIFNode(ILIF_BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = Quant(), detach_reset: bool = False,
                 step_mode='s', backend='torch', tau: float = 2., store_v_seq: bool = False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        
    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)
        
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

    def forward(self, x: torch.Tensor):
        return super().forward(x)