import copy
import math
import numpy as np
from abc import abstractmethod
from typing import Callable, Optional
from spikingjelly.activation_based import surrogate

import torch
import torch.nn as nn

class StepModule:
    def supported_step_mode(self):
        """
        * :ref:`API in English <StepModule.supported_step_mode-en>`

        .. _StepModule.supported_step_mode-cn:

        :return: 包含支持的后端的tuple
        :rtype: tuple[str]

        返回此模块支持的步进模式。

        * :ref:`中文 API <StepModule.supported_step_mode-cn>`

        .. _StepModule.supported_step_mode-en:

        :return: a tuple that contains the supported backends
        :rtype: tuple[str]

        """
        return ('s', 'm')

    @property
    def step_mode(self):
        """
        * :ref:`API in English <StepModule.step_mode-en>`

        .. _StepModule.step_mode-cn:

        :return: 模块当前使用的步进模式
        :rtype: str

        * :ref:`中文 API <StepModule.step_mode-cn>`

        .. _StepModule.step_mode-en:

        :return: the current step mode of this module
        :rtype: str
        """
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: str):
        """
        * :ref:`API in English <StepModule.step_mode-setter-en>`

        .. _StepModule.step_mode-setter-cn:

        :param value: 步进模式
        :type value: str

        将本模块的步进模式设置为 ``value``

        * :ref:`中文 API <StepModule.step_mode-setter-cn>`

        .. _StepModule.step_mode-setter-en:

        :param value: the step mode
        :type value: str

        Set the step mode of this module to be ``value``

        """
        if value not in self.supported_step_mode():
            raise ValueError(f'step_mode can only be {self.supported_step_mode()}, but got "{value}"!')
        self._step_mode = value

class MemoryModule(nn.Module, StepModule):
    def __init__(self):
        """
        * :ref:`API in English <MemoryModule.__init__-en>`

        .. _MemoryModule.__init__-cn:

        ``MemoryModule`` 是SpikingJelly中所有有状态（记忆）模块的基类。

        * :ref:`中文API <MemoryModule.__init__-cn>`

        .. _MemoryModule.__init__-en:

        ``MemoryModule`` is the base class of all stateful modules in SpikingJelly.

        """
        super().__init__()
        self._memories = {}
        self._memories_rv = {}
        self._backend = 'torch'
        self.step_mode = 's'

    @property
    def supported_backends(self):
        """
        * :ref:`API in English <MemoryModule.supported_backends-en>`

        .. _MemoryModule.supported_backends-cn:

        返回支持的后端，默认情况下只有 `('torch', )`

        :return: 支持的后端
        :rtype: tuple[str]

        * :ref:`中文API <MemoryModule.supported_backends-cn>`

        .. _MemoryModule.supported_backends-en:

        Return the supported backends. The default return value is `('torch', )`

        :return: supported backends
        :rtype: tuple[str]

        """
        return ('torch',)

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value: str):
        if value not in self.supported_backends:
            raise NotImplementedError(f'{value} is not a supported backend of {self._get_name()}!')
        self._backend = value

    @abstractmethod
    def single_step_forward(self, x: torch.Tensor, *args, **kwargs):
        """
        * :ref:`API in English <MemoryModule.single_step_forward-en>`

        .. _MemoryModule.single_step_forward-cn:

        :param x: input tensor with ``shape = [N, *] ``
        :type x: torch.Tensor

        本模块的单步的前向传播函数


        * :ref:`中文 API <MemoryModule.single_step_forward-cn>`

        .. _MemoryModule.single_step_forward-en:

        :param x: input tensor with ``shape = [N, *] ``
        :type x: torch.Tensor

        The single-step forward function for this module

        """
        pass

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        """
        * :ref:`API in English <MemoryModule.multi_step_forward-en>`

        .. _MemoryModule.multi_step_forward-cn:

        :param x_seq: input tensor with ``shape = [T, N, *] ``
        :type x_seq: torch.Tensor

        本模块的多步的前向传播函数，通过调用 ``T`` 次 ``single_step_forward(x[t], *args, **kwargs)`` 实现


        * :ref:`中文 API <MemoryModule.multi_step_forward-cn>`

        .. _MemoryModule.multi_step_forward-en:

        :param x_seq: input tensor with ``shape = [T, N, *] ``
        :type x_seq: torch.Tensor

        The multi-step forward function for this module, which is implemented by calling ``single_step_forward(x[t], *args, **kwargs)`` over ``T`` times

        """

        T = x_seq.shape[0]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t], *args, **kwargs)
            y_seq.append(y.unsqueeze(0))

        return torch.cat(y_seq, 0)

    def forward(self, *args, **kwargs):
        if self.step_mode == 's':
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return f'step_mode={self.step_mode}, backend={self.backend}'

    def register_memory(self, name: str, value):
        """
        * :ref:`API in English <MemoryModule.register_memory-en>`

        .. _MemoryModule.register_memory-cn:

        :param name: 变量的名字
        :type name: str
        :param value: 变量的值
        :type value: any

        将变量存入用于保存有状态变量（例如脉冲神经元的膜电位）的字典中。这个变量的重置值会被设置为 ``value``。每次调用 ``self.reset()``
        函数后， ``self.name`` 都会被重置为 ``value``。

        * :ref:`中文API <MemoryModule.register_memory-cn>`

        .. _MemoryModule.register_memory-en:

        :param name: variable's name
        :type name: str
        :param value: variable's value
        :type value: any

        Register the variable to memory dict, which saves stateful variables (e.g., the membrane potential of a
        spiking neuron). The reset value of this variable will be ``value``. ``self.name`` will be set to ``value`` after
        each calling of ``self.reset()``.

        """
        assert not hasattr(self, name), f'{name} has been set as a member variable!'
        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self):
        """
        * :ref:`API in English <MemoryModule.reset-en>`

        .. _MemoryModule.reset-cn:

        重置所有有状态变量为默认值。

        * :ref:`中文API <MemoryModule.reset-cn>`

        .. _MemoryModule.reset-en:

        Reset all stateful variables to their default values.
        """
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def set_reset_value(self, name: str, value):
        self._memories_rv[name] = copy.deepcopy(value)

    def __getattr__(self, name: str):
        if '_memories' in self.__dict__:
            memories = self.__dict__['_memories']
            if name in memories:
                return memories[name]

        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        _memories = self.__dict__.get('_memories')
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._memories:
            del self._memories[name]
            del self._memories_rv[name]
        else:
            return super().__delattr__(name)

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        memories = list(self._memories.keys())
        keys = module_attrs + attrs + parameters + modules + buffers + memories

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def memories(self):
        """
        * :ref:`API in English <MemoryModule.memories-en>`

        .. _MemoryModule.memories-cn:

        :return: 返回一个所有状态变量的迭代器
        :rtype: Iterator

        * :ref:`中文API <MemoryModule.memories-cn>`

        .. _MemoryModule.memories-en:

        :return: an iterator over all stateful variables
        :rtype: Iterator
        """
        for name, value in self._memories.items():
            yield value

    def named_memories(self):
        """
        * :ref:`API in English <MemoryModule.named_memories-en>`

        .. _MemoryModule.named_memories-cn:

        :return: 返回一个所有状态变量及其名称的迭代器
        :rtype: Iterator

        * :ref:`中文API <MemoryModule.named_memories-cn>`

        .. _MemoryModule.named_memories-en:

        :return: an iterator over all stateful variables and their names
        :rtype: Iterator
        """

        for name, value in self._memories.items():
            yield name, value

    def detach(self):
        """
        * :ref:`API in English <MemoryModule.detach-en>`

        .. _MemoryModule.detach-cn:

        从计算图中分离所有有状态变量。

        .. tip::

            可以使用这个函数实现TBPTT(Truncated Back Propagation Through Time)。


        * :ref:`中文API <MemoryModule.detach-cn>`

        .. _MemoryModule.detach-en:

        Detach all stateful variables.

        .. admonition:: Tip
            :class: tip

            We can use this function to implement TBPTT(Truncated Back Propagation Through Time).

        """

        for key in self._memories.keys():
            if isinstance(self._memories[key], torch.Tensor):
                self._memories[key].detach_()

    def _apply(self, fn):
        for key, value in self._memories.items():
            if isinstance(value, torch.Tensor):
                self._memories[key] = fn(value)
        # do not apply on default values
        # for key, value in self._memories_rv.items():
        #     if isinstance(value, torch.Tensor):
        #         self._memories_rv[key] = fn(value)
        return super()._apply(fn)

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica._memories = self._memories.copy()
        return replica
    

class GatedLIFNode(MemoryModule):
    def __init__(self, T: int, inplane = None,
                 init_linear_decay = None, init_v_subreset = None, init_tau: float = 0.25, init_v_threshold: float = 0.5, init_conduct: float = 0.5,
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
        return torch.stack(y_seq)