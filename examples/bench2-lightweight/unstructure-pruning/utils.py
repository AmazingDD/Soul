import os
import time
import errno
import datetime
import logging
from math import nan

import torch
import torch.nn as nn

from model.layers import ConvBlock
from typing import List

def safe_makedirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def setup_logger(output_dir):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s',
                                  datefmt=r'%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    return logger

class DVStransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = torch.from_numpy(img).float()
        shape = [img.shape[0], img.shape[1]]
        img = img.flatten(0, 1)
        img = self.transform(img)
        shape.extend(img.shape[1:])
        return img.view(shape)

class DatasetSplitter(torch.utils.data.Dataset):
    '''To split CIFAR10DVS into training dataset and test dataset'''
    def __init__(self, parent_dataset, rate=0.1, train=True):

        self.parent_dataset = parent_dataset
        self.rate = rate
        self.train = train
        self.it_of_original = len(parent_dataset) // 10
        self.it_of_split = int(self.it_of_original * rate)

    def __len__(self):
        return int(len(self.parent_dataset) * self.rate)

    def __getitem__(self, index):
        base = (index // self.it_of_split) * self.it_of_original
        off = index % self.it_of_split
        if not self.train:
            off = self.it_of_original - off - 1
        item = self.parent_dataset[base + off]

        return item
    
class Timer:
    def __init__(self, timer_name, logger):
        self.timer_name = timer_name
        self.logger = logger

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # seconds
        self.logger.debug('{} spent: {}.'.format(
            self.timer_name, str(datetime.timedelta(seconds=int(self.interval)))))
        
class DatasetWarpper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.trasnform = transform

    def __getitem__(self, index):
        return self.trasnform(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)
    
def set_pruning_mode(model: nn.Module, mode: bool = False):
    for m in model.modules():
        if isinstance(m, ConvBlock):
            m._pruning(mode)

def init_mask(model: nn.Module, weights_mean: float, neurons_mean: float, weights_std: float = 0, neurons_std: float = 0):
    mask_list = []
    for m in model.modules():
        if isinstance(m, ConvBlock):
            masks = m.init_mask(weights_mean, neurons_mean, weights_std, neurons_std)
            for mask in masks:
                mask_list.append(mask)
    return mask_list
        
class PenaltyTerm(nn.Module):
    def __init__(self, model: nn.Module, lmbda: float) -> None:
        super(PenaltyTerm, self).__init__()
        self.layers: List[ConvBlock] = []
        self.model = model
        for m in model.modules():
            if isinstance(m, ConvBlock):
                self.layers.append(m)
        self.lmbda = lmbda
        model.calc_c()

    def forward(self):
        loss = 0
        for layer in self.layers:
            if layer.sparse_neurons:
                loss = loss + (self.lmbda * layer.neuron_mask.lmbda) * (torch.sigmoid(
                    layer.neuron_mask.mask_value * layer.neuron_mask.temp)).sum()
            if layer.sparse_weights:
                loss = loss + (self.lmbda * layer.weight_mask.lmbda) * (torch.sigmoid(
                    layer.weight_mask.mask_value * layer.weight_mask.temp)).sum()
        return loss
    
def left_neurons(model: nn.Module):
    conn = 0
    total = 0
    for m in model.modules():
        if isinstance(m, ConvBlock):
            c, t = m.neuron_mask.left()
            conn, total = conn + c, total + t
    return conn, total


def left_weights(model: nn.Module):
    conn = 0
    total = 0
    for m in model.modules():
        if isinstance(m, ConvBlock):
            c, t = m.weight_mask.left()
            conn, total = conn + c, total + t
    return conn, total

class BaseMonitor:
    def __init__(self):
        self.hooks = []
        self.monitored_layers = []
        self.records = []
        self.name_records_index = {}
        self._enable = True

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        elif isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        else:
            raise ValueError(i)

    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.name_records_index.items():
            v.clear()

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def is_enable(self):
        return self._enable

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove_hooks()

class SOPMonitor(BaseMonitor):
    def __init__(self, net: nn.Module):
        super().__init__()
        for name, m in net.named_modules():
            if name in net.skip:
                continue
            if isinstance(m, ConvBlock):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                # conv.weight [C_out, C_in, H_k, W_k]
                if m.sparse_weights:
                    connects = (m.weight_mask(m.conv.weight) != 0).float()
                else:
                    connects = torch.ones_like(m.conv.weight)
                if m.sparse_neurons:
                    mask = (m.neuron_mask.mask_value > 0).float().squeeze(0)
                else:
                    mask = None
                self.hooks.append(m.register_forward_hook(self.create_hook(name, connects, mask)))

    def cal_sop(self, x: torch.Tensor, connects: torch.Tensor, mask: torch.Tensor, m: nn.Conv2d):
        out = torch.nn.functional.conv2d(x, connects, None, m.stride, m.padding, m.dilation,
                                         m.groups)
        if mask is None:
            sop = out.sum()
        else:
            sop = (out * mask).sum()
        return sop.unsqueeze(0)

    def create_hook(self, name, connects, mask):
        def hook(m: ConvBlock, x: torch.Tensor, y: torch.Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(
                    self.cal_sop(
                        snn_to_ann_input(unpack_len1_tuple(x)).detach(), connects, mask, m.conv))

        return hook
    
def snn_to_ann_input(x: torch.Tensor):
    if len(x.shape) == 5:
        return x.flatten(0, 1)
    else:
        return x
    
def unpack_len1_tuple(x):
    if isinstance(x, tuple) and x.__len__() == 1:
        return x[0]
    else:
        return x
