import torch
import torch.nn as nn

import copy
import pandas as pd

class PruningNetworkManager(object):
    def __init__(self, model):
        self.pruning_layers = self.get_pruning_layers(model)
        self.layers = self.get_granularity(model)

        self.all_num = 0.0
        self.grow_num = 0.0
        self.prune_num = 0.0

        self.masks = []
        self.mask_grows = []

        self.grads=[]
        self.count=1

    def get_granularity(self, model): # Channel-wise
        layers = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
                layers.append(module)
        return layers

    def get_pruning_layers(self, model):
        pruning_layers = []
        for module in model.modules():
            if isinstance(module, PruningLayer):
                pruning_layers.append(module)

        return pruning_layers
    
    def evaling(self):
        for pruning_layer in self.pruning_layers:
            pruning_layer.set_eval()

    def training(self):
        for prunning_layer in self.pruning_layers:
            prunning_layer.set_train()

    def reset_zeros(self):
        for pruning_layer in self.pruning_layers:
            pruning_layer.reset_zero()

    def update_masks(self,model, a, b):
        '''
        a : float
            ratio of channel pruning
        b : float
            ratio of regrowth
        '''
        old = copy.deepcopy(self.masks)
        acts = []
        for pruning_layer in self.pruning_layers:
            activation = pruning_layer.get_activations()
            acts.append(activation)

        sorted_indices = torch.argsort(torch.cat(acts), descending=True)
        print(sorted_indices.shape) # num of all filters
        num_elements = int(len(sorted_indices) * a)
        threshold_indice = sorted_indices[num_elements]
        threshold = torch.cat(acts)[threshold_indice] # the threshold of activation value indicates pruning

        i = 0
        for i in range(len(acts)): # select masks layer by layer
            if len(self.masks) != 13: # number of prunelayer
                mask = (acts[i] > threshold).float().detach()
                self.masks.append(mask)
            else:
                self.masks[i] = (acts[i] > threshold).float().detach()
            i += 1

        i = 0
        prune_num = 0
        for i in range(len(self.masks)):
            prune_num += torch.sum(self.masks[i] == 0).item()
            i += 1
        print(prune_num)


class PruningLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.activation_means = None
        self.mask = None

        self.v_accumulated = None
        self.prune_count = 0  # recording the number of pruned neurons
        self.restore_count = 0

        self.count_input = 1 # counting how many inputs fed into current layer
        self.training_state = True
        self.spikes = 0
        self.membrance = 0

        self.p1 = 0
        self.p2 = 0
        self.aa = 0
        self.a = 0

    def set_eval(self):
        self.training_state = False

    def set_train(self):
        self.training_state = True

    def get_mask(self):
        return self.mask

    def forward(self, x, v):
        '''
        Parameters
        ----------
        x : input at each time step
        v : membrane potential after each time step
        '''
        # determine important filter at training phase 
        if self.training_state: 
            self.spikes = x.detach()
            self.membrance = v.abs().detach()
            v_temp = (self.spikes + self.membrance).detach()
            if self.aa == 0 or self.v_accumulated is None:
                # in conv-pruning, the shape of membrane (T, B, C, H, W)
                # prove the element in a filter has spikes (important) in this batch input
                # self.v_accumulated = torch.mean(v_temp, dim=(0, 1, 3, 4)) # -> (C)
                self.v_accumulated = torch.sum(v_temp, dim=(0, 1, 3, 4)) # -> (C)
            else: 
                # self.v_accumulated = self.v_accumulated * self.count_input + torch.mean(v_temp, dim=(0, 1, 3, 4))
                self.count_input += 1
                # self.v_accumulated /= self.count_input
                self.v_accumulated += torch.sum(v_temp, dim=(0, 1, 3, 4)) # -> (C)

        return x
    
    def reset_zero(self):
        self.v_accumulated = None
        self.count_input = 1
        self.aa = 1

    def get_prunenum(self):
        num = torch.sum(self.mask == 0).item()
        return num
    
    def get_allnum(self):
        num = self.mask.numel()
        return num
    
    def get_activations(self):
        # return self.v_accumulated
        return self.v_accumulated / self.count_input


    def save_csv(self):
        self.mask_zero_percentage = torch.sum(self.mask == 0).item() / self.mask.numel() * 100
        data = {'Mask Percentage': [self.mask_zero_percentage]}
        df = pd.DataFrame(data)
        df.to_csv('mask_info.csv', mode='a', index=False)
        datathre = {'threshold': [self.threshold], 'reset_threshold': self.restore_threshold}
        dfthre = pd.DataFrame(datathre)
        dfthre.to_csv('threshold.csv', mode='a', index=False)

        prune_indices = (self.mask == 0).nonzero()
        self.prune_count = len(prune_indices)

        restore_indices = (self.mask == 1).nonzero()
        self.restore_count = len(restore_indices)
        print('Pruned neurons:', self.prune_count)
        print('Restored neurons:', self.restore_count)
        print('P neurons:', self.p1)
        print('P neurons:', self.p2)
        print('A neurons:', self.a)
        print('total:', self.mask.numel())
        print(self.threshold)
        print(self.restore_threshold)
