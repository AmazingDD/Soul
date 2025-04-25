import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate
import torch.nn.utils.prune as prune
from copy import deepcopy

class StructuredPruner(object):
    def __init__(self, model):
        self.model_type = model.model_type

        self.original_model = deepcopy(model)
        self.prune_ratio = None
        self.bn_layers = []
        self.bn_weights = []
        self.masks = []

    def collect_bn_weights(self):
        self.bn_layers = []
        self.bn_weights = []

        for m in self.original_model.features.modules():
            if isinstance(m, (nn.BatchNorm2d, layer.BatchNorm2d)):
                self.bn_layers.append(m)
                self.bn_weights.append(m.weight.data.abs().clone())

    def compute_prune_masks(self):
        all_weights = torch.cat(self.bn_weights)
        num_total = all_weights.shape[0]
        num_prune = int(num_total * self.prune_ratio)
        threshold = all_weights.sort()[0][num_prune]

        self.masks = [(w > threshold) for w in self.bn_weights]

    def prune_conv_bn(self, conv, bn, mask_out, mask_in=None):
        idx_out = torch.where(mask_out)[0]
        idx_in = torch.arange(conv.in_channels) if mask_in is None else torch.where(mask_in)[0]

        new_conv = layer.Conv2d(len(idx_in), len(idx_out), 
                                kernel_size=conv.kernel_size,
                                stride=conv.stride,
                                padding=conv.padding,
                                dilation=conv.dilation,
                                groups=conv.groups,
                                bias=(conv.bias is not None))
        new_bn = layer.BatchNorm2d(len(idx_out))

        # weight pruning
        new_conv.weight.data = conv.weight.data[idx_out][:, idx_in, :, :].clone()
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data[idx_out].clone()

        # bn pruning
        new_bn.weight.data = bn.weight.data[idx_out].clone()
        new_bn.bias.data = bn.bias.data[idx_out].clone()
        new_bn.running_mean = bn.running_mean[idx_out].clone()
        new_bn.running_mean = bn.running_mean[idx_out].clone()

        return new_conv, new_bn, idx_out

    def rebuild_vgg(self):
        H, W = self.original_model.H, self.original_model.W
        new_features = []
        last_mask = None
        mask_idx = 0
        for l in self.original_model.features:
            if isinstance(l, layer.Conv2d):
                bn = None
                # find next bn
                for i in range(mask_idx, len(self.original_model.features)):
                    if isinstance(self.original_model.features[i], layer.BatchNorm2d):
                        bn = self.original_model.features[i]
                        break
                assert bn is not None, 'Each Conv2d should be followed by a BatchNorm2d'
                new_conv, new_bn, last_mask = self.prune_conv_bn(l, bn, self.masks[mask_idx], last_mask)
                new_features.append(new_conv)
                new_features.append(new_bn)
                mask_idx += 1
            elif isinstance(l, neuron.LIFNode):
                new_features.append(neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()))
            elif isinstance(l, layer.MaxPool2d):
                new_features.append(l)
                H //= 2
                W //= 2

        pruned_model = deepcopy(self.original_model)
        pruned_model.features = nn.Sequential(*new_features)

        out_channels = last_mask.sum().item()
        last_feature_dim = out_channels * H * W

        old_fc = self.original_model.fc
        new_fc = nn.Sequential(
            layer.Linear(last_feature_dim, old_fc[0].out_features),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()),
            old_fc[2],
        )
        pruned_model.fc = new_fc

        return pruned_model
    
    def prune(self, ratio=0.4):
        self.prune_ratio = ratio

        self.collect_bn_weights()
        self.compute_prune_masks()
        if self.model_type == 'vgg':
            return self.rebuild_vgg()
        else:
            raise NotImplementedError(f'Model type "{self.model_type}" not supported yet.')
    
    @staticmethod
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

class UnstructuredPruner(object):
    def __init__(self, model):
        self.model = model
        
    def apply_pruning(self, ratio=0.3):
        for name, module in self.model.features.named_modules():
            if isinstance(module, (layer.Conv2d, nn.Conv2d, layer.Linear, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=ratio)

    def remove_pruning(self):
        for name, module in self.model.features.named_modules():
            if isinstance(module, (layer.Conv2d, nn.Conv2d, layer.Linear, nn.Linear)):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass

    def compute_sparsity(self):
        total = 0
        zeros = 0
        for module in self.model.features.modules():
            
            if isinstance(module, (nn.Conv2d, layer.Conv2d, nn.Linear, layer.Linear)):
                weight = module.weight.data
                zeros += torch.sum(weight == 0).item()
                total += weight.numel()
        sparsity = 100. * zeros / total
        print(f'[Sparsity] {zeros}/{total} = {sparsity:.2f}%')
        return sparsity
    
    def get_model(self):
        return self.model

    def freeze(self):
        for name, param in self.model.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
