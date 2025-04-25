import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional
import torch.nn.utils.prune as prune

class StructuredPruner(object):
    def __init__(self, model, device):
        self.model_type = model.model_type

        self.model = model
        self.device = device
        self.model.to(device)

        self.C, self.H, self.W = model.C, model.H, model.W
        self.num_classes = model.num_classes
        self.prune_ratio = None

    def get_model(self):
        return self.model

    def freeze(self):
        for name, param in self.model.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def apply_pruning(self, ratio=0.4):
        # self.pruned_model = deepcopy(self.model)

        bn_weights = []

        for m in self.model.features:
            if isinstance(m, layer.BatchNorm2d):
                bn_weights.append(m.weight.data.abs().clone())
        all_weights = torch.cat(bn_weights)
        threshold = torch.quantile(all_weights, ratio)

        # 为每个通道生成保留 mask（按顺序）
        masks = []
        for bn_w in bn_weights:
            masks.append(bn_w.ge(threshold).float())

        # 构造新的通道配置
        new_features = []
        masks_idx = 0
        cin = self.C
        in_mask = torch.ones(cin).bool()
        H, W = self.H, self.W
        for m in self.model.features:
            if isinstance(m, layer.Conv2d):
                cout = int(masks[masks_idx].sum().item())
                new_m = layer.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False)

                out_mask = masks[masks_idx]
                idx_out = torch.where(out_mask > 0)[0]
                idx_in = torch.where(in_mask > 0)[0]

                new_m.weight.data = m.weight.data[idx_out][:, idx_in, :, :].clone()
                if m.bias is not None:
                    new_m.bias.data = m.bias.data[idx_out].clone()

                new_features.append(new_m)
                
            elif isinstance(m, layer.MaxPool2d):
                new_features.append(layer.MaxPool2d(2, 2))
                H //= 2
                W //= 2
            elif isinstance(m, neuron.LIFNode):
                new_features.append(neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()))
            elif isinstance(m, layer.BatchNorm2d):
                cout = int(masks[masks_idx].sum().item())
                new_m = layer.BatchNorm2d(cout)

                out_mask = masks[masks_idx]
                idx = torch.where(out_mask > 0)[0]
                new_m.weight.data = m.weight.data[idx].clone()
                new_m.bias.data = m.bias.data[idx].clone()
                new_m.running_mean = m.running_mean[idx].clone()
                new_m.running_var = m.running_var[idx].clone()

                new_features.append(new_m)

                cin = cout
                in_mask = masks[masks_idx]
                masks_idx += 1 

        self.model.features = nn.Sequential(*new_features)

        # 构建新的分类头用来fine-tune
        self.model.fc = nn.Sequential(
            layer.Linear(int(in_mask.sum().item()) * H * W, 1024, bias=False),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()),
            nn.Linear(1024, self.num_classes)
        )

        functional.set_step_mode(self.model, 'm')
        self.model.to(self.device)


class UnstructuredPruner(object):
    def __init__(self, model, device):
        self.model = model
        self.model.to(device)
        
    def apply_pruning(self, ratio=0.3):
        # pruning and replace all mask to 0.
        for _, module in self.model.features.named_modules():
            if isinstance(module, (layer.Conv2d, nn.Conv2d, layer.Linear, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=ratio)

    # def remove_pruning(self):
        for _, module in self.model.features.named_modules():
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
