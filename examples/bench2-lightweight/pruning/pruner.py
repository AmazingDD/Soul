import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional
import torch.nn.utils.prune as prune

from model.resnet import BasicBlock

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

        if self.model_type == 'vgg':
            self.rebuild_vgg(ratio)
        elif self.model_type == 'resnet':
            self.rebuild_resnet(ratio)
        else:
            raise ValueError(f'Unsupported model type: {self.model_type}')

        functional.set_step_mode(self.model, 'm')
        self.model.to(self.device)

    def rebuild_resnet(self, ratio):
        bn_weights = []
        for m in self.model.modules():
            if isinstance(m, BasicBlock):
                bn_weights.append(m.conv1[1].weight.data.abs().clone())
                bn_weights.append(m.conv2[1].weight.data.abs().clone())
        bn_weights = torch.cat(bn_weights)
        threshold = torch.quantile(bn_weights, ratio)

        def prune_bn_mask(bn_layer):
            return bn_layer.weight.data.abs() >= threshold
        
        def copy_conv(conv_old, conv_new, in_idx, out_idx):
            # print(conv_old.weight.data.shape)
            # print(conv_new.weight.data.shape)
            # print(out_idx, in_idx)

            conv_new.weight.data = conv_old.weight.data[out_idx][:, in_idx, :, :].clone()

        def copy_bn(bn_old, bn_new, idx):
            bn_new.weight.data = bn_old.weight.data[idx].clone()
            bn_new.bias.data = bn_old.bias.data[idx].clone()
            bn_new.running_mean = bn_old.running_mean[idx].clone()
            bn_new.running_var = bn_old.running_var[idx].clone()

        # 记录每个block的通道mask
        block_masks = []

        def prune_layer(blocks, in_channels):
            new_blocks = []
            for block in blocks:
                # 原始 block 通道数
                out_channels = block.conv2[1].num_features

                # 获得 mask
                mask1 = prune_bn_mask(block.conv1[1])
                mask2 = prune_bn_mask(block.conv2[1])
                c1, c2 = int(mask1.sum()), int(mask2.sum())

                new_conv1 = nn.Conv2d(in_channels, c1, 3, stride=block.conv1[0].stride, padding=1, bias=False)
                new_bn1 = nn.BatchNorm2d(c1)
                
                new_conv2 = nn.Conv2d(c1, c2, 3, stride=1, padding=1, bias=False)
                new_bn2 = nn.BatchNorm2d(c2)
                
                if block.downsample is not None:
                    ds_conv = block.downsample[0][0]
                    ds_bn = block.downsample[0][1]
                    new_ds_conv = nn.Conv2d(in_channels, c2, 1, stride=ds_conv.stride, bias=False)
                    new_ds_bn = nn.BatchNorm2d(c2)
                    copy_conv(ds_conv, new_ds_conv, torch.arange(in_channels), torch.where(mask2)[0])
                    copy_bn(ds_bn, new_ds_bn, torch.where(mask2)[0])
                    new_downsample = nn.Sequential(
                        layer.SeqToANNContainer(
                            new_ds_conv, 
                            new_ds_bn
                        ),
                        neuron.LIFNode(detach_reset=True)
                    )
                else:
                    if in_channels != c2:
                        new_ds_conv = nn.Conv2d(in_channels, c2, 1, stride=1, bias=False)
                        new_ds_bn = nn.BatchNorm2d(c2)
                        nn.init.kaiming_normal_(new_ds_conv.weight, mode='fan_out', nonlinearity='relu')
                        new_ds_bn.weight.data.fill_(1.0)
                        new_ds_bn.bias.data.zero_()
                        new_downsample = nn.Sequential(
                            layer.SeqToANNContainer(
                                new_ds_conv, 
                                new_ds_bn
                            ),
                            neuron.LIFNode(detach_reset=True)
                        )
                    else:
                        new_downsample = None

                # 创建新的block
                new_block = BasicBlock(in_channels, c2, stride=block.stride, downsample=new_downsample, connect_f=self.model.connect_f)
                new_block.conv1 = layer.SeqToANNContainer(
                    new_conv1, 
                    new_bn1
                )
                new_block.conv2 = layer.SeqToANNContainer(
                    new_conv2,
                    new_bn2
                )

                # 复制权重
                idx_in = torch.arange(in_channels)
                idx_c1 = torch.where(mask1)[0]
                idx_c2 = torch.where(mask2)[0]
                copy_conv(block.conv1[0], new_conv1, idx_in, idx_c1)
                copy_bn(block.conv1[1], new_bn1, idx_c1)
                copy_conv(block.conv2[0], new_conv2, idx_c1, idx_c2)
                copy_bn(block.conv2[1], new_bn2, idx_c2)

                new_blocks.append(new_block)
                in_channels = c2

            return nn.Sequential(*new_blocks), in_channels
        
        conv1_mask = prune_bn_mask(self.model.conv1[1])
        c1 = int(conv1_mask.sum())
        new_conv1 = nn.Conv2d(self.model.C, c1, kernel_size=7, stride=2, padding=3, bias=False)
        new_bn1 = nn.BatchNorm2d(c1)
        copy_conv(self.model.conv1[0], new_conv1, torch.arange(self.model.C), torch.where(conv1_mask)[0])
        copy_bn(self.model.conv1[1], new_bn1, torch.where(conv1_mask)[0])

        self.model.conv1 = layer.SeqToANNContainer(
            new_conv1,
            new_bn1
        )

        in_c = c1
        self.model.layer1, in_c = prune_layer(self.model.layer1, in_c)
        self.model.layer2, in_c = prune_layer(self.model.layer2, in_c)
        self.model.layer3, in_c = prune_layer(self.model.layer3, in_c)
        self.model.layer4, in_c = prune_layer(self.model.layer4, in_c)

        self.model.features = nn.Sequential(
            self.model.conv1, 
            self.model.sn1,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
            self.model.avgpool,
        )

        # 新建分类头
        if self.model.num_classes * 10 < in_c * BasicBlock.expansion:
            self.model.fc = nn.Sequential(
                nn.Linear(in_c * BasicBlock.expansion, self.model.num_classes * 10),
                nn.AvgPool1d(10, 10)
            )
        else:
            self.model.fc = nn.Linear(in_c * BasicBlock.expansion, self.model.num_classes)

        # print(self.model.features)

    def rebuild_vgg(self, ratio):
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
