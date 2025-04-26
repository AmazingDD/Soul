from l_quantize import Aggregated_Spiking_Layer as ASL_xs
import torch
from spikingjelly.activation_based import layer
@torch.no_grad()
def fuse_rateBatchNorm_xs(module):
    if not isinstance(module,ASL_xs):
        return
    _layers=module._layer
    if _layers is None or (not isinstance(_layers,(torch.nn.Conv2d,torch.nn.ConvTranspose2d))) or not isinstance(module._norm, torch.nn.BatchNorm2d):
        return
    
    _new_conv= fuse_conv2d(_layers, module._norm)
    module._layer= _new_conv
    module._norm= torch.nn.Identity()

def check_norm_xs(module):
    if isinstance(module,ASL_xs):
        assert module._norm is None

def fuse_conv2d(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    if isinstance(conv,torch.nn.Conv2d):
        fused_conv = layer.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
            step_mode="m"
        )
    elif isinstance(conv,torch.nn.ConvTranspose2d):
        fused_conv = layer.ConvTranspose2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
            step_mode="m"
        )

    fused_conv.weight = torch.nn.Parameter(w)
    fused_conv.bias = torch.nn.Parameter(b)
    return fused_conv