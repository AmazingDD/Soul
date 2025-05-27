import os
import torch
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODULE_SOP_DICT = {}

def img2col(X, kernel_size, stride=1, pad=0):
    """
    Convert a 4D input tensor into a 2D column matrix (img2col operation).
    Params:
        X: Input tensor with shape (N, C, H, W)
        kernel_size: Convolution kernel size (int or tuple)
        stride: Stride (int or tuple)
        pad: Padding size
    Returns:
        2D column matrix with shape (N*out_h*out_w, C*kh*kw)
    """

    N, C, H, W = X.shape
    kh = kw = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    sh = sw = stride if isinstance(stride, int) else stride[0]
    
    # implement padding
    X_pad = np.pad(X, [(0,0), (0,0), (pad, pad), (pad, pad)], mode='constant')
    
    # calculate output shape
    H_out = (H + 2*pad - kh) // sh + 1
    W_out = (W + 2*pad - kw) // sw + 1
    
    # generate sliding window view
    windows = sliding_window_view(X_pad, (kh, kw), axis=(2, 3))
    
    # slicing by step
    windows = windows[:, :, ::sh, ::sw]
    
    # rebuild column-wise matrix
    cols = windows.reshape(N, C, H_out, W_out, kh*kw)
    cols = cols.transpose(0, 2, 3, 1, 4).reshape(N*H_out*W_out, -1)
    
    return cols

def conv_forward_with_sparsity(X, W, b, stride=1, pad=0):
    """
    Implement convolutional forward propagation using img2col and compute the effective computation ratio  
    Params:
        X: Input data (N, C, H, W)  
        W: Convolution kernel (F, C, KH, KW)  
        b: Bias (F,)  
        stride: Stride  
        pad: Padding
    Returns:
        output: Convolution result (N, F, OH, OW)  
        effective_ratio: Effective computation ratio (0.0-1.0)  
    """
    # img2col transformation for input
    cols = img2col(X, W.shape[2:], stride, pad)
    
    # change conv kernel to 2D-matrix (including rotation 180 degree)
    F, C, KH, KW = W.shape
    W_rot = np.rot90(W, 2, axes=(2, 3))  #  rotate 180 degrees in H and W dim
    W_reshaped = W_rot.reshape(F, -1).T  # shape (C*KH*KW, F)
    
    # implement matrix calculation 
    out = cols @ W_reshaped + b
    
    # sparsity calculation --------------------------------------------------
    # generate nonzero masks
    cols_nonzero = (cols != 0)        # nonzero index for input
    W_nonzero = (W_reshaped != 0)     # idx of nonzero element for kernel 
    
    # count number of multiply operations
    effective_matrix = cols_nonzero.astype(np.int64) @ W_nonzero.astype(np.int64)
    total_effective = effective_matrix.sum()
    
    N_samples = cols.shape[0]         # number of samples（N*OH*OW）
    K_size = cols.shape[1]            # expanded dimension per sample（C*KH*KW）
    F_size = W_reshaped.shape[1]      # number of filters
    total_multiplies = N_samples * K_size * F_size
    
    # calculate effective ratio
    effective_ratio = total_effective / total_multiplies if total_multiplies != 0 else 0.0
    # ------------------------------------------------------------
    
    # rebuild output 
    N, _, H, W = X.shape
    OH = (H + 2*pad - KH) // stride + 1
    OW = (W + 2*pad - KW) // stride + 1
    out = out.reshape(N, OH, OW, F).transpose(0, 3, 1, 2)
    
    return out, effective_ratio

def ops_monitor(net, is_sop=False):
    m_dict = dict(net.named_modules())
    for key in m_dict.keys():
        if key == "":
            continue
        m = m_dict[key]
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_hook(ops_hook_fn(key + ".weight", is_sop))


# this function is especially prepared for lasr and ac attr
def ops_hook_fn(module_name, is_sop):
    def hook(m, inputs, outputs):
        inputs = inputs[0]
        max_v = torch.max(inputs)
        is_mac = max_v.dtype == torch.int64 or not torch.floor(max_v) == max_v
        ran = max_v.detach().cpu().numpy().astype(int)
        lsar = 0
        stride = m.stride[0]
        padding = m.padding[0]
        kn, kn = m.kernel_size
        hn, wn = inputs.shape[-2:]
        in_channels = m.in_channels
        out_channels = m.out_channels

        B, C, H, W = inputs.shape
        for i in range(1, ran + 1):
            lsar += len(torch.where(inputs == float(i))[0]) / inputs.numel() * i
        if lsar == 0:
            lsar = inputs.count_nonzero() / inputs.numel()
        if is_sop and isinstance(m,torch.nn.Conv2d):
            weight = m.weight.data
            weight = weight.detach().cpu().numpy()
            # inputs = inputs.reshape(B,C,H,W)
            inputs = inputs.reshape(-1, C, H, W)
            inputs = inputs.detach().cpu().numpy()
            _, lsar = conv_forward_with_sparsity(inputs,weight,0,stride,padding)
            pass
        if module_name not in MODULE_SOP_DICT.keys():
            MODULE_SOP_DICT[module_name] = lsar * kn * kn * hn * wn *  in_channels * out_channels
        else:
            MODULE_SOP_DICT[module_name] += lsar * kn * kn * hn * wn *  in_channels * out_channels
    return hook
