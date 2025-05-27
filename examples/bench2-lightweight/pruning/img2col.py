import torch
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def img2col(X, kernel_size, stride=1, pad=0):
    """
    将4D输入张量转换为2D列矩阵
    参数:
        X: 输入张量，形状为(N, C, H, W)
        kernel_size: 卷积核大小（整数或元组）
        stride: 步长（整数或元组）
        pad: 填充大小
    返回:
        2D列矩阵，形状为(N*out_h*out_w, C*kh*kw)
    """
    # 解析参数
    N, C, H, W = X.shape
    kh = kw = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    sh = sw = stride if isinstance(stride, int) else stride[0]
    
    # 执行填充
    X_pad = np.pad(X, [(0,0), (0,0), (pad, pad), (pad, pad)], mode='constant')
    
    # 计算输出尺寸
    H_out = (H + 2*pad - kh) // sh + 1
    W_out = (W + 2*pad - kw) // sw + 1
    
    # 生成滑动窗口视图
    windows = sliding_window_view(X_pad, (kh, kw), axis=(2, 3))
    
    # 步长切片
    windows = windows[:, :, ::sh, ::sw]
    
    # 重塑为列矩阵
    cols = windows.reshape(N, C, H_out, W_out, kh*kw)
    cols = cols.transpose(0, 2, 3, 1, 4).reshape(N*H_out*W_out, -1)
    
    return cols

def conv_forward_with_sparsity(X, W, b, stride=1, pad=0):
    """
    使用img2col实现卷积前向传播，并计算有效计算比例
    参数:
        X: 输入数据 (N, C, H, W)
        W: 卷积核 (F, C, KH, KW)
        b: 偏置 (F,)
        stride: 步长
        pad: 填充
    返回:
        output: 卷积结果 (N, F, OH, OW)
        effective_ratio: 有效计算比例 (0.0-1.0)
    """
    # 通过img2col转换输入
    cols = img2col(X, W.shape[2:], stride, pad)
    
    # 转换卷积核为2D矩阵（包含旋转180度）
    F, C, KH, KW = W.shape
    W_rot = np.rot90(W, 2, axes=(2, 3))  # 在H和W维度旋转180度
    W_reshaped = W_rot.reshape(F, -1).T  # 形状 (C*KH*KW, F)
    
    # 执行矩阵乘法
    out = cols @ W_reshaped + b
    
    # 稀疏度检测 --------------------------------------------------
    # 生成非零掩码
    cols_nonzero = (cols != 0)        # 输入数据的非零位置
    W_nonzero = (W_reshaped != 0)     # 卷积核的非零位置
    
    # 计算有效乘法次数（利用矩阵乘法统计符合条件的位置）
    effective_matrix = cols_nonzero.astype(np.int64) @ W_nonzero.astype(np.int64)
    total_effective = effective_matrix.sum()
    
    # 计算总乘法次数
    N_samples = cols.shape[0]         # 样本数量（N*OH*OW）
    K_size = cols.shape[1]            # 每个样本的展开维度（C*KH*KW）
    F_size = W_reshaped.shape[1]      # 滤波器数量
    total_multiplies = N_samples * K_size * F_size
    
    # 计算有效比例
    effective_ratio = total_effective / total_multiplies if total_multiplies != 0 else 0.0
    # ------------------------------------------------------------
    
    # 重塑输出结果
    N, _, H, W = X.shape
    OH = (H + 2*pad - KH) // stride + 1
    OW = (W + 2*pad - KW) // stride + 1
    out = out.reshape(N, OH, OW, F).transpose(0, 3, 1, 2)
    
    return out, effective_ratio