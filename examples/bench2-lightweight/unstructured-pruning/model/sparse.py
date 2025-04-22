import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate

sparse_function = True

def create_lif():
    return neuron.LIFNode(
        v_threshold=1., v_reset=0., surrogate_function=surrogate.ATan(), detach_reset=True)

class PseudoRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.requires_grad:
            ctx.save_for_backward(x)
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output
        return grad_x, None

pseudoRelu = PseudoRelu.apply

def softThreshold(x, s):
    return torch.sign(x) * pseudoRelu(torch.abs(x) - s)

def softThresholdinv(x, s):
    return torch.sign(x) * (torch.abs(x) + s)

class PConv(nn.Conv2d):
    def __init__(self, thr, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sparse_function = sparse_function
        self.flat_width = thr

        with torch.no_grad():
            if self.sparse_function:
                self.mapping = lambda x: softThreshold(x, self.flat_width)
            else:
                self.mapping = lambda x: x
    
    def forward(self, x):
        sparseWeight = self.mapping(self.weight)
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x

    @torch.no_grad()
    def getSparsity(self):
        sparseWeight = self.mapping(self.weight)
        temp = sparseWeight.detach().cpu()
        return (temp == 0).sum(), temp.numel()

    @torch.no_grad()
    def getSparseWeight(self):
        return self.mapping(self.weight)
        
    @torch.no_grad()
    def setFlatWidth(self, width):
        if self.sparse_function:
            self.mapping = lambda x: softThreshold(x, width)
        else:
            pass

class PLinear(nn.Linear):
    def __init__(self, thr, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.sparse_function = sparse_function
        self.flat_width = thr

        with torch.no_grad():
            if self.sparse_function:
                self.mapping = lambda x: softThreshold(x, self.flat_width)
            else:
                self.mapping = lambda x: x

    def forward(self, x):
        sparseWeight = self.mapping(self.weight)
        x = F.linear(x, sparseWeight)

        return x

    @torch.no_grad()
    def getSparsity(self):
        sparseWeight = self.mapping(self.weight)
        temp = sparseWeight.detach().cpu()
        return (temp == 0).sum(), temp.numel()

    @torch.no_grad()
    def getSparseWeight(self):
        return self.mapping(self.weight)
        
    @torch.no_grad()
    def setFlatWidth(self, width):
        if self.sparse_function:
            self.mapping = lambda x: softThreshold(x, width)
        else:
            pass