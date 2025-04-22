import torch
from spikingjelly.activation_based import layer, neuron, surrogate


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

class PConv(layer.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', step_mode='s', *args, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, step_mode)

        self.sparse_function = kwargs['sparse_function'] if kwargs['sparse_function'] is not None else 'st'
        self.gradual = kwargs['gradual'] if kwargs['gradual'] is not None else None
        self.flat_width = kwargs['flat_width'] if kwargs['flat_width'] is not None else 1.0

        with torch.no_grad():
            if self.sparse_function == 'st' and self.gradual is None:
                self.mapping = lambda x: softThreshold(x, self.flat_width)
            else:
                self.mapping = lambda x: x

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
        if self.sparse_function == 'st':
            self.mapping = lambda x: softThreshold(x, width)
        else:
            pass

class PLinear(layer.Linear):
    def __init__(self, in_features, out_features, bias=True, step_mode='s', *args, **kwargs):
        super().__init__(in_features, out_features, bias, step_mode)
    
        self.sparse_function = kwargs['sparse_function'] if kwargs['sparse_function'] is not None else 'st'
        self.gradual = kwargs['gradual'] if kwargs['gradual'] is not None else None
        self.flat_width = kwargs['flat_width'] if kwargs['flat_width'] is not None else 1.0

        with torch.no_grad():
            if self.sparse_function == 'st' and self.gradual is None:
                self.mapping = lambda x: softThreshold(x, self.flat_width)
            else:
                self.mapping = lambda x: x

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
        if self.sparse_function == 'st':
            self.mapping = lambda x: softThreshold(x, width)
        else:
            pass