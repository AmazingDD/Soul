import torch
import torch.nn as nn

class SurrogateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_output * tmp

        return grad_input, None

class LMH(nn.Module):
    def __init__(self, scale=1.):
        super(LMH, self).__init__()
        self.v_threshold = nn.Parameter(torch.tensor([scale]), requires_grad=False)
        self.t = 0
        self.v_d = None
        self.v_s = None

        self.alpha_1 = nn.Parameter(torch.tensor([0.]), requires_grad=True)  # True
        self.beta_1 = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        self.alpha_2 = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        self.beta_2 = nn.Parameter(torch.tensor([0.]), requires_grad=True)

        self.act = SurrogateFunction.apply
        # self.act = ZO.apply
        self.gama = 1.

    def forward(self, x):
        # reset
        self.v_d = torch.ones_like(x[0]) * 0. * (self.v_threshold) # (B, C, H, W)
        self.v_s = torch.ones_like(x[0]) * 0.5 * (self.v_threshold) # (B, C, H, W)

        spike_pot = []
        for t in range(x.shape[0]): # T
            self.v_d = (self.alpha_1.sigmoid() - 0.5) * self.v_d + (self.beta_1.sigmoid() - 0.5) * self.v_s + x[t]
            self.v_s = (self.alpha_2.sigmoid() + 0.5) * self.v_s + (self.beta_2.sigmoid() + 0.5) * self.v_d

            output = self.act(self.v_s - (self.v_threshold), self.gama) * (self.v_threshold)
            self.v_s -= output.detach()
            spike_pot.append(output)

        x = torch.stack(spike_pot, dim=0)

        return x
