import torch, math

class firing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, psp,tau,vth):
        ctx.save_for_backward(psp)
        ctx.vth=vth
        ctx.tau=tau
        output,U=_forward(psp,tau,vth)
        return output,U

    @staticmethod
    def backward(ctx, dy,dU):
        psp,= ctx.saved_tensors
        return _back(psp, ctx.tau,dy,dU,ctx.vth),None,None

@torch.jit.script
def _forward(psp,tau:float,vth:float):
    output = torch.gt(psp, vth).type_as(psp)
    U = tau * (1-output) * psp
    # U = psp - output * psp
    return output,U

@torch.jit.script
def _back(psp,tau:float,dy,dU,vth:float):

    _o = torch.gt(psp,vth).type_as(dy)
    foo = dU * (1-_o)
    dp= foo * tau
    # dp = foo

    # 近似梯度 arctan
    alpha=2
    frac=(3.1415926 /2 * alpha)**2
    hu = alpha / (2 * (1 + (psp-vth)**2 * frac))

    return dy*hu + dp

class D3LIF(torch.nn.Module):
    duplicate=1
    g=firing.apply
    id=0
    def __init__(self,vth_base=1.,tau=0.5,**kwargs):
        super().__init__()
        
        # serial number
        self.id=D3LIF.id
        D3LIF.id+=1

        self.vth=vth_base
        self.tau=tau

        self.reset()
        self.state_hooks=[]

    def reset(self):
        self.U=None
    
    def forward(self,input):

        rslt=self._update(input)

        if(len(self.state_hooks) != 0):
            for hooks in self.state_hooks:
                hooks(self.id,input,rslt)
        return rslt

    def _update(self,input):
        if(self.U is None):
            PSP=input
        else:
            # t=torch.zeros(*input.shape)+self.U
            PSP=self.U + input
        o,self.U=D3LIF.g(PSP,self.tau,self.vth)
        self.U=torch.relu(self.U + self.vth) - self.vth
        return o

    def extra_repr(self):
        s = (f'id={self.id},vth={self.vth},tau={self.tau}')
        return s

class D3LIF_for_delay(D3LIF):
    def forward(self,input):
        if self._neuromorphic_states['current_step']< self.id:
            return torch.zeros_like(input)
        else:
            return super().forward(input)

class D3LIF_Quant(D3LIF):
    def __init__(self, id=0, vth_base=1, tau=1, bias=0, **kwargs):
        super().__init__(vth_base, tau, **kwargs)
        self.id=id
        self.bias=bias #torch.nn.Parameter(bias,requires_grad=False)
        self.vth=vth_base # torch.nn.Parameter(vth_base,requires_grad=False)

        self._neuromorphic_states = {'delay': None, 'current_step': None}

    def forward(self,input):
        # return self.forward_impl(input)
        if self._neuromorphic_states['delay']:
            if self._neuromorphic_states['current_step'] < self.id:
                return torch.zeros_like(input)
            else:
                return self.forward_impl(input)
        else:
            return self.forward_impl(input)
        
    def forward_impl(self, input:torch.Tensor):
        with torch.no_grad():
            _orignal_type=input.dtype
            input = input.short() + self.bias.short()
            if(self.U is None):
                PSP=input
            else:
                PSP=self.U + input
            o=torch.gt(PSP,self.vth).type(_orignal_type)
            self.U = torch.floor((1-o) * PSP.float() * self.tau).short()
            self.U = torch.clamp_min(self.U, - self.vth)
            return o