from D3LIF import D3LIF, D3LIF_Quant
import torch,math,einops
from enum import Enum
from torch import nn
import numpy as np
import copy
from spikingjelly.activation_based.neuron import LIFNode

class Aggregated_Spiking_Layer(nn.Module):
    def __init__(self, conv_layer, bn_layer, lif_layer, layer_id):
        super().__init__()
        self._layer = conv_layer
        self.bn = bn_layer
        self._neuron_model = lif_layer
        self.id = layer_id  # 记录唯一的ID

    def forward(self, x):
        x = self._layer(x)
        x = self.bn(x)
        x = self._neuron_model(x)
        return x

def get_results(q,weight,lower_bound,upper_bound):
    top9=torch.abs(weight).flatten(1).quantile(dim=1,q=q)
    scale_factor= upper_bound / top9
    new_weight= torch.clamp(torch.round(scale_factor[:,None,None,None] * weight),lower_bound,upper_bound)
    reconstruct = new_weight / scale_factor[:,None,None,None]
    score= torch.nn.functional.mse_loss(weight,reconstruct)
    return score, (scale_factor,new_weight)

def search_mean(search_range:list,step,eval:callable,**kwds):
    min_score=None
    min_results=None
    for _i in np.arange(search_range[0],search_range[1],step):
        _score,_results=eval(_i,**kwds)
        if min_score is None or _score< min_score:
            min_score=_score
            min_results=_results
    return min_results

class Symmetric_Quantize:
    class Target_Precision(Enum):
        INT8=127
        INT9=255
        INT10=511
        INT11=1023
        
    def __init__(self,precision) -> None:
        self.upper_bound=precision.value
        self.lower_bound=-precision.value
        self._scale_factor = None
        self._bias = None  
    
    def __call__(self, module):
        if not isinstance(module, Aggregated_Spiking_Layer):
            print(module)
            print('Not Aggregated Spiking Layer')
            return
        if module._layer is not None and isinstance(module._layer,torch.nn.Conv2d):
            _weight = module._layer.weight.data

            _scale_factor, _new_weight = search_mean([0.97, 1.0], 0.001, eval=get_results, weight=_weight,
                                                     lower_bound=self.lower_bound, upper_bound=self.upper_bound)
            print('search one layer')

            module._layer.weight.data = _new_weight

            # assert isinstance(module._neuron_model, D3LIF)
            _bias = module._layer.bias.data
            self._bias = _bias
            module._layer.bias.data = torch.zeros_like(_bias)
            self._scale_factor = _scale_factor
        
        if module._neuron_model is not None:
            _new_bias = torch.round(self._bias * self._scale_factor)
            _new_vth = torch.round(module._neuron_model.v * self._scale_factor)
            _new_neuron = D3LIF_Quant(
                id=module._neuron_model.id,
                vth_base=_new_vth[None, :, None, None],
                tau=module._neuron_model.tau,
                bias=_new_bias.clone(),
                scale_factor=self._scale_factor,
            )
            module._neuron_model = _new_neuron