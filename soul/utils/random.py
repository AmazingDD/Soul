import torch
import random
import numpy as np

def init_seed(seed, reproducibility=True):
    '''
    init random seed for random functions in numpy, torch, cuda and cudnn

    Parameters
    ----------
    seed : int
        random seed
    reproducibility : bool
        Whether to require reproducibility
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False