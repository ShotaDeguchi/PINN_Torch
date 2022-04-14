"""
********************************************************************************
initial & boundary conditions
********************************************************************************
"""

import numpy as np
import torch

def initial(x):
    if torch.is_tensor(x) == True:
        pass
    else:
        x = torch.from_numpy(x)
    y = -torch.sin(np.pi * x)
    return y

def boundary(t):
    if torch.is_tensor(t) == True:
        pass
    else:
        t = torch.from_numpy(t)
    y = torch.zeros(size = [t.shape[0]])
    return y

