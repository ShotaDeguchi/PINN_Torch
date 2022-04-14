"""
********************************************************************************
activations
********************************************************************************
"""

import torch

def act_func(
    name
):
    if name == "tanh":
        activation = torch.nn.Tanh()
    elif name == "silu":
        activation = torch.nn.SiLU()
    elif name == "gelu":
        activation = torch.nn.GELU()
    elif name == "mish":
        activation = torch.nn.Mish()
    else:
        raise NotImplementedError(">>>>> act_func")
    return activation
