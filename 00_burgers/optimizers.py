"""
********************************************************************************
optimizers
********************************************************************************
"""

import torch

def opt_alg(params, lr, name):
    if name == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False
            )
    elif name == "RMSprop":
        optimizer = torch.optim.RMSprop(
            params, lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False
            )
    elif name == "Adam":
        optimizer = torch.optim.Adam(
            params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
            )
    elif name == "Adamax":
        optimizer = torch.optim.Adamax(
            params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
            )
    elif name == "Nadam":
        optimizer = torch.optim.NAdam(
            params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004
            )
    elif name == "Adamw":
        optimizer = torch.optim.AdamW(
            params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False
            )
    elif name == "Radam":
        optimizer = torch.optim.RAdam(
            params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
            )
    else:
        raise NotImplementedError(">>>>> opt_alg")
    return optimizer
