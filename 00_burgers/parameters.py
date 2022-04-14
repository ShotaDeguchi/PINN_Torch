"""
********************************************************************************
params
********************************************************************************
"""

import sys
import os
import numpy as np
import random
import torch

# network structure
f_in  = 2
f_out = 1
width = 2 ** 8   # 2 ** 6 = 64, 2 ** 8 = 256
depth = 5

# training setting
n_epch = int(1e5)
n_btch = 2 ** 12   # 2 ** 8 = 256, 2 ** 10 = 1024
c_tol = 1e-8

# initializers
w_init = "Glorot"
b_init = "zeros"
act = "tanh"

# optimization
lr = 5e-4
opt = "Adam"
f_scl = "minmax"   # "minmax" or "mean"

# system params
nu = .01 / np.pi

# weights
w_ini = 1.
w_bnd = 1.
w_pde = 1.

# boundary condition 
bc = "Dir"   # "Dir" for Dirichlet, "Neu" for Neumann

# rarely changed params
f_mntr = 10
r_seed = 0
d_type = torch.float32

def params():
    print(f"python : {sys.version}")
    print(f"pytorch: {torch.__version__}")
    print(f"r_seed : {r_seed}")
    print(f"d_type : {d_type}")
    os.environ["PYTHONHASHSEED"] = str(r_seed)
    np.random.seed(r_seed)
    random.seed(r_seed)
    torch.manual_seed(r_seed)
    torch.set_default_dtype(d_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    return \
        f_in, f_out, width, depth, \
        w_init, b_init, act, \
        lr, opt, \
        f_scl, nu, \
        w_ini, w_bnd, w_pde, bc, \
        f_mntr, r_seed, d_type, device, \
        n_epch, n_btch, c_tol

