"""
********************************************************************************
prepare data (initial, boundary, PDE residual points)
********************************************************************************
"""

import torch
import numpy as np

def prp_dat():
    # set domain
    xmin, xmax, nx = -1., 1., int(1e2) + 1
    tmin, tmax, nt =  0., 1., int(2e2) + 1
    x = np.linspace(xmin, xmax, nx)
    t = np.linspace(tmin, tmax, nt)
    x, t = np.meshgrid(x, t)
    x, t = x.reshape(-1, 1), t.reshape(-1, 1)
    XT = np.c_[x, t]

    # find lower & upper bound
    lb = [min(XT[:,0]), min(XT[:,1])]
    ub = [max(XT[:,0]), max(XT[:,1])]

    # fraction for training, validation, test set
    frac_trn = .6
    frac_val = .2
    frac_tst = .2

    # initial condition evaluation
    N_ini = nx
    x_ini = np.random.uniform(low=lb[0], high=ub[0], size=N_ini)
    t_ini = lb[1] * np.ones((N_ini))
    XT_ini = np.c_[x_ini, t_ini]
    
    N_ini = XT_ini.shape[0]
    N_ini_trn = int(frac_trn * N_ini)
    N_ini_val = int(frac_val * N_ini)
    N_ini_tst = int(frac_tst * N_ini)

    idx_ini = np.random.choice(N_ini, N_ini, replace=False)
    idx_ini_trn = idx_ini[0:N_ini_trn]
    idx_ini_val = idx_ini[N_ini_trn:N_ini_val]
    idx_ini_tst = idx_ini[N_ini_val:N_ini_tst]

    XT_ini_trn = XT_ini[idx_ini_trn,:]
    XT_ini_val = XT_ini[idx_ini_val,:]
    XT_ini_tst = XT_ini[idx_ini_tst,:]

    # boundary condition evaluation
    N_bnd = nt
    x_lb = lb[0] * np.ones((int(.5 * N_bnd)))
    x_ub = ub[0] * np.ones((int(.5 * N_bnd)))
    x_bnd = np.append(x_lb, x_ub)
    t_bnd = np.random.uniform(low=lb[1], high=ub[1], size=N_bnd-1)
    XT_bnd = np.c_[x_bnd, t_bnd]

    N_bnd = XT_bnd.shape[0]
    N_bnd_trn = int(frac_trn * N_bnd)
    N_bnd_val = int(frac_val * N_bnd)
    N_bnd_tst = int(frac_tst * N_bnd)

    idx_bnd = np.random.choice(N_bnd, N_bnd, replace=False)
    idx_bnd_trn = idx_bnd[0:N_bnd_trn]
    idx_bnd_val = idx_bnd[N_bnd_trn:N_bnd_val]
    idx_bnd_tst = idx_bnd[N_bnd_val:N_bnd_tst]

    XT_bnd_trn = XT_bnd[idx_bnd_trn,:]
    XT_bnd_val = XT_bnd[idx_bnd_val,:]
    XT_bnd_tst = XT_bnd[idx_bnd_tst,:]

    # PDE-residual evaluation
    N_pde = XT.shape[0]
    N_pde_trn = int(frac_trn * N_pde)
    N_pde_val = int(frac_val * N_pde)
    N_pde_tst = int(frac_tst * N_pde)

    idx_pde = np.random.choice(N_pde, N_pde, replace=False)
    idx_pde_trn = idx_pde[0:N_pde_trn]
    idx_pde_val = idx_pde[N_pde_trn:N_pde_val]
    idx_pde_tst = idx_pde[N_pde_val:N_pde_tst]

    XT_pde_trn = XT[idx_pde_trn,:]
    XT_pde_val = XT[idx_pde_val,:]
    XT_pde_tst = XT[idx_pde_tst,:]

    XT_ini_trn, XT_ini_val, XT_ini_tst = torch.from_numpy(XT_ini_trn), \
                                         torch.from_numpy(XT_ini_val), \
                                         torch.from_numpy(XT_ini_tst)
    XT_bnd_trn, XT_bnd_val, XT_bnd_tst = torch.from_numpy(XT_bnd_trn), \
                                         torch.from_numpy(XT_bnd_val), \
                                         torch.from_numpy(XT_bnd_tst)
    XT_pde_trn, XT_pde_val, XT_pde_tst = torch.from_numpy(XT_pde_trn), \
                                         torch.from_numpy(XT_pde_val), \
                                         torch.from_numpy(XT_pde_tst)

    return XT_ini_trn, XT_ini_val, XT_ini_tst, \
            XT_bnd_trn, XT_bnd_val, XT_bnd_tst, \
            XT_pde_trn, XT_pde_val, XT_pde_tst
