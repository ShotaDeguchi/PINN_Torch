"""
********************************************************************************
main file to execute
********************************************************************************
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from pinn import PINN
from preparations import prp_dat
from parameters import params
from icbc import initial, boundary

def main():
    f_in, f_out, width, depth, \
    w_init, b_init, act, \
    lr, opt, \
    f_scl, nu, \
    w_ini, w_bnd, w_pde, bc, \
    f_mntr, r_seed, d_type, device, \
    n_epch, n_btch, c_tol = params()

    XT_ini_trn, XT_ini_val, XT_ini_tst, \
    XT_bnd_trn, XT_bnd_val, XT_bnd_tst, \
    XT_pde_trn, XT_pde_val, XT_pde_tst = prp_dat()

    # training set
    u_ini_trn = initial_condition(XT_ini_trn[:,0])
    u_bnd_trn = boundary_condition(XT_bnd_trn[:,1])

    # validation set
    u_ini_val = initial(XT_ini_val[:,0])
    u_bnd_val = boundary(XT_bnd_val[:,1])
    
    pinn = PINN(
        XT_ini_trn[:,0], XT_ini_trn[:,1], u_ini_trn, 
        XT_bnd_trn[:,0], XT_bnd_trn[:,1], u_bnd_trn, 
        XT_pde_trn[:,0], XT_pde_trn[:,1], 
        XT_ini_val[:,0], XT_ini_val[:,1], u_ini_val, 
        XT_bnd_val[:,0], XT_bnd_val[:,1], u_bnd_val, 
        XT_pde_val[:,0], XT_pde_val[:,1], 
        f_in, f_out, width, depth,
        w_init, b_init, act, 
        lr, opt, 
        f_scl, nu, 
        w_ini, w_bnd, w_pde, bc, 
        f_mntr, r_seed, d_type, device
    )
    pinn.to(device)
    pinn.train(epoch = n_epch, batch = n_btch, tol = c_tol)

    plt.figure(figsize = (8, 4))
    plt.plot(pinn.loss_trn_log, alpha = .7, label = "loss_trn")
    plt.plot(pinn.loss_val_log, alpha = .7, label = "loss_val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.legend(loc = "upper right")
    plt.grid(alpha = .5)
    plt.show()

    nx, nt = 101, 101
    x, t = np.linspace(-1, 1, nx), np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    X, T = X.reshape(-1, 1), T.reshape(-1, 1)
    XT = np.hstack((X, T)).astype(np.float32)
    u_ = pinn.infer(XT)
    u_ = np.reshape(u_, (nx, nt))

    x_eval = np.ravel(X.T).reshape(-1, 1)
    t_eval = np.ravel(T.T).reshape(-1, 1)
    xmax = np.max(x_eval)
    tmax = np.max(t_eval)

    plt.figure(figsize = (8, 4))
    plt.scatter(t_eval, x_eval, c = u_, vmin = -1, vmax = 1, cmap="seismic")
    plt.colorbar()
    plt.xlim( 0, 1)
    plt.ylim(-1, 1)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("PINN solution")
    plt.show()

if __name__ == "__main__":
    main()
