"""
********************************************************************************
Author: Shota DEGUCHI
        Structural Analysis Lab. Kyushu Univ. (Feb. 15th, 2022)
implementation of PINN - Physics-Informed Neural Network on PyTorch
********************************************************************************
"""

import os
import numpy as np
import random
import torch
import time
import datetime

# from initializers import weight_init, bias_init
from optimizers import opt_alg
from activations import act_func

class PINN(torch.nn.Module):
    def __init__(
        self, 
        x_ini_trn, t_ini_trn, u_ini_trn, 
        x_bnd_trn, t_bnd_trn, u_bnd_trn, 
        x_pde_trn, t_pde_trn, 
        x_ini_val, t_ini_val, u_ini_val, 
        x_bnd_val, t_bnd_val, u_bnd_val, 
        x_pde_val, t_pde_val, 
        f_in, f_out, width, depth,
        w_init, b_init, act, 
        lr, opt, 
        f_scl, nu, 
        w_ini, w_bnd, w_pde, bc, 
        f_mntr, r_seed, d_type, device
    ):
        # init
        super().__init__()
        self.f_in   = f_in
        self.f_out  = f_out
        self.width  = width
        self.depth  = depth
        self.w_init = w_init
        self.b_init = b_init
        self.act    = act
        self.lr     = lr
        self.opt    = opt
        self.f_scl  = f_scl
        self.f_mntr = f_mntr
        self.r_seed = r_seed
        self.d_type = d_type
        self.device = device
        self.setup(r_seed, d_type)

        # training set
        self.x_ini_trn = x_ini_trn.float().reshape(-1, 1).to(self.device)
        self.t_ini_trn = t_ini_trn.float().reshape(-1, 1).to(self.device)
        self.u_ini_trn = u_ini_trn.float().reshape(-1, 1).to(self.device)
        self.x_bnd_trn = x_bnd_trn.float().reshape(-1, 1).to(self.device)
        self.t_bnd_trn = t_bnd_trn.float().reshape(-1, 1).to(self.device)
        self.u_bnd_trn = u_bnd_trn.float().reshape(-1, 1).to(self.device)
        self.x_pde_trn = x_pde_trn.float().reshape(-1, 1).to(self.device)
        self.t_pde_trn = t_pde_trn.float().reshape(-1, 1).to(self.device)

        # validation set
        self.x_ini_val = x_ini_val.float().reshape(-1, 1).to(self.device)
        self.t_ini_val = t_ini_val.float().reshape(-1, 1).to(self.device)
        self.u_ini_val = u_ini_val.float().reshape(-1, 1).to(self.device)
        self.x_bnd_val = x_bnd_val.float().reshape(-1, 1).to(self.device)
        self.t_bnd_val = t_bnd_val.float().reshape(-1, 1).to(self.device)
        self.u_bnd_val = u_bnd_val.float().reshape(-1, 1).to(self.device)
        self.x_pde_val = x_pde_val.float().reshape(-1, 1).to(self.device)
        self.t_pde_val = t_pde_val.float().reshape(-1, 1).to(self.device)

        # weights
        self.w_ini = w_ini
        self.w_bnd = w_bnd
        self.w_pde = w_pde

        # setup
        self.nu = nu
        self.bc = bc

        # bounds
        self.bounds = torch.cat((self.x_pde_trn, self.t_pde_trn), dim = 1)
        self.lb = torch.min(self.bounds, dim = 0).values.to(self.device)
        self.ub = torch.max(self.bounds, dim = 0).values.to(self.device)
        self.mn = torch.mean(self.bounds, dim = 0).values

        self.arch = [self.f_in] + [self.width] * (self.depth - 1) + [self.f_out]
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(self.arch[d], self.arch[d + 1]) for d in range(self.depth)]
            )
        for d in range(0, self.depth):
            torch.nn.init.xavier_normal_(self.layers[d].weight.data, gain=1.)
            torch.nn.init.zeros_(self.layers[d].bias.data)

        # self.model = self.dnn_init(self.arch)
        self.activation = act_func(self.act)
        self.optimzer = opt_alg(self.layers.parameters(), self.lr, self.opt)
        self.loss_func = torch.nn.MSELoss(reduction = "mean")
        self.loss_trn_log = []
        self.loss_val_log = []

        print("\n************************************************************")
        print("****************     MAIN PROGRAM START     ****************")
        print("************************************************************")
        print(">>>>> start time:", datetime.datetime.now())
        print(">>>>> configuration;")
        print("         r_seed       :", self.r_seed)
        print("         d_type       :", self.d_type)
        print("         weight init  :", self.w_init)
        print("         bias   init  :", self.b_init)
        print("         activation   :", self.act)
        print("         learning rate:", self.lr)
        print("         optimizer    :", self.optimzer)
        print("         layers       :", self.layers)
        
    def setup(
        self, r_seed, d_type
    ):
        os.environ["PYTHONHASHSEED"] = str(r_seed)
        np.random.seed(r_seed)
        random.seed(r_seed)
        torch.manual_seed(r_seed)
        torch.set_default_dtype(d_type)

    def forward(
        self, x
    ):
        # convert to tensor
        if torch.is_tensor(x) == True:
            pass
        else:
            x = torch.from_numpy(x)

        # feature scaling
        x = x.to(self.device)
        if self.f_scl == "minmax":
            z = 2. * (x - self.lb) / (self.ub - self.lb) - 1.
        elif self.f_scl == "mean":
            z = (x - self.mn) / (self.ub - self.lb)
        else:
            raise NotImplementedError(">>>>> forward_pass (f_scl)")

        # forward pass
        for d in range(0, self.depth - 1):
            u = self.layers[d](z)
            z = self.activation(u)
        y = self.layers[-1](z)
        return y

    def loss_ini(
        self, x, y
    ):
        x_ = x.clone()
        x_.requires_grad = False
        y_ = self.forward(x_)
        loss = self.loss_func(y, y_)
        return loss

    def loss_bnd(
        self, x, y
    ):
        if self.bc == "Dir":
            x_ = x.clone()
            x_.requires_grad = True
            y_ = self.forward(x_)
            loss = self.loss_func(y, y_)
        elif self.bc == "Neu":
            x_ = x.clone()
            x_.requires_grad = True
            u = self.forward(x_)
            u_x_t = torch.autograd.grad(u, x_, torch.ones([x.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
            u_x   = u_x_t[:,[0]]
            loss = self.loss_func(y, u_x)
        else:
            raise NotImplementedError(">>>>> loss_bnd")
        return loss

    def loss_pde(
        self, x
    ):
        x_ = x.clone()
        x_.requires_grad = True
        u = self.forward(x_)
        u_x_t   = torch.autograd.grad(u,     x_, torch.ones([x.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        u_xx_tt = torch.autograd.grad(u_x_t, x_, torch.ones(x.shape).to(self.device), create_graph=True)[0]
        u_x  = u_x_t[:,[0]]
        u_t  = u_x_t[:,[1]]
        u_xx = u_xx_tt[:,[0]]
        gveq = u_t + u * u_x - self.nu * u_xx
        _ = torch.zeros_like(gveq)
        loss = self.loss_func(_, gveq)
        return loss

    def loss_glb(
        self, 
        x_ini, t_ini, u_ini, 
        x_bnd, t_bnd, u_bnd, 
        x_pde, t_pde
    ):
        XT_ini = torch.cat((x_ini, t_ini), dim = 1)
        XT_bnd = torch.cat((x_bnd, t_bnd), dim = 1)
        XT_pde = torch.cat((x_pde, t_pde), dim = 1)
        loss_ini = self.loss_ini(XT_ini, u_ini)
        loss_bnd = self.loss_bnd(XT_bnd, u_bnd)
        loss_pde = self.loss_pde(XT_pde)
        loss = self.w_ini * loss_ini \
                + self.w_bnd * loss_bnd \
                + self.w_pde * loss_pde
        return loss

    def train(
        self, epoch = int(1e3), batch = 2 ** 8, tol = 1e-5
    ):
        print("\n************************************************************")
        print("******************     TRAINING START     ******************")
        print("************************************************************")
        t0 = time.time()
        for ep in range(epoch):
            loss_trn = self.loss_glb(
                self.x_ini_trn, self.t_ini_trn, self.u_ini_trn, 
                self.x_bnd_trn, self.t_bnd_trn, self.u_bnd_trn, 
                self.x_pde_trn, self.t_pde_trn
            )
            loss_val = self.loss_glb(
                self.x_ini_val, self.t_ini_val, self.u_ini_val, 
                self.x_bnd_val, self.t_bnd_val, self.u_bnd_val, 
                self.x_pde_val, self.t_pde_val
            )
            self.optimzer.zero_grad()
            loss_trn.backward()
            self.optimzer.step()
            ep_loss_trn = loss_trn.item()
            ep_loss_val = loss_val.item()
            self.loss_trn_log.append(ep_loss_trn)
            self.loss_val_log.append(ep_loss_val)
            if ep % self.f_mntr == 0:
                elps = time.time() - t0
                print("epoch: %d, loss_trn: %.3e, loss_val: %.3e, elps: %.3f"
                        % (ep, ep_loss_trn, ep_loss_val, elps))
                t0 = time.time()
            if ep_loss_val < tol:
                print(">>>>> program terminating with the loss converging to the tolerance.")
                print("\n************************************************************")
                print("*******************     TRAINING END     *******************")
                print("************************************************************")
                break
        print("\n************************************************************")
        print("*******************     TRAINING END     *******************")
        print("************************************************************")

    def infer(
        self, x
    ):
        # x = x.to(self.device)
        u_ = self.forward(x)
        u_ = u_.cpu().detach().numpy()
        return u_
