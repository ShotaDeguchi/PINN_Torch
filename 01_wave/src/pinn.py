"""
********************************************************************************
Author: Shota DEGUCHI
        Structural Analysis Lab. Kyushu Univ. (Feb. 15th, 2022)
Implementation of PINN in PyTorch
********************************************************************************
"""

import os
from turtle import forward
import numpy as np
import torch
import time
import datetime

class PINN(torch.nn.Module):
	def __init__(
		self, 
		t_ic0_train, x_ic0_train, u_ic0_train, 
		t_ic1_train, x_ic1_train, u_ic1_train, 
		t_bc0_train, x_bc0_train, u_bc0_train,    # Dirichlet boundary
		t_bc1_train, x_bc1_train, u_bc1_train,    # Neumann boundary
		t_pde_train, x_pde_train, 
		t_ic0_val, x_ic0_val, u_ic0_val, 
		t_ic1_val, x_ic1_val, u_ic1_val, 
		t_bc0_val, x_bc0_val, u_bc0_val, 
		t_bc1_val, x_bc1_val, u_bc1_val, 
		t_pde_val, x_pde_val, 
		f_in=2, f_out=1, f_hid=2**6, depth=6, 
		w_init="Glorot", b_init="zeros", act="tanh", 
		lr=5e-4, opt = "Adam", 
		f_scl="minmax", c=1.,
		w_ic=1., w_bc=1., w_pde=1., 
		d_type=torch.float32, r_seed=1234
	):
		# initialization
		super().__init__()
		self.f_in   = f_in
		self.f_out  = f_out
		self.f_hid  = f_hid
		self.depth  = depth
		self.w_init = w_init
		self.b_init = b_init
		self.act    = act
		self.lr     = lr
		self.opt    = opt
		self.f_scl  = f_scl
		self.c      = c
		self.w_ic   = w_ic
		self.w_bc   = w_bc
		self.w_pde  = w_pde
		self.d_type = d_type
		self.r_seed = r_seed

		# set random seed and float dtype
		os.environ["PYTHONHASHSEED"] = str(self.r_seed)
		np.random.seed(self.r_seed)
		torch.manual_seed(self.r_seed)
		torch.set_default_dtype(self.d_type)

		# training set
		self.t_ic0_train = t_ic0_train.astype(np.float32)
		self.x_ic0_train = x_ic0_train
		self.u_ic0_train = u_ic0_train
		self.t_ic1_train = t_ic1_train
		self.x_ic1_train = x_ic1_train
		self.u_ic1_train = u_ic1_train
		self.t_bc0_train = t_bc0_train
		self.x_bc0_train = x_bc0_train
		self.u_bc0_train = u_bc0_train
		self.t_bc1_train = t_bc1_train
		self.x_bc1_train = x_bc1_train
		self.u_bc1_train = u_bc1_train
		self.t_pde_train = t_pde_train
		self.x_pde_train = x_pde_train

		# validation set
		self.t_ic0_val = t_ic0_val
		self.x_ic0_val = x_ic0_val
		self.u_ic0_val = u_ic0_val
		self.t_ic1_val = t_ic1_val
		self.x_ic1_val = x_ic1_val
		self.u_ic1_val = u_ic1_val
		self.t_bc0_val = t_bc0_val
		self.x_bc0_val = x_bc0_val
		self.u_bc0_val = u_bc0_val
		self.t_bc1_val = t_bc1_val
		self.x_bc1_val = x_bc1_val
		self.u_bc1_val = u_bc1_val
		self.t_pde_val = t_pde_val
		self.x_pde_val = x_pde_val




	def forward(
		self, x
	):
		if torch.is_tensor(x) == True:
			pass
		else:
			x = torch.from_numpy(x)



		z = x
		y = z
		return y



	def train(
		self, 
	):




	def infer(
		self, 
	):


