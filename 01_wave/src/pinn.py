"""
********************************************************************************
Author: Shota DEGUCHI
        Structural Analysis Lab. Kyushu Univ. (Feb. 15th, 2022)
Implementation of PINN in PyTorch
********************************************************************************
"""

import os
import numpy as np
import torch
import time

class PINN(torch.nn.Module):
	def __init__(
		self, 
		t_ic0_train, x_ic0_train, u_ic0_train, 
		t_ic1_train, x_ic1_train, u_ic1_train, 
		t_bc0_train, x_bc0_train, u_bc0_train, 
		t_bc1_train, x_bc1_train, u_bc1_train, 
		t_pde_train, x_pde_train, 
		t_ic0_val, x_ic0_val, u_ic0_val, 
		t_ic1_val, x_ic1_val, u_ic1_val, 
		t_bc0_val, x_bc0_val, u_bc0_val, 
		t_bc1_val, x_bc1_val, u_bc1_val, 
		t_pde_val, x_pde_val, 
		f_in=2, f_out=1, f_hid=2**5, depth=4, 
		w_init="Glorot", b_init="zeros", act="tanh", 
		lr=5e-4, opt = "Adam", 
		f_scl="minmax", c=1.,
		w_ic=1., w_bc=1., w_pde=1., 
		d_type=torch.float32, r_seed=1234
	):
		# initialization
		super().__init__()
		self.f_in   = f_in     # input feature
		self.f_out  = f_out    # output feature
		self.f_hid  = f_hid    # hidden feature
		self.depth  = depth    # depth
		self.w_init = w_init   # weight initializer
		self.b_init = b_init   # bias initializer
		self.act    = act      # element-wise activation
		self.lr     = lr       # learning rate
		self.opt    = opt      # optimizer
		self.f_scl  = f_scl    # feature scaling
		self.c      = c        # system parameter
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
		torch.cuda.manual_seed(self.d_type)

		# training set
		self.t_ic0_train = t_ic0_train.astype(np.float32)
		self.x_ic0_train = x_ic0_train
		self.u_ic0_train = u_ic0_train
		self.t_ic1_train = t_ic1_train
		self.x_ic1_train = x_ic1_train
		self.u_ic1_train = u_ic1_train
		self.t_bc0_train = t_bc0_train   # Dirichlet boundary
		self.x_bc0_train = x_bc0_train   # Dirichlet boundary
		self.u_bc0_train = u_bc0_train   # Dirichlet boundary
		self.t_bc1_train = t_bc1_train   # Neumann boundary
		self.x_bc1_train = x_bc1_train   # Neumann boundary
		self.u_bc1_train = u_bc1_train   # Neumann boundary
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

	def dnn_initializer(
		self, 
	):
		dnn = 9999.
		return dnn

	def opt_algorithm(self, lr, opt, params):
		print(">>>>> opt_algorithm")
		print("         learning rate:", lr)
		print("         optimizer    :", opt)
		if opt == "SGD":
			optimizer = torch.optim.SGD(params, lr=lr)
		elif opt == "RMSprop":
			optimizer = torch.optim.RMSprop(params, lr=lr)
		elif opt == "Adam":
			optimizer = torch.optim.Adam(params, lr=lr)
		elif opt == "Adamax":
			optimizer = torch.optim.Adamax(params, lr=lr)
		elif opt == "Nadam":
			optimizer = torch.optim.NAdam(params, lr=lr)
		else:
			raise NotImplementedError(">>>>> opt_algorithm")
		return optimizer

	def lr_schedule(
		self, 
		gamma, sch, opt
	):
		raise NotImplementedError
		if sch == "Exponential":
			scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma, last_epoch=- 1, verbose=False)
		elif sch == "Cosine":
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max, eta_min=0, last_epoch=- 1)
		elif sch == "CosineWR":
			scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0, T_mult=1, eta_min=0, last_epoch=- 1, verbose=False)
		return scheduler

	def forward(self, x):
		# if torch.is_tensor(x) == True:
		# 	pass
		# else:
		# 	x = torch.from_numpy(x)
		if self.f_scl == None or "linear":
			z = x
		elif self.f_scl == "minmax":
			z = 2. * () - 1.
		elif self.f_scl == "mean":
			raise NotImplementedError

		z = x
		y_hat = z
		return y_hat

	def compute_pde(self, x):
		u_hat = self.forward(x)
		u_x_hat = torch.autograd.grad(
			outputs=u_hat, inputs=x, 
			grad_outputs=None, 
			retain_graph=None, 
			create_graph=True,    # True for higher order derivatives
			only_inputs=True, 
			allow_unused=False, 
			is_grads_batched=False
		)
		u_xx_hat = torch.autograd.grad(
			outputs=u_x_hat, inputs=x, 
			grad_outputs=None, 
			retain_graph=None, 
			create_graph=False,    # 2nd order is sufficient
			only_inputs=True, 
			allow_unused=False, 
			is_grads_batched=False
		)
		gv_hat = 6.
		return gv_hat

	def train(self, n_epoch, b_size, es_pat):
		print(">>>>> train")
		print("         n_epoch:", n_epoch)
		print("         b_size :", b_size)
		print("         es_pat :", es_pat)



	def infer(self, x):
		print(">>>>> train")
		u_hat = self.forward(x)
		return u_hat

