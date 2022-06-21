# PINN_Torch
This repository implements [PINN](https://doi.org/10.1016/j.jcp.2018.10.045) in [PyTorch](https://pytorch.org/) environment to solve Burgers equation and 1D wave equation. [Automatic differentiation](https://arxiv.org/abs/1502.05767), which is a generalization of [back-propagation](https://doi.org/10.1038/323533a0), is utilized to leverage the convenctional neural network architecture's representation power and to satisfy govering equations, initial, and boundary conditions. 

Please note this repository is not intended to reproduce the results of [PINN_TF2](https://github.com/ShotaDeguchi/PINN_TF2). 

## Examples
Burgers equation solution inferred by PINN (found in <code>./00_burgers/</code>):
<img src="./00_burgers/figures/infered_solution.svg">

## References
[1] Raissi, M., Perdikaris, P., Karniadakis, G.E.: Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, *Journal of Computational Physics*, Vol. 378, pp. 686-707, 2019. ([paper](https://doi.org/10.1016/j.jcp.2018.10.045))
<br>
[2] Baydin, A.G., Pearlmutter, B.A., Radul, A.A., Siskind, J.M.: Automatic Differentiation in Machine Learning: A Survey, *Journal of Machine Learning Research*, Vol. 18, No. 1, pp. 5595–5637, 2018. ([paper](https://arxiv.org/abs/1502.05767))
<br>
[3] Rumelhart, D., Hinton, G., Williams, R.: Learning representations by back-propagating errors, *Nature*, Vol. 323, pp. 533–536, 1986. ([paper](https://doi.org/10.1038/323533a0))

