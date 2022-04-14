# Burgers equation

Implements [PINN(s)](https://doi.org/10.1016/j.jcp.2018.10.045) in PyTorch to solve Burgers equation. 

## Solution
The following is the PINN-derived solution under a specific initial and boundary conditions (IC: sine wave, BC: zero Dirichlet). 
<img src="./figures/PINN_Burgers.png">

## Dependencies
Tested with
|Package|Version|
|:---:|:---:|
|python|3.8.10|
|numpy|1.22.1|
|scipy|1.8.0|
|torch|1.7.1 + cu110|

## Reference
[1] Raissi, M., Perdikaris, P., Karniadakis, G.E.: Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, *Journal of Computational Physics*, Vol. 378, pp. 686-707, 2019. 
<br>
[2] Baydin, A.G., Pearlmutter, B.A., Radul, A.A., Siskind, J.M.: Automatic Differentiation in Machine Learning: A Survey, *Journal of Machine Learning Research*, Vol. 18, No. 1, pp. 5595–5637, 2018. 
<br>
[3] Rumelhart, D., Hinton, G., Williams, R.: Learning representations by back-propagating errors, *Nature*, Vol. 323, pp. 533–536, 1986. 
