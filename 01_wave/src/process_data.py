"""
********************************************************************************
process reference data
********************************************************************************
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data():
    t = pd.read_csv("../input/t.csv", index_col=0).to_numpy()
    x = pd.read_csv("../input/x.csv", index_col=0).to_numpy()
    u = pd.read_csv("../input/u.csv", index_col=0).to_numpy()
    return t, x, u

def visualize_data():
    t, x, u = load_data()
    t, x = np.meshgrid(t, x)

    plt.figure(figsize=(8, 4))
    plt.scatter(t, x, c=u, cmap="turbo", vmin=-1., vmax=1.)
    plt.colorbar()
    plt.xlim(0., 5.)
    plt.ylim(0., 1.)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("FDM")
    plt.savefig("../output/reference.png")

def ic_data():
    t, x, u = load_data()
    t, x = np.meshgrid(t, x)

    t_ic0 = t[:,0]
    x_ic0 = x[:,0]
    u_ic0 = u[:,0]

    t_ic1 = t[:,1]
    x_ic1 = x[:,1]
    u_ic1 = u[:,1]
    u_ic1 = (u_ic1 - u_ic0) / (t_ic1 - t_ic0)

    return \
        t_ic0, x_ic0, u_ic0, \
        t_ic1, x_ic1, u_ic1

def bc_data():
    t, x, u = load_data()
    t, x = np.meshgrid(t, x)

    t_bc0 = t[0,1:]
    x_bc0 = x[0,1:]
    u_bc0 = u[0,1:]

    t_bc1 = t[-1,1:]
    x_bc1 = x[-1,1:]
    u_bc1 = (u[-1,1:] - u[-2,1:]) / (x_bc1 - x_bc0)

    return \
        t_bc0, x_bc0, u_bc0, \
        t_bc1, x_bc1, u_bc1

def pde_data():
    t, x, u = load_data()
    t, x = np.meshgrid(t, x)

    t_pde = t[1:-1,1:]
    x_pde = x[1:-1,1:]
    # u_pde = u[1:-1,1:]

    plt.scatter(t_pde, x_pde)
    plt.show()

    return \
        t_pde, x_pde

def split_data(x, seed=1234):
    

    print(x)
    print(x.reshape(-1, 1))

    x1, x2 = train_test_split(x,  test_size=.4, random_state=seed)
    x2, x3 = train_test_split(x2, test_size=.5, random_state=seed)

    return x1, x2, x3

    # x1, x2, x3 = train_test_split

    # t = t.reshape(-1, 1)
    # x = x.reshape(-1, 1)
    # u = u.reshape(-1, 1)

    # return \
    #     t_train, x_train, u_train, 

    # t_ic0_train, x_ic0_train, u_ic0_train, 
    # t_ic1_train, x_ic1_train, u_ic1_train, 
    # t_bc0_train, x_bc0_train, u_bc0_train, 
    # t_bc1_train, x_bc1_train, u_bc1_train, 
    # t_pde_train, x_pde_train, 
    # t_ic0_val, x_ic0_val, u_ic0_val, 
    # t_ic1_val, x_ic1_val, u_ic1_val, 
    # t_bc0_val, x_bc0_val, u_bc0_val, 
    # t_bc1_val, x_bc1_val, u_bc1_val, 
    # t_pde_val, x_pde_val, 

if __name__ == "__main__":
    t_bc0, x_bc0, u_bc0, \
    t_bc1, x_bc1, u_bc1 = bc_data()

    x1, x2, x3 = split_data(t_bc0)
    # print(x1)
    # print(x2)
    # print(x3)

    # plt.scatter(x1, x1)
    # plt.scatter(x2, x2)
    # plt.scatter(x3, x3)
    # plt.show()





