"""
********************************************************************************
make reference data
********************************************************************************
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # computational domain
    tmin, tmax = 0., 5.
    xmin, xmax = 0., 1.

    # parameters
    c  = 1.
    dx = 1e-2
    dt = 5e-3
    dt_star = (.5 * dx ** 2 / c ** 2) ** .5
    if dt < dt_star:
        print("CFL satisfied")
        print("dt: %.6e, dt_star: %.6e" % (dt, dt_star))
    else:
        print("CFL NOT satisfied")
        dt = .9 * dt_star
        print("dt: %.6e, dt_star: %.6e" % (dt, dt_star))

    t = np.arange(tmin, tmax, dt)
    x = np.arange(xmin, xmax, dx)
    nt = len(t)   # == int((tmax - tmin) / dt)
    nx = len(x)   # == int((xmax - xmin) / dx)
    beta = c ** 2 * dt ** 2 / dx ** 2

    # df
    df_t = pd.DataFrame(data=t.reshape(-1, 1), columns=["t"])
    df_x = pd.DataFrame(data=x.reshape(-1, 1), columns=["x"])
    df_u = pd.DataFrame()

    u = np.ones(shape=(nx))
    u_ic = 0.
    u *= u_ic
    u  = np.sin(np.pi / 2 * x)

    # boundary condition 
    u[0]  = u_ic
    u[-1] = u[-2]

    # time integration, u - step n+1 
    v = np.copy(u)   #  v - step n
    w = np.copy(u)   #  w - step n-1
    for n in range(nt):
        u[1:-1] = 2. * v[1:-1] - w[1:-1] \
                    + beta * (v[0:-2] - 2. * v[1:-1] + v[2:])
        u[0]  = u_ic
        u[-1] = u[-2]
        w = np.copy(v)
        v = np.copy(u)
        df_u[str(n)] = u   # output data

        if n % 100 == 0:
            plt.plot(x, u)
            plt.xlim(-0.2, 1.2)
            plt.ylim(-1.2, 1.2)
            plt.grid(alpha=.5)
            plt.savefig("../input/" + str(n))
            plt.clf()
            plt.close()

    df_t.to_csv("../input/t.csv")
    df_x.to_csv("../input/x.csv")
    df_u.to_csv("../input/u.csv")

if __name__ == "__main__":
    main()

