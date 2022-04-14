"""
********************************************************************************
initializers
********************************************************************************
"""

import numpy as np
import torch

def weight_init(shape, depth, name):
    if name == "Glorot":
        std = np.sqrt(2 / (shape[0] + shape[1]))
    elif name == "He":
        std = np.sqrt(2 / shape[0])
    elif name == "LeCun":
        std = np.sqrt(1 / shape[0])
    else:
        raise NotImplementedError(">>>>> weight_init")
    weight = tf.Variable(
        tf.random.truncated_normal(shape = [shape[0], shape[1]], \
        mean = 0., stddev = std, dtype = self.dat_typ), \
        dtype = self.dat_typ, name = "w" + str(depth)
        )
    return weight

def bias_init(shape, depth, name):
    in_dim  = shape[0]
    out_dim = shape[1]
    if name == "zeros":
        bias = tf.Variable(
            tf.zeros(shape = [in_dim, out_dim], dtype = self.dat_typ), \
            dtype = self.dat_typ, name = "b" + str(depth)
            )
    elif name == "ones":
        bias = tf.Variable(
            tf.ones(shape = [in_dim, out_dim], dtype = self.dat_typ), \
            dtype = self.dat_typ, name = "b" + str(depth)
            )
    else:
        raise NotImplementedError(">>>>> bias_init")
    return bias
