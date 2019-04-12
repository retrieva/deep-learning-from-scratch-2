#!/usr/bin/env python

import numpy as np


def load_data(seed=1984):
    np.random.seed(seed)
    N = 100     # number of sample each class
    DIM = 2     # number of data elements
    CLS_NUM = 3 # number of class

    x = np.zeros((N * CLS_NUM, DIM))
    t = np.zeros((N * CLS_NUM, CLS_NUM), dtype=np.int)

    for j in range(CLS_NUM):
        for i in range(N):
            rate = i / N
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2

            ix = N * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)], dtype=np.float32).flatten()
            t[ix, j] = 1

    return x, t
