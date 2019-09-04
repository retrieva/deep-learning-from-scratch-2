# coding: utf-8
# from common.np import *
import numpy as np
import cupy as cp


def sigmoid(x):
    return 1 / (1 + cp.exp(-x))


def relu(x):
    return cp.maximum(0, x)


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = cp.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - cp.max(x)
        x = cp.exp(x) / cp.sum(cp.exp(x))

    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]

    return -cp.sum(cp.log(y[cp.arange(batch_size), t] + 1e-7)) / batch_size
