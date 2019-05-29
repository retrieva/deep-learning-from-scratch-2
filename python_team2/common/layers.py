# -*- coding: utf-8

import numpy as np

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

# https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py
class Relu:
    def __init__(self):
        self.mask = None
        self.params, self.grads = [], []

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

class AffineMM:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.MM = MatMul(W)

    def forward(self, x):
        b = self.params[1]
        out = self.MM.forward(x) + b
        return out

    def backward(self, dout):
        b = self.params[1]
        dx = self.MM.backward(dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = self.MM.grads[0]
        self.grads[1][...] = db
        return dx


# https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/master/common/functions.py
# からパチった
def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/master/common/layers.py
class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

         # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx




if __name__ == '__main__':
    print('MatMul 形状チェック')

    W = np.random.randn(3, 4)
    mm = MatMul(W)
    x = np.random.randn(1, 3)
    out = mm.forward(x)
    print('mm.forward().shape', out.shape)
    grad = mm.backward(out)
    print('mm.backward().shape', grad.shape)


    print("Affine, AffineMM 実装チェック")

    W = np.random.randn(3, 2)
    b = np.random.randn(2)
    aff = Affine(W, b)
    amm = AffineMM(W, b)

    x = np.random.randn(10, 3)
    out1 = aff.forward(x)
    out2 = amm.forward(x)
    print("out equal", (out1 == out2).all())

    grad1 = aff.backward(out1)
    grad2 = amm.backward(out2)
    print("grad equal", (grad1 == grad2).all())
