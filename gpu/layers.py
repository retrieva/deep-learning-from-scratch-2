import cupy as np


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
        dW = np.dot(x.T, dout)
        self.grads[0][...] = dW
        return dx

class Sigmoid:
    def __init__(self):
        self.params, self.grad = [], []
        self.x = None

    def forward(self, x):
        out = 1. / (1.0 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grad = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W,b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W,b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

class SoftMaxWithLoss:
    def __init__(self):
        self.params, self.grad = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        y = np.exp(x) / sum(np.exp(x))
        self.y = y
        loss = np.sum(t * np.log(y))
        self.t = t
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        return dx
        
