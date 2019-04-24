import cupy as cp

class MatMul:

    def __init__(self, W):
        self.params = [W]
        self.grads = [cp.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = cp.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = cp.dot(dout, W.T)
        dW = cp.dot(x.T, dout)
        self.grads[0][...] = dW
        return dx

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.x = None

    def forward(self, x):
        out = 1. / (1.0 + cp.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [cp.zeros_like(W), cp.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = cp.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W,b = self.params
        dx = cp.dot(dout, W.T)
        dW = cp.dot(self.x.T, dout)
        db = cp.sum(dout, axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        if x.ndim == 2: # ミニバッチ使用時
            x = x - x.max(axis=1, keepdims=True)
            x = cp.exp(x)
            y = x / x.sum(axis=1, keepdims=True)
        elif x.ndim == 1:
            x = x - cp.max(x)
            y = cp.exp(x) / cp.sum(cp.exp(x))

        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        loss = - 1.0 * cp.sum(t * cp.log(y[cp.arange(batch_size), t] + 1e-7)) / batch_size
        self.y = y
        self.t = t
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        dx[cp.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size
        return dx
