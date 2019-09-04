import numpy as np
import cupy as cp
from common.functions import softmax, cross_entropy_error


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
        dW = cp.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


# 他の章でも使うようなので ch1/forward_net.py からコピーしてbackwardを実装
class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []

    def forward(self, x):
        self.out = 1 / (1 + cp.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * (1 - self.out) * self.out
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoidの出力
        self.t = None  # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + cp.exp(-x))

        self.loss = cross_entropy_error(cp.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoidの出力
        self.t = None  # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + cp.exp(-x))

        self.loss = cross_entropy_error(cp.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx




class Affine:
	def __init__(self, W, b):
		self.mm = MatMul(W)  # MatMulを使って実装してみる
		self.params = [W, b]
		self.grads = [
		  self.mm.grads[0],  # modelが初期化した直後のgradsを参照するため、MatMulのgradsを参照するようにする
		  cp.zeros_like(b),
		]

	def forward(self, x):
		_, b = self.params
		out = self.mm.forward(x) + b
		return out

	def backward(self, dout):
		dx = self.mm.backward(dout)
		db = cp.sum(dout, axis = 0)
		# self.grads[0] はmm.backwardで更新される
		self.grads[1] = db.copy()
		return dx

class TwoLayerNet:
	def __init__(self, input_size, hidden_size, output_size):
		I, H, O = input_size, hidden_size, output_size

		# 重みとバイアスの初期化
		W1 = cp.random.randn(I, H)
		b1 = cp.random.randn(H)
		W2 = cp.random.randn(H, O)
		b2 = cp.random.randn(O)

		# レイヤの生成
		self.layers = [
			Affine(W1, b1),
			Sigmoid(),
			Affine(W2, b2)
		]

		# すべての重みをリストにまとめる
		self.params = []
		for layer in self.layers:
			self.params += layer.params

	def predict(self, x):
		for layer in self.layers:
			x = layer.forward(x)
		return x


# FROM https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/master/common/layers.py
class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = cp.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル

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
        dx[cp.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx

    
class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [cp.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        cp.cupyx.scatter_add(dW, self.idx, dout)
        return None
