import numpy as np

class MatMul:
	def __init__(self, W):
		self.params = [W]
		self.grads = [np.zeros_like(W)]
		self.x = None

	def forward(self, x):
		W, = self.params
		self.x = x
		return np.dot(x, W)

	def backward(self, dout):
		W, = self.params
		dx = np.dot(dout, W.transport())
		dW = np.dot(self.x.T, dout)
		grads[0] = dW.copy()
		return dx

# 他の章でも使うようなので ch1/forward_net.py からコピーしてbackwardを実装
class Sigmoid:
	def __init__(self):
		self.params = []
		self.grads = []

	def forward(self, x):
		self.out = 1 / (1 + np.exp(-x))
		return self.out

	def backward(self, dout):
		dx = dout * (1 - self.out) * self.out
		return dx


class Affine:
	def __init__(self, W, b):
		self.mm = MatMul(W)  # MatMulを使って実装してみる
		self.params = [W, b]
		self.grads = [
		  self.mm.grads[0],  # modelが初期化した直後のgradsを参照するため、MatMulのgradsを参照するようにする
		  np.zeros_like(b),
		]

	def forward(self, x):
		_, b = self.params
		out = self.mm.forward(x) + b
		return out

	def backward(self, dout):
		dx = self.mm.backward(dout)
		db = np.sum(dout, axis = 0)
		# self.grads[0] はmm.backwardで更新される
		self.grads[1] = db.copy()
		return dx

class TwoLayerNet:
	def __init__(self, input_size, hidden_size, output_size):
		I, H, O = input_size, hidden_size, output_size

		# 重みとバイアスの初期化
		W1 = np.random.randn(I, H)
		b1 = np.random.randn(H)
		W2 = np.random.randn(H, O)
		b2 = np.random.randn(O)

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
