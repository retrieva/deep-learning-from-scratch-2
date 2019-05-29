import sys
sys.path.append('..')
import numpy as np
from common.layers import Affine, Sigmoid  # Affine, Sigmoidは自前実装を使う
from upstream.common.layers import SoftmaxWithLoss  # SoftmaxWithLossは教科書のを使う

class TwoLayerNet:
	def __init__(self, input_size, hidden_size, output_size):
		I, H, O = input_size, hidden_size, output_size
		W1 = 0.01 * np.random.randn(I, H)
		b1 = np.zeros(H)
		W2 = 0.01 * np.random.randn(H, O)
		b2 = np.zeros(O)

		self.layers = [
			Affine(W1, b1),
			Sigmoid(),
			Affine(W2, b2),
		]
		self.loss_layer = SoftmaxWithLoss()

		self.params, self.grads = [], []
		for layer in self.layers:
			self.params += layer.params
			self.grads += layer.grads

	def predict(self, x):
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def forward(self, x, t):
		score = self.predict(x)
		loss = self.loss_layer.forward(score, t)
		return loss

	def backward(self, dout=1):
		dout = self.loss_layer.backward(dout)
		for layer in reversed(self.layers):
			dout = layer.backward(dout)
		return dout

