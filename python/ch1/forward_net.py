
import numpy as np

class Sigmoid:
	def __init__(self):
		self.params = []

	def forward(self, x):
		return 1 / (1 + np.exp(-x))

	def backward(self):
		pass



class Affine:
	def __init__(self, W, b):
		self.params = [W, b]

	def forward(self, x):
		W, b = self.params
		out = np.dot(x, W) + b
		return out

	def backward(self):
		pass



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


if __name__ == '__main__':
	sig = Sigmoid()
	
	print(sig.forward(3))
	print(sig.forward(0))
	print(sig.forward(-3))
	
	print(sig.forward(3) + sig.forward(-3))
	
	# numpyっぽい書き方
	print(sig.forward(np.array([3,0,-3])))

	# text p18
	x = np.random.randn(10, 2)
	model = TwoLayerNet(2, 4, 3)
	s = model.predict(x)
	print(s)


