
import numpy as np

class Affine:
	def __init__(self, W, b):
		self.params = [W, b]

	def forward(self, x):
		W, b = self.params
		out = np.dot(x, W) + b
		return out

