
import numpy as np

class Sigmoid:
	def __init__(self):
		pass  # 何もしない

	def forward(self, x):
		return 1 / (1 + np.exp(-x))

	def backward(self):
		pass


sig = Sigmoid()

print(sig.forward(3))
print(sig.forward(0))
print(sig.forward(-3))

print(sig.forward(3) + sig.forward(-3))

# numpyっぽい書き方
print(sig.forward(np.array([3,0,-3])))

