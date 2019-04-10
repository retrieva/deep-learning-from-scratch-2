
import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

print(sigmoid(3))
print(sigmoid(0))
print(sigmoid(-3))

print(sigmoid(3) + sigmoid(-3))

# numpyっぽい書き方
print(sigmoid(np.array([3,0,-3])))

# P.13の例
x  = np.random.randn(10, 2)
W1 = np.random.randn(2, 4)
b1 = np.random.randn(4)
W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

h = np.dot(x, W1) + b1  # これで1層の計算
a = sigmoid(h)
s = np.dot(a, W2) + b2

print("h=", h)
print("a=", a)
print("s=", s)

print(h.shape)
print(a.shape)
print(s.shape)

