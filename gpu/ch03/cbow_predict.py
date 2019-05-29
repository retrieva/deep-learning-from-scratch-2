import sys
sys.path.append('..')
import cupy as cp
from ch01.layers import MatMul

#サンプルのコンテキストデータ
c0 = cp.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = cp.array([[0, 0, 0, 1, 0, 0, 0]])

W_in = cp.random.randn(7, 3)
W_out = cp.random.randn(3, 7)

in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)

print(s)
