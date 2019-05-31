# p.100 3.1.3

import sys
sys.path.append('../book')
import numpy as np
from common.layers import MatMul

c = np.array([[1, 0, 0, 0, 0, 0, 0]]) # 入力 "you"
W = np.random.randn(7, 3)             # 重み
layer = MatMul(W)
h = layer.forward(c)
print(h)
print(W)
