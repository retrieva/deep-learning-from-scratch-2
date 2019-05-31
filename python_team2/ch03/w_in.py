import numpy as np
# 3.1.3 p.99

c = np.array([[1, 0, 0, 0, 0, 0, 0]]) # 入力 "you"
W = np.random.randn(7, 3)             # 重み
h = np.dot(c, W)                      # 中間ノード
print(h)
print(W)
