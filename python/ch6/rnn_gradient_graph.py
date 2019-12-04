import numpy as np
import matplotlib.pyplot as plt

N = 2 # ミニバッチサイズ
H = 3 # 隠れ状態ベクトルの次元数
T = 20 # 時系列データの長さ

dh = np.ones((N, H))
np.random.seed(3) # 再現性のため乱数のシードを固定
Wh = np.random.randn(H, H)
#Wh = np.random.randn(H, H) * 0.5


max_norm = 5.0

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


norm_list = []
for t in range(T):
    dh = np.dot(dh, Wh.T)
    clip_grads(dh, max_norm)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)

plt.plot(norm_list)
plt.show()
