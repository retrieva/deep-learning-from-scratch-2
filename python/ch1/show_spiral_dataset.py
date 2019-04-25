import sys
sys.path.append('..')
from upstream.dataset import spiral  # 教科書のリポジトリのを使う

import matplotlib.pyplot as plt

x, t = spiral.load_data()

print('x', x.shape)
print('t', t.shape)

N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):  # python3向け。python2ではrangeの挙動が違うので注意。
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=10, marker=markers[i])
plt.show()
