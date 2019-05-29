import numpy as np

class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr  # 学習率

    def update(self, prams, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
