# -*- coding: utf-8

import sys
sys.path.append('..')
import numpy as np
from common.optimizer import SGD
from book.dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
from plots import plotResults

# ハイパーパラメータの設定
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# データの読み込み
x, t = spiral.load_data()

def train(x, t):
    # 学習で使用する変数
    data_size = len(x)
    max_iters = data_size // batch_size
    total_loss = 0
    loss_count = 0
    loss_list = []

    for epoch in range(max_epoch):
        # データのシャッフル
        idx = np.random.permutation(data_size)
        x = x[idx]
        t = t[idx]

        for iters in range(max_iters):
            batch_x = x[iters*batch_size:(iters+1)*batch_size]
            batch_t = t[iters*batch_size:(iters+1)*batch_size]

            # 勾配を求めパラメターを更新
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)

            total_loss += loss
            loss_count += 1

            # 定期的に学習経過を出力
            if (iters+1) % 10 == 0:
                avg_loss = total_loss / loss_count
                print('| epoch %d |  iter %d / %d | loss %.2f'
                      % (epoch + 1, iters + 1, max_iters, avg_loss))
                loss_list.append(avg_loss)
                total_loss, loss_count = 0, 0

    return loss_list




if __name__ == '__main__':
    # 学習試行1

    # モデルとオプティマイザの生成
    optimizer = SGD(lr=learning_rate)
    model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)

    loss_list = train(x, t)
    plt.figure(figsize=(10,4))
    plotResults(model, loss_list, x)


    ## 学習試行2
    #
    ## モデルとオプティマイザの生成
    #learning_rate = 2.0
    #optimizer = SGD(lr=learning_rate)
    #model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
    #
    #loss_list = train(x, t)
    #plt.figure(figsize=(10,4))
    #plotResults(model, loss_list, x)


    # グラフ表示
    plt.show()
