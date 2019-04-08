#!/usr/bin/env python

import numpy as np
from optimizer import SGD
import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet


if __name__ == '__main__':
    # 1. hyper parameter settings
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    learning_rate = 1.0

    # 2. load data and generate model and optimizer
    x, t = spiral.load_data()
    model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
    optimizer = SGD(lr=learning_rate)

    # variables for learning
    data_size = len(x)
    max_iters = data_size // batch_size
    total_loss = 0
    loss_count = 0
    loss_list = []

    for epoch in range(max_epoch):
        # 3. shuffle data
        idx = np.random.permutation(data_size)
        x = x[idx]
        t = t[idx]

        for iters in range(max_iters):
            batch_x = x[iters * batch_size: (iters + 1) * batch_size]
            batch_t = t[iters * batch_size: (iters + 1) * batch_size]

            # 4. process grads and update parameters
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)

            total_loss += loss
            loss_count += 1

            # 5. output learning result
            if (iters + 1) % 10 == 0:
                avg_loss = total_loss / loss_count
                print('| epoch %d | iter %d / %d | loss %.2f'
                      % (epoch + 1, iters + 1, max_iters, avg_loss))
                loss_list.append(avg_loss)
                total_loss, loss_count = 0, 0

    # plot learning result
    plt.plot(np.arange(len(loss_list)), loss_list, label='train')
    plt.xlabel('iterations (x10)')
    plt.ylabel('loss')
    plt.show()

    # plot boundary
    h = 0.001
    x_min, x_max = np.min(x[:, 0]) - .1, np.max(x[:, 0]) + .1
    y_min, y_max = np.min(x[:, 1]) - .1, np.max(x[:, 1]) + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    score = model.predict(X)
    predict_cls = np.argmax(score, axis=1)
    Z = predict_cls.reshape(xx.shape)
    plt.contourf(xx, yy, Z)
    plt.axis('off')

    # plot data points
    x, t = spiral.load_data()
    N = 100
    CLS_NUM = 3
    markers = ['o', 'x', '^']
    for i in range(CLS_NUM):
        plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
    plt.show()
