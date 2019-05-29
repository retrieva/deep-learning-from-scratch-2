# -*- coding: utf-8

# 参考
# scikit-learn - matplotlib を使って分類問題の決定境界を描画する - Pynote
# http://pynote.hatenablog.com/entry/sklearn-plot-decision-boundary
# 機械学習の分類結果を可視化！決定境界 - 見習いデータサイエンティストの隠れ家
# http://www.dskomei.com/entry/2018/03/04/125249

import numpy as np
import matplotlib.pyplot as plt



def plotResults(model, loss_list, x):
    # 学習経過をプロット
    plt.subplot(1,2,1)
    plt.plot(loss_list)

    # 決定境界をプロット
    plt.subplot(1,2,2)
    plotDecisionBoundary(model, x)


# 決定境界のプロット
def plotDecisionBoundary(model, x):
    # グリッドの座標を作る
    x_min, x_max = x[:, 0].min(), x[:, 0].max()
    y_min, y_max = x[:, 1].min(), x[:, 1].max()
    x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                 np.arange(y_min, y_max, 0.01))
    grid = np.array([x_mesh.ravel(), y_mesh.ravel()]).T

    # グリッドの推論結果を集める
    pred = model.predict(grid)
    z = np.array(x_mesh.ravel())
    for i in range(len(pred)):
        z[i] = pred[i].argmax()
    z = z.reshape(x_mesh.shape)

    # 等高線描画
    plt.contourf(x_mesh, y_mesh, z, alpha=0.3)
    plt.xlim(x_mesh.min(), x_mesh.max())
    plt.ylim(y_mesh.min(), y_mesh.max())

    # データ点のプロット
    N = 100
    CLS_NUM = 3
    markers = ['o', 'x', '^']
    for i in range(CLS_NUM):
        plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
