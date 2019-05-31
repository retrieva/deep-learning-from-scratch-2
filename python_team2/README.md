# このディレクトリについて

 * 「ゼロから作るDeep Learning 2 自然言語編 読書会」のサンプルをPythonで実装してみるチームの作業ディレクトリです。

## メモ
 * PythonはPython 3.7.3
 * 依存ライブラリインストールにはpipenvを使っている

## ToDo
 - [X] ディレクトリを作る
 - [X] numpyのインストール
 - [X] 何を作るか確認する
   - [X] シグモイド関数(P.13)
   - [X] シグモイド関数をクラス化(P.15)
   - [X] Affineレイヤの実装(P.15)
   - [X] TwoLayerNetの作成(P.17)
   - [ ] 1.3 ニューラルネットの学習
     - [X] 1.3.4 計算グラフ
       - [X] MatMul
     - [X] 1.3.5 勾配の導出と逆伝播の実装
       - [X] Sigmoid
       - [X] Affine
       - [X] SoftmaxWithLoss
     - [X] 1.3.6 重みの更新
   - [X] 1.4 ニューラルネットワークで問題を解く
     - [X] 1.4.1 スパイラルデータセット
     - [X] 1.4.2 ニューラルネットワークの実装
     - [X] 1.4.3 学習用のソースコード
     - [X] 1.4.4 Trainerクラス
 - [ ] 3章 Word2vec
    - [X] 3.1 推論ベースの手法とニューラルネットワーク
       - [X] 3.1.3 全結合層による計算
       - [X] 3.1.3 MatMulによる計算
    - [ ] 3.2 シンプルなword2vec
       - [X] 3.2.1 推論処理


## 開発環境の構築

```
# 公式のソースコードをbookでチェックアウト
$ git clone https://github.com/oreilly-japan/deep-learning-from-scratch-2.git book
```

```
# for mac
$ brew install pipenv
$ cd {CURRENT_DIRECTORY}
$ pipenv sync
$ pipenv shell
```

### 参考

- [Python環境構築ベストプラクティス2019 - ばいおいんふぉっぽいの！](https://www.natsukium.com/blog/2019-02-18/python/)
- [Pipenv で起きる Matplotlib まわりのエラー - Qiita](https://qiita.com/utahkaA/items/ad9aa825832c5909575a)
    - mac でpipenv + matplotlib で使う場合は次の設定が必要
