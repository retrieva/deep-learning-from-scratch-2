require "numo/narray"
require "mat_mul"
require "softmax_with_loss"
require "embedding"

class SimpleCBow
  attr_reader :params, :grads, :word_vecs

  def initialize(vocab_size, hidden_size)
    v, h = vocab_size, hidden_size

    # 重みの初期化
    w_in = 0.01 * Numo::DFloat.new(v, h).rand
    w_out = 0.01 * Numo::DFloat.new(h, v).rand

    # レイヤの生成
    @in_layer0 = Embedding.new(w_in)
    @in_layer1 = Embedding.new(w_in)
    # @in_layer0 = MatMul.new(w_in)
    # @in_layer1 = MatMul.new(w_in)
    @out_layer = MatMul.new(w_out)
    @loss_layer = SoftmaxWithLoss.new

    # すべての重みと勾配をリストにまとめる
    layers = [@in_layer0, @in_layer1, @out_layer]
    @params, @grads = layers.reduce([[], []]) do |acc, layer|
      [acc[0] + layer.params, acc[1] + layer.grads]
    end

    # メンバ変数に単語の分散表現を設定
    @word_vecs = w_in
  end

  def forward(contexts, target)
    h0 = @in_layer0.forward(contexts[true, 0])
    h1 = @in_layer1.forward(contexts[true, 1])
    h = (h0 + h1) * 0.5
    score = @out_layer.forward(h)
    loss = @loss_layer.forward(score, target)
    return loss
  end

  def backward(dout=1)
    ds = @loss_layer.backward(dout)
    da = @out_layer.backward(ds)
    da *= 0.5
    @in_layer1.backward(da)
    @in_layer0.backward(da)
    nil
  end
end
