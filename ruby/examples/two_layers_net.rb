require_relative '../lib/affine'
require_relative '../lib/sigmoid'
require_relative '../lib/softmax_with_loss'

class TwoLayersNet
  attr_reader :layers, :loss_layer
  attr_accessor :params, :grads

  def initialize(input_size:, hidden_size:, output_size:)
    w1 = 0.01 * Numo::SFloat.new(input_size, hidden_size).rand
    b1 = Numo::SFloat.zeros(hidden_size)
    w2 = 0.01 * Numo::SFloat.new(hidden_size, output_size).rand
    b2 = Numo::SFloat.zeros(output_size)

    @layers = [
      Affine.new(w1, b1),
      Sigmoid.new,
      Affine.new(w2, b2),
    ]
    @loss_layer = SoftmaxWithLoss.new

    @params, @grads  = @layers.reduce([[], []]) do |acc, layer|
      acc[0] += layer.params
      acc[1] += layer.grads
      acc
    end
  end

  def predict(x)
    @layers.each do |layer|
      x = layer.forward(x)
    end
    x
  end

  def forward(x, t)
    score = predict(x)
    @loss_layer.forward(score, t)
  end

  def backward(dout = 1)
    dout = @loss_layer.backward(dout)
    @layers.reverse.each do |layer|
      dout = layer.backward(dout)
    end
    dout
  end
end
