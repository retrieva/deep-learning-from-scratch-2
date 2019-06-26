require "numo/narray"

class Affine
  attr_accessor :params, :grads, :x

  def initialize(weight, bias)
    @params = [weight, bias]
    @grads  = [weight.new_zeros, bias.new_zeros]
    @x = nil
  end

  def forward(x)
    weight, bias = @params
    @x = x
    x.dot(weight) + bias
  end

  def backward(dout)
    weight, _ = @params
    dx = dout.dot(weight.transpose)
    dW = @x.transpose.dot(dout)
    db = dout.sum(axis: 0)

    @grads[0].store dW
    @grads[1].store db
    dx
  end
end
