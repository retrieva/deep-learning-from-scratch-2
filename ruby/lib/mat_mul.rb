require 'numo/narray'

class MatMul
  attr_accessor :params, :grads, :x
  def initialize(w)
    @params = [w]
    @grads = [w.new_zeros]
    @x = nil
  end

  def forward(x)
    w = @params.first
    @x = x
    x.dot(w)
  end

  def backward(dout)
    w = @params.first
    dx = dout.dot(w.transpose)
    dw = @x.transpose.dot(dout)
    @grads[0][] = dw
    dx
  end
end
