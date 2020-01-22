# frozen_string_literal: true

require 'numo/narray'
require 'embedding'

class TimeEmbedding
  attr_accessor :params, :grads

  def initialize(w)
    @params = [w]
    @grads = [w.new_zeros]
  end

  def forward(idx)
    w = @params.first
    n, t = idx.shape
    @idx = idx
    out = Numo::SFloat.zeros(n, t, w.shape.last)
    @layers = []
    t.times do |ti|
      layer = Embedding.new(w)
      out[true, ti, true] = layer.forward(idx[true, ti])
      @layers << layer
    end
    out
  end

  def backward(dout)
    _n, t, _d = dout.shape
    w = @params.first

    grad = w.new_zeros

    (t - 1).downto(0) do |ti|
      layer = @layers[ti]
      layer.backward(dout[true, ti, true])
      grad.inplace + layer.grads[0]
    end

    @grads[0].store(grad)
    nil
  end
end

