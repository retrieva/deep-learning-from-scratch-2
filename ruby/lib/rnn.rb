require 'numo/narray'

class Rnn

  attr_accessor :params, :grads, :cache

  def initialize(wx, wh, b)
    @params = [wx, wh, b]
    @grads = [wx.new_zeros, wh.new_zeros, b.new_zeros]
  end

  def forward(x, h_prev)
    wx, wh, b = @params
    t = h_prev.dot(wh) + x.dot(wx) + b
    h_next = Numo::NMath::tanh(t)

    @cache = [x, h_prev, h_next]

    h_next
  end

  def backward(dh_next)
    wx, wh, _b = @params
    x, h_prev, h_next = @cache

    dt = dh_next * (1 - h_next**2)
    db = dt.sum(axis: 0)
    dwh = h_prev.transpose.dot(dt)
    dh_prev = dt.dot(wh.transpose)
    dwx = x.transpose.dot(dt)
    dx = dt.dot(wx.transpose)

    @grads[0].store(dwx)
    @grads[1].store(dwh)
    @grads[2].store(db)

    [dx, dh_prev]
  end
end